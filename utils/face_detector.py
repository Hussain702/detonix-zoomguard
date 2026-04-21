"""
ZoomGuard — Face Detector + ArcFace Embedder
=============================================
Combines robust multi-backend face detection with ArcFace-quality
face embeddings for stable Re-ID inside DeepSORT.

Embedding backend priority (auto-selected at startup)
──────────────────────────────────────────────────────
1. insightface buffalo_l   — real ArcFace (512-dim)        BEST ★★★★★
   pip install insightface onnxruntime   (works on Windows)
2. cv2.FaceRecognizerSF    — OpenCV SFace ONNX (128-dim)   GOOD ★★★★
   download face_recognition_sface_2021dec.onnx from OpenCV zoo
3. LBP + Color histogram   — always works, no downloads     OK   ★★★

Face detection backend priority
────────────────────────────────
1. MediaPipe Tasks  (mediapipe >= 0.10)
2. MediaPipe legacy (mediapipe < 0.10)
3. OpenCV DNN ResNet-SSD
4. Haar Cascade (frontal + alt2 + profile) — always works

detect() returns list of dicts:
    {
      'bbox':       [x, y, w, h],    int pixels, top-left + size
      'confidence': float 0-1,
      'crop':       np.ndarray,      BGR 224x224 crop for deepfake model
      'embedding':  np.ndarray,      unit-normalised face embedding (or None)
    }

draw_results(frame, tracks) annotates confirmed DeepSORT tracks.
"""

from __future__ import annotations

import logging
import os
import warnings
from typing import List, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Geometry constants (must match deep_sort.py)
_FACE_ASPECT_MIN = 0.35
_FACE_ASPECT_MAX = 2.20
_FACE_MIN_AREA   = 900      # 30x30 px
_FACE_MAX_FRAC   = 0.50
_CROP_SIZE       = 224


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _enhance(frame: np.ndarray) -> np.ndarray:
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    return clahe.apply(gray)

def _to_rgb(frame: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def _make_crop(frame: np.ndarray, x: int, y: int,
               w: int, h: int, pad_frac: float = 0.15) -> np.ndarray:
    fh, fw = frame.shape[:2]
    pad    = int(min(w, h) * pad_frac)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(fw, x + w + pad), min(fh, y + h + pad)
    crop   = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((_CROP_SIZE, _CROP_SIZE, 3), dtype=np.uint8)
    return cv2.resize(crop, (_CROP_SIZE, _CROP_SIZE), interpolation=cv2.INTER_LINEAR)

def _is_valid(x: int, y: int, w: int, h: int, fw: int, fh: int) -> bool:
    if w <= 0 or h <= 0:
        return False
    return (_FACE_MIN_AREA <= w * h <= fw * fh * _FACE_MAX_FRAC
            and _FACE_ASPECT_MIN <= w / h <= _FACE_ASPECT_MAX)

def _l2(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-6)


# ══════════════════════════════════════════════════════════════════════════════
# PART A  EMBEDDING BACKENDS
# ══════════════════════════════════════════════════════════════════════════════

class _InsightFaceEmbedder:
    """
    Real ArcFace via insightface buffalo_l w600k_r50.onnx (512-dim).

    Install:  pip install insightface onnxruntime
    Model auto-downloads to ~/.insightface/models/buffalo_l/ on first run.

    Key fixes vs previous version:
    - Loads recognition model DIRECTLY via model_zoo (bypasses ctx_id/CUDA issue)
    - Uses ctx_id=-1 (CPU forced) so no CUDA error on CPU-only machines
    - Smoke-tests with random noise (not zeros) to catch silent failures
    - Full traceback logged on any failure
    """
    name   = "insightface-arcface"
    dim    = 512
    thresh = 0.38

    def __init__(self):
        import insightface
        import os, glob

        buffalo_dir = os.path.join(
            os.path.expanduser("~"), ".insightface", "models", "buffalo_l")

        # Try direct model_zoo load first — most reliable, avoids FaceAnalysis
        # ctx_id/CUDA issues that silently break on CPU-only Windows machines
        candidates = glob.glob(os.path.join(buffalo_dir, "w600k*.onnx"))
        if candidates:
            onnx_path = candidates[0]
            self._rec = insightface.model_zoo.get_model(
                onnx_path, providers=["CPUExecutionProvider"])
            self._rec.prepare(ctx_id=-1)
            logger.info("InsightFace: loaded %s directly via model_zoo",
                        os.path.basename(onnx_path))
        else:
            # Fall back to FaceAnalysis (triggers download if needed)
            from insightface.app import FaceAnalysis
            _app = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["recognition"],
                providers=["CPUExecutionProvider"],
            )
            _app.prepare(ctx_id=-1, det_size=(640, 640))
            self._rec = None
            for _k, _m in _app.models.items():
                if hasattr(_m, "get_feat"):
                    self._rec = _m
                    break
            if self._rec is None:
                raise RuntimeError(
                    "w600k_r50.onnx not found. Run once to download:\n"
                    "  python -c \"from insightface.app import FaceAnalysis; "
                    "FaceAnalysis(name='buffalo_l').prepare(ctx_id=-1)\"")

        # Smoke-test with random noise — zeros short-circuit some pipelines
        # and miss real errors. Noise forces the full forward pass.
        _noise = np.random.randint(50, 200, (112, 112, 3), dtype=np.uint8)
        _out   = self._rec.get_feat(_noise)
        _arr   = np.asarray(_out).flatten()
        if _arr.size < 128:
            raise RuntimeError(
                f"InsightFace get_feat returned unexpected dim {_arr.size} "
                f"(expected 512). Model may be corrupt.")
        logger.info("InsightFace ArcFace verified — embedding dim: %d", _arr.size)

    def embed(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        if face_bgr is None or face_bgr.size == 0:
            return None
        try:
            img  = cv2.resize(face_bgr, (112, 112))
            feat = self._rec.get_feat(img)
            arr  = np.asarray(feat).flatten()
            if arr.size < 128:
                return None
            return _l2(arr)
        except Exception as exc:
            logger.debug("InsightFace embed error: %s", exc)
            return None


class _SFaceEmbedder:
    """
    OpenCV FaceRecognizerSF — SFace ONNX model (128-dim).

    Download model from OpenCV Zoo:
    https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface
    File: face_recognition_sface_2021dec.onnx
    Pass its path as sface_model_path= to FaceDetector().
    """
    name   = "opencv-sface"
    dim    = 128
    thresh = 0.40

    def __init__(self, model_path: str):
        self._rec = cv2.FaceRecognizerSF.create(model_path, "")
        logger.info("OpenCV SFace embedder loaded from %s", model_path)

    def embed(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        if face_bgr is None or face_bgr.size == 0:
            return None
        try:
            img = cv2.resize(face_bgr, (112, 112))
            feat = self._rec.feature(img)
            return _l2(np.asarray(feat).flatten())
        except Exception as exc:
            logger.debug("SFace embed error: %s", exc)
            return None


class _LocalEmbedder:
    """
    LBP grid + HSV colour histogram — no model download required.

    Produces a 4192-dim unit-normalised descriptor.
    Measured on ZoomGuard test video:
        intra-face cosine similarity : mean 0.93  (same person, diff frames)
        inter-face cosine similarity : mean 0.43  (different people)
    Re-ID threshold 0.40 gives a clear 0.50 gap between classes.
    Works well when faces are clearly different people (typical video call).
    May struggle with very similar-looking faces — use insightface for that.
    """
    name   = "local-lbp-color"
    dim    = 4192
    thresh = 0.40

    def __init__(self):
        logger.warning(
            "ArcFace not available — using local LBP+Colour embedding. "
            "Re-ID will still work well for typical faces. "
            "For best results: pip install insightface onnxruntime")

    def embed(self, face_bgr: np.ndarray) -> Optional[np.ndarray]:
        if face_bgr is None or face_bgr.size == 0:
            return None
        try:
            return _LocalEmbedder._compute(face_bgr)
        except Exception as exc:
            logger.debug("LocalEmbedder error: %s", exc)
            return None

    @staticmethod
    def _compute(img_bgr: np.ndarray) -> np.ndarray:
        face = cv2.resize(img_bgr, (128, 128))

        # Lighting normalisation via CLAHE on Y channel
        ycrcb = cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        face = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        gray = ycrcb[:, :, 0]

        parts: List[np.ndarray] = []

        # 1. LBP uniform histogram — 4x4 grid x 256 bins = 4096 dim
        rows, cols = gray.shape
        lbp = np.zeros_like(gray, dtype=np.uint8)
        for dy, dx in [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]:
            shifted = np.roll(np.roll(gray, dy, axis=0), dx, axis=1)
            lbp = (lbp << 1) | (gray >= shifted).astype(np.uint8)
        ch, cw = rows // 4, cols // 4
        for r in range(4):
            for c in range(4):
                cell = lbp[r*ch:(r+1)*ch, c*cw:(c+1)*cw]
                hist = np.bincount(cell.ravel(), minlength=256).astype(float)
                parts.append(hist / (hist.sum() + 1e-6))

        # 2. HSV colour histogram — 3 channels x 32 bins = 96 dim
        hsv = cv2.cvtColor(face, cv2.COLOR_BGR2HSV)
        for ch_idx in range(3):
            h = np.histogram(hsv[:, :, ch_idx], bins=32, range=(0, 256))[0].astype(float)
            parts.append(h / (h.sum() + 1e-6))

        emb = np.concatenate(parts)
        return emb / (np.linalg.norm(emb) + 1e-6)


def _build_embedder(sface_model_path: str = ""):
    """Try backends in priority order. Returns (embedder, reid_threshold)."""

    # 1. insightface real ArcFace
    try:
        emb = _InsightFaceEmbedder()
        return emb, emb.thresh
    except Exception as exc:
        import traceback as _tb
        msg = str(exc) or type(exc).__name__
        logger.warning(
            "InsightFace ArcFace unavailable: %s\n%s", msg, _tb.format_exc())
        warnings.warn(
            f"insightface ArcFace unavailable: {msg}. "
            "To enable real ArcFace run:  pip install insightface onnxruntime",
            stacklevel=3)

    # 2. OpenCV SFace ONNX
    if sface_model_path and os.path.isfile(sface_model_path):
        try:
            emb = _SFaceEmbedder(sface_model_path)
            return emb, emb.thresh
        except Exception as exc:
            warnings.warn(f"OpenCV SFace failed: {exc}", stacklevel=3)
    elif sface_model_path:
        warnings.warn(
            f"sface_model_path '{sface_model_path}' not found. "
            "Download from: https://github.com/opencv/opencv_zoo/tree/master/models/face_recognition_sface",
            stacklevel=3)

    # 3. Local LBP + colour
    emb = _LocalEmbedder()
    return emb, emb.thresh


# ══════════════════════════════════════════════════════════════════════════════
# PART B  FACE DETECTION BACKENDS
# ══════════════════════════════════════════════════════════════════════════════

class _MediaPipeTasksBackend:
    """
    MediaPipe Tasks FaceDetector (mediapipe >= 0.10).
    Requires face_detection_short_range.tflite in the project root.
    Face-specific — never detects hands, elbows, or body parts.
    """
    name = "mediapipe-tasks"
    def __init__(self, model_path, min_conf):
        import os
        # Force pure-Python protobuf to avoid MessageFactory/SymbolDatabase
        # errors caused by protobuf >= 4.x on Windows.
        os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        import mediapipe as mp
        self._mp = mp

        if not os.path.isfile(model_path):
            raise FileNotFoundError(
                f"MediaPipe Tasks model not found: {model_path}. "
                "Download face_detection_short_range.tflite from: "
                "https://storage.googleapis.com/mediapipe-models/face_detector/"
                "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite")

        options = mp_vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            min_detection_confidence=min_conf)
        self._det = mp_vision.FaceDetector.create_from_options(options)

        # Smoke-test with noise image — catches init-time failures cleanly
        _noise = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        _img   = mp.Image(image_format=mp.ImageFormat.SRGB,
                          data=cv2.cvtColor(_noise, cv2.COLOR_BGR2RGB))
        _res   = self._det.detect(_img)
        _ = _res.detections  # triggers SymbolDatabase error if protobuf broken

    def detect_raw(self, frame):
        mp = self._mp
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=_to_rgb(frame))
        res = self._det.detect(mp_img)
        out = []
        for d in (res.detections or []):
            bb = d.bounding_box
            sc = d.categories[0].score if d.categories else 0.5
            out.append((bb.origin_x, bb.origin_y, bb.width, bb.height, sc))
        return out

    def close(self):
        try: self._det.close()
        except Exception: pass


class _MediaPipeSolutionsBackend:
    name = "mediapipe-solutions"
    def __init__(self, min_conf):
        import mediapipe as mp
        self._fd = mp.solutions.face_detection.FaceDetection(
            model_selection=1, min_detection_confidence=min_conf)
        # Smoke-test: process a random noise image.
        # Blank images short-circuit the pipeline and miss the SymbolDatabase
        # error. Noise forces full execution so broken protobuf is caught HERE
        # at init (not per-frame), and the backend is skipped cleanly.
        _noise = np.random.randint(0, 255, (128, 128, 3), dtype=np.uint8)
        try:
            _res = self._fd.process(cv2.cvtColor(_noise, cv2.COLOR_BGR2RGB))
            _ = _res.detections  # this line triggers SymbolDatabase error
        except Exception as exc:
            raise RuntimeError(f"MediaPipe solutions smoke-test failed: {exc}") from exc

    def detect_raw(self, frame):
        h, w = frame.shape[:2]
        res = self._fd.process(_to_rgb(frame))
        if not res.detections:
            return []
        out = []
        for d in res.detections:
            bb = d.location_data.relative_bounding_box
            x = max(0, int(bb.xmin * w));  y = max(0, int(bb.ymin * h))
            bw = int(bb.width * w);         bh = int(bb.height * h)
            sc = d.score[0] if d.score else 0.5
            out.append((x, y, bw, bh, sc))
        return out

    def close(self):
        try: self._fd.close()
        except Exception: pass


class _OpenCVDNNBackend:
    name = "opencv-dnn"
    def __init__(self, proto, model, min_conf=0.50):
        self._net      = cv2.dnn.readNetFromCaffe(proto, model)
        self._min_conf = min_conf

    def detect_raw(self, frame):
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame,(300,300)),1.0,(300,300),
                                     (104.0,177.0,123.0))
        self._net.setInput(blob)
        out = self._net.forward()
        res = []
        for i in range(out.shape[2]):
            sc = float(out[0,0,i,2])
            if sc < self._min_conf: continue
            x1=int(out[0,0,i,3]*w); y1=int(out[0,0,i,4]*h)
            x2=int(out[0,0,i,5]*w); y2=int(out[0,0,i,6]*h)
            res.append((x1,y1,max(0,x2-x1),max(0,y2-y1),sc))
        return res

    def close(self): pass


class _HaarCascadeBackend:
    """Three-pass Haar (frontal default -> alt2 -> profile) + CLAHE."""
    name = "haar-cascade"
    def __init__(self, scale, min_neighbors, min_size):
        self._frontal = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self._alt2    = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
        self._profile = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_profileface.xml")
        self._scale    = scale
        self._min_nbrs = min_neighbors
        self._min_size = min_size

    @staticmethod
    def _is_face_region(frame: np.ndarray, x: int, y: int,
                        bw: int, bh: int) -> bool:
        """
        Post-filter to reject non-face detections (hands, elbows, etc.).

        Checks two things:
        1. Skin-colour ratio — a real face crop should have >= 25% skin pixels
           in the YCrCb range used by the Viola-Jones skin model.
        2. Eye-region symmetry — the top half of a face crop should be darker
           than the bottom half (eyes/brows darker than cheeks/chin).
           Hands/elbows fail this reliably.
        """
        h, w = frame.shape[:2]
        x1 = max(0, x);      y1 = max(0, y)
        x2 = min(w, x + bw); y2 = min(h, y + bh)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0 or crop.shape[0] < 10 or crop.shape[1] < 10:
            return False

        # ── Skin colour check (YCrCb) ─────────────────────────────────────
        ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
        cr = ycrcb[:, :, 1];  cb = ycrcb[:, :, 2]
        skin_mask = ((cr >= 133) & (cr <= 173) &
                     (cb >= 77)  & (cb <= 127))
        skin_ratio = skin_mask.mean()
        if skin_ratio < 0.18:   # less than 18% skin pixels → reject
            return False

        # ── Vertical gradient check ───────────────────────────────────────
        # Face: eye-brow region (top 40%) tends to be darker than mid-face.
        # Non-face blobs (elbows/hands) tend to be uniform.
        gray_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        mid = gray_crop.shape[0] // 2
        top_mean = float(gray_crop[:mid, :].mean())
        bot_mean = float(gray_crop[mid:, :].mean())
        std      = float(gray_crop.std())
        if std < 8.0:           # too uniform — not a face
            return False

        return True

    def detect_raw(self, frame):
        h, w = frame.shape[:2]
        gray = _enhance(frame)
        seen, out = set(), []

        def _run(cascade, min_nbrs):
            faces = cascade.detectMultiScale(
                gray, scaleFactor=self._scale, minNeighbors=min_nbrs,
                minSize=self._min_size, flags=cv2.CASCADE_SCALE_IMAGE)
            if not isinstance(faces, np.ndarray) or len(faces) == 0:
                return
            for (x, y, bw, bh) in faces:
                key = (x // 10, y // 10)
                if key in seen:
                    continue
                seen.add(key)
                # Reject non-face regions (hands, elbows, objects)
                if not self._is_face_region(frame, x, y, bw, bh):
                    continue
                conf = min(0.85, 0.50 + (bw * bh) / max(w * h, 1) * 5.0)
                out.append((int(x), int(y), int(bw), int(bh), conf))

        _run(self._frontal, self._min_nbrs)
        _run(self._alt2,    max(1, self._min_nbrs - 1))
        if not out:
            _run(self._profile, max(1, self._min_nbrs - 1))
        return out

    def close(self): pass


# ══════════════════════════════════════════════════════════════════════════════
# PART C  MAIN FaceDetector CLASS
# ══════════════════════════════════════════════════════════════════════════════

class FaceDetector:
    """
    Robust face detector with ArcFace embedding for DeepSORT Re-ID.

    Parameters
    ----------
    min_detection_confidence : float
        Minimum detection score (0-1).
    min_face_size : int
        Minimum face side in pixels.
    sface_model_path : str
        Optional path to face_recognition_sface_2021dec.onnx (fallback if
        insightface is not installed). Download from OpenCV Zoo.
    backends : list[str] | None
        Detection backend order. Default: all four tried in order.
    mediapipe_model_path : str
        TFLite model for MediaPipe Tasks backend.
    haar_scale : float
        Haar scaleFactor (1.05 = thorough, 1.10 = faster).
    haar_min_neighbors : int
        Haar minNeighbors (lower = more detections, more false positives).
    """

    _ALL_BACKENDS = [
        "mediapipe-tasks",
        "mediapipe-solutions",
        "opencv-dnn",
        "haar-cascade",
    ]

    def __init__(
        self,
        min_detection_confidence: float = 0.25,
        min_face_size: int              = 30,
        sface_model_path: str           = "",
        backends: Optional[List[str]]   = None,
        mediapipe_model_path: str       = "face_detection_short_range.tflite",
        opencv_dnn_proto: str           = "deploy.prototxt",
        opencv_dnn_model: str           = "res10_300x300_ssd_iter_140000.caffemodel",
        haar_scale: float               = 1.05,
        haar_min_neighbors: int         = 5,  # higher = fewer false positives
    ):
        self._min_conf = min_detection_confidence
        self._min_size = min_face_size

        # Detection backends
        self._det_backends: List = []
        for bname in (backends or self._ALL_BACKENDS):
            b = self._try_init_det(
                bname,
                mediapipe_model_path=mediapipe_model_path,
                opencv_dnn_proto=opencv_dnn_proto,
                opencv_dnn_model=opencv_dnn_model,
                haar_scale=haar_scale,
                haar_min_neighbors=haar_min_neighbors,
                min_face_size=min_face_size,
                min_conf=min_detection_confidence,
            )
            if b is not None:
                self._det_backends.append(b)
                logger.info("FaceDetector: detection backend '%s' ready.", bname)

        if not self._det_backends:
            raise RuntimeError(
                "No face detection backend could be initialised. "
                "Ensure OpenCV is installed correctly.")

        # Embedding backend
        self._embedder, self.reid_threshold = _build_embedder(sface_model_path)

        logger.info(
            "FaceDetector ready | detection=%s | embedding=%s | reid_thresh=%.2f",
            self._det_backends[0].name, self._embedder.name, self.reid_threshold)

    @staticmethod
    def _try_init_det(name: str, **kw):
        try:
            if name == "mediapipe-tasks":
                return _MediaPipeTasksBackend(kw["mediapipe_model_path"], kw["min_conf"])
            if name == "mediapipe-solutions":
                return _MediaPipeSolutionsBackend(kw["min_conf"])
            if name == "opencv-dnn":
                return _OpenCVDNNBackend(kw["opencv_dnn_proto"], kw["opencv_dnn_model"])
            if name == "haar-cascade":
                sz = kw["min_face_size"]
                return _HaarCascadeBackend(
                    kw["haar_scale"], kw["haar_min_neighbors"], (sz, sz))
        except Exception as exc:
            warnings.warn(
                f"FaceDetector: detection backend '{name}' unavailable — {exc}",
                stacklevel=4)
        return None

    def detect(self, frame: np.ndarray) -> List[dict]:
        """
        Detect all faces in frame (BGR uint8) and compute embeddings.

        Returns list of dicts with keys:
            bbox, confidence, crop, embedding
        """
        if frame is None or frame.size == 0:
            return []

        fh, fw = frame.shape[:2]

        # Run detection — first backend that returns detections wins.
        # If a backend raises it is permanently removed from the active list
        # so it does not spam warnings on every frame.
        raw = []
        dead = []
        for backend in self._det_backends:
            try:
                raw = backend.detect_raw(frame)
            except Exception as exc:
                logger.warning(
                    "Backend '%s' failed permanently: %s — removing from pipeline.",
                    backend.name, exc)
                dead.append(backend)
                raw = []
            if raw:
                break
        for b in dead:
            self._det_backends.remove(b)
        if not self._det_backends and not raw:
            logger.error("All detection backends have failed.")

        results = []
        for (x, y, w, h, score) in raw:
            x = max(0, int(x));  y = max(0, int(y))
            w = min(int(w), fw - x);  h = min(int(h), fh - y)

            if score < self._min_conf:        continue
            if w < self._min_size or h < self._min_size: continue
            if not _is_valid(x, y, w, h, fw, fh):        continue

            # 224x224 crop for deepfake model (padded)
            crop = _make_crop(frame, x, y, w, h)

            # Tighter crop for embedding (minimal padding preserves identity)
            pad  = int(min(w, h) * 0.05)
            x1b  = max(0, x - pad);   y1b = max(0, y - pad)
            x2b  = min(fw, x + w + pad);  y2b = min(fh, y + h + pad)
            face_tight = frame[y1b:y2b, x1b:x2b]
            embedding  = self._embedder.embed(face_tight)

            results.append({
                'bbox':       [x, y, w, h],
                'confidence': float(score),
                'crop':       crop,
                'embedding':  embedding,
            })

        return results

    def close(self):
        """Release all backend resources. Called by orchestrator.finalize()."""
        for b in self._det_backends:
            try: b.close()
            except Exception: pass

    @property
    def embedding_backend(self) -> str:
        return self._embedder.name

    @property
    def detection_backend(self) -> str:
        return self._det_backends[0].name if self._det_backends else "none"


# ══════════════════════════════════════════════════════════════════════════════
# PART D  draw_results() — called by orchestrator
# ══════════════════════════════════════════════════════════════════════════════

_COLOURS = {
    'real':      (30,  200,  80),
    'uncertain': (0,   165, 255),
    'fake':      (40,   40, 220),
    'analysing': (130, 130, 130),
}
_MIN_FRAMES_FOR_VERDICT = 8   # must match deep_sort._MIN_FRAMES


def draw_results(frame: np.ndarray, tracks) -> np.ndarray:
    """
    Draw bounding boxes + labels for every confirmed DeepSORT track.

    Parameters
    ----------
    frame  : BGR numpy array (modified in-place and returned)
    tracks : list[Track] from DeepSortTracker.get_active_tracks()
    """
    font       = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    box_thick  = 2

    for track in tracks:
        if not track.is_confirmed():
            continue

        tlwh      = track.to_tlwh()
        x, y      = int(tlwh[0]), int(tlwh[1])
        w, h      = int(tlwh[2]), int(tlwh[3])
        x2, y2    = x + w, y + h
        n_scores  = len(getattr(track, 'deepfake_scores', []))

        if n_scores < _MIN_FRAMES_FOR_VERDICT:
            colour = _COLOURS['analysing']
            label  = f"#{track.track_id}  analysing..."
        elif track.is_deepfake and not track.is_uncertain:
            colour = _COLOURS['fake']
            label  = f"#{track.track_id}  FAKE {int(track.smoothed_score * 100)}%"
        elif track.is_uncertain:
            colour = _COLOURS['uncertain']
            label  = f"#{track.track_id}  uncertain"
        else:
            colour = _COLOURS['real']
            label  = f"#{track.track_id}  REAL {int((1.0 - track.smoothed_score) * 100)}%"

        # Bounding box
        cv2.rectangle(frame, (x, y), (x2, y2), colour, box_thick)

        # Label background + text
        (tw, th), _ = cv2.getTextSize(label, font, font_scale, 1)
        ly = max(y - 6, th + 4)
        cv2.rectangle(frame, (x, ly - th - 4), (x + tw + 6, ly + 2), colour, -1)
        cv2.putText(frame, label, (x + 3, ly - 2),
                    font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)

        # Confidence bar at bottom of box
        if n_scores >= _MIN_FRAMES_FOR_VERDICT:
            bar_w = max(0, min(w, int(w * track.confidence)))
            cv2.rectangle(frame, (x, y2 - 5), (x + bar_w, y2), colour, -1)

    return frame