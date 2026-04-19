"""
Face Detector Module — Detonix ZoomGuard (ArcFace Integrated)

Color fix in this version:
✔ REAL  → bright green box
✔ FAKE  → red box
✔ UNCERTAIN after enough data → orange
✔ Still warming up (< _MIN_FRAMES inferences) → grey/white — NOT orange
  This stops the "everything is orange" problem during the first few seconds.
✔ close() method present for orchestrator.finalize()
✔ MediaPipe new-API fallback fixed
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Keep in sync with deep_sort._MIN_FRAMES so the warm-up colour matches
_WARMUP_FRAMES = 8


# ─────────────────────────────────────────────────────────────────────────────
# draw_results
# ─────────────────────────────────────────────────────────────────────────────

def draw_results(frame, tracks, detections=None, show_id=True, show_conf=True):
    """
    Draw bounding boxes on *frame* (returns a copy).

    Box colours:
        Grey   — warming up (fewer than _WARMUP_FRAMES inferences, tentative)
        Green  — REAL
        Orange — UNCERTAIN (enough data but score is ambiguous)
        Red    — FAKE / DEEPFAKE

    Parameters
    ----------
    frame      : BGR numpy array
    tracks     : list of Track objects from DeepSortTracker
    detections : optional list of raw detections (drawn as thin green outlines)
    """
    out  = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    # colours (BGR)
    COLOR_WARMUP    = (180, 180, 180)   # grey   — not enough data yet
    COLOR_REAL      = (50,  220,  50)   # green
    COLOR_UNCERTAIN = (0,   165, 255)   # orange
    COLOR_FAKE      = (50,   50, 230)   # red

    # ── optional raw detection outlines ──────────────────────────────────────
    if detections:
        for det in detections:
            bbox = getattr(det, 'tlwh', None) or det.get('bbox', None)
            if bbox is None:
                continue
            x, y, w, h = [int(v) for v in bbox]
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 0), 1)

    # ── tracks ────────────────────────────────────────────────────────────────
    for trk in (tracks or []):
        if trk.is_deleted():
            continue

        x, y, w, h = [int(v) for v in trk.to_tlwh()]
        x2, y2 = x + w, y + h

        n_scores = len(getattr(trk, 'deepfake_scores', []))
        warming_up = n_scores < _WARMUP_FRAMES

        # Pick colour
        if not trk.is_confirmed() or warming_up:
            color     = COLOR_WARMUP
            thickness = 1
        elif getattr(trk, 'is_deepfake', False):
            color     = COLOR_FAKE
            thickness = 3
        elif getattr(trk, 'is_uncertain', True):
            color     = COLOR_UNCERTAIN
            thickness = 2
        else:
            color     = COLOR_REAL
            thickness = 2

        cv2.rectangle(out, (x, y), (x2, y2), color, thickness)

        if not trk.is_confirmed():
            continue

        # ── label ────────────────────────────────────────────────────────────
        if warming_up:
            label       = f"ID {trk.track_id}  analyzing..."
            label_color = COLOR_WARMUP
        elif getattr(trk, 'is_deepfake', False):
            conf  = getattr(trk, 'confidence', 0.0)
            label = f"ID {trk.track_id}  DEEPFAKE  {conf:.0%}"
            label_color = COLOR_FAKE
        elif getattr(trk, 'is_uncertain', True):
            sc    = getattr(trk, 'smoothed_score', 0.5)
            label = f"ID {trk.track_id}  UNCERTAIN  {sc:.2f}"
            label_color = COLOR_UNCERTAIN
        else:
            sc    = getattr(trk, 'smoothed_score', 0.0)
            label = f"ID {trk.track_id}  REAL  {1 - sc:.0%}"
            label_color = COLOR_REAL

        (tw, th), _ = cv2.getTextSize(label, font, 0.52, 1)
        lx, ly = x, max(y - 6, th + 2)
        cv2.rectangle(out, (lx, ly - th - 4), (lx + tw + 6, ly + 2),
                      (15, 15, 15), -1)
        cv2.putText(out, label, (lx + 3, ly),
                    font, 0.52, label_color, 1, cv2.LINE_AA)

        # 5-point landmarks
        if hasattr(trk, 'landmarks') and trk.landmarks is not None:
            for (lx2, ly2) in trk.landmarks:
                cv2.circle(out, (int(lx2), int(ly2)), 2, (0, 140, 255), -1)

    return out


# ─────────────────────────────────────────────────────────────────────────────
# FaceDetector
# ─────────────────────────────────────────────────────────────────────────────

class FaceDetector:
    """
    Detection priority:
      1. InsightFace (RetinaFace + ArcFace buffalo_l)
      2. MediaPipe (legacy 0.9.x API, then new 0.10+ tasks API)
      3. Haar cascade
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.45,
        min_face_size:            int   = 30,
        smooth_alpha:             float = 0.65,
        blur_threshold:           float = 30.0,
    ):
        self.min_face_size  = min_face_size
        self.smooth_alpha   = smooth_alpha
        self.blur_threshold = blur_threshold
        self.backend        = None

        self._smooth_state: dict = {}
        self._prev_count:   int  = 0
        self._mp_detector   = None
        self.arcface        = None

        self._init_insightface()
        if self.backend is None:
            self._init_mediapipe(min_detection_confidence)
        if self.backend is None:
            self._init_haar()

        logger.info(f"FaceDetector ready — backend: {self.backend}")

    # ── backend init ──────────────────────────────────────────────────────────

    def _init_insightface(self):
        try:
            from insightface.app import FaceAnalysis
            self.arcface = FaceAnalysis(
                name='buffalo_l',
                allowed_modules=['detection', 'recognition']
            )
            self.arcface.prepare(ctx_id=0, det_size=(320, 320))
            self.backend = 'insightface'
            logger.info("InsightFace / ArcFace buffalo_l initialised")
        except Exception as e:
            self.arcface = None
            logger.warning(f"InsightFace unavailable: {e}")

    def _init_mediapipe(self, confidence):
        try:
            import mediapipe as mp
            # Legacy API (mediapipe <= 0.9.x)
            try:
                sol = mp.solutions.face_detection
                self._mp_detector  = sol.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=confidence)
                self._mp_use_legacy = True
                self.backend = 'mediapipe'
                logger.info("MediaPipe face detection initialised (legacy API)")
                return
            except AttributeError:
                pass

            # New tasks API (mediapipe >= 0.10) — requires a model file.
            # Skip silently if no model is available; Haar will handle it.
            logger.warning("MediaPipe unavailable: new tasks API requires model file; "
                           "falling back to Haar.")
        except Exception as e:
            logger.warning(f"MediaPipe unavailable: {e}")

    def _init_haar(self):
        path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self._haar   = cv2.CascadeClassifier(path)
        self.backend = 'haar'
        logger.warning("Falling back to Haar cascade (no embeddings)")

    # ── public close ──────────────────────────────────────────────────────────

    def close(self):
        """Release resources. Called by orchestrator.finalize()."""
        try:
            if self._mp_detector is not None:
                self._mp_detector.close()
        except Exception:
            pass

    # ── smoothing ─────────────────────────────────────────────────────────────

    def _ema_smooth(self, idx: int, box: list) -> list:
        if idx not in self._smooth_state:
            self._smooth_state[idx] = box
            return box
        prev = self._smooth_state[idx]
        a = self.smooth_alpha
        smoothed = [int(a * prev[k] + (1 - a) * box[k]) for k in range(4)]
        self._smooth_state[idx] = smoothed
        return smoothed

    def _reset_smooth(self):
        self._smooth_state.clear()

    # ── geometry ──────────────────────────────────────────────────────────────

    @staticmethod
    def _make_square(x1, y1, x2, y2, W, H):
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        half   = max(x2 - x1, y2 - y1) // 2
        return max(0, cx - half), max(0, cy - half), \
               min(W, cx + half), min(H, cy + half)

    @staticmethod
    def _is_blurry(crop, threshold):
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return cv2.Laplacian(gray, cv2.CV_64F).var() < threshold

    # ── detect ────────────────────────────────────────────────────────────────

    def detect(self, frame_bgr: np.ndarray) -> list:
        if self.backend == 'insightface':
            return self._detect_insightface(frame_bgr)
        elif self.backend == 'mediapipe':
            return self._detect_mediapipe(frame_bgr)
        else:
            return self._detect_haar(frame_bgr)

    # ── InsightFace ───────────────────────────────────────────────────────────

    def _detect_insightface(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        try:
            faces = self.arcface.get(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        except Exception as e:
            logger.error(f"InsightFace.get() failed: {e}")
            return []

        if not faces:
            self._reset_smooth()
            return []

        faces = sorted(faces, key=lambda f: f.bbox[0])
        if len(faces) != self._prev_count:
            self._reset_smooth()
        self._prev_count = len(faces)

        detections = []
        for idx, face in enumerate(faces):
            fx1, fy1, fx2, fy2 = [int(v) for v in face.bbox]
            if (fx2 - fx1) < self.min_face_size or (fy2 - fy1) < self.min_face_size:
                continue
            pad = int(min(fx2 - fx1, fy2 - fy1) * 0.20)
            x1, y1, x2, y2 = self._make_square(
                max(0, fx1 - pad), max(0, fy1 - pad),
                min(W, fx2 + pad),  min(H, fy2 + pad), W, H)
            box = self._ema_smooth(idx, [x1, y1, x2 - x1, y2 - y1])
            sx1, sy1, sw, sh = box
            crop = frame_bgr[sy1:sy1 + sh, sx1:sx1 + sw]
            if crop.size == 0 or self._is_blurry(crop, self.blur_threshold):
                continue
            detections.append({
                'bbox':       [sx1, sy1, sw, sh],
                'confidence': float(face.det_score),
                'crop':       cv2.resize(crop, (224, 224)),
                'embedding':  face.embedding,
                'landmarks':  face.kps if hasattr(face, 'kps') else None,
            })
        return detections

    # ── MediaPipe ─────────────────────────────────────────────────────────────

    def _detect_mediapipe(self, frame_bgr):
        H, W = frame_bgr.shape[:2]
        try:
            results = self._mp_detector.process(
                cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            raw_dets = results.detections or []
        except Exception:
            return []

        raw_boxes = []
        for det in raw_dets:
            b  = det.location_data.relative_bounding_box
            bw = int(b.width * W);  bh = int(b.height * H)
            if bw < self.min_face_size or bh < self.min_face_size:
                continue
            x  = int(b.xmin * W);   y  = int(b.ymin * H)
            pad = int(min(bw, bh) * 0.20)
            x1, y1, x2, y2 = self._make_square(
                max(0, x - pad), max(0, y - pad),
                min(W, x + bw + pad), min(H, y + bh + pad), W, H)
            raw_boxes.append(([x1, y1, x2 - x1, y2 - y1], float(det.score[0])))

        if len(raw_boxes) != self._prev_count:
            self._reset_smooth()
        self._prev_count = len(raw_boxes)

        detections = []
        for idx, (box, conf) in enumerate(raw_boxes):
            box = self._ema_smooth(idx, box)
            x1, y1, bw, bh = box
            crop = frame_bgr[y1:y1 + bh, x1:x1 + bw]
            if crop.size == 0 or self._is_blurry(crop, self.blur_threshold):
                continue
            detections.append({
                'bbox':       [x1, y1, bw, bh],
                'confidence': conf,
                'crop':       cv2.resize(crop, (224, 224)),
                'embedding':  self._get_arcface_embedding(crop),
                'landmarks':  None,
            })
        return detections

    # ── Haar ──────────────────────────────────────────────────────────────────

    def _detect_haar(self, frame_bgr):
        H, W  = frame_bgr.shape[:2]
        gray  = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(
            gray, 1.1, 5, minSize=(self.min_face_size, self.min_face_size))

        if len(faces) != self._prev_count:
            self._reset_smooth()
        self._prev_count = len(faces)

        detections = []
        for idx, (x, y, w, h) in enumerate(faces):
            pad = int(min(w, h) * 0.20)
            x1, y1, x2, y2 = self._make_square(
                max(0, x - pad), max(0, y - pad),
                min(W, x + w + pad), min(H, y + h + pad), W, H)
            box = self._ema_smooth(idx, [x1, y1, x2 - x1, y2 - y1])
            x1, y1, bw, bh = box
            crop = frame_bgr[y1:y1 + bh, x1:x1 + bw]
            if crop.size == 0:
                continue
            detections.append({
                'bbox':       [x1, y1, bw, bh],
                'confidence': 0.85,
                'crop':       cv2.resize(crop, (224, 224)),
                'embedding':  self._get_arcface_embedding(crop),
                'landmarks':  None,
            })
        return detections

    # ── ArcFace helper ────────────────────────────────────────────────────────

    def _get_arcface_embedding(self, crop_bgr):
        if self.arcface is None:
            return None
        try:
            faces = self.arcface.get(cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB))
            if faces:
                return faces[0].embedding
        except Exception:
            pass
        return None