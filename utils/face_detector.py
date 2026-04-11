"""
Face Detector Module
Uses MediaPipe for robust multi-face detection.
Extracts and aligns face crops for deepfake analysis.
"""

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Multi-face detector using MediaPipe or OpenCV Haar Cascades as fallback.
    Returns bounding boxes compatible with DeepSort tracker.
    """

    def __init__(self, min_detection_confidence=0.5, min_face_size=40):
        self.min_face_size = min_face_size
        self.detector = None
        self._init_detector(min_detection_confidence)

    def _init_detector(self, confidence):
        """Initialize MediaPipe face detection with version compatibility."""
        try:
            import mediapipe as mp

            # New MediaPipe API (0.10+): mp.tasks.vision
            if hasattr(mp, 'tasks'):
                from mediapipe.tasks import python as mp_python
                from mediapipe.tasks.python import vision as mp_vision
                import urllib.request, os

                model_path = os.path.join(os.path.dirname(__file__), 'blaze_face_short_range.tflite')
                if not os.path.exists(model_path):
                    logger.info("Downloading MediaPipe face detection model...")
                    url = ("https://storage.googleapis.com/mediapipe-models/"
                           "face_detector/blaze_face_short_range/float16/1/"
                           "blaze_face_short_range.tflite")
                    urllib.request.urlretrieve(url, model_path)

                options = mp_vision.FaceDetectorOptions(
                    base_options=mp_python.BaseOptions(model_asset_path=model_path),
                    min_detection_confidence=confidence
                )
                self.detector = mp_vision.FaceDetector.create_from_options(options)
                self.backend = 'mediapipe_new'
                logger.info("FaceDetector initialized with MediaPipe (new API)")

            # Old MediaPipe API (< 0.10): mp.solutions
            elif hasattr(mp, 'solutions'):
                self.mp_face = mp.solutions.face_detection
                self.detector = self.mp_face.FaceDetection(
                    model_selection=1,
                    min_detection_confidence=confidence
                )
                self.backend = 'mediapipe'
                logger.info("FaceDetector initialized with MediaPipe (legacy API)")
            else:
                raise AttributeError("Unrecognized MediaPipe version")

        except Exception as e:
            logger.warning(f"MediaPipe init failed ({e}). Falling back to OpenCV Haar Cascades.")
            self._init_haar()

    def _init_haar(self):
        """Fallback to OpenCV Haar Cascade detector."""
        self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.backend = 'haar'
        logger.info("FaceDetector initialized with OpenCV Haar Cascades")

    def detect(self, frame_bgr):
        """
        Detect all faces in a frame.
        
        Args:
            frame_bgr: numpy array in BGR format (OpenCV default)
            
        Returns:
            List of dicts: [{
                'bbox': [x, y, w, h],   # top-left x,y + width,height
                'confidence': float,
                'crop': numpy array     # cropped face BGR
            }]
        """
        if self.backend == 'mediapipe':
            return self._detect_mediapipe(frame_bgr)
        elif self.backend == 'mediapipe_new':
            return self._detect_mediapipe_new(frame_bgr)
        else:
            return self._detect_haar(frame_bgr)

    def _detect_mediapipe(self, frame_bgr):
        """MediaPipe detection."""
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.detector.process(frame_rgb)

        detections = []
        if not results.detections:
            return detections

        for det in results.detections:
            bbox = det.location_data.relative_bounding_box
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            bw = int(bbox.width * w)
            bh = int(bbox.height * h)

            # Add padding for better context
            pad = int(min(bw, bh) * 0.15)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)

            if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
                continue

            crop = frame_bgr[y1:y2, x1:x2].copy()
            detections.append({
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'confidence': det.score[0],
                'crop': crop
            })

        return detections

    def _detect_mediapipe_new(self, frame_bgr):
        """New MediaPipe Tasks API (0.10+) detection."""
        import mediapipe as mp
        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self.detector.detect(mp_image)

        detections = []
        if not results.detections:
            return detections

        for det in results.detections:
            bbox = det.bounding_box
            x, y, bw, bh = bbox.origin_x, bbox.origin_y, bbox.width, bbox.height

            pad = int(min(bw, bh) * 0.15)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + bw + pad)
            y2 = min(h, y + bh + pad)

            if (x2 - x1) < self.min_face_size or (y2 - y1) < self.min_face_size:
                continue

            crop = frame_bgr[y1:y2, x1:x2].copy()
            score = det.categories[0].score if det.categories else 0.9
            detections.append({
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'confidence': score,
                'crop': crop
            })

        return detections

    def _detect_haar(self, frame_bgr):
        """Haar cascade detection."""
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        h, w = frame_bgr.shape[:2]
        faces = self.detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size)
        )

        detections = []
        if not isinstance(faces, np.ndarray) or len(faces) == 0:
            return detections

        for (x, y, fw, fh) in faces:
            pad = int(min(fw, fh) * 0.1)
            x1 = max(0, x - pad)
            y1 = max(0, y - pad)
            x2 = min(w, x + fw + pad)
            y2 = min(h, y + fh + pad)

            crop = frame_bgr[y1:y2, x1:x2].copy()
            detections.append({
                'bbox': [x1, y1, x2 - x1, y2 - y1],
                'confidence': 0.9,
                'crop': crop
            })

        return detections

    def close(self):
        if self.backend in ('mediapipe', 'mediapipe_new') and self.detector:
            try:
                self.detector.close()
            except Exception:
                pass


def draw_results(frame, tracks, show_score=True):
    """
    Draw bounding boxes and labels on frame for all tracked faces.
    
    Args:
        frame: BGR numpy array
        tracks: list of Track objects from DeepSort
        show_score: whether to show confidence score
        
    Returns:
        annotated frame
    """
    annotated = frame.copy()
    
    for track in tracks:
        if track.is_deleted():
            continue

        tlbr = track.to_tlbr()
        x1, y1, x2, y2 = int(tlbr[0]), int(tlbr[1]), int(tlbr[2]), int(tlbr[3])

        # Color: Red for deepfake, Green for real, Yellow for tentative
        if track.is_tentative():
            color = (0, 200, 255)   # Yellow-orange
            label = f"ID-{track.track_id}: Analyzing..."
        elif track.is_deepfake:
            color = (0, 0, 220)     # Red
            label = f"ID-{track.track_id}: DEEPFAKE {track.smoothed_score:.0%}"
        else:
            color = (0, 180, 0)     # Green
            label = f"ID-{track.track_id}: REAL {(1-track.smoothed_score):.0%}"

        # Draw box
        thickness = 3 if track.is_confirmed() else 1
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        # Draw label background
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        text_size, _ = cv2.getTextSize(label, font, font_scale, 2)
        tw, th = text_size
        cv2.rectangle(annotated, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4), font, font_scale, (255, 255, 255), 2)

        # Draw small score bar
        if track.is_confirmed() and len(track.deepfake_scores) > 0:
            bar_width = x2 - x1
            filled = int(bar_width * track.smoothed_score)
            cv2.rectangle(annotated, (x1, y2 + 2), (x2, y2 + 8), (50, 50, 50), -1)
            bar_color = (0, 0, 200) if track.is_deepfake else (0, 180, 0)
            cv2.rectangle(annotated, (x1, y2 + 2), (x1 + filled, y2 + 8), bar_color, -1)

    return annotated