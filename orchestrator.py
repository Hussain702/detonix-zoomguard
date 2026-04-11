"""
Detection Orchestrator
Coordinates frame processing, face detection, tracking, and deepfake inference.

Fixes applied:
- Track faces on EVERY frame (not just every N) to prevent ID reset
- Deepfake inference still runs every N frames for performance
- HUD shows all active persons (not just confirmed)
- Summary consolidates IDs properly
"""

import cv2
import numpy as np
import logging
import os
from datetime import datetime

from utils.deep_sort import DeepSortTracker, Detection
from utils.face_detector import FaceDetector, draw_results
from utils.deepfake_model import DeepfakeDetector
from utils.logger import SessionLogger

logger = logging.getLogger(__name__)


class DetectionOrchestrator:
    """
    Main pipeline coordinator for Detonix ZoomGuard.
    Handles: Frame → Face Detection → DeepSort Tracking → Deepfake Inference → Alert
    """

    def __init__(self, config):
        self.config = config
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_logger = SessionLogger(self.session_id, config.get('log_dir', 'logs'))

        logger.info("Initializing Detonix ZoomGuard pipeline...")

        self.face_detector = FaceDetector(
            min_detection_confidence=config.get('face_confidence', 0.6),
            min_face_size=config.get('min_face_size', 60)
        )

        self.deepfake_detector = DeepfakeDetector(
            model_path=config.get('model_path', None),
            device=config.get('device', None)
        )

        # KEY FIX: lower n_init so tracks confirm faster (2 instead of 3)
        # KEY FIX: higher max_age so tracks survive between inference frames
        self.tracker = DeepSortTracker(
            max_iou_distance=config.get('max_iou_distance', 0.7),
            max_age=config.get('max_age', 150),   # was 30 — give tracks more time to survive
            n_init=config.get('n_init', 2)        # was 3 — confirm after 2 hits not 3
        )

        self.deepfake_threshold = config.get('deepfake_threshold', 0.5)
        self.infer_every_n = config.get('process_every_n_frames', 3)  # inference cadence
        self.frame_count = 0
        self.track_results = {}

        # Store last crops per track for score assignment
        self._last_crops = {}

        logger.info("Pipeline initialized successfully.")

    def process_frame(self, frame_bgr, frame_number, video_name="video"):
        """
        Process a single video frame through the full pipeline.

        Tracking runs EVERY frame → stable IDs.
        Deepfake inference runs every N frames → manageable CPU load.
        """
        self.frame_count += 1

        # ── Step 1: Always predict tracker forward ──────────────────────────
        self.tracker.predict()

        # ── Step 2: Detect faces every frame for stable tracking ─────────────
        face_detections_raw = self.face_detector.detect(frame_bgr)

        face_detections = []
        crops = []
        for fd in face_detections_raw:
            face_detections.append(Detection(fd['bbox'], fd['confidence']))
            crops.append(fd['crop'])

        # ── Step 3: Update tracker with detections ───────────────────────────
        self.tracker.update(face_detections)

        # ── Step 4: Run deepfake inference every N frames ────────────────────
        if crops and self.frame_count % self.infer_every_n == 0:
            scores = self.deepfake_detector.predict_batch(crops)
            active_tracks = self.tracker.get_active_tracks()
            assigned = set()

            for det, score, crop in zip(face_detections, scores, crops):
                best_track = None
                best_iou = 0.1   # lowered from 0.3 → more forgiving match

                for track in active_tracks:
                    if track.track_id in assigned:
                        continue
                    iou_val = _iou(track.to_tlwh(), det.tlwh)
                    if iou_val > best_iou:
                        best_iou = iou_val
                        best_track = track

                if best_track is not None:
                    best_track.add_deepfake_score(score)
                    assigned.add(best_track.track_id)

                    # ── Logging & alert ──────────────────────────────────────
                    tid = best_track.track_id
                    self.session_logger.log_detection(
                        tid, best_track.is_deepfake,
                        best_track.smoothed_score, frame_number, video_name
                    )

                    if tid not in self.track_results:
                        self.track_results[tid] = {
                            'is_deepfake': False, 'confidence': 0.0,
                            'frames_analyzed': 0, 'alerted': False
                        }

                    if best_track.is_deepfake and best_track.smoothed_score > self.deepfake_threshold:
                        if not self.track_results[tid]['alerted']:
                            self.session_logger.log_alert(
                                tid, best_track.smoothed_score, video_name, frame_number
                            )
                            self.track_results[tid]['alerted'] = True

                    self.track_results[tid]['is_deepfake'] = best_track.is_deepfake
                    self.track_results[tid]['confidence'] = best_track.confidence
                    self.track_results[tid]['frames_analyzed'] += 1

        # ── Step 5: Draw results ─────────────────────────────────────────────
        active_tracks = self.tracker.get_active_tracks()
        annotated = draw_results(frame_bgr, active_tracks)
        annotated = self._draw_hud(annotated, frame_number, video_name)
        return annotated

    def _draw_hud(self, frame, frame_number, video_name):
        """Draw heads-up display overlay."""
        h, w = frame.shape[:2]

        # Semi-transparent top bar
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

        active = self.tracker.get_active_tracks()

        # Count all active (not just confirmed) — fixes "Persons: 0"
        n_active = len(active)
        n_confirmed = sum(1 for t in active if t.is_confirmed())
        n_fake = sum(1 for t in active if t.is_confirmed() and t.is_deepfake)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "DETONIX ZOOMGUARD", (8, 27), font, 0.7, (0, 200, 100), 2)

        status_text = (f"Frame: {frame_number}  |  "
                       f"Persons: {n_active}  |  "
                       f"Confirmed: {n_confirmed}  |  "
                       f"Deepfakes: {n_fake}  |  "
                       f"{os.path.basename(video_name)}")
        cv2.putText(frame, status_text, (w // 3, 27), font, 0.45, (180, 180, 180), 1)

        if n_fake > 0:
            overlay2 = frame.copy()
            cv2.rectangle(overlay2, (0, h - 34), (w, h), (0, 0, 160), -1)
            cv2.addWeighted(overlay2, 0.85, frame, 0.15, 0, frame)
            cv2.putText(frame,
                        f"  WARNING: DEEPFAKE DETECTED — {n_fake} person(s) flagged",
                        (8, h - 10), font, 0.6, (255, 255, 255), 2)

        return frame

    def get_summary(self):
        return self.track_results

    def finalize(self):
        self.session_logger.print_summary(self.track_results)
        path = self.session_logger.save_summary()
        self.face_detector.close()
        return path


def _iou(box_a, box_b):
    """Compute IoU between two [x, y, w, h] boxes."""
    ax2 = box_a[0] + box_a[2]
    ay2 = box_a[1] + box_a[3]
    bx2 = box_b[0] + box_b[2]
    by2 = box_b[1] + box_b[3]

    ix1 = max(box_a[0], box_b[0])
    iy1 = max(box_a[1], box_b[1])
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter
    return inter / union if union > 0 else 0.0