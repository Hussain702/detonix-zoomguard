"""
Detection Orchestrator — Detonix ZoomGuard
Coordinates the full pipeline and pushes live data to dashboard.
"""

import cv2
import numpy as np
import logging
import os
import time
from datetime import datetime

from utils.deep_sort import DeepSortTracker, Detection
from utils.face_detector import FaceDetector, draw_results
from utils.deepfake_model import DeepfakeDetector
from utils.logger import SessionLogger

logger = logging.getLogger(__name__)


def _push(event_type, data):
    """Send event to dashboard if server is running."""
    try:
        from dashboard_server import push_event
        push_event(event_type, data)
    except Exception:
        pass


class DetectionOrchestrator:

    def __init__(self, config):
        self.config = config
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_logger = SessionLogger(
            self.session_id, config.get('log_dir', 'logs'))

        logger.info("Initializing Detonix ZoomGuard pipeline...")

        self.face_detector = FaceDetector(
            min_detection_confidence=config.get('face_confidence', 0.6),
            min_face_size=config.get('min_face_size', 60)
        )
        self.deepfake_detector = DeepfakeDetector(
            model_path=config.get('model_path', None),
            device=config.get('device', None)
        )
        self.tracker = DeepSortTracker(
            max_iou_distance=config.get('max_iou_distance', 0.75),
            max_age=config.get('max_age', 150),
            n_init=config.get('n_init', 2)
        )

        self.deepfake_threshold = config.get('deepfake_threshold', 0.65)
        self.infer_every_n = config.get('process_every_n_frames', 3)
        self.frame_count = 0
        self.track_results = {}
        self._last_fps = 0.0
        self._t_start = time.time()

        logger.info("Pipeline initialized successfully.")

    # ── called once per video before the frame loop ───────────────────────────
    def start_session(self, video_name, total_frames, fps, resolution):
        self._t_start = time.time()
        _push('session_start', {
            'session_id':   self.session_id,
            'video_name':   os.path.basename(video_name),
            'total_frames': total_frames,
            'fps':          fps,
            'resolution':   resolution,
            'threshold':    self.deepfake_threshold,
        })

    # ── called once per video after the frame loop ────────────────────────────
    def end_session(self):
        _push('session_end', {'session_id': self.session_id})

    # ── called for every frame ────────────────────────────────────────────────
    def process_frame(self, frame_bgr, frame_number, video_name="video"):

        self.frame_count += 1

        # Step 1 — Kalman predict
        self.tracker.predict()

        # Step 2 — detect faces every N frames for speed, every frame for tracking
        run_inference = (self.frame_count % self.infer_every_n == 0)
        run_detection = (self.frame_count % 2 == 0)  # detect every 2nd frame (speed)

        face_detections = []
        crops = []

        if run_detection:
            face_detections_raw = self.face_detector.detect(frame_bgr)
            for fd in face_detections_raw:
                face_detections.append(Detection(fd['bbox'], fd['confidence']))
                crops.append(fd['crop'])

        # Step 3 — update tracker
        self.tracker.update(face_detections)

        # Step 4 — deepfake inference every N frames
        if crops and run_inference:
            scores = self.deepfake_detector.predict_batch(crops)
            active_tracks = self.tracker.get_active_tracks()
            assigned = set()

            for det, score in zip(face_detections, scores):
                best_track = None
                best_iou   = 0.1

                for track in active_tracks:
                    if track.track_id in assigned:
                        continue
                    iou_val = _iou(track.to_tlwh(), det.tlwh)
                    if iou_val > best_iou:
                        best_iou   = iou_val
                        best_track = track

                if best_track is None:
                    continue

                best_track.add_deepfake_score(score)
                assigned.add(best_track.track_id)
                tid = best_track.track_id

                self.session_logger.log_detection(
                    tid, best_track.is_deepfake,
                    best_track.smoothed_score, frame_number, video_name)

                # ensure entry exists
                if tid not in self.track_results:
                    self.track_results[tid] = {
                        'is_deepfake': False, 'confidence': 0.0,
                        'frames_analyzed': 0,  'alerted': False
                    }

                # fire alert once
                if (best_track.is_deepfake
                        and best_track.smoothed_score > self.deepfake_threshold
                        and not self.track_results[tid]['alerted']):
                    self.session_logger.log_alert(
                        tid, best_track.smoothed_score, video_name, frame_number)
                    self.track_results[tid]['alerted'] = True
                    _push('alert', {
                        'id':    tid,
                        'score': round(best_track.smoothed_score, 4),
                        'frame': frame_number,
                        'video': os.path.basename(video_name),
                    })

                self.track_results[tid]['is_deepfake']     = best_track.is_deepfake
                self.track_results[tid]['confidence']      = best_track.confidence
                self.track_results[tid]['frames_analyzed'] += 1

                # push person update to dashboard
                _push('person_update', {
                    'id':              tid,
                    'score':           round(best_track.smoothed_score, 4),
                    'confidence':      round(best_track.confidence, 4),
                    'frames_analyzed': self.track_results[tid]['frames_analyzed'],
                    'is_deepfake':     best_track.is_deepfake,
                })

        # Step 5 — push frame progress every 5 frames
        if self.frame_count % 5 == 0:
            elapsed = time.time() - self._t_start
            fps = self.frame_count / elapsed if elapsed > 0 else 0
            self._last_fps = fps
            _push('frame', {
                'frame':    frame_number,
                'fps':      round(fps, 1),
                'duration': int(elapsed),
            })

        # Step 6 — annotate and return frame
        active_tracks = self.tracker.get_active_tracks()
        annotated = draw_results(frame_bgr, active_tracks)
        annotated = self._draw_hud(annotated, frame_number, video_name)
        return annotated

    def _draw_hud(self, frame, frame_number, video_name):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 42), (13, 15, 16), -1)
        cv2.addWeighted(overlay, 0.82, frame, 0.18, 0, frame)

        active    = self.tracker.get_active_tracks()
        n_active  = len(active)
        n_fake    = sum(1 for t in active if t.is_confirmed() and t.is_deepfake)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "DETONIX ZOOMGUARD",
                    (10, 27), font, 0.62, (31, 199, 120), 2)
        info = (f"Frame {frame_number}  |  Persons: {n_active}  |  "
                f"Deepfakes: {n_fake}  |  {self._last_fps:.0f} fps  |  "
                f"{os.path.basename(video_name)}")
        cv2.putText(frame, info, (w // 3, 27), font, 0.42, (140, 148, 158), 1)

        if n_fake > 0:
            ov2 = frame.copy()
            cv2.rectangle(ov2, (0, h - 32), (w, h), (20, 10, 10), -1)
            cv2.addWeighted(ov2, 0.88, frame, 0.12, 0, frame)
            cv2.putText(frame,
                        f"  WARNING: DEEPFAKE DETECTED — {n_fake} person(s)",
                        (8, h - 10), font, 0.55, (240, 82, 82), 2)
        return frame

    def get_summary(self):
        return self.track_results

    def finalize(self):
        self.session_logger.print_summary(self.track_results)
        path = self.session_logger.save_summary()
        self.face_detector.close()
        return path


def _iou(box_a, box_b):
    ax2 = box_a[0] + box_a[2];  ay2 = box_a[1] + box_a[3]
    bx2 = box_b[0] + box_b[2];  by2 = box_b[1] + box_b[3]
    ix1 = max(box_a[0], box_b[0]); iy1 = max(box_a[1], box_b[1])
    ix2 = min(ax2, bx2);           iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = box_a[2]*box_a[3] + box_b[2]*box_b[3] - inter
    return inter / union if union > 0 else 0.0