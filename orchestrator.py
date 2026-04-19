"""
Detection Orchestrator — Detonix ZoomGuard (Production Quality)

Fixes in this version:
✔ Alert confidence threshold raised to 0.92 to avoid false alerts
✔ Alert only fires when track has enough frames (n_scores >= _MIN_FRAMES)
✔ Summary verdict now correctly reads is_deepfake after enough frames
✔ HUD shows "analyzing..." count during warm-up (not UNCERTAIN)
✔ draw_results called with correct signature
✔ ArcFace embedding passed through Detection to tracker
✔ face_detector.close() safe
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

# Must match deep_sort._MIN_FRAMES
_MIN_FRAMES_FOR_VERDICT = 8


def _push(event_type, data):
    try:
        from dashboard_server import push_event
        push_event(event_type, data)
    except Exception:
        pass


def _iou(box_a, box_b):
    ax2 = box_a[0] + box_a[2];  ay2 = box_a[1] + box_a[3]
    bx2 = box_b[0] + box_b[2];  by2 = box_b[1] + box_b[3]
    ix1 = max(box_a[0], box_b[0]); iy1 = max(box_a[1], box_b[1])
    ix2 = min(ax2, bx2);           iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter
    return inter / union if union > 0 else 0.0


class DetectionOrchestrator:

    def __init__(self, config):
        self.config     = config
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_logger = SessionLogger(
            self.session_id, config.get('log_dir', 'logs'))

        logger.info("Initializing Detonix ZoomGuard pipeline...")

        self.face_detector = FaceDetector(
            min_detection_confidence=config.get('face_confidence', 0.4),
            min_face_size=config.get('min_face_size', 30)
        )
        self.deepfake_detector = DeepfakeDetector(
            model_path=config.get('model_path', None),
            device=config.get('device', None)
        )
        self.tracker = DeepSortTracker(
            max_iou_distance=config.get('max_iou_distance', 0.75),
            max_age=config.get('max_age', 60),
            n_init=config.get('n_init', 3)
        )

        self.deepfake_threshold = config.get('deepfake_threshold', 0.65)
        self.infer_every_n  = config.get('process_every_n_frames', 3)
        self.frame_count    = 0
        self.track_results  = {}
        self._last_fps      = 0.0
        self._t_start       = time.time()

        logger.info("Pipeline initialized successfully.")

    # ── Session lifecycle ─────────────────────────────────────────────────────

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

    def end_session(self):
        _push('session_end', {'session_id': self.session_id})

    # ── Main frame processing ─────────────────────────────────────────────────

    def process_frame(self, frame_bgr, frame_number, video_name="video"):
        self.frame_count += 1

        # Step 1: Kalman predict
        self.tracker.predict()

        # Step 2: Face detection
        face_detections = []
        crops           = []

        h_orig, w_orig = frame_bgr.shape[:2]

        if w_orig > 640:
            scale = 640 / w_orig
            small = cv2.resize(frame_bgr,
                               (640, int(h_orig * scale)),
                               interpolation=cv2.INTER_LINEAR)
        else:
            scale = 1.0
            small = frame_bgr

        raw_faces = self.face_detector.detect(small)

        for fd in raw_faces:
            bx, by, bw, bh = fd['bbox']

            if scale != 1.0:
                bx = int(bx / scale); by = int(by / scale)
                bw = int(bw / scale); bh = int(bh / scale)
                pad = int(min(bw, bh) * 0.10)
                x1  = max(0, bx - pad);   y1 = max(0, by - pad)
                x2  = min(w_orig, bx + bw + pad)
                y2  = min(h_orig, by + bh + pad)
                hq_crop = frame_bgr[y1:y2, x1:x2]
                if hq_crop.size > 0:
                    fd['crop'] = cv2.resize(hq_crop, (224, 224))
                fd['bbox'] = [bx, by, bw, bh]

            det = Detection(
                tlwh=[bx, by, bw, bh],
                confidence=fd['confidence'],
                embedding=fd.get('embedding')         # ArcFace embedding
            )
            face_detections.append(det)
            crops.append(fd['crop'])

        # Step 3: Update tracker
        self.tracker.update(face_detections)

        # Step 4: Deepfake inference every N frames
        if crops and self.frame_count % self.infer_every_n == 0:
            scores        = self.deepfake_detector.predict_batch(crops)
            active_tracks = self.tracker.get_active_tracks()
            assigned      = set()

            for det, score in zip(face_detections, scores):
                best_track, best_iou = None, 0.1
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
                n_scores = len(best_track.deepfake_scores)

                self.session_logger.log_detection(
                    tid, best_track.is_deepfake,
                    best_track.smoothed_score, frame_number, video_name)

                if tid not in self.track_results:
                    self.track_results[tid] = {
                        'is_deepfake':     False,
                        'is_uncertain':    True,
                        'confidence':      0.0,
                        'frames_analyzed': 0,
                        'alerted':         False,
                    }

                # Alert — only after enough frames, high confidence, not uncertain
                enough_data = n_scores >= _MIN_FRAMES_FOR_VERDICT
                if (enough_data
                        and best_track.is_deepfake
                        and not best_track.is_uncertain
                        and best_track.confidence > 0.92
                        and not self.track_results[tid]['alerted']):
                    self.session_logger.log_alert(
                        tid, best_track.confidence, video_name, frame_number)
                    self.track_results[tid]['alerted'] = True
                    _push('alert', {
                        'id':    tid,
                        'score': round(best_track.confidence, 4),
                        'frame': frame_number,
                        'video': os.path.basename(video_name),
                    })

                self.track_results[tid]['is_deepfake']      = best_track.is_deepfake
                self.track_results[tid]['is_uncertain']     = best_track.is_uncertain
                self.track_results[tid]['confidence']       = best_track.confidence
                self.track_results[tid]['frames_analyzed'] += 1

                _push('person_update', {
                    'id':              tid,
                    'score':           round(best_track.smoothed_score, 4),
                    'confidence':      round(best_track.confidence, 4),
                    'frames_analyzed': self.track_results[tid]['frames_analyzed'],
                    'is_deepfake':     best_track.is_deepfake,
                    'is_uncertain':    best_track.is_uncertain,
                })

        # Step 5: FPS push
        if self.frame_count % 5 == 0:
            elapsed = time.time() - self._t_start
            self._last_fps = self.frame_count / elapsed if elapsed > 0 else 0
            _push('frame', {
                'frame':    frame_number,
                'fps':      round(self._last_fps, 1),
                'duration': int(elapsed),
            })

        # Step 6: Annotate
        active_tracks = self.tracker.get_active_tracks()
        annotated     = draw_results(frame_bgr, active_tracks)
        annotated     = self._draw_hud(annotated, frame_number, video_name)
        return annotated

    # ── HUD ───────────────────────────────────────────────────────────────────

    def _draw_hud(self, frame, frame_number, video_name):
        h, w = frame.shape[:2]

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 44), (10, 12, 14), -1)
        cv2.addWeighted(overlay, 0.80, frame, 0.20, 0, frame)

        active = self.tracker.get_active_tracks()

        n_analyzing = sum(
            1 for t in active
            if t.is_confirmed()
            and len(getattr(t, 'deepfake_scores', [])) < _MIN_FRAMES_FOR_VERDICT
        )
        n_fake = sum(
            1 for t in active
            if t.is_confirmed()
            and len(getattr(t, 'deepfake_scores', [])) >= _MIN_FRAMES_FOR_VERDICT
            and t.is_deepfake
        )
        n_uncertain = sum(
            1 for t in active
            if t.is_confirmed()
            and len(getattr(t, 'deepfake_scores', [])) >= _MIN_FRAMES_FOR_VERDICT
            and t.is_uncertain
        )
        n_real = sum(
            1 for t in active
            if t.is_confirmed()
            and len(getattr(t, 'deepfake_scores', [])) >= _MIN_FRAMES_FOR_VERDICT
            and not t.is_deepfake
            and not t.is_uncertain
        )

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, "DETONIX ZOOMGUARD",
                    (10, 28), font, 0.60, (31, 199, 120), 2, cv2.LINE_AA)

        info = (f"Frame {frame_number}  |  "
                f"REAL:{n_real}  UNCERTAIN:{n_uncertain}  "
                f"FAKE:{n_fake}  ANALYZING:{n_analyzing}  |  "
                f"{self._last_fps:.0f} fps  |  {os.path.basename(video_name)}")
        cv2.putText(frame, info, (w // 3 + 10, 28),
                    font, 0.38, (160, 165, 170), 1, cv2.LINE_AA)

        if n_fake > 0:
            ov2 = frame.copy()
            cv2.rectangle(ov2, (0, h - 34), (w, h), (18, 8, 8), -1)
            cv2.addWeighted(ov2, 0.88, frame, 0.12, 0, frame)
            cv2.putText(frame,
                        f"  DEEPFAKE DETECTED  {n_fake} person(s) flagged",
                        (8, h - 10), font, 0.55, (50, 50, 230), 2, cv2.LINE_AA)
        elif n_uncertain > 0:
            ov2 = frame.copy()
            cv2.rectangle(ov2, (0, h - 34), (w, h), (18, 12, 4), -1)
            cv2.addWeighted(ov2, 0.85, frame, 0.15, 0, frame)
            cv2.putText(frame,
                        f"  UNCERTAIN  {n_uncertain} person(s) — reviewing",
                        (8, h - 10), font, 0.50, (0, 165, 255), 2, cv2.LINE_AA)

        return frame

    # ── Summary / cleanup ─────────────────────────────────────────────────────

    def get_summary(self):
        return self.track_results

    def finalize(self):
        self.session_logger.print_summary(self.track_results)
        path = self.session_logger.save_summary()
        self.face_detector.close()
        return path