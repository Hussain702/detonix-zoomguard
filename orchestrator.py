"""
Detection Orchestrator — Detonix ZoomGuard (Production Quality)

Fixes applied:
✔ FaceDetector instantiated with correct kwarg names
  (min_detection_confidence, min_face_size)
✔ tracker.update() called with frame_w / frame_h so the face-only gate works
✔ face_detector.detect() returns dicts → access via fd['bbox'] etc.
✔ draw_results / close() wired correctly
✔ Alert confidence threshold raised to 0.92 (avoid false alerts)
✔ HUD shows "analysing" count during warm-up, not UNCERTAIN
✔ Classification crop now taken AFTER tracker.update(), from the Kalman-
  smoothed track box (track.to_tlwh()) instead of the raw per-frame detector
  box. Raw detector boxes jitter frame-to-frame (a few px of jitter shifts
  the crop's framing), and XceptionNet's score turned out to be very
  sensitive to that framing — this was causing the same real face to swing
  between ~0.01 and ~0.99 raw fake-probability frame to frame. Cropping from
  the smoothed box removes that jitter at the source, before it ever reaches
  the classifier, instead of trying to smooth the resulting noisy scores
  after the fact.
✔ Classification is now applied directly to the tracks DeepSORT actually
  matched this frame (track.time_since_update == 0), rather than
  independently re-matching raw detections back to tracks via a second,
  greedy IoU pass. The old greedy pass could occasionally disagree with
  DeepSORT's own Hungarian assignment when two faces were close together,
  scoring the wrong track.
"""

import cv2
import numpy as np
import logging
import os
import time
from datetime import datetime

from utils.deep_sort import DeepSortTracker, Detection
from utils.face_detector import (
    FaceDetector, draw_results, _MIN_FRAMES_FOR_VERDICT, _make_crop,
)
from utils.deepfake_model import DeepfakeDetector
from utils.logger import SessionLogger
from utils.temporal_classifier import TemporalAggregator

logger = logging.getLogger(__name__)


def _push(event_type, data):
    try:
        from dashboard_server import push_event
        push_event(event_type, data)
    except Exception:
        pass


class DetectionOrchestrator:

    def __init__(self, config):
        self.config     = config
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_logger = SessionLogger(
            self.session_id, config.get('log_dir', 'logs'))

        logger.info("Initializing Detonix ZoomGuard pipeline...")

        # ── FaceDetector: use the exact kwarg names the class defines ─────────
        self.face_detector = FaceDetector(
            min_detection_confidence=config.get('face_confidence', 0.4),
            min_face_size=config.get('min_face_size', 30),
        )

        self.deepfake_detector = DeepfakeDetector(
            model_path=config.get('model_path', None),
            device=config.get('device', None),
        )
        self.tracker = DeepSortTracker(
            max_iou_distance=config.get('max_iou_distance', 0.75),
            max_age=config.get('max_age', 60),
            n_init=config.get('n_init', 3),
        )

        # NOTE: this is reporting-only. The single source of truth for the
        # actual FAKE decision boundary is TemporalAggregator.FAKE_RAW_THR —
        # read live (not cached here) wherever it's reported, so it can
        # never drift out of sync with the value the classifier is really
        # using (config['deepfake_threshold'] previously fed a threshold
        # into the dashboard that the decision logic never consulted).
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
            # Live values from the single source of truth, not a copy that
            # can drift: 'threshold' is the actual raw-space FAKE gate the
            # classifier is using right now; 'display_threshold' is the same
            # boundary mapped into the calibrated space that smoothed_score
            # (the value plotted per-frame) actually lives in.
            'threshold':         TemporalAggregator.FAKE_RAW_THR,
            'display_threshold': TemporalAggregator.display_threshold(),
        })

    def end_session(self):
        _push('session_end', {'session_id': self.session_id})

    # ── Main frame processing ─────────────────────────────────────────────────

    def process_frame(self, frame_bgr, frame_number, video_name="video"):
        self.frame_count += 1

        h_orig, w_orig = frame_bgr.shape[:2]

        # Step 1: Kalman predict
        self.tracker.predict()

        # Step 2: Face detection
        # Downscale for speed, then map boxes back to original resolution
        if w_orig > 640:
            scale = 640 / w_orig
            small = cv2.resize(frame_bgr,
                               (640, int(h_orig * scale)),
                               interpolation=cv2.INTER_LINEAR)
        else:
            scale = 1.0
            small = frame_bgr

        raw_faces = self.face_detector.detect(small)

        face_detections = []

        for fd in raw_faces:
            bx, by, bw, bh = fd['bbox']

            # Map back to original resolution (needed for the tracker itself,
            # independent of anything to do with classification crops).
            if scale != 1.0:
                bx = int(bx / scale);  by = int(by / scale)
                bw = int(bw / scale);  bh = int(bh / scale)

            det = Detection(
                tlwh=[bx, by, bw, bh],
                confidence=fd['confidence'],
                embedding=fd.get('embedding'),   # ArcFace embedding (may be None)
                frame_w=w_orig,
                frame_h=h_orig,
            )
            face_detections.append(det)

        # Step 3: Update tracker — pass frame dims so the face-only gate fires.
        # After this call, every track's Kalman state (track.to_tlwh()) has
        # fused in this frame's detection, so it's a smoothed box, not a raw
        # jittery one.
        self.tracker.update(face_detections, frame_w=w_orig, frame_h=h_orig)

        # Step 4: Deepfake inference every N frames.
        # Crop directly from the tracks DeepSORT just matched this frame —
        # using each track's smoothed box, not the raw detector box — instead
        # of re-matching raw detections back to tracks after the fact.
        if self.frame_count % self.infer_every_n == 0:
            active_tracks = self.tracker.get_active_tracks()
            live_tracks   = [t for t in active_tracks if t.time_since_update == 0]

            if live_tracks:
                crops = []
                for track in live_tracks:
                    x, y, w, h = track.to_tlwh()
                    x = max(0, int(x));  y = max(0, int(y))
                    w = max(1, int(w));  h = max(1, int(h))
                    crops.append(_make_crop(frame_bgr, x, y, w, h))

                scores = self.deepfake_detector.predict_batch(crops)

                for track, score in zip(live_tracks, scores):
                    tid = track.track_id

                    # A failed inference (score is None) must NEVER be
                    # treated as evidence of REAL. Discard the observation
                    # entirely rather than feeding a fabricated value into
                    # the temporal aggregator — and log it clearly so
                    # failures are visible, not silent.
                    if score is None:
                        logger.warning(
                            "Deepfake inference failed for track %s at "
                            "frame %s (%s) — observation discarded, not "
                            "counted as REAL or FAKE evidence.",
                            tid, frame_number, video_name)
                        continue

                    track.add_deepfake_score(score)
                    n_scores = len(track.deepfake_scores)

                    self.session_logger.log_detection(
                        tid, track.is_deepfake,
                        track.smoothed_score, frame_number, video_name)

                    if tid not in self.track_results:
                        self.track_results[tid] = {
                            'is_deepfake':     False,
                            'is_uncertain':    True,
                            'confidence':      0.0,
                            'frames_analyzed': 0,
                            'alerted':         False,
                        }

                    # Alert — only after enough frames, high confidence
                    enough_data = n_scores >= _MIN_FRAMES_FOR_VERDICT
                    if (enough_data
                            and track.is_deepfake
                            and not track.is_uncertain
                            and track.confidence > 0.92
                            and not self.track_results[tid]['alerted']):
                        self.session_logger.log_alert(
                            tid, track.confidence, video_name, frame_number)
                        self.track_results[tid]['alerted'] = True
                        _push('alert', {
                            'id':    tid,
                            'score': round(track.confidence, 4),
                            'frame': frame_number,
                            'video': os.path.basename(video_name),
                        })

                    self.track_results[tid]['is_deepfake']      = track.is_deepfake
                    self.track_results[tid]['is_uncertain']     = track.is_uncertain
                    self.track_results[tid]['confidence']       = track.confidence
                    self.track_results[tid]['frames_analyzed'] += 1

                    _push('person_update', {
                        'id':              tid,
                        'score':           round(track.smoothed_score, 4),
                        'confidence':      round(track.confidence, 4),
                        'frames_analyzed': self.track_results[tid]['frames_analyzed'],
                        'is_deepfake':     track.is_deepfake,
                        'is_uncertain':    track.is_uncertain,
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

        # Step 6: Annotate frame
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

        n_analysing = sum(
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
                f"FAKE:{n_fake}  ANALYSING:{n_analysing}  |  "
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