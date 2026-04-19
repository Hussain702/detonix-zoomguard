"""
Session Logger Module
Logs detection events, alerts, and session summaries.
"""

import json
import os
import logging
from datetime import datetime


def setup_logging(log_dir="logs"):
    """Configure application-wide logging."""
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"zoomguard_{timestamp}.log")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file


class SessionLogger:
    """Logs all detection events during a session."""

    def __init__(self, session_id, log_dir="logs"):
        self.session_id = session_id
        self.log_dir = log_dir
        self.detections = []
        self.alerts = []
        self.start_time = datetime.now()
        self.logger = logging.getLogger("SessionLogger")
        os.makedirs(log_dir, exist_ok=True)

    def log_detection(self, track_id, is_deepfake, score, frame_number, video_name):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "frame": frame_number,
            "video": video_name,
            "track_id": track_id,
            "is_deepfake": is_deepfake,
            "score": round(score, 4)
        }
        self.detections.append(entry)

    def log_alert(self, track_id, score, video_name, frame_number):
        alert = {
            "timestamp": datetime.now().isoformat(),
            "type": "DEEPFAKE_DETECTED",
            "track_id": track_id,
            "score": round(score, 4),
            "video": video_name,
            "frame": frame_number
        }
        self.alerts.append(alert)
        self.logger.warning(
            f"ALERT | Video: {video_name} | Frame: {frame_number} | "
            f"Person ID-{track_id} | Score: {score:.2%}"
        )

    def save_summary(self):
        summary = {
            "session_id": self.session_id,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "total_detections": len(self.detections),
            "total_alerts": len(self.alerts),
            "alerts": self.alerts,
            "detections": self.detections[-100:]  # Last 100 entries
        }
        path = os.path.join(self.log_dir, f"session_{self.session_id}.json")
        with open(path, 'w') as f:
            json.dump(summary, f, indent=2)
        self.logger.info(f"Session summary saved to {path}")
        return path

    def print_summary(self, track_results):
        """Print final per-person summary."""
        print("\n" + "="*60)
        print("  DETONIX ZOOMGUARD — SESSION SUMMARY")
        print("="*60)
        print(f"  Session ID  : {self.session_id}")
        print(f"  Total Alerts: {len(self.alerts)}")
        print(f"  Duration    : {(datetime.now() - self.start_time).seconds}s")
        print("-"*60)
        if track_results:
            print(f"  {'Person':<12} {'Verdict':<14} {'Confidence':<12} {'Frames'}")
            print("-"*60)
            for tid, info in sorted(track_results.items()):
                if info['is_deepfake']:
                    verdict = "DEEPFAKE"
                elif info.get('is_uncertain', False):
                    verdict = "? UNCERTAIN"
                else:
                    verdict = "REAL"
                print(f"  ID-{tid:<9} {verdict:<14} {info['confidence']:.1%}        {info['frames_analyzed']}")
        else:
            print("  No persons tracked.")
        print("="*60 + "\n")