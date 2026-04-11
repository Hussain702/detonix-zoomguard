"""
Debug script - shows exactly what face detector sees frame by frame.
Run this to diagnose why IDs keep resetting.
"""
import cv2
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from utils.face_detector import FaceDetector
from utils.deep_sort import DeepSortTracker, Detection

# Find first video in input_videos/
videos = [f for f in os.listdir('input_videos') 
          if f.lower().endswith(('.mp4','.avi','.mov','.mkv'))]
if not videos:
    print("No video found in input_videos/")
    sys.exit(1)

video_path = os.path.join('input_videos', videos[0])
print(f"Analyzing: {video_path}\n")

cap = cv2.VideoCapture(video_path)
detector = FaceDetector(min_detection_confidence=0.3, min_face_size=20)
tracker  = DeepSortTracker(max_iou_distance=0.8, max_age=120, n_init=2)

detection_counts = []   # how many faces detected per frame
id_per_frame     = []   # which track IDs were active per frame
missed_frames    = 0

total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames: {total}")
print(f"{'Frame':<8} {'Detections':<14} {'Active IDs':<30} {'Miss streak'}")
print("-" * 65)

streak = 0
for fn in range(min(total, 300)):   # check first 300 frames
    ret, frame = cap.read()
    if not ret:
        break

    faces = detector.detect(frame)
    dets  = [Detection(f['bbox'], f['confidence']) for f in faces]

    tracker.predict()
    tracker.update(dets)

    active_ids = [t.track_id for t in tracker.get_active_tracks()]
    n_det = len(faces)
    detection_counts.append(n_det)

    if n_det == 0:
        streak += 1
        missed_frames += 1
    else:
        streak = 0

    if fn < 60 or fn % 30 == 0:   # print first 60 frames + every 30th
        id_str = str(active_ids) if active_ids else "[]"
        print(f"{fn:<8} {n_det:<14} {id_str:<30} {streak if streak>0 else ''}")

cap.release()
detector.close()

print("\n" + "="*65)
print(f"Frames with NO face detected : {missed_frames}/{min(total,300)} "
      f"({missed_frames/min(total,300)*100:.1f}%)")
print(f"Max consecutive missed frames: {max(0, streak)}")
print(f"Avg detections per frame     : {sum(detection_counts)/len(detection_counts):.2f}")
print(f"Total unique IDs created     : {tracker._next_id - 1}")
print()
if missed_frames / min(total, 300) > 0.3:
    print(">> ROOT CAUSE: Face detector is missing face on >30% of frames.")
    print("   Fix: lower detection confidence or use Haar cascade fallback.")
else:
    print(">> Face detection is OK. Issue is in matching threshold.")
