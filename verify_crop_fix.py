"""
verify_crop_fix.py (v3 — instrumented, folder mode)

Pinpoints WHERE the pipeline is failing: video I/O, raw face detection,
the post-detection validity filter, or the classifier itself.

Usage — scan every video already in your input_videos/ folder (no labels
needed, just tells you the diagnosis + score stats for each one):
    python verify_crop_fix.py --folder input_videos

Usage — compare a specific known-real clip against a known-fake clip:
    python verify_crop_fix.py --real path/to/known_real.mp4 --fake path/to/known_fake.mp4
"""
import argparse, os, sys, cv2, numpy as np
sys.path.insert(0, '.')
from utils.face_detector import FaceDetector, _make_crop
from utils.deepfake_model import DeepfakeDetector

SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}


def diagnose_clip(label, video_path, det, clf, n=20):
    print(f"\n=== {label}: {video_path} ===")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [FAIL] cv2.VideoCapture could not open this file at all. "
              f"Check the path is correct and the codec is supported by "
              f"your OpenCV build.")
        return []

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"  Opened OK — {width}x{height} @ {fps:.1f}fps, "
          f"{total} frames reported")

    step = max(1, total // n) if total > 0 else 1
    frames_read     = 0
    raw_faces_seen  = 0     # everything FaceDetector.detect() returned
    valid_faces     = 0
    none_scores     = 0     # classifier returned None (inference failure)
    scores          = []
    i = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames_read += 1
        if i % step == 0:
            faces = det.detect(frame)
            raw_faces_seen += len(faces)
            for fd in faces:
                valid_faces += 1
                x, y, w, h = fd['bbox']
                crop = _make_crop(frame, x, y, w, h)
                s = clf.predict(crop)
                if s is None:
                    none_scores += 1
                else:
                    scores.append(s)
        i += 1
        if len(scores) + none_scores >= n and frames_read >= n:
            break
    cap.release()

    print(f"  Frames actually read from file : {frames_read}")
    print(f"  Face detections returned       : {raw_faces_seen} "
          f"(across {min(n, frames_read)} sampled frames)")
    print(f"  Classifier inference failures  : {none_scores}")
    print(f"  Usable scores                  : {len(scores)}")

    if frames_read == 0:
        print("  -> DIAGNOSIS: file opened but produced 0 readable frames. "
              "Likely a codec/container issue with this OpenCV build.")
    elif raw_faces_seen == 0:
        print("  -> DIAGNOSIS: frames read fine, but the face detector "
              "found NO faces at all in this clip. This is a face-"
              "detection problem, not a classification problem — check "
              "which backend is actually active (see the startup log: "
              "mediapipe / opencv-dnn / haar) and whether it's tuned for "
              "this clip's resolution/lighting.")
    elif len(scores) == 0:
        print("  -> DIAGNOSIS: faces WERE detected, but every single "
              "classifier call failed (returned None). Check the ERROR-"
              "level log lines just above this for the actual exception.")
    else:
        print(f"  -> mean={np.mean(scores):.3f}  median={np.median(scores):.3f}  "
              f"(>0.90 raw-space = DEEPFAKE-leaning, matches TemporalAggregator.FAKE_RAW_THR)")

    return scores


def run_folder(folder, det, clf, n):
    if not os.path.isdir(folder):
        print(f"[ERROR] '{folder}' is not a folder.")
        sys.exit(1)
    videos = sorted(
        os.path.join(folder, f) for f in os.listdir(folder)
        if os.path.splitext(f)[1].lower() in SUPPORTED_FORMATS
    )
    if not videos:
        print(f"[ERROR] No supported video files found in '{folder}'. "
              f"Supported: {', '.join(sorted(SUPPORTED_FORMATS))}")
        sys.exit(1)

    print(f"Found {len(videos)} video(s) in '{folder}':")
    for v in videos:
        print(f"  - {os.path.basename(v)}")

    results = {}
    for v in videos:
        scores = diagnose_clip(os.path.basename(v), v, det, clf, n=n)
        results[v] = scores

    print("\n=== SUMMARY (all videos in folder) ===")
    for v, scores in results.items():
        name = os.path.basename(v)
        if scores:
            print(f"  {name:<40} mean={np.mean(scores):.3f}  "
                  f"n={len(scores)}")
        else:
            print(f"  {name:<40} NO USABLE SCORES — see DIAGNOSIS above")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--folder', default=None,
                   help="Scan every video in this folder (e.g. input_videos). "
                        "No labels needed — just reports diagnosis + scores "
                        "per file.")
    p.add_argument('--real', default=None,
                   help="Path to one known-real clip (use with --fake).")
    p.add_argument('--fake', default=None,
                   help="Path to one known-fake clip (use with --real).")
    p.add_argument('--model', default='models/deepfake_c0_xception.pkl')
    p.add_argument('--samples', type=int, default=20,
                   help="How many frames to sample per video (default 20).")
    args = p.parse_args()

    if not args.folder and not (args.real and args.fake):
        p.error("Provide either --folder input_videos, or both --real and --fake.")

    det = FaceDetector(min_detection_confidence=0.4, min_face_size=24)
    clf = DeepfakeDetector(model_path=args.model, device='cpu')

    if args.folder:
        run_folder(args.folder, det, clf, args.samples)
    else:
        real_scores = diagnose_clip('REAL', args.real, det, clf, n=args.samples)
        fake_scores = diagnose_clip('FAKE', args.fake, det, clf, n=args.samples)

        print("\n=== SUMMARY ===")
        if real_scores:
            print(f"REAL: mean={np.mean(real_scores):.3f}")
        if fake_scores:
            print(f"FAKE: mean={np.mean(fake_scores):.3f}")
        if not real_scores or not fake_scores:
            print("One or both clips produced no usable scores — see the "
                  "DIAGNOSIS lines above for exactly where it broke, and "
                  "paste that output back so we can fix the actual cause "
                  "instead of guessing.")