"""
Detonix ZoomGuard — Main Entry Point
Processes all videos from input_videos/ folder and saves annotated output.

Usage:
    python main.py                         # Process all videos with default settings
    python main.py --input my_videos/      # Custom input folder
    python main.py --video single.mp4      # Single video
    python main.py --no-display            # Headless mode (no preview window)
    python main.py --threshold 0.6         # Custom deepfake threshold
"""

import cv2
import os
import sys
import argparse
import logging
import time
from pathlib import Path
from datetime import datetime

from orchestrator import DetectionOrchestrator
from utils.logger import setup_logging


SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Detonix ZoomGuard — Real-Time Deepfake Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py
  python main.py --video input_videos/test.mp4
  python main.py --input input_videos/ --no-display
  python main.py --threshold 0.55 --skip 2
        """
    )
    parser.add_argument('--input', default='input_videos', help='Folder with input videos')
    parser.add_argument('--video', default=None, help='Single video file to process')
    parser.add_argument('--output', default='output_results', help='Output folder for annotated videos')
    parser.add_argument('--model', default=None, help='Path to custom XceptionNet model weights (.pth)')
    parser.add_argument('--threshold', type=float, default=0.65, help='Deepfake detection threshold (0-1)')
    parser.add_argument('--skip', type=int, default=3, help='Process every N frames (higher=faster, less accurate)')
    parser.add_argument('--no-display', action='store_true', help='Disable preview window (for headless/server mode)')
    parser.add_argument('--face-conf', type=float, default=0.5, help='Minimum face detection confidence')
    parser.add_argument('--max-age', type=int, default=30, help='Max frames before track is removed')
    return parser.parse_args()


def process_video(video_path, output_path, config, show_display=True):
    """
    Process a single video file through the ZoomGuard pipeline.
    Returns dict with per-track results.
    """
    logger = logging.getLogger("VideoProcessor")
    logger.info(f"Processing: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video: {video_path}")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    logger.info(f"Video info: {width}x{height} @ {fps:.1f}fps, {total_frames} frames")

    # Setup video writer
    os.makedirs(output_path.parent, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    # Initialize fresh orchestrator for each video
    orchestrator = DetectionOrchestrator(config)
    orchestrator.is_running = True

    frame_num = 0
    t_start = time.time()

    print(f"\n{'='*60}")
    print(f"  Processing: {os.path.basename(video_path)}")
    print(f"  Resolution: {width}x{height} | FPS: {fps:.1f} | Frames: {total_frames}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_num += 1

        # Process frame
        annotated = orchestrator.process_frame(frame, frame_num, str(video_path))
        writer.write(annotated)

        # Show preview
        if show_display:
            display = annotated
            # Scale down for display if needed
            if width > 1280 or height > 720:
                scale = min(1280 / width, 720 / height)
                display = cv2.resize(annotated, (int(width * scale), int(height * scale)))
            cv2.imshow("Detonix ZoomGuard — Press Q to skip", display)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                logger.info("User skipped video.")
                break

        # Progress bar
        if frame_num % 30 == 0 or frame_num == total_frames:
            elapsed = time.time() - t_start
            fps_proc = frame_num / elapsed if elapsed > 0 else 0
            pct = (frame_num / total_frames * 100) if total_frames > 0 else 0
            bar_filled = int(pct / 5)
            bar = "█" * bar_filled + "░" * (20 - bar_filled)
            eta = (total_frames - frame_num) / fps_proc if fps_proc > 0 else 0
            print(f"\r  [{bar}] {pct:.0f}% | Frame {frame_num}/{total_frames} | "
                  f"{fps_proc:.1f} fps | ETA: {eta:.0f}s  ", end="", flush=True)

    print()  # newline after progress bar
    cap.release()
    writer.release()

    # Finalize and save logs
    log_path = orchestrator.finalize()
    summary = orchestrator.get_summary()

    elapsed = time.time() - t_start
    logger.info(f"Done in {elapsed:.1f}s. Output saved: {output_path}")

    return summary


def print_banner():
    banner = """
╔══════════════════════════════════════════════════════════╗
║                                                          ║
║          DETONIX  ZOOMGUARD                              ║
║          Real-Time Deepfake Detection System             ║
║                                                          ║
║          Quaid-i-Azam University | CS Final Year         ║
║          Supervised by: Dr. Syed M Naqi                  ║
║                                                          ║
╚══════════════════════════════════════════════════════════╝
"""
    print(banner)


def main():
    print_banner()
    args = parse_args()

    # Setup logging
    log_file = setup_logging('logs')
    logger = logging.getLogger("Main")
    logger.info("Detonix ZoomGuard starting...")

    # Build config
    config = {
        'model_path': args.model,
        'deepfake_threshold': args.threshold,
        'process_every_n_frames': args.skip,
        'face_confidence': args.face_conf,
        'min_face_size': 40,
        'max_age': args.max_age,
        'n_init': 3,
        'max_iou_distance': 0.7,
        'log_dir': 'logs',
    }

    show_display = not args.no_display
    os.makedirs(args.output, exist_ok=True)

    # Collect videos to process
    if args.video:
        videos = [Path(args.video)]
    else:
        input_dir = Path(args.input)
        if not input_dir.exists():
            print(f"\n[ERROR] Input folder '{input_dir}' does not exist.")
            print(f"  Create it and drop your videos there:\n  mkdir {input_dir}\n")
            sys.exit(1)

        videos = [p for p in sorted(input_dir.iterdir())
                  if p.suffix.lower() in SUPPORTED_FORMATS]

    if not videos:
        print(f"\n[ERROR] No supported videos found.")
        print(f"  Supported formats: {', '.join(SUPPORTED_FORMATS)}")
        print(f"  Drop your videos into: {args.input}/\n")
        sys.exit(1)

    print(f"\n  Found {len(videos)} video(s) to process:")
    for v in videos:
        print(f"    → {v.name}")
    print(f"\n  Settings:")
    print(f"    Deepfake threshold : {args.threshold}")
    print(f"    Frame skip         : every {args.skip} frames")
    print(f"    Display preview    : {'Yes (press Q to skip)' if show_display else 'No (headless)'}")
    print(f"    Output folder      : {args.output}/")
    print()

    all_results = {}

    for video_path in videos:
        output_filename = f"zoomguard_{video_path.stem}_{datetime.now().strftime('%H%M%S')}.mp4"
        output_path = Path(args.output) / output_filename

        result = process_video(video_path, output_path, config, show_display)
        if result is not None:
            all_results[video_path.name] = result

    # Final overall summary
    print("\n" + "="*60)
    print("  ALL VIDEOS PROCESSED")
    print("="*60)
    total_deepfakes = sum(
        sum(1 for info in res.values() if info.get('is_deepfake'))
        for res in all_results.values()
    )
    print(f"  Videos processed : {len(all_results)}")
    print(f"  Total deepfakes  : {total_deepfakes} person-tracks flagged")
    print(f"  Output saved to  : {args.output}/")
    print(f"  Logs saved to    : logs/")
    print("="*60)

    if show_display:
        cv2.destroyAllWindows()

    logger.info("Detonix ZoomGuard session complete.")


if __name__ == "__main__":
    main()