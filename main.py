"""
Detonix ZoomGuard — Main Entry Point

Usage:
    python main.py                    # terminal only
    python main.py --dashboard        # open live browser dashboard
    python main.py --video clip.mp4   # single video
    python main.py --no-display       # headless (no OpenCV window)
"""

# ── Suppress protobuf / mediapipe SymbolDatabase warning ──────────────────────
# Caused by protobuf version mismatch on some Windows installs.
# Safe to silence — the Haar cascade backend takes over automatically.
import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import cv2, sys, argparse, logging, time, threading, queue
from pathlib import Path
from datetime import datetime

from orchestrator import DetectionOrchestrator
from utils.logger import setup_logging

SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}


def _check_display() -> bool:
    """
    Return True only if cv2.imshow actually works on this machine.

    Uses only bare Python exception handling so it cannot itself crash,
    even if cv2 is partially broken (headless build, missing attributes, etc.).
    """
    try:
        # namedWindow is the lightest call that exercises the GUI backend.
        # It raises on headless builds before any image data is needed.
        cv2.namedWindow("__zg_probe__", cv2.WINDOW_AUTOSIZE)
        cv2.destroyWindow("__zg_probe__")
        return True
    except Exception:
        # Catches cv2.error, AttributeError, OSError — anything cv2 can throw
        return False


def parse_args():
    p = argparse.ArgumentParser(description="Detonix ZoomGuard")
    p.add_argument('--input',      default='input_videos')
    p.add_argument('--video',      default=None)
    p.add_argument('--output',     default='output_results')
    p.add_argument('--model',      default=None)
    p.add_argument('--threshold',  type=float, default=0.65)
    p.add_argument('--skip',       type=int,   default=3)
    p.add_argument('--no-display', action='store_true',
                   help='Disable live preview window (required for headless OpenCV)')
    p.add_argument('--dashboard',  action='store_true',
                   help='Launch live browser dashboard at http://localhost:5050')
    p.add_argument('--face-conf',  type=float, default=0.4)
    p.add_argument('--max-age',    type=int,   default=150)
    return p.parse_args()


def process_video(video_path, output_path, config, show_display=True):
    logger = logging.getLogger("VideoProcessor")
    logger.info(f"Processing: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open: {video_path}")
        return None

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    res    = f"{width}x{height}"

    os.makedirs(output_path.parent, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps,
        (width, height),
    )

    orch = DetectionOrchestrator(config)
    orch.start_session(str(video_path), total, fps, res)

    frame_num = 0
    t_start   = time.time()
    display_q = queue.Queue(maxsize=4)
    stop_flag = threading.Event()

    print(f"\n{'='*60}")
    print(f"  {os.path.basename(video_path)}  |  {res} @ {fps:.0f}fps  |  {total} frames")
    if not show_display:
        print(f"  Display: OFF  (headless mode — output saved to {output_path.name})")
    print(f"{'='*60}")

    # ── Worker thread: read + process frames ──────────────────────────────────
    def worker():
        nonlocal frame_num
        while not stop_flag.is_set():
            ret, frame = cap.read()
            if not ret:
                display_q.put(None)   # sentinel — signals completion
                break
            frame_num += 1
            annotated = orch.process_frame(frame, frame_num, str(video_path))
            writer.write(annotated)

            # Progress bar
            if frame_num % 30 == 0 or frame_num == total:
                el  = time.time() - t_start
                fp  = frame_num / el if el > 0 else 0
                pct = frame_num / total * 100 if total else 0
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                eta = (total - frame_num) / fp if fp > 0 else 0
                print(f"\r  [{bar}] {pct:.0f}%  {fp:.1f} fps  ETA {eta:.0f}s  ",
                      end="", flush=True)

            if show_display:
                try:
                    display_q.put_nowait(annotated)
                except queue.Full:
                    pass   # drop frame — display thread is slow, that's fine

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    # ── Main thread: display loop ─────────────────────────────────────────────
    target_ms = max(1, int(1000 / fps))

    if show_display:
        while True:
            try:
                frame_out = display_q.get(timeout=5.0)
            except queue.Empty:
                break
            if frame_out is None:
                break

            disp = frame_out
            if width > 960 or height > 540:
                s    = min(960 / width, 540 / height)
                disp = cv2.resize(frame_out, (int(width * s), int(height * s)))

            try:
                cv2.imshow("Detonix ZoomGuard  (Q to skip)", disp)
                key = cv2.waitKey(target_ms) & 0xFF
                if key in (ord('q'), 27):
                    stop_flag.set()
                    break
            except cv2.error as exc:
                # imshow not supported (opencv-python-headless) — switch to headless
                logger.warning(
                    "cv2.imshow unavailable (%s). "
                    "Switching to headless mode.\n"
                    "  Fix: pip uninstall opencv-python-headless && "
                    "pip install opencv-python", exc)
                show_display = False
                # drain the queue and let worker finish
                while not display_q.empty():
                    try: display_q.get_nowait()
                    except queue.Empty: break
                break
    else:
        t.join()

    stop_flag.set()
    t.join(timeout=10)
    print()   # newline after progress bar

    cap.release()
    writer.release()
    orch.end_session()
    log_path = orch.finalize()
    return orch.get_summary()


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║          DETONIX  ZOOMGUARD                              ║
║          Real-Time Deepfake Detection System             ║
║          QAU CS Final Year  |  Dr. Syed M Naqi           ║
╚══════════════════════════════════════════════════════════╝""")


def main():
    print_banner()
    args   = parse_args()
    setup_logging('logs')
    logger = logging.getLogger("Main")

    # ── Display capability check ──────────────────────────────────────────────
    show_display = not args.no_display
    if show_display:
        show_display = _check_display()
        if not show_display:
            print(
                "\n  [INFO] OpenCV window support not available on this install.\n"
                "         Running in headless mode (output saved to video file).\n"
                "         To enable live preview:\n"
                "           pip uninstall opencv-python-headless\n"
                "           pip install opencv-python\n"
            )

    # ── Dashboard ─────────────────────────────────────────────────────────────
    if args.dashboard:
        try:
            from dashboard_server import start_server
            start_server(port=5050)
            import webbrowser
            webbrowser.open("http://localhost:5050")
        except Exception as exc:
            logger.warning("Dashboard could not start: %s", exc)

    # ── Build config ──────────────────────────────────────────────────────────
    config = {
        'model_path':             args.model,
        'deepfake_threshold':     args.threshold,
        'process_every_n_frames': args.skip,
        'face_confidence':        args.face_conf,
        'min_face_size':          60,
        'max_age':                args.max_age,
        'n_init':                 2,
        'max_iou_distance':       0.75,
        'log_dir':                'logs',
    }

    os.makedirs(args.output, exist_ok=True)

    # ── Collect video files ───────────────────────────────────────────────────
    if args.video:
        videos = [Path(args.video)]
    else:
        inp = Path(args.input)
        if not inp.exists():
            print(f"\n  [ERROR] Input folder '{inp}' not found.\n"
                  f"          Create it and drop your videos in, or use:\n"
                  f"          python main.py --video path/to/clip.mp4\n")
            sys.exit(1)
        videos = sorted([p for p in inp.iterdir()
                         if p.suffix.lower() in SUPPORTED_FORMATS])

    if not videos:
        print(f"\n  [ERROR] No supported videos found in '{args.input}/'.\n"
              f"          Supported formats: {', '.join(sorted(SUPPORTED_FORMATS))}\n")
        sys.exit(1)

    print(f"\n  Found {len(videos)} video(s)  |  "
          f"threshold: {args.threshold}  |  "
          f"skip: {args.skip}  |  "
          f"display: {'on' if show_display else 'off'}")
    if args.dashboard:
        print("  Dashboard: http://localhost:5050")
    print()

    # ── Process each video ────────────────────────────────────────────────────
    all_results = {}
    for vp in videos:
        ts  = datetime.now().strftime('%H%M%S')
        out = Path(args.output) / f"zoomguard_{vp.stem}_{ts}.mp4"
        res = process_video(vp, out, config, show_display)
        if res:
            all_results[vp.name] = res

    # ── Final summary ─────────────────────────────────────────────────────────
    fakes = sum(
        sum(1 for info in r.values() if info.get('is_deepfake'))
        for r in all_results.values()
    )
    print(f"\n  Done  |  {len(all_results)} video(s) processed  |  "
          f"{fakes} deepfake track(s) flagged")
    print(f"  Output : {args.output}/")
    print(f"  Logs   : logs/\n")

    if show_display:
        cv2.destroyAllWindows()

    if args.dashboard:
        print("  Dashboard still running at http://localhost:5050")
        print("  Press Ctrl+C to exit.\n")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()