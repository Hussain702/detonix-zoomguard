"""
Quick installation test for Detonix ZoomGuard.
Run this first to verify all components work.
"""

import sys
import os


def check(name, fn):
    try:
        fn()
        print(f"  ✓  {name}")
        return True
    except Exception as e:
        print(f"  ✗  {name}: {e}")
        return False


def main():
    print("\n╔══════════════════════════════════════════════╗")
    print("║  DETONIX ZOOMGUARD — Installation Test      ║")
    print("╚══════════════════════════════════════════════╝\n")

    results = []

    results.append(check("Python version", lambda: (
        (_ := sys.version_info),
        (_ >= (3, 8)) or (_.__setitem__(0, 1/0))  # force error if too old
    )))

    results.append(check("OpenCV", lambda: (
        __import__('cv2'),
        print(f"       cv2 version: {__import__('cv2').__version__}", end="")
    )))

    results.append(check("NumPy", lambda: __import__('numpy')))

    results.append(check("PyTorch", lambda: (
        torch := __import__('torch'),
        print(f"       torch version: {torch.__version__}", end="")
    )))

    results.append(check("TorchVision", lambda: __import__('torchvision')))

    results.append(check("PIL / Pillow", lambda: __import__('PIL')))

    results.append(check("SciPy", lambda: __import__('scipy')))

    # MediaPipe is optional
    mp_ok = check("MediaPipe (optional)", lambda: __import__('mediapipe'))
    if not mp_ok:
        print("       (Will use OpenCV Haar Cascades instead)")

    print()

    results.append(check("ZoomGuard: DeepSort", lambda: (
        sys.path.insert(0, os.path.dirname(__file__)),
        __import__('utils.deep_sort', fromlist=['DeepSortTracker'])
    )))

    results.append(check("ZoomGuard: FaceDetector", lambda:
        __import__('utils.face_detector', fromlist=['FaceDetector'])
    ))

    results.append(check("ZoomGuard: DeepfakeDetector", lambda:
        __import__('utils.deepfake_model', fromlist=['DeepfakeDetector'])
    ))

    results.append(check("ZoomGuard: Orchestrator", lambda:
        __import__('orchestrator', fromlist=['DetectionOrchestrator'])
    ))

    print()

    # Try a quick inference
    try:
        import numpy as np
        from utils.deepfake_model import DeepfakeDetector
        det = DeepfakeDetector()
        dummy_face = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        score = det.predict(dummy_face)
        print(f"  ✓  Model inference test (score: {score:.4f})")
        results.append(True)
    except Exception as e:
        print(f"  ✗  Model inference test: {e}")
        results.append(False)

    # Check input folder
    os.makedirs("input_videos", exist_ok=True)
    os.makedirs("output_results", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    print("  ✓  Folders created: input_videos/, output_results/, logs/")

    passed = sum(1 for r in results if r)
    total = len(results)

    print(f"\n{'='*50}")
    print(f"  Results: {passed}/{total} checks passed")

    if passed == total:
        print("  ✓  All checks passed! ZoomGuard is ready.")
        print("\n  Next steps:")
        print("  1. Drop video(s) into: input_videos/")
        print("  2. Run: python main.py")
        print("  3. Check: output_results/")
    else:
        print("  Some checks failed. Run:")
        print("  pip install -r requirements.txt")

    print('='*50 + "\n")


if __name__ == "__main__":
    main()
