"""
Model Downloader for Detonix ZoomGuard
Downloads a pretrained deepfake detection model (FaceForensics++ trained XceptionNet).

Options:
1. EfficientNet-based deepfake detector (recommended, small)
2. XceptionNet FaceForensics++ (if available publicly)

Run: python download_model.py
"""

import os
import sys
import urllib.request
import hashlib
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def download_file(url, dest_path, expected_md5=None):
    """Download a file with progress bar."""
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)

    def progress_hook(count, block_size, total_size):
        pct = min(100, count * block_size * 100 // total_size) if total_size > 0 else 0
        bar = "█" * (pct // 5) + "░" * (20 - pct // 5)
        print(f"\r  [{bar}] {pct}%  ", end="", flush=True)

    try:
        logger.info(f"Downloading from: {url}")
        urllib.request.urlretrieve(url, dest_path, progress_hook)
        print()

        if expected_md5:
            with open(dest_path, 'rb') as f:
                md5 = hashlib.md5(f.read()).hexdigest()
            if md5 != expected_md5:
                logger.error("MD5 mismatch! File may be corrupted.")
                return False

        logger.info(f"Saved to: {dest_path}")
        return True

    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def main():
    print("""
╔══════════════════════════════════════════════════════════╗
║       DETONIX ZOOMGUARD — MODEL DOWNLOADER               ║
╚══════════════════════════════════════════════════════════╝

This script helps you set up a pretrained deepfake detection model.

NOTE: Without a custom model, ZoomGuard uses a pretrained MobileNetV2
backbone with random final layer weights (untrained for deepfakes).
For real deepfake detection, you need a fine-tuned model.

Options:
  1. Use built-in pretrained backbone (works out of box, no deepfake training)
  2. Provide your own model path (recommended for real results)
  3. Instructions to get FaceForensics++ trained XceptionNet model

""")

    print("="*60)
    print("RECOMMENDED: Get the FaceForensics++ XceptionNet model")
    print("="*60)
    print("""
1. Visit: https://github.com/ondyari/FaceForensics
   Request access to FaceForensics++ dataset and pretrained models.

2. Or use this open-source alternative:
   https://github.com/IIIT-SEHGAL/FakeVideo-Detector
   
3. Or train your own on these datasets:
   - FaceForensics++ (ff++): https://github.com/ondyari/FaceForensics
   - Celeb-DF: https://github.com/yuezunli/celeb-deepfakeforensics
   - DFDC: https://ai.facebook.com/datasets/dfdc/

Once you have a .pth model file, run:
   python main.py --model path/to/your_model.pth

For testing WITHOUT a trained model (to verify the pipeline works):
   python main.py
   (will use pretrained MobileNetV2 backbone — detection won't be accurate)
""")

    print("="*60)
    print("QUICK START without trained model:")
    print("="*60)
    print("""
The system will still run and track all faces correctly.
The deepfake detection scores will be based on generic
ImageNet features — not reliable for deepfake detection,
but the pipeline, tracking, and visualization all work.

To verify the full pipeline:
  1. Drop a video in input_videos/
  2. Run: python main.py
  3. Check output_results/ for annotated video
""")


if __name__ == "__main__":
    main()
