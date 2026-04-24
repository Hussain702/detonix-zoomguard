"""
calibrate.py — Detonix ZoomGuard
One-shot script to fit temperature T and decision thresholds from your own
video data, then save them to calibration.json for use at runtime.

Usage:
──────────────────────────────────────────────────────────────────────────────
1.  Collect ground-truth scores from a validation set of videos you know are
    real or fake.  The easiest way is to run ZoomGuard on those videos and
    log (raw_score, label) pairs.

2.  Run this script:
        python calibrate.py --real real_scores.txt --fake fake_scores.txt

    Each .txt file is one raw XceptionNet probability per line (0.0–1.0).

3.  The script writes  calibration.json  in the current directory.

4.  Load at runtime — add this near the top of main.py  (after imports):

        import json
        from utils.temporal_classifier import TemporalAggregator
        try:
            d = json.load(open('calibration.json'))
            TemporalAggregator.load_calibration(d['T'], d['fake_thr'], d['real_thr'])
            print(f"[ZoomGuard] Calibration loaded: T={d['T']:.3f}  "
                  f"fake_thr={d['fake_thr']:.3f}  real_thr={d['real_thr']:.3f}")
        except FileNotFoundError:
            print("[ZoomGuard] No calibration.json found — using defaults (T=1.8).")

Minimum data:
    ~20 real scores + ~20 fake scores is enough for basic fitting.
    ~200+ of each gives reliable thresholds.

How to collect scores quickly:
    Add this to orchestrator.py inside process_frame(), after predict_batch():
        for score, label in zip(scores, your_labels):
            with open('val_scores.csv', 'a') as f:
                f.write(f'{score},{label}\\n')
    Then split the CSV into real/fake text files by label.
"""

import argparse
import json
import sys
import numpy as np


def load_scores(path):
    scores = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                try:
                    scores.append(float(line))
                except ValueError:
                    pass
    return np.array(scores, dtype=float)


def main():
    p = argparse.ArgumentParser(
        description="Fit temperature scaling calibration for Detonix ZoomGuard"
    )
    p.add_argument('--real',       required=True,
                   help='Text file: one raw real-video score per line')
    p.add_argument('--fake',       required=True,
                   help='Text file: one raw fake-video score per line')
    p.add_argument('--output',     default='calibration.json',
                   help='Output JSON file (default: calibration.json)')
    p.add_argument('--target-fpr', type=float, default=0.02,
                   help='Target false-positive rate for fake threshold (default: 0.02)')
    args = p.parse_args()

    real_scores = load_scores(args.real)
    fake_scores = load_scores(args.fake)

    print(f"Loaded {len(real_scores)} real scores, {len(fake_scores)} fake scores")

    if len(real_scores) < 10 or len(fake_scores) < 10:
        print("ERROR: need at least 10 scores per class.", file=sys.stderr)
        sys.exit(1)

    probs  = np.concatenate([real_scores, fake_scores])
    labels = np.concatenate([np.zeros(len(real_scores)), np.ones(len(fake_scores))])

    # Import here so script works from project root
    sys.path.insert(0, '.')
    try:
        from utils.temporal_classifier import TemporalAggregator
    except ImportError:
        from temporal_classifier import TemporalAggregator

    info = TemporalAggregator.fit_from_validation(probs, labels, args.target_fpr)

    print(f"\nFitted calibration:")
    print(f"  Temperature T : {info['T']:.4f}")
    print(f"  Fake threshold: {info['fake_thr']:.4f}  (at FPR <= {args.target_fpr:.0%})")
    print(f"  Real threshold: {info['real_thr']:.4f}  (50th pct of calibrated real)")

    # Quick sanity check
    from scipy.special import expit, logit
    def scale(p, T): return float(expit(logit(max(1e-7, min(1-1e-7, p))) / T))
    cal_real = np.array([scale(p, info['T']) for p in real_scores])
    cal_fake = np.array([scale(p, info['T']) for p in fake_scores])
    fp = (cal_real >= info['fake_thr']).mean()
    fn = (cal_fake <  info['fake_thr']).mean()
    print(f"\nSanity check on your data:")
    print(f"  False positives (real flagged FAKE) : {fp*100:.1f}%")
    print(f"  False negatives (fake missed)       : {fn*100:.1f}%")

    with open(args.output, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"\nSaved to: {args.output}")
    print("\nNext step — add to main.py:")
    print("  import json")
    print("  from utils.temporal_classifier import TemporalAggregator")
    print("  d = json.load(open('calibration.json'))")
    print("  TemporalAggregator.load_calibration(d['T'], d['fake_thr'], d['real_thr'])")


if __name__ == '__main__':
    main()