"""
Temporal Classifier — Detonix ZoomGuard
Implements proper temporal aggregation over XceptionNet frame embeddings.

Architecture:
  XceptionNet -> frame logits -> EMA aggregator -> video-level decision

This replaces all fixed-threshold / short-window rule-based logic with a
proper probabilistic temporal model per tracked identity.
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)


class TemporalAggregator:
    """
    Per-track temporal aggregation using Exponential Moving Average (EMA)
    with Bayesian-style confidence accumulation.

    Design principles:
    - Treats each XceptionNet output as a noisy observation of the true
      deepfake probability, not a hard classification signal.
    - Uses EMA to weight recent observations more heavily while maintaining
      historical context.
    - Confidence = how CONSISTENT the signal has been (low variance = high confidence).
    - Final label derived from probability + confidence, not raw thresholds.
    - Minimum observation window before any label is committed.

    Three-class output:
      REAL        — aggregated probability < 0.35 with confidence >= 0.5
      DEEPFAKE    — aggregated probability > 0.85 with confidence >= 0.6
      UNCERTAIN   — everything else
    """

    # EMA decay: alpha=0.25 means recent frames matter more but history matters too
    # Higher alpha = more responsive but noisier
    # Lower alpha = more stable but slower to react
    EMA_ALPHA = 0.30

    # Minimum frames before we commit to any non-UNCERTAIN label
    MIN_FRAMES_REAL     = 6   # need 10 frames below threshold to confirm REAL
    MIN_FRAMES_DEEPFAKE = 6    # need 6 consecutive frames above threshold

    PROB_DEEPFAKE = 0.75
    PROB_REAL     = 0.30
    CONF_MIN      = 0.40


    def __init__(self):
        self.ema_prob      = None   # EMA of deepfake probability
        self.obs_count     = 0      # number of observations received
        self.variance_ema  = 0.0    # EMA of squared deviation (for consistency)
        self._raw_history  = []     # last 30 raw scores for variance computation
        self._label        = "Analyzing..."
        self._is_deepfake  = False
        self._is_uncertain = True
        self._confidence   = 0.0
        self._frames_above = 0      # consecutive frames above threshold
        self._frames_below = 0      # consecutive frames below threshold
    def _calibrate(self, p: float) -> float:
       """Simple probability calibration (fixes model bias)."""
       return max(0.0, min(1.0, (p - 0.1) / 0.8))
    
    def update(self, raw_prob: float):
        """
        Update temporal model with a new XceptionNet output probability.

        Args:
            raw_prob: float in [0, 1], XceptionNet fake probability for this frame
        """
        raw_prob = self._calibrate(raw_prob)
        self.obs_count += 1

        # Initialize EMA on first observation
        if self.ema_prob is None:
            self.ema_prob     = raw_prob
            self.variance_ema = 0.0
        else:
            # Update EMA
            prev_ema      = self.ema_prob
            self.ema_prob = (self.EMA_ALPHA * raw_prob
                             + (1 - self.EMA_ALPHA) * self.ema_prob)

            # Update variance EMA (measure of signal consistency)
            deviation          = (raw_prob - prev_ema) ** 2
            self.variance_ema  = (0.15 * deviation
                                  + 0.85 * self.variance_ema)

        # Track raw history for additional statistics
        self._raw_history.append(raw_prob)
        if len(self._raw_history) > 30:
            self._raw_history.pop(0)

        # Update consecutive frame counters
        if self.ema_prob > self.PROB_DEEPFAKE:
            self._frames_above += 1
            self._frames_below  = 0
        elif self.ema_prob < self.PROB_REAL:
            self._frames_below += 1
            self._frames_above  = 0
        else:
            # Borderline: decay both counters
            self._frames_above = max(0, self._frames_above - 1)
            self._frames_below = max(0, self._frames_below - 1)

        # Compute confidence from signal consistency
        # Low variance = consistent signal = high confidence
        # variance_ema near 0 means signal is stable
        # variance_ema near 0.25 (max for binary) means completely noisy
        self._confidence = max(0.2, min(1.0, 1.0 - 2.5 * self.variance_ema))

        # Commit to label
        self._decide()

    def _decide(self):
        """
        Video-level decision from aggregated probability + confidence.
        No single frame dominates. Must satisfy BOTH probability AND confidence.
        """
        p    = self.ema_prob
        conf = self._confidence
        n    = self.obs_count

        if n < 3:
            # Not enough data
            self._label        = "Analyzing..."
            self._is_deepfake  = False
            self._is_uncertain = True
            self._confidence   = 0.0
            return
        deepfake_thresh = 0.75 if not self._is_deepfake else 0.70
        real_thresh     = 0.30 if self._is_deepfake else 0.35
        # DEEPFAKE: high aggregated probability + consistent signal + enough frames
        if (p > self.PROB_DEEPFAKE
                and conf >= self.CONF_MIN
                and self._frames_above >= self.MIN_FRAMES_DEEPFAKE):
            self._is_deepfake  = True
            self._is_uncertain = False
            self._label        = f"DEEPFAKE ({p:.2f})"

        # REAL: low aggregated probability + consistent signal + enough frames
        elif (p < self.PROB_REAL
                  and conf >= self.CONF_MIN
                  and self._frames_below >= self.MIN_FRAMES_REAL):
            self._is_deepfake  = False
            self._is_uncertain = False
            self._label        = f"REAL ({1-p:.2f})"

        # UNCERTAIN: everything else
        else:
            self._is_deepfake  = False
            self._is_uncertain = True
            self._label        = f"UNCERTAIN ({p:.2f})"

    def reset_window(self):
        """
        Called when face disappears from frame (potential person change).
        Soft reset: keep history but reset consecutive counters so a new
        person appearing doesn't inherit the previous person's verdict.
        """
        self._frames_above = 0
        self._frames_below = 0
        # Partial reset of EMA — move toward neutral
        if self.ema_prob is not None:
             self.ema_prob = 0.8 * self.ema_prob + 0.2 * 0.5
        self.variance_ema = min(0.1, self.variance_ema)

    @property
    def label(self)        -> str:   return self._label
    @property
    def is_deepfake(self)  -> bool:  return self._is_deepfake
    @property
    def is_uncertain(self) -> bool:  return self._is_uncertain
    @property
    def confidence(self)   -> float: return self._confidence
    @property
    def smoothed_score(self) -> float:
        return self.ema_prob if self.ema_prob is not None else 0.5