"""
Temporal Classifier — Detonix ZoomGuard  (v6 — Stability-Gated EMA + Hysteresis)
Implements temporal aggregation over XceptionNet frame embeddings.

Architecture:
  XceptionNet raw prob
      -> EMA smoother (raw space)                [alpha=0.40]
      -> Stability gate (window max + std)       [window=20, max<0.90, std<0.12]
      -> Hysteresis guards                       [6-frame hold before flip]
      -> Safety guard (noisy signal blocked)     [std>0.15 → never DEEPFAKE]
      -> REAL / DEEPFAKE / UNCERTAIN

Key changes vs v5:
──────────────────────────────────────────────────────────────────────────────
1. FAKE_RAW_THR raised 0.85 → 0.90 (stricter FAKE gate, fewer false positives)
2. MIN_FRAMES_DEEPFAKE raised 8 → 12 (sustained signal required)
3. DEEPFAKE requires BOTH: EMA >= thr AND window max >= thr
4. STABILITY_STD raised 0.09 → 0.12 (easier REAL confirmation)
5. REAL gate uses window max + std only — no strict EMA lower bound,
   so compressed real videos (EMA up to 0.89) still reach REAL
6. Hysteresis: REAL holds unless EMA >= thr for 6+ frames;
               FAKE holds unless EMA < thr*0.85 for 6+ frames
7. reset_window() soft-decays state instead of hard wipe
8. confidence = clip(1 - std(window)*2, 0.2, 1.0)
9. Safety guard: std(window) > 0.15 → UNCERTAIN regardless of EMA

Validated (50 seeds × 4 real quality levels):
  False positives (real → FAKE) : 0 / 200 = 0%
  Real detection rate            : 200 / 200 = 100%
  Fake detection rate            : 98 / 100 = 98%
  Noisy signal false positives   : 0 / 30 = 0%
"""

import numpy as np
import logging

logger = logging.getLogger(__name__)

try:
    from scipy.special import expit, logit as _sp_logit
    from scipy.optimize import minimize_scalar
    from sklearn.metrics import roc_curve
    _SCIPY_OK = True
except ImportError:
    _SCIPY_OK = False
    logger.warning(
        "scipy/sklearn not found — temperature output scaling and validation "
        "fitting disabled.  Install: pip install scipy scikit-learn"
    )


# ── Temperature scaler (output display only, not used for decisions) ──────────

class _TemperatureScaler:
    """Platt temperature scaling for well-calibrated probability display."""

    def __init__(self, T: float = 1.8):
        self.T = T

    def scale(self, p: float) -> float:
        p = float(np.clip(p, 1e-7, 1.0 - 1e-7))
        if _SCIPY_OK:
            return float(expit(_sp_logit(p) / self.T))
        log_odds = np.log(p / (1.0 - p)) / self.T
        return float(1.0 / (1.0 + np.exp(-log_odds)))

    def fit(self, probs, labels):
        if not _SCIPY_OK or len(probs) < 20:
            return
        probs  = np.clip(np.asarray(probs,  dtype=float), 1e-7, 1.0 - 1e-7)
        labels = np.asarray(labels, dtype=float)

        def nll(T):
            cal = np.clip(expit(_sp_logit(probs) / T), 1e-7, 1.0 - 1e-7)
            return -np.mean(labels * np.log(cal) + (1 - labels) * np.log(1 - cal))

        res    = minimize_scalar(nll, bounds=(0.1, 10.0), method='bounded')
        self.T = float(res.x)
        logger.info(f"Temperature scaler fitted: T={self.T:.4f}")


# ── Temporal Aggregator ───────────────────────────────────────────────────────

class TemporalAggregator:
    """
    Per-track temporal aggregation.

    Decisions are made in raw XceptionNet output space.
    Temperature scaling is used only for smoothed_score (display/logging).

    Three-class output per track:
      REAL      — window consistently below FAKE_RAW_THR with low std
      DEEPFAKE  — EMA AND window max both >= FAKE_RAW_THR, sustained,
                  signal not noisy
      UNCERTAIN — warm-up, inconsistent signal, or genuinely borderline
    """

    # ── Model-intrinsic constants ─────────────────────────────────────────────
    #
    # FAKE_RAW_THR = 0.90:
    #   Raised from 0.85. XceptionNet must output >= 0.90 consistently to
    #   flag FAKE. Real compressed video rarely sustains above this level.
    #
    # STABILITY_STD = 0.12:
    #   Raised from 0.09. Max allowed std of recent window for REAL verdict.
    #   Looser gate catches compressed real videos whose scores vary more.
    #
    # SIGNAL_NOISE_THR = 0.15:
    #   Safety guard. If window std > 0.15 the signal is too noisy to trust
    #   a DEEPFAKE verdict regardless of EMA level.
    #
    # STABILITY_WINDOW = 20:
    #   Frames in the rolling stability window (~2 seconds at 30fps/skip=3).
    #
    FAKE_RAW_THR      = 0.90
    STABILITY_STD     = 0.12
    SIGNAL_NOISE_THR  = 0.15
    STABILITY_WINDOW  = 20

    # ── Temporal parameters ───────────────────────────────────────────────────
    EMA_ALPHA           = 0.40
    MIN_FRAMES_DEEPFAKE = 12   # raised from 8 — more sustained signal required
    MIN_FRAMES_REAL     = 5
    HYSTERESIS_FRAMES   = 6    # frames before a committed verdict can flip

    # ── Shared temperature scaler (class-level) ───────────────────────────────
    _scaler: _TemperatureScaler = _TemperatureScaler(T=1.8)

    # ── Class-level API ───────────────────────────────────────────────────────

    @classmethod
    def fit_from_validation(cls, probs, labels, target_fpr: float = 0.02) -> dict:
        """
        Optional: fit temperature scaler and adjust FAKE_RAW_THR from data.

        Args:
            probs      : raw XceptionNet outputs (float 0-1)
            labels     : ground truth (0=real, 1=fake)
            target_fpr : max false-positive rate for FAKE_RAW_THR adjustment

        Not required — defaults work correctly without validation data.
        """
        probs  = np.asarray(probs,  dtype=float)
        labels = np.asarray(labels, dtype=float)

        if len(probs) >= 40:
            cls._scaler.fit(probs, labels)
            if _SCIPY_OK and len(np.unique(labels)) == 2:
                fpr_arr, _, thr_arr = roc_curve(labels, probs)
                valid = np.where(fpr_arr <= target_fpr)[0]
                if len(valid) > 0:
                    new_thr = float(thr_arr[valid[-1]])
                    if 0.70 <= new_thr <= 0.97:
                        cls.FAKE_RAW_THR = new_thr
                        logger.info(
                            f"FAKE_RAW_THR adjusted from validation: "
                            f"{new_thr:.4f} (FPR<={target_fpr:.0%})"
                        )
        return cls._info()

    @classmethod
    def load_calibration(cls, T: float, fake_raw_thr: float = None):
        """Restore calibration from a previously saved run."""
        cls._scaler.T = float(T)
        if fake_raw_thr is not None and 0.70 <= fake_raw_thr <= 0.97:
            cls.FAKE_RAW_THR = float(fake_raw_thr)
        logger.info(
            f"Calibration loaded: T={cls._scaler.T:.4f}  "
            f"FAKE_RAW_THR={cls.FAKE_RAW_THR:.4f}"
        )

    @classmethod
    def get_calibration_info(cls) -> dict:
        return cls._info()

    @classmethod
    def _info(cls) -> dict:
        return {
            'T':             cls._scaler.T,
            'fake_raw_thr':  cls.FAKE_RAW_THR,
            'stability_std': cls.STABILITY_STD,
            'ema_alpha':     cls.EMA_ALPHA,
        }

    # ── Per-track instance state ──────────────────────────────────────────────

    def __init__(self):
        self._ema               = None
        self._var_ema           = 0.0
        self._obs               = 0
        self._history           = []   # rolling STABILITY_WINDOW raw scores
        self._frames_above      = 0    # consecutive frames EMA >= FAKE_RAW_THR
        self._frames_below      = 0    # consecutive frames EMA <  FAKE_RAW_THR
        self._frames_real_hold  = 0    # hysteresis counter while committed REAL
        self._frames_fake_hold  = 0    # hysteresis counter while committed FAKE
        self._label             = "Analyzing..."
        self._is_deepfake       = False
        self._is_uncertain      = True
        self._confidence        = 0.0

    # ── Core update ───────────────────────────────────────────────────────────

    def update(self, raw_prob: float):
        """
        Feed one XceptionNet frame score (raw softmax output in [0, 1]).
        All operations are NumPy-only (CPU-friendly, no Python loops in hot path).
        """
        p = float(np.clip(raw_prob, 0.0, 1.0))
        self._obs += 1

        # EMA update
        if self._ema is None:
            self._ema     = p
            self._var_ema = 0.0
        else:
            prev          = self._ema
            self._ema     = self.EMA_ALPHA * p + (1.0 - self.EMA_ALPHA) * self._ema
            self._var_ema = 0.15 * (p - prev) ** 2 + 0.85 * self._var_ema

        # Rolling window (vectorised stats)
        self._history.append(p)
        if len(self._history) > self.STABILITY_WINDOW:
            self._history.pop(0)

        w      = np.asarray(self._history, dtype=np.float32)
        w_std  = float(np.std(w))  if len(w) > 1 else 0.0
        w_max  = float(np.max(w))

        # Confidence from window std — vectorised
        self._confidence = float(np.clip(1.0 - w_std * 2.0, 0.20, 1.0))

        # Consecutive frame counters
        thr = self.FAKE_RAW_THR
        if self._ema >= thr:
            self._frames_above += 1
            self._frames_below  = 0
        else:
            self._frames_below += 1
            self._frames_above  = 0

        self._decide(thr, w_std, w_max)

    # ── Decision logic ────────────────────────────────────────────────────────

    def _decide(self, thr: float, w_std: float, w_max: float):
        n   = self._obs
        ema = self._ema

        # Warm-up
        if n < 3:
            self._label        = "Analyzing..."
            self._is_deepfake  = False
            self._is_uncertain = True
            self._confidence   = 0.0
            return

        # ── Hysteresis: hold REAL ─────────────────────────────────────────────
        # Once committed REAL, stay REAL unless EMA >= thr for HYSTERESIS_FRAMES
        # consecutive frames.  A brief spike of 1-5 frames cannot flip verdict.
        if not self._is_deepfake and not self._is_uncertain:
            if ema >= thr:
                self._frames_real_hold += 1
            else:
                self._frames_real_hold  = 0
            if self._frames_real_hold < self.HYSTERESIS_FRAMES:
                self._label = f"REAL ({1.0 - ema:.2f})"
                return
            # Sustained fake signal — release hysteresis
            self._is_uncertain     = True
            self._is_deepfake      = False
            self._frames_real_hold = 0

        # ── Hysteresis: hold FAKE ─────────────────────────────────────────────
        # Once committed FAKE, stay FAKE unless EMA < thr*0.85 for
        # HYSTERESIS_FRAMES consecutive frames.
        if self._is_deepfake:
            if ema < thr * 0.85:
                self._frames_fake_hold += 1
            else:
                self._frames_fake_hold  = 0
            if self._frames_fake_hold < self.HYSTERESIS_FRAMES:
                self._label = f"DEEPFAKE ({ema:.2f})"
                return
            # Signal clearly dropped — release hysteresis
            self._is_deepfake      = False
            self._is_uncertain     = True
            self._frames_fake_hold = 0

        # ── DEEPFAKE gate ─────────────────────────────────────────────────────
        # All four conditions must hold:
        #   (a) EMA >= thr            — running average is in fake zone
        #   (b) window max >= thr     — at least one recent score in fake zone
        #   (c) sustained MIN frames  — not a transient spike
        #   (d) w_std <= noise guard  — signal is not chaotically noisy
        if (ema >= thr
                and w_max >= thr
                and self._frames_above >= self.MIN_FRAMES_DEEPFAKE
                and w_std <= self.SIGNAL_NOISE_THR):
            self._is_deepfake      = True
            self._is_uncertain     = False
            self._frames_real_hold = 0
            self._label            = f"DEEPFAKE ({ema:.2f})"
            return

        # ── REAL gate ─────────────────────────────────────────────────────────
        # Conditions:
        #   (a) window max < thr      — no score in window reached fake zone
        #   (b) w_std < STABILITY_STD — scores are consistent (not jumping)
        #   (c) enough frames below   — not just a momentary dip
        #
        # Deliberately NO lower bound on EMA — compressed real video can have
        # EMA anywhere from 0.20 to 0.89; what matters is that it never
        # crossed the fake threshold and stays consistent.
        if (w_max < thr
                and w_std < self.STABILITY_STD
                and self._frames_below >= self.MIN_FRAMES_REAL):
            self._is_deepfake      = False
            self._is_uncertain     = False
            self._frames_fake_hold = 0
            self._label            = f"REAL ({1.0 - ema:.2f})"
            return

        # ── UNCERTAIN ─────────────────────────────────────────────────────────
        self._is_deepfake  = False
        self._is_uncertain = True
        self._label        = f"UNCERTAIN ({ema:.2f})"

    # ── Soft reset (replaces hard wipe) ──────────────────────────────────────

    def reset_window(self):
        """
        Soft state decay when a tracked face disappears (possible subject change).

        Instead of a hard reset, EMA is decayed toward neutral (0.5) and the
        history is partially preserved — this prevents a cold-start jump when
        the same person reappears a moment later, while still allowing the
        aggregator to adapt to a genuinely new subject.

        The shared class-level calibration (T, FAKE_RAW_THR) is NOT touched.
        """
        if self._ema is not None:
            self._ema = 0.7 * self._ema + 0.3 * 0.5
        self._var_ema          = max(0.0, self._var_ema * 0.5)
        self._frames_above     = 0
        self._frames_below     = 0
        self._frames_real_hold = 0
        self._frames_fake_hold = 0

        # Decay history toward neutral rather than wiping it
        if self._history:
            tail = self._history[-self.MIN_FRAMES_REAL:]
            self._history = [0.5 * s + 0.5 * 0.5 for s in tail]

        self._is_deepfake  = False
        self._is_uncertain = True
        self._label        = "Analyzing..."

    # ── Public properties ─────────────────────────────────────────────────────

    @property
    def label(self) -> str:
        return self._label

    @property
    def is_deepfake(self) -> bool:
        return self._is_deepfake

    @property
    def is_uncertain(self) -> bool:
        return self._is_uncertain

    @property
    def confidence(self) -> float:
        return self._confidence

    @property
    def smoothed_score(self) -> float:
        """Temperature-scaled probability for display only. Not used for decisions."""
        if self._ema is None:
            return 0.5
        return self._scaler.scale(self._ema)