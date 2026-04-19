"""
DeepSORT Tracker — Detonix ZoomGuard

Scoring fixes in this version:
✔ EMA alpha reduced 0.25 → 0.08  (single bad frame cannot spike the score)
✔ Minimum 8 frames before ANY verdict (was 2 — way too few)
✔ REAL threshold lowered  0.40 → 0.35  (easier to confirm REAL)
✔ FAKE threshold raised   0.65 → 0.72  (harder to trigger FAKE)
✔ Rolling median guard: last 5 raw scores median must exceed 0.55 to call FAKE
  — this directly kills "nanosecond FAKE on sudden movement" bug
✔ Track starts with smoothed_score=0.5, is_uncertain=True (neutral until data)
✔ All tracker / Kalman / matching logic unchanged from previous version
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from utils.temporal_classifier import TemporalAggregator
except ImportError:
    class TemporalAggregator:
        def update(self, _): pass
        def get(self): return None


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_distance(a, b):
    if a is None or b is None:
        return 1.0
    an = a / (np.linalg.norm(a) + 1e-6)
    bn = b / (np.linalg.norm(b) + 1e-6)
    return float(1.0 - np.dot(an, bn))


def _gallery_distance(gallery, embedding):
    if not gallery or embedding is None:
        return 1.0
    return float(min(_cosine_distance(g, embedding) for g in gallery))


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    union = aw * ah + bw * bh - inter
    return inter / max(union, 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# Kalman Filter  (constant-velocity model)
# ─────────────────────────────────────────────────────────────────────────────

class KalmanFilter:
    ndim, dt = 4, 1.0

    def __init__(self):
        n = self.ndim
        self.F = np.eye(2 * n)
        for i in range(n):
            self.F[i, n + i] = self.dt
        self.H = np.eye(n, 2 * n)
        self.Q = np.diag([0.01] * n + [0.1] * n)
        self.R = np.eye(n) * 1.5

    def initiate(self, measurement):
        mean = np.r_[measurement, np.zeros(self.ndim)]
        cov  = np.diag([10.0] * self.ndim + [100.0] * self.ndim)
        return mean, cov

    def predict(self, mean, cov):
        return self.F @ mean, self.F @ cov @ self.F.T + self.Q

    def update(self, mean, cov, measurement):
        S = self.H @ cov @ self.H.T + self.R
        K = cov @ self.H.T @ np.linalg.inv(S)
        mean = mean + K @ (measurement - self.H @ mean)
        cov  = (np.eye(len(mean)) - K @ self.H) @ cov
        return mean, cov


# ─────────────────────────────────────────────────────────────────────────────
# Scoring constants  (all tunable in one place)
# ─────────────────────────────────────────────────────────────────────────────

_EMA_ALPHA       = 0.08   # slow decay — motion blur won't spike score
_MIN_FRAMES      = 8      # frames required before issuing any verdict
_FAKE_THRESH     = 0.72   # smoothed score must exceed this to call FAKE
_REAL_THRESH     = 0.35   # smoothed score must be below this to call REAL
_MIN_CONFIDENCE  = 0.20   # |score - 0.5| * 2 must exceed this
_MEDIAN_WINDOW   = 5      # rolling window size for motion-spike guard
_MEDIAN_FAKE_MIN = 0.55   # rolling median must also exceed this to call FAKE


# ─────────────────────────────────────────────────────────────────────────────
# Track
# ─────────────────────────────────────────────────────────────────────────────

class Track:
    TENTATIVE, CONFIRMED, DELETED = 1, 2, 3
    _GALLERY_SIZE = 20

    def __init__(self, mean, cov, track_id, n_init=3, max_age=60):
        self.mean, self.covariance = mean, cov
        self.track_id = track_id
        self.hits = 1
        self.age  = 1
        self.time_since_update = 0
        self.state    = Track.TENTATIVE
        self._n_init  = n_init
        self._max_age = max_age

        self.embedding = None
        self._gallery: list = []
        self.deepfake_scores: list = []
        self._temporal = TemporalAggregator()

        # Result fields — start neutral/uncertain until enough frames seen
        self.is_deepfake    = False
        self.is_uncertain   = True    # stays True until _MIN_FRAMES reached
        self.smoothed_score = 0.5     # neutral starting point
        self.confidence     = 0.0

    # ── geometry ──────────────────────────────────────────────────────────────

    def to_xyah(self):
        return self.mean[:4].copy()

    def to_tlwh(self):
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, w, h])

    def center(self):
        return np.array([self.mean[0], self.mean[1]])

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age               += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.time_since_update = 0
        self.hits += 1

        emb = getattr(detection, 'embedding', None)
        if emb is not None:
            alpha = max(0.10, 0.30 - 0.005 * self.hits)
            self.embedding = emb.copy() if self.embedding is None \
                             else (1 - alpha) * self.embedding + alpha * emb
            self._gallery.append(emb.copy())
            if len(self._gallery) > self._GALLERY_SIZE:
                self._gallery.pop(0)

        if self.state == Track.TENTATIVE and self.hits >= self._n_init:
            self.state = Track.CONFIRMED

    def mark_missed(self):
        if self.state == Track.TENTATIVE:
            self.state = Track.DELETED
        elif self.time_since_update > self._max_age:
            self.state = Track.DELETED

    def is_deleted(self):   return self.state == Track.DELETED
    def is_confirmed(self): return self.state == Track.CONFIRMED

    # ── deepfake scoring ──────────────────────────────────────────────────────

    def add_deepfake_score(self, score: float):
        """
        Update verdict from a new model score.

        Three-layer protection against false FAKE calls:
          1. Slow EMA   — a single high score barely moves the average.
          2. Min frames — no verdict at all until enough data is collected.
          3. Median guard — even if EMA drifts up, the RECENT raw scores
             must also be high. Rapid movement produces 1-2 bad frames then
             returns to normal; the median stays low → stays REAL.
        """
        if self.hits < 2:
            return

        self._temporal.update(score)
        self.deepfake_scores.append(score)

        # 1. Slow EMA
        self.smoothed_score = (
            score
            if len(self.deepfake_scores) == 1
            else _EMA_ALPHA * score + (1 - _EMA_ALPHA) * self.smoothed_score
        )

        # 2. Confidence = distance from 0.5, normalised to [0, 1]
        self.confidence = abs(self.smoothed_score - 0.5) * 2.0

        # 3. Not enough frames yet → stay UNCERTAIN, never flash FAKE
        if len(self.deepfake_scores) < _MIN_FRAMES:
            self.is_uncertain = True
            self.is_deepfake  = False
            return

        # 4. Rolling median of most recent raw scores (motion-spike guard)
        recent_median = float(np.median(
            self.deepfake_scores[-_MEDIAN_WINDOW:]
        ))

        # 5. Verdict
        if (self.smoothed_score >= _FAKE_THRESH
                and self.confidence  >= _MIN_CONFIDENCE
                and recent_median    >= _MEDIAN_FAKE_MIN):
            self.is_deepfake  = True
            self.is_uncertain = False

        elif (self.smoothed_score <= _REAL_THRESH
              and self.confidence  >= _MIN_CONFIDENCE):
            self.is_deepfake  = False
            self.is_uncertain = False

        else:
            # Score between thresholds or not confident enough yet
            self.is_uncertain = True
            self.is_deepfake  = False


# ─────────────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────────────

class Detection:
    def __init__(self, tlwh, confidence: float, embedding=None):
        self.tlwh       = np.asarray(tlwh, dtype=float)
        self.confidence = float(confidence)
        self.embedding  = embedding

    def to_xyah(self):
        x, y, w, h = self.tlwh
        return np.array([x + w / 2, y + h / 2, w / max(h, 1e-6), h])

    def center(self):
        x, y, w, h = self.tlwh
        return np.array([x + w / 2, y + h / 2])


# ─────────────────────────────────────────────────────────────────────────────
# Matching
# ─────────────────────────────────────────────────────────────────────────────

def _build_appearance_cost(tracks, detections, t_idx, d_idx):
    cost = np.ones((len(t_idx), len(d_idx)))
    for i, ti in enumerate(t_idx):
        for j, di in enumerate(d_idx):
            cost[i, j] = _gallery_distance(
                tracks[ti]._gallery, detections[di].embedding)
    return cost


def _build_fused_cost(tracks, detections, t_idx, d_idx):
    cost = np.ones((len(t_idx), len(d_idx)))
    for i, ti in enumerate(t_idx):
        t      = tracks[ti]
        t_box  = t.to_tlwh()
        t_ctr  = t.center()
        t_diag = np.hypot(t_box[2], t_box[3]) + 1e-6
        for j, di in enumerate(d_idx):
            d = detections[di]
            iou_c  = 1.0 - _iou(t_box, d.tlwh)
            dist_c = min(np.linalg.norm(t_ctr - d.center()) / t_diag, 1.0)
            app_c  = _gallery_distance(t._gallery, d.embedding)
            cost[i, j] = 0.50 * iou_c + 0.25 * dist_c + 0.25 * app_c
    return cost


def _hungarian(cost, t_idx, d_idx, threshold):
    if cost.size == 0:
        return [], list(t_idx), list(d_idx)
    rows, cols = linear_sum_assignment(cost)
    matches, unmatched_t, unmatched_d = [], set(t_idx), set(d_idx)
    for r, c in zip(rows, cols):
        if cost[r, c] <= threshold:
            matches.append((t_idx[r], d_idx[c]))
            unmatched_t.discard(t_idx[r])
            unmatched_d.discard(d_idx[c])
    return matches, list(unmatched_t), list(unmatched_d)


def _match(tracks, detections, max_iou_distance=0.75):
    if not tracks or not detections:
        return [], list(range(len(tracks))), list(range(len(detections)))

    confirmed_t   = [i for i, t in enumerate(tracks) if t.is_confirmed()]
    unconfirmed_t = [i for i, t in enumerate(tracks)
                     if not t.is_confirmed() and not t.is_deleted()]
    all_d = list(range(len(detections)))

    if confirmed_t:
        m1, ut1, ud1 = _hungarian(
            _build_appearance_cost(tracks, detections, confirmed_t, all_d),
            confirmed_t, all_d, threshold=0.55)
    else:
        m1, ut1, ud1 = [], list(confirmed_t), all_d

    remaining_t = ut1 + unconfirmed_t
    if remaining_t and ud1:
        m2, ut2, ud2 = _hungarian(
            _build_fused_cost(tracks, detections, remaining_t, ud1),
            remaining_t, ud1, threshold=max_iou_distance)
    else:
        m2, ut2, ud2 = [], remaining_t, ud1

    return m1 + m2, ut2, ud2


# ─────────────────────────────────────────────────────────────────────────────
# DeepSortTracker
# ─────────────────────────────────────────────────────────────────────────────

class DeepSortTracker:
    def __init__(
        self,
        n_init:           int   = 3,
        max_age:          int   = 60,
        max_iou_distance: float = 0.75,
    ):
        self.kf            = KalmanFilter()
        self.tracks: list  = []
        self.next_id       = 1
        self._n_init       = n_init
        self._max_age      = max_age
        self._max_iou_dist = max_iou_distance

    def predict(self):
        for t in self.tracks:
            t.predict(self.kf)

    def update(self, detections: list):
        matches, unmatched_t, unmatched_d = _match(
            self.tracks, detections, self._max_iou_dist)

        for ti, di in matches:
            self.tracks[ti].update(self.kf, detections[di])

        for ti in unmatched_t:
            self.tracks[ti].mark_missed()

        for di in unmatched_d:
            det = detections[di]
            mean, cov = self.kf.initiate(det.to_xyah())
            t = Track(mean, cov, self.next_id,
                      n_init=self._n_init, max_age=self._max_age)
            if det.embedding is not None:
                t.embedding = det.embedding.copy()
                t._gallery  = [det.embedding.copy()]
            self.tracks.append(t)
            self.next_id += 1

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def get_active_tracks(self) -> list:
        return list(self.tracks)

    def get_tracks(self) -> list:
        return self.get_active_tracks()

    def get_confirmed_tracks(self) -> list:
        return [t for t in self.tracks if t.is_confirmed()]