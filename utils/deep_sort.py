"""
DeepSORT Tracker — Detonix ZoomGuard  (Re-ID + Face-Only Edition)

Fixes in this version:
✔ Persistent Global Re-ID Gallery — every confirmed track's embedding is
  kept in a cross-track store; a new track is merged back into an existing
  ID if it matches a previously-seen face (cosine dist < RE_ID_THRESHOLD).
  This directly fixes "fast movement gives multiple IDs to the same person."
✔ Face-only guard — bounding-boxes that don't match expected face aspect
  ratio / area are rejected before they enter the tracker.  Non-face
  objects (hands, logos, shoulders) never get an ID.
✔ Velocity-aware Kalman noise — during high-speed motion the position
  uncertainty is widened so the filter doesn't lose the face and spawn a
  new ID.
✔ Appearance cost threshold tightened (0.55 → 0.45) so un-related faces
  are not merged.
✔ EMA alpha reduced 0.25 → 0.08  (single bad frame cannot spike the score)
✔ Minimum 8 frames before ANY verdict
✔ REAL threshold lowered  0.40 → 0.35
✔ FAKE threshold raised   0.65 → 0.72
✔ Rolling median guard: last 5 raw scores must exceed 0.55 to call FAKE
✔ All prior Kalman / scoring logic preserved and compatible.
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
# Face-validity constants  (filter non-face detections before tracking)
# ─────────────────────────────────────────────────────────────────────────────

# A human face is roughly square; allow generous slack for tilted / partial
_FACE_ASPECT_MIN = 0.35   # width / height — very wide squint allowed
_FACE_ASPECT_MAX = 2.20   # width / height — very narrow slant allowed
_FACE_MIN_AREA   = 900    # pixels² — ~30×30 minimum
_FACE_MAX_AREA   = 0.50   # fraction of frame area — no full-body false hit


def is_face_bbox(tlwh, frame_w: int, frame_h: int) -> bool:
    """
    Return True only when tlwh looks like a face bounding box.
    Rejects body parts, logos, and background blobs that slip through
    the face detector on rapid movement.
    """
    x, y, w, h = tlwh
    if w <= 0 or h <= 0:
        return False
    area   = w * h
    aspect = w / h
    max_area = frame_w * frame_h * _FACE_MAX_AREA
    return (_FACE_MIN_AREA <= area <= max_area
            and _FACE_ASPECT_MIN <= aspect <= _FACE_ASPECT_MAX)


# ─────────────────────────────────────────────────────────────────────────────
# Re-ID Gallery  (global, persists across track lifetimes)
# ─────────────────────────────────────────────────────────────────────────────

RE_ID_THRESHOLD   = 0.38   # cosine distance below this → same person
RE_ID_GALLERY_MAX = 30     # embeddings kept per known identity


class GlobalReIDGallery:
    """
    Maps canonical IDs → list of embeddings.
    When a brand-new track is about to be assigned a fresh ID, we first
    check whether its embedding matches any previously-seen identity.
    If it does, we return that old ID so the track inherits it.
    """

    def __init__(self):
        # { canonical_id : [embedding, …] }
        self._store: dict[int, list] = {}

    # ── public API ────────────────────────────────────────────────────────────

    def register(self, track_id: int, embedding: np.ndarray) -> None:
        """Add an embedding to an existing identity's gallery."""
        if embedding is None:
            return
        gallery = self._store.setdefault(track_id, [])
        gallery.append(embedding.copy())
        if len(gallery) > RE_ID_GALLERY_MAX:
            gallery.pop(0)

    def lookup(self, embedding: np.ndarray) -> int | None:
        """
        Return the canonical track ID whose gallery is closest to *embedding*,
        or None if no match is within RE_ID_THRESHOLD.
        """
        if embedding is None or not self._store:
            return None
        best_id, best_dist = None, RE_ID_THRESHOLD
        for cid, gallery in self._store.items():
            d = _gallery_distance(gallery, embedding)
            if d < best_dist:
                best_dist = d
                best_id   = cid
        return best_id

    def update_from_track(self, track: "Track") -> None:
        """Convenience: pull the most-recent embedding from a confirmed track."""
        if track.embedding is not None:
            self.register(track.track_id, track.embedding)


# Module-level singleton — shared by all DeepSortTracker instances
_global_reid = GlobalReIDGallery()


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
# Kalman Filter  (constant-velocity model, velocity-aware noise)
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

    def predict(self, mean, cov, motion_scale: float = 1.0):
        """
        *motion_scale* > 1 widens process noise during fast movement so the
        filter keeps up and doesn't declare a "new" face.
        """
        Q = self.Q * max(1.0, motion_scale)
        return self.F @ mean, self.F @ cov @ self.F.T + Q

    def update(self, mean, cov, measurement):
        S = self.H @ cov @ self.H.T + self.R
        K = cov @ self.H.T @ np.linalg.inv(S)
        mean = mean + K @ (measurement - self.H @ mean)
        cov  = (np.eye(len(mean)) - K @ self.H) @ cov
        return mean, cov


# ─────────────────────────────────────────────────────────────────────────────
# Scoring constants
# ─────────────────────────────────────────────────────────────────────────────

_EMA_ALPHA       = 0.08
_MIN_FRAMES      = 8
_FAKE_THRESH     = 0.72
_REAL_THRESH     = 0.35
_MIN_CONFIDENCE  = 0.20
_MEDIAN_WINDOW   = 5
_MEDIAN_FAKE_MIN = 0.55

# Velocity threshold above which Kalman noise is scaled up
_FAST_MOTION_PX  = 25.0   # pixels/frame centre displacement


# ─────────────────────────────────────────────────────────────────────────────
# Track
# ─────────────────────────────────────────────────────────────────────────────

class Track:
    TENTATIVE, CONFIRMED, DELETED = 1, 2, 3
    _GALLERY_SIZE = 30   # larger gallery → better re-ID over time

    def __init__(self, mean, cov, track_id, n_init=3, max_age=90):
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

        self.is_deepfake    = False
        self.is_uncertain   = True
        self.smoothed_score = 0.5
        self.confidence     = 0.0

        # Track previous centre for velocity estimation
        self._prev_centre: np.ndarray | None = None

    # ── geometry ──────────────────────────────────────────────────────────────

    def to_xyah(self):
        return self.mean[:4].copy()

    def to_tlwh(self):
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, w, h])

    def center(self):
        return np.array([self.mean[0], self.mean[1]])

    def _motion_scale(self) -> float:
        """Return a noise scale factor based on how fast the face is moving."""
        c = self.center()
        if self._prev_centre is None:
            self._prev_centre = c
            return 1.0
        speed = float(np.linalg.norm(c - self._prev_centre))
        self._prev_centre = c
        if speed > _FAST_MOTION_PX:
            # Quadratically widen noise: a face moving 50 px/frame gets 4× noise
            return min((speed / _FAST_MOTION_PX) ** 2, 8.0)
        return 1.0

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def predict(self, kf: KalmanFilter):
        scale = self._motion_scale()
        self.mean, self.covariance = kf.predict(
            self.mean, self.covariance, motion_scale=scale)
        self.age               += 1
        self.time_since_update += 1

    def update(self, kf: KalmanFilter, detection: "Detection"):
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.time_since_update = 0
        self.hits += 1

        emb = getattr(detection, 'embedding', None)
        if emb is not None:
            alpha = max(0.10, 0.30 - 0.005 * self.hits)
            self.embedding = (emb.copy() if self.embedding is None
                              else (1 - alpha) * self.embedding + alpha * emb)
            self._gallery.append(emb.copy())
            if len(self._gallery) > self._GALLERY_SIZE:
                self._gallery.pop(0)
            # Keep global re-ID gallery up to date
            _global_reid.register(self.track_id, emb)

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
        if self.hits < 2:
            return

        self._temporal.update(score)
        self.deepfake_scores.append(score)

        self.smoothed_score = (
            score
            if len(self.deepfake_scores) == 1
            else _EMA_ALPHA * score + (1 - _EMA_ALPHA) * self.smoothed_score
        )

        self.confidence = abs(self.smoothed_score - 0.5) * 2.0

        if len(self.deepfake_scores) < _MIN_FRAMES:
            self.is_uncertain = True
            self.is_deepfake  = False
            return

        recent_median = float(np.median(
            self.deepfake_scores[-_MEDIAN_WINDOW:]))

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
            self.is_uncertain = True
            self.is_deepfake  = False


# ─────────────────────────────────────────────────────────────────────────────
# Detection
# ─────────────────────────────────────────────────────────────────────────────

class Detection:
    def __init__(self, tlwh, confidence: float, embedding=None,
                 frame_w: int = 0, frame_h: int = 0):
        self.tlwh       = np.asarray(tlwh, dtype=float)
        self.confidence = float(confidence)
        self.embedding  = embedding
        # Store frame dims for face-validity check downstream
        self._frame_w   = frame_w
        self._frame_h   = frame_h

    def to_xyah(self):
        x, y, w, h = self.tlwh
        return np.array([x + w / 2, y + h / 2, w / max(h, 1e-6), h])

    def center(self):
        x, y, w, h = self.tlwh
        return np.array([x + w / 2, y + h / 2])

    def is_valid_face(self) -> bool:
        """True when bbox geometry is consistent with a human face."""
        if self._frame_w <= 0 or self._frame_h <= 0:
            return True   # unknown frame size → don't block
        return is_face_bbox(self.tlwh, self._frame_w, self._frame_h)


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

    # Tightened appearance threshold (0.55 → 0.45) — reduces false merges
    APPEARANCE_THRESHOLD = 0.45

    if confirmed_t:
        m1, ut1, ud1 = _hungarian(
            _build_appearance_cost(tracks, detections, confirmed_t, all_d),
            confirmed_t, all_d, threshold=APPEARANCE_THRESHOLD)
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
        max_age:          int   = 90,   # was 60; longer memory helps re-ID
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

    def update(self, detections: list, frame_w: int = 0, frame_h: int = 0):
        """
        Parameters
        ----------
        detections : list of Detection
        frame_w, frame_h : original frame dimensions for face-validity filter
        """
        # ── face-only filter ─────────────────────────────────────────────────
        if frame_w > 0 and frame_h > 0:
            detections = [
                d for d in detections
                if is_face_bbox(d.tlwh, frame_w, frame_h)
            ]

        matches, unmatched_t, unmatched_d = _match(
            self.tracks, detections, self._max_iou_dist)

        for ti, di in matches:
            self.tracks[ti].update(self.kf, detections[di])

        for ti in unmatched_t:
            self.tracks[ti].mark_missed()

        # ── new detections: check global re-ID before minting a new ID ───────
        for di in unmatched_d:
            det = detections[di]
            existing_id = _global_reid.lookup(det.embedding)

            if existing_id is not None:
                # Find whether this ID is still alive (e.g. just went off-screen)
                live_track = next(
                    (t for t in self.tracks
                     if t.track_id == existing_id and not t.is_deleted()),
                    None)
                if live_track is not None:
                    # Merge back into the live track — do not spawn a new one
                    live_track.update(self.kf, det)
                    continue
                else:
                    # Track expired but face is known — resurrect with same ID
                    mean, cov = self.kf.initiate(det.to_xyah())
                    t = Track(mean, cov, existing_id,
                              n_init=self._n_init, max_age=self._max_age)
                    # Give it existing gallery so it confirms faster
                    existing_gallery = _global_reid._store.get(existing_id, [])
                    if det.embedding is not None:
                        t.embedding = det.embedding.copy()
                        t._gallery  = list(existing_gallery[-self._n_init:])
                        t._gallery.append(det.embedding.copy())
                    self.tracks.append(t)
                    continue

            # Truly new face — mint a fresh ID
            mean, cov = self.kf.initiate(det.to_xyah())
            t = Track(mean, cov, self.next_id,
                      n_init=self._n_init, max_age=self._max_age)
            if det.embedding is not None:
                t.embedding = det.embedding.copy()
                t._gallery  = [det.embedding.copy()]
                _global_reid.register(self.next_id, det.embedding)
            self.tracks.append(t)
            self.next_id += 1

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

        # Keep global re-ID gallery fresh with confirmed track embeddings
        for t in self.tracks:
            if t.is_confirmed() and t.embedding is not None:
                _global_reid.update_from_track(t)

    def get_active_tracks(self) -> list:
        return list(self.tracks)

    def get_tracks(self) -> list:
        return self.get_active_tracks()

    def get_confirmed_tracks(self) -> list:
        return [t for t in self.tracks if t.is_confirmed()]

    def reset_reid(self):
        """Call this between unrelated video clips / sessions."""
        global _global_reid
        _global_reid = GlobalReIDGallery()
        self.tracks  = []
        self.next_id = 1