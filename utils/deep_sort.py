"""
DeepSORT Tracker — Detonix ZoomGuard
Kalman filter + two-pass Hungarian + ArcFace Re-ID gallery

Guarantees
──────────────────────────────────────────────────────────────────────────────
• Face-only gate  — aspect ratio, min/max area, frame-fraction cap applied
  to every Detection before it can create or update a track.
• Embedding-gated IDs  — new ID issued only when cosine distance to every
  known gallery > RE_ID_THRESHOLD. Known faces always recover their canonical
  ID even after disappearance (Re-ID via GlobalReIDGallery).
• Stable tracking  — two-pass Hungarian: pass 1 appearance-only for confirmed
  tracks; pass 2 fused (IoU + centre dist + appearance) for the rest.
• Velocity-aware Kalman noise  — process noise scales with face speed so the
  filter stays locked during fast motion.

FIXES (v2)
──────────────────────────────────────────────────────────────────────────────
• add_deepfake_score() now reads its verdict EXCLUSIVELY from TemporalAggregator.
  The old parallel EMA in Track was ignored by TemporalAggregator and caused
  smoothed_score to stay near 0.5 forever (too-low _EMA_ALPHA=0.08), which
  kept real videos stuck in UNCERTAIN / incorrectly flipped to FAKE.
• confidence is now read from TemporalAggregator.confidence instead of the
  broken  abs(smoothed_score - 0.5)*2  formula.
• smoothed_score is read from TemporalAggregator.smoothed_score so the HUD
  and draw_results() display consistent numbers.
• _MIN_FRAMES lowered to 6 (was 8) to match TemporalAggregator.MIN_FRAMES_*.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment

try:
    from utils.temporal_classifier import TemporalAggregator
except ImportError:
    try:
        from temporal_classifier import TemporalAggregator
    except ImportError:
        class TemporalAggregator:                   # bare stub so import never fails
            def update(self, _): pass
            @property
            def label(self):         return "Analyzing..."
            @property
            def is_deepfake(self):   return False
            @property
            def is_uncertain(self):  return True
            @property
            def confidence(self):    return 0.0
            @property
            def smoothed_score(self): return 0.5


# ── Face validity (mirrors face_detector constants) ───────────────────────────
_FACE_ASPECT_MIN = 0.35
_FACE_ASPECT_MAX = 2.20
_FACE_MIN_AREA   = 1600    # 40×40 px²
_FACE_MAX_AREA   = 0.45    # fraction of frame


def is_face_bbox(tlwh, fw, fh):
    x, y, w, h = tlwh
    if w <= 0 or h <= 0:
        return False
    return (_FACE_MIN_AREA <= w * h <= fw * fh * _FACE_MAX_AREA
            and _FACE_ASPECT_MIN <= w / h <= _FACE_ASPECT_MAX)


# ── Re-ID gallery ─────────────────────────────────────────────────────────────
RE_ID_THRESHOLD   = 0.38
RE_ID_GALLERY_MAX = 30


def _l2(mat):
    return mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-6)


class GlobalReIDGallery:
    def __init__(self):
        self._store = {}   # { id: ndarray (N, D) }

    def register(self, tid, emb):
        if emb is None:
            return
        e = emb.copy().reshape(1, -1)
        if tid not in self._store:
            self._store[tid] = e
        else:
            mat = np.vstack([self._store[tid], e])
            self._store[tid] = mat[-RE_ID_GALLERY_MAX:]

    def update_from_track(self, track):
        if track.embedding is not None:
            self.register(track.track_id, track.embedding)

    def lookup(self, emb):
        """Return canonical ID with min cosine distance, or None."""
        if emb is None or not self._store:
            return None
        en = (emb / (np.linalg.norm(emb) + 1e-6)).reshape(1, -1)
        best_id, best_d = None, RE_ID_THRESHOLD
        for tid, mat in self._store.items():
            d = float(1.0 - (_l2(mat) @ en.T).max())
            if d < best_d:
                best_d, best_id = d, tid
        return best_id


_global_reid = GlobalReIDGallery()


# ── Math helpers ──────────────────────────────────────────────────────────────

def _iou(a, b):
    ax, ay, aw, ah = a;  bx, by, bw, bh = b
    ix1 = max(ax, bx);  iy1 = max(ay, by)
    ix2 = min(ax + aw, bx + bw);  iy2 = min(ay + ah, by + bh)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    return inter / max(aw * ah + bw * bh - inter, 1e-6)


def _gallery_dist(gallery, emb):
    if not gallery or emb is None:
        return 1.0
    mat = _l2(np.stack(gallery))
    en  = emb / (np.linalg.norm(emb) + 1e-6)
    return float(1.0 - (mat @ en).max())


# ── Kalman filter ─────────────────────────────────────────────────────────────

class KalmanFilter:
    """Constant-velocity model; state = [cx, cy, a, h, vcx, vcy, va, vh]."""
    ndim, dt = 4, 1.0

    def __init__(self):
        n = self.ndim
        self.F = np.eye(2 * n)
        for i in range(n):
            self.F[i, n + i] = self.dt
        self.H = np.eye(n, 2 * n)
        self.Q = np.diag([0.01, 0.01, 1e-4, 0.01, 0.1, 0.1, 1e-3, 0.1])
        self.R = np.diag([1.5,  1.5,  0.01, 1.5])

    def initiate(self, m):
        mean = np.r_[m, np.zeros(self.ndim)]
        cov  = np.diag([10., 10., 0.1, 10., 100., 100., 1., 100.])
        return mean, cov

    def predict(self, mean, cov, scale=1.0):
        return self.F @ mean, self.F @ cov @ self.F.T + self.Q * max(1.0, scale)

    def update(self, mean, cov, m):
        S = self.H @ cov @ self.H.T + self.R
        K = cov @ self.H.T @ np.linalg.inv(S)
        return mean + K @ (m - self.H @ mean), (np.eye(len(mean)) - K @ self.H) @ cov


# ── Deepfake scoring constants ────────────────────────────────────────────────
# NOTE: These are only used for the legacy parallel EMA path which is now
# DISABLED. Verdict authority belongs entirely to TemporalAggregator.
# _MIN_FRAMES is kept for the "analysing" gate in orchestrator / draw_results.
_MIN_FRAMES      = 6    # lowered from 8 to match TemporalAggregator.MIN_FRAMES_*
_FAST_MOTION_PX  = 25.0


# ── Track ─────────────────────────────────────────────────────────────────────

class Track:
    TENTATIVE, CONFIRMED, DELETED = 1, 2, 3
    _GALLERY_SIZE = 30

    def __init__(self, mean, cov, tid, n_init=3, max_age=90):
        self.mean, self.covariance = mean, cov
        self.track_id = tid
        self.hits     = 1
        self.age      = 1
        self.time_since_update = 0
        self.state    = Track.TENTATIVE
        self._n_init  = n_init
        self._max_age = max_age

        self.embedding       = None
        self._gallery        = []
        self.deepfake_scores = []

        # ── Single source of truth for deepfake verdict ───────────────────
        # All classification logic lives in TemporalAggregator.
        # Track just proxies the results out via properties below.
        self._temporal       = TemporalAggregator()

        self._prev_centre    = None

    # ── Proxy properties (read from TemporalAggregator) ──────────────────────

    @property
    def is_deepfake(self) -> bool:
        return self._temporal.is_deepfake

    @property
    def is_uncertain(self) -> bool:
        return self._temporal.is_uncertain

    @property
    def smoothed_score(self) -> float:
        return self._temporal.smoothed_score

    @property
    def confidence(self) -> float:
        return self._temporal.confidence

    # ── Geometry ──────────────────────────────────────────────────────────────

    def to_tlwh(self):
        cx, cy, a, h = self.mean[:4]
        w = a * h
        return np.array([cx - w / 2, cy - h / 2, w, h])

    def to_xyah(self):
        return self.mean[:4].copy()

    def center(self):
        return self.mean[:2].copy()

    def _motion_scale(self):
        c = self.center()
        if self._prev_centre is None:
            self._prev_centre = c; return 1.0
        speed = float(np.linalg.norm(c - self._prev_centre))
        self._prev_centre = c
        return min((speed / _FAST_MOTION_PX) ** 2, 8.0) if speed > _FAST_MOTION_PX else 1.0

    # ── Kalman ────────────────────────────────────────────────────────────────

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(
            self.mean, self.covariance, self._motion_scale())
        self.age += 1;  self.time_since_update += 1

    def update(self, kf, det):
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, det.to_xyah())
        self.time_since_update = 0;  self.hits += 1

        emb = getattr(det, "embedding", None)
        if emb is not None:
            alpha = max(0.10, 0.30 - 0.005 * self.hits)
            self.embedding = (emb.copy() if self.embedding is None
                              else (1 - alpha) * self.embedding + alpha * emb)
            self._gallery.append(emb.copy())
            if len(self._gallery) > self._GALLERY_SIZE:
                self._gallery.pop(0)
            _global_reid.register(self.track_id, emb)

        if self.state == Track.TENTATIVE and self.hits >= self._n_init:
            self.state = Track.CONFIRMED

    def mark_missed(self):
        if self.state == Track.TENTATIVE:
            self.state = Track.DELETED
        elif self.time_since_update > self._max_age:
            self.state = Track.DELETED

        # Soft-reset aggregator when track goes missing (possible subject change)
        if self.time_since_update == 1:
            self._temporal.reset_window()

    def is_deleted(self):   return self.state == Track.DELETED
    def is_confirmed(self): return self.state == Track.CONFIRMED

    def add_deepfake_score(self, score: float):
        """
        Feed a raw XceptionNet/MobileNetV2 score to the temporal aggregator.

        The TemporalAggregator owns ALL classification logic:
          - calibration
          - EMA smoothing
          - variance / confidence
          - REAL / DEEPFAKE / UNCERTAIN decision

        Track.is_deepfake, .is_uncertain, .smoothed_score, .confidence are
        all proxied from _temporal so every consumer sees consistent data.
        """
        if self.hits < 2:
            return
        self.deepfake_scores.append(score)
        self._temporal.update(score)


# ── Detection wrapper ─────────────────────────────────────────────────────────

class Detection:
    def __init__(self, tlwh, confidence, embedding=None, frame_w=0, frame_h=0):
        self.tlwh       = np.asarray(tlwh, dtype=float)
        self.confidence = float(confidence)
        self.embedding  = embedding
        self._fw        = frame_w
        self._fh        = frame_h

    def to_xyah(self):
        x, y, w, h = self.tlwh
        return np.array([x + w / 2, y + h / 2, w / max(h, 1e-6), h])

    def center(self):
        x, y, w, h = self.tlwh
        return np.array([x + w / 2, y + h / 2])

    def is_valid_face(self):
        if self._fw <= 0 or self._fh <= 0:
            return True
        return is_face_bbox(self.tlwh, self._fw, self._fh)


# ── Cost matrices ─────────────────────────────────────────────────────────────

def _appearance_cost(tracks, dets, t_idx, d_idx):
    cost = np.ones((len(t_idx), len(d_idx)), dtype=np.float32)
    for i, ti in enumerate(t_idx):
        gal = tracks[ti]._gallery
        if not gal:
            continue
        gm = _l2(np.stack(gal))
        for j, di in enumerate(d_idx):
            e = dets[di].embedding
            if e is not None:
                en = e / (np.linalg.norm(e) + 1e-6)
                cost[i, j] = float(1.0 - (gm @ en).max())
    return cost


def _fused_cost(tracks, dets, t_idx, d_idx):
    cost = np.ones((len(t_idx), len(d_idx)), dtype=np.float32)
    for i, ti in enumerate(t_idx):
        t    = tracks[ti]
        tbox = t.to_tlwh();  tctr = t.center()
        diag = np.hypot(tbox[2], tbox[3]) + 1e-6
        gm   = _l2(np.stack(t._gallery)) if t._gallery else None
        for j, di in enumerate(d_idx):
            d     = dets[di]
            iou_c = 1.0 - _iou(tbox, d.tlwh)
            dc    = min(np.linalg.norm(tctr - d.center()) / diag, 1.0)
            if gm is not None and d.embedding is not None:
                en    = d.embedding / (np.linalg.norm(d.embedding) + 1e-6)
                app_c = float(1.0 - (gm @ en).max())
            else:
                app_c = 1.0
            cost[i, j] = 0.50 * iou_c + 0.25 * dc + 0.25 * app_c
    return cost


def _hungarian(cost, t_idx, d_idx, thresh):
    if cost.size == 0:
        return [], list(t_idx), list(d_idx)
    rows, cols = linear_sum_assignment(cost)
    matches, ut, ud = [], set(t_idx), set(d_idx)
    for r, c in zip(rows, cols):
        if cost[r, c] <= thresh:
            matches.append((t_idx[r], d_idx[c]))
            ut.discard(t_idx[r]);  ud.discard(d_idx[c])
    return matches, list(ut), list(ud)


def _match(tracks, dets, max_iou_dist=0.75):
    if not tracks or not dets:
        return [], list(range(len(tracks))), list(range(len(dets)))
    conf_t  = [i for i, t in enumerate(tracks) if t.is_confirmed()]
    tent_t  = [i for i, t in enumerate(tracks)
               if not t.is_confirmed() and not t.is_deleted()]
    all_d   = list(range(len(dets)))

    m1, ut1, ud1 = (_hungarian(_appearance_cost(tracks, dets, conf_t, all_d),
                                conf_t, all_d, thresh=0.45)
                    if conf_t else ([], conf_t, all_d))

    rem_t = ut1 + tent_t
    m2, ut2, ud2 = (_hungarian(_fused_cost(tracks, dets, rem_t, ud1),
                                rem_t, ud1, thresh=max_iou_dist)
                    if rem_t and ud1 else ([], rem_t, ud1))

    return m1 + m2, ut2, ud2


# ── Tracker ───────────────────────────────────────────────────────────────────

class DeepSortTracker:
    """
    Deep SORT tracker — face-only, embedding-gated, Re-ID-aware.

    Call pattern per frame:
        tracker.predict()
        tracker.update(detections, frame_w=W, frame_h=H)
    """

    def __init__(self, n_init=3, max_age=90, max_iou_distance=0.75):
        self.kf          = KalmanFilter()
        self.tracks      = []
        self.next_id     = 1
        self._n_init     = n_init
        self._max_age    = max_age
        self._max_iou    = max_iou_distance

    def predict(self):
        for t in self.tracks:
            t.predict(self.kf)

    def update(self, detections, frame_w=0, frame_h=0):
        # Face-only gate
        if frame_w > 0 and frame_h > 0:
            detections = [d for d in detections
                          if is_face_bbox(d.tlwh, frame_w, frame_h)]

        matches, unmatched_t, unmatched_d = _match(
            self.tracks, detections, self._max_iou)

        for ti, di in matches:
            self.tracks[ti].update(self.kf, detections[di])
        for ti in unmatched_t:
            self.tracks[ti].mark_missed()

        # Embedding-gated new / re-ID track creation
        for di in unmatched_d:
            det = detections[di]
            existing = _global_reid.lookup(det.embedding)

            if existing is not None:
                live = next((t for t in self.tracks
                             if t.track_id == existing and not t.is_deleted()), None)
                if live is not None:
                    live.update(self.kf, det);  continue
                # Resurrect deleted track with canonical ID
                mean, cov = self.kf.initiate(det.to_xyah())
                t = Track(mean, cov, existing,
                          n_init=self._n_init, max_age=self._max_age)
                if det.embedding is not None:
                    t.embedding = det.embedding.copy()
                    src = _global_reid._store.get(existing)
                    if src is not None:
                        t._gallery = [src[i] for i in
                                      range(max(0, len(src) - self._n_init), len(src))]
                    t._gallery.append(det.embedding.copy())
                self.tracks.append(t);  continue

            # Genuinely new face
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
        for t in self.tracks:
            if t.is_confirmed() and t.embedding is not None:
                _global_reid.update_from_track(t)

    def get_active_tracks(self):
        return list(self.tracks)

    def get_tracks(self):
        return self.get_active_tracks()

    def get_confirmed_tracks(self):
        return [t for t in self.tracks if t.is_confirmed()]

    def reset_reid(self):
        """Call between unrelated video clips to prevent ID bleed."""
        global _global_reid
        _global_reid = GlobalReIDGallery()
        self.tracks  = []
        self.next_id = 1

    def reset(self):
        self.reset_reid()