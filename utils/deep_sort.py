"""
DeepSort Tracker - with Re-ID and center-distance matching.
Key improvements:
- Tracks survive 150 frames without detection (handles brief disappearances)
- Combined IoU + center-distance cost (handles head movement)
- Re-ID: when a new detection appears near a recently-deleted track's last position,
  reuse the same ID instead of creating a new one
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanFilter:
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean = np.r_[measurement, np.zeros_like(measurement)]
        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3],
        ]
        return mean, np.diag(np.square(std))

    def predict(self, mean, covariance):
        std_pos = [self._std_weight_position * mean[3],
                   self._std_weight_position * mean[3],
                   1e-2,
                   self._std_weight_position * mean[3]]
        std_vel = [self._std_weight_velocity * mean[3],
                   self._std_weight_velocity * mean[3],
                   1e-5,
                   self._std_weight_velocity * mean[3]]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = (np.linalg.multi_dot(
            [self._motion_mat, covariance, self._motion_mat.T]) + motion_cov)
        return mean, covariance

    def project(self, mean, covariance):
        std = [self._std_weight_position * mean[3],
               self._std_weight_position * mean[3],
               1e-1,
               self._std_weight_position * mean[3]]
        innovation_cov = np.diag(np.square(std))
        projected_mean = np.dot(self._update_mat, mean)
        projected_cov = (np.linalg.multi_dot(
            [self._update_mat, covariance, self._update_mat.T]) + innovation_cov)
        return projected_mean, projected_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol = np.linalg.cholesky(projected_cov)
        tmp = np.linalg.solve(chol, np.dot(self._update_mat, covariance.T))
        kalman_gain = np.linalg.solve(chol, tmp).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_cov = covariance - np.linalg.multi_dot(
            [kalman_gain, projected_cov, kalman_gain.T])
        return new_mean, new_cov


class Track:
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED   = 3

    def __init__(self, mean, covariance, track_id, n_init=2, max_age=150):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = Track.TENTATIVE
        self._n_init = n_init
        self._max_age = max_age

        self.deepfake_scores = []
        self.is_deepfake = False
        self.smoothed_score = 0.0
        self.confidence = 0.0
        self.label = "Analyzing..."

        # Last known position (for re-ID after track deletion)
        self.last_tlwh = None

    def to_tlwh(self):
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def center(self):
        tlwh = self.to_tlwh()
        return np.array([tlwh[0] + tlwh[2] / 2,
                         tlwh[1] + tlwh[3] / 2])

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah())
        self.last_tlwh = self.to_tlwh().copy()
        self.hits += 1
        self.time_since_update = 0
        if self.state == Track.TENTATIVE and self.hits >= self._n_init:
            self.state = Track.CONFIRMED

    def mark_missed(self):
        self.last_tlwh = self.to_tlwh().copy()
        if self.state == Track.TENTATIVE:
            self.state = Track.DELETED
        elif self.time_since_update > self._max_age:
            self.state = Track.DELETED

    def is_tentative(self): return self.state == Track.TENTATIVE
    def is_confirmed(self):  return self.state == Track.CONFIRMED
    def is_deleted(self):    return self.state == Track.DELETED

    def add_deepfake_score(self, score):
        self.deepfake_scores.append(score)
        if len(self.deepfake_scores) > 15:
            self.deepfake_scores.pop(0)
        self.smoothed_score = float(np.mean(self.deepfake_scores))
        self.is_deepfake = self.smoothed_score > 0.5
        self.confidence = (self.smoothed_score if self.is_deepfake
                           else 1.0 - self.smoothed_score)
        self.label = (f"DEEPFAKE ({self.smoothed_score:.2f})"
                      if self.is_deepfake
                      else f"REAL ({1-self.smoothed_score:.2f})")


class Detection:
    def __init__(self, tlwh, confidence):
        self.tlwh = np.asarray(tlwh, dtype=float)
        self.confidence = float(confidence)

    def to_tlbr(self):
        ret = self.tlwh.copy(); ret[2:] += ret[:2]; return ret

    def to_xyah(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    def center(self):
        return np.array([self.tlwh[0] + self.tlwh[2] / 2,
                         self.tlwh[1] + self.tlwh[3] / 2])


# ── helpers ───────────────────────────────────────────────────────────────────

def _iou(a, b):
    ax2 = a[0]+a[2]; ay2 = a[1]+a[3]
    bx2 = b[0]+b[2]; by2 = b[1]+b[3]
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(ax2, bx2);   iy2 = min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1: return 0.0
    inter = (ix2-ix1)*(iy2-iy1)
    union = a[2]*a[3] + b[2]*b[3] - inter
    return inter/union if union > 0 else 0.0


def _center_dist_cost(track_ctr, track_diag, det_ctr):
    dist = float(np.linalg.norm(track_ctr - det_ctr))
    return min(dist / (track_diag + 1e-6), 1.0)


def _build_cost_matrix(tracks, detections, track_indices, detection_indices):
    cost = np.ones((len(track_indices), len(detection_indices)), dtype=float)
    for i, ti in enumerate(track_indices):
        t = tracks[ti]
        t_box  = t.to_tlwh()
        t_ctr  = t.center()
        t_diag = np.sqrt(t_box[2]**2 + t_box[3]**2)
        for j, di in enumerate(detection_indices):
            d_box = detections[di].tlwh
            d_ctr = detections[di].center()
            iou_cost  = 1.0 - _iou(t_box, d_box)
            dist_cost = _center_dist_cost(t_ctr, t_diag, d_ctr)
            # Weighted: 40% IoU + 60% distance (distance more robust for movement)
            cost[i, j] = 0.4 * iou_cost + 0.6 * dist_cost
    return cost


def _match_detections(max_cost, tracks, detections, track_indices, detection_indices):
    if not track_indices or not detection_indices:
        return [], list(track_indices), list(detection_indices)

    cost = _build_cost_matrix(tracks, detections, track_indices, detection_indices)
    rows, cols = linear_sum_assignment(cost)

    matches, unmatched_t, unmatched_d = [], list(track_indices), list(detection_indices)
    for r, c in zip(rows, cols):
        if cost[r, c] > max_cost:
            continue
        ti = track_indices[r]; di = detection_indices[c]
        matches.append((ti, di))
        unmatched_t.remove(ti)
        unmatched_d.remove(di)

    return matches, unmatched_t, unmatched_d


# ── Main tracker ──────────────────────────────────────────────────────────────

class DeepSortTracker:
    """
    Multi-face tracker with Re-ID.
    - Tracks survive 150 frames of missed detection
    - Combined IoU + center-distance matching
    - Re-ID: if a new detection appears near a recently lost track's last position,
      it reuses the same ID (fixes the 'same person gets new ID' problem)
    """

    def __init__(self, max_iou_distance=0.75, max_age=150, n_init=2):
        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.max_cost = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

        # Graveyard: recently deleted tracks for re-ID
        # {track_id: {'last_tlwh': ..., 'scores': [...], 'frames_since_delete': 0}}
        self._graveyard = {}
        self._graveyard_ttl = 60   # keep deleted tracks for 60 frames for re-ID

    def predict(self):
        for t in self.tracks:
            t.predict(self.kf)
        # Age graveyard entries
        expired = [tid for tid, g in self._graveyard.items()
                   if g['frames_since_delete'] > self._graveyard_ttl]
        for tid in expired:
            del self._graveyard[tid]
        for g in self._graveyard.values():
            g['frames_since_delete'] += 1

    def update(self, detections):
        # Step 1: match existing tracks
        matches, unmatched_tracks, unmatched_dets = self._match(detections)

        for ti, di in matches:
            self.tracks[ti].update(self.kf, detections[di])

        for ti in unmatched_tracks:
            self.tracks[ti].mark_missed()

        # Step 2: try re-ID on remaining unmatched detections
        still_unmatched = self._try_reid(detections, unmatched_dets)

        # Step 3: create brand new tracks for truly unmatched detections
        for di in still_unmatched:
            self._initiate_track(detections[di])

        # Step 4: move newly deleted tracks to graveyard
        for t in self.tracks:
            if t.is_deleted():
                self._graveyard[t.track_id] = {
                    'last_tlwh': t.last_tlwh if t.last_tlwh is not None else t.to_tlwh(),
                    'scores': list(t.deepfake_scores),
                    'frames_since_delete': 0,
                    'hits': t.hits,
                }

        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def _match(self, detections):
        confirmed   = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]
        all_dets    = list(range(len(detections)))

        m_a, unm_conf, remaining = _match_detections(
            self.max_cost, self.tracks, detections, confirmed, all_dets)
        m_b, unm_unconf, remaining = _match_detections(
            0.75, self.tracks, detections, unconfirmed, remaining)

        return m_a + m_b, unm_conf + unm_unconf, remaining

    def _try_reid(self, detections, unmatched_det_indices):
        """
        Try to match unmatched detections to recently deleted tracks.
        If a detection is close to where a deleted track last was, reuse its ID.
        """
        if not self._graveyard or not unmatched_det_indices:
            return unmatched_det_indices

        still_unmatched = list(unmatched_det_indices)
        matched_dets = set()

        for di in unmatched_det_indices:
            det = detections[di]
            det_ctr = det.center()
            det_diag = np.sqrt(det.tlwh[2]**2 + det.tlwh[3]**2) + 1e-6

            best_tid = None
            best_dist = 0.5   # max normalized distance for re-ID (0.5 = half face diagonal)

            for tid, g in self._graveyard.items():
                last = g['last_tlwh']
                if last is None:
                    continue
                last_ctr = np.array([last[0] + last[2]/2, last[1] + last[3]/2])
                dist = float(np.linalg.norm(det_ctr - last_ctr)) / det_diag
                if dist < best_dist:
                    best_dist = dist
                    best_tid = tid

            if best_tid is not None:
                # Revive this track with the same ID
                g = self._graveyard.pop(best_tid)
                mean, cov = self.kf.initiate(det.to_xyah())
                t = Track(mean, cov, best_tid, self.n_init, self.max_age)
                t.hits = g['hits']
                t.state = Track.CONFIRMED   # revived tracks are immediately confirmed
                t.deepfake_scores = g['scores']
                if t.deepfake_scores:
                    t.smoothed_score = float(np.mean(t.deepfake_scores))
                    t.is_deepfake = t.smoothed_score > 0.5
                    t.confidence = t.smoothed_score if t.is_deepfake else 1-t.smoothed_score
                self.tracks.append(t)
                matched_dets.add(di)

        return [di for di in still_unmatched if di not in matched_dets]

    def _initiate_track(self, detection):
        mean, cov = self.kf.initiate(detection.to_xyah())
        self.tracks.append(
            Track(mean, cov, self._next_id, self.n_init, self.max_age))
        self._next_id += 1

    def get_active_tracks(self):
        return [t for t in self.tracks if not t.is_deleted()]