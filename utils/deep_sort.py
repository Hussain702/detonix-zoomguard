"""
DeepSort Tracker - Custom Implementation
Tracks multiple faces across video frames using Kalman Filter + IoU matching
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


class KalmanFilter:
    """
    Kalman Filter for tracking bounding boxes.
    State: [cx, cy, w, h, vx, vy, vw, vh]
    """
    def __init__(self):
        ndim, dt = 4, 1.0
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt

        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1.0 / 20
        self._std_weight_velocity = 1.0 / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

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
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3],
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3],
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot([self._motion_mat, covariance, self._motion_mat.T]) + motion_cov
        return mean, covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3],
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot([self._update_mat, covariance, self._update_mat.T])
        return mean, covariance + innovation_cov

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor = np.linalg.cholesky(projected_cov)
        kalman_gain = np.linalg.solve(
            chol_factor,
            np.linalg.solve(chol_factor, np.dot(covariance, self._update_mat.T).T).T
        ).T
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot([kalman_gain, projected_cov, kalman_gain.T])
        return new_mean, new_covariance

    def gating_distance(self, mean, covariance, measurements):
        projected_mean, projected_cov = self.project(mean, covariance)
        d = measurements - projected_mean
        cholesky_factor = np.linalg.cholesky(projected_cov)
        z = np.linalg.solve(cholesky_factor, d.T).T
        squared_maha = np.sum(z * z, axis=1)
        return squared_maha


class Track:
    """Single track for one person/face."""
    
    _id_counter = 0
    
    TENTATIVE = 1
    CONFIRMED = 2
    DELETED = 3

    def __init__(self, mean, covariance, track_id, n_init=3, max_age=30):
        self.mean = mean
        self.covariance = covariance
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.state = Track.TENTATIVE
        self._n_init = n_init
        self._max_age = max_age

        # Deepfake detection history
        self.deepfake_scores = []
        self.is_deepfake = False
        self.smoothed_score = 0.0
        self.label = "Analyzing..."
        self.confidence = 0.0

    def to_tlwh(self):
        """Convert to (top-left x, top-left y, width, height)."""
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret

    def to_tlbr(self):
        """Convert to (min x, min y, max x, max y)."""
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf):
        self.mean, self.covariance = kf.predict(self.mean, self.covariance)
        self.age += 1
        self.time_since_update += 1

    def update(self, kf, detection):
        self.mean, self.covariance = kf.update(self.mean, self.covariance, detection.to_xyah())
        self.hits += 1
        self.time_since_update = 0
        if self.state == Track.TENTATIVE and self.hits >= self._n_init:
            self.state = Track.CONFIRMED

    def mark_missed(self):
        if self.state == Track.TENTATIVE:
            self.state = Track.DELETED
        elif self.time_since_update > self._max_age:
            self.state = Track.DELETED

    def is_tentative(self):
        return self.state == Track.TENTATIVE

    def is_confirmed(self):
        return self.state == Track.CONFIRMED

    def is_deleted(self):
        return self.state == Track.DELETED

    def add_deepfake_score(self, score):
        """Add deepfake probability score and compute smoothed result."""
        self.deepfake_scores.append(score)
        # Keep last 10 scores for smoothing
        if len(self.deepfake_scores) > 10:
            self.deepfake_scores.pop(0)
        self.smoothed_score = np.mean(self.deepfake_scores)
        self.is_deepfake = self.smoothed_score > 0.5
        self.confidence = self.smoothed_score if self.is_deepfake else (1.0 - self.smoothed_score)
        self.label = f"DEEPFAKE ({self.smoothed_score:.2f})" if self.is_deepfake else f"REAL ({1-self.smoothed_score:.2f})"


class Detection:
    """Single face detection."""

    def __init__(self, tlwh, confidence):
        self.tlwh = np.asarray(tlwh, dtype=float)
        self.confidence = float(confidence)

    def to_tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def to_xyah(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret


def iou(bbox, candidates):
    """Compute IoU between bbox and candidates."""
    bbox_tl = bbox[:2]
    bbox_br = bbox[:2] + bbox[2:]
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, :2] + candidates[:, 2:]

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0]),
               np.maximum(bbox_tl[1], candidates_tl[:, 1])]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0]),
               np.minimum(bbox_br[1], candidates_br[:, 1])]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox[2:].prod()
    area_candidates = candidates[:, 2:].prod(axis=1)
    return area_intersection / (area_bbox + area_candidates - area_intersection)


def iou_cost(tracks, detections, track_indices, detection_indices):
    cost_matrix = np.zeros((len(track_indices), len(detection_indices)))
    for row, track_idx in enumerate(track_indices):
        if tracks[track_idx].time_since_update > 1:
            cost_matrix[row, :] = 1e5
            continue
        bbox = tracks[track_idx].to_tlwh()
        candidates = np.asarray([detections[i].tlwh for i in detection_indices])
        cost_matrix[row, :] = 1.0 - iou(bbox, candidates)
    return cost_matrix


def min_cost_matching(cost_matrix, max_distance, tracks, detections, track_indices, detection_indices):
    if len(track_indices) == 0 or len(detection_indices) == 0:
        return [], track_indices, detection_indices

    cost_matrix = cost_matrix[track_indices][:, detection_indices]  # This is wrong, fix:
    cost_matrix_sub = np.zeros((len(track_indices), len(detection_indices)))
    for i, ti in enumerate(track_indices):
        for j, di in enumerate(detection_indices):
            bbox = tracks[ti].to_tlwh()
            candidates = np.array([detections[di].tlwh])
            cost_matrix_sub[i, j] = 1.0 - iou(bbox, candidates)[0]

    row_indices, col_indices = linear_sum_assignment(cost_matrix_sub)
    matches, unmatched_tracks, unmatched_detections = [], [], []

    for col, detection_idx in enumerate(detection_indices):
        if col not in col_indices:
            unmatched_detections.append(detection_idx)

    for row, track_idx in enumerate(track_indices):
        if row not in row_indices:
            unmatched_tracks.append(track_idx)

    for row, col in zip(row_indices, col_indices):
        track_idx = track_indices[row]
        detection_idx = detection_indices[col]
        if cost_matrix_sub[row, col] > max_distance:
            unmatched_tracks.append(track_idx)
            unmatched_detections.append(detection_idx)
        else:
            matches.append((track_idx, detection_idx))

    return matches, unmatched_tracks, unmatched_detections


class DeepSortTracker:
    """
    Multi-object tracker using DeepSort algorithm with Kalman Filter.
    Adapted for face tracking across video frames.
    """

    def __init__(self, max_iou_distance=0.7, max_age=30, n_init=3):
        self.kf = KalmanFilter()
        self.tracks = []
        self._next_id = 1
        self.max_iou_distance = max_iou_distance
        self.max_age = max_age
        self.n_init = n_init

    def predict(self):
        for track in self.tracks:
            track.predict(self.kf)

    def update(self, detections):
        """Update tracker with list of Detection objects."""
        # Step 1: Match detections to existing tracks
        matches, unmatched_tracks, unmatched_detections = self._match(detections)

        # Step 2: Update matched tracks
        for track_idx, detection_idx in matches:
            self.tracks[track_idx].update(self.kf, detections[detection_idx])

        # Step 3: Mark unmatched tracks as missed
        for track_idx in unmatched_tracks:
            self.tracks[track_idx].mark_missed()

        # Step 4: Create new tracks for unmatched detections
        for detection_idx in unmatched_detections:
            self._initiate_track(detections[detection_idx])

        # Step 5: Remove deleted tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]

    def _match(self, detections):
        confirmed_tracks = [i for i, t in enumerate(self.tracks) if t.is_confirmed()]
        unconfirmed_tracks = [i for i, t in enumerate(self.tracks) if not t.is_confirmed()]

        # Match confirmed tracks using IoU
        matches_a, unmatched_tracks_a, unmatched_detections = min_cost_matching(
            None, self.max_iou_distance, self.tracks, detections,
            confirmed_tracks, list(range(len(detections)))
        )

        # Match unconfirmed tracks to remaining detections
        matches_b, unmatched_tracks_b, unmatched_detections = min_cost_matching(
            None, 0.5, self.tracks, detections,
            unconfirmed_tracks, unmatched_detections
        )

        matches = matches_a + matches_b
        unmatched_tracks = list(set(unmatched_tracks_a + unmatched_tracks_b))
        return matches, unmatched_tracks, unmatched_detections

    def _initiate_track(self, detection):
        mean, covariance = self.kf.initiate(detection.to_xyah())
        self.tracks.append(Track(mean, covariance, self._next_id, self.n_init, self.max_age))
        self._next_id += 1

    def get_active_tracks(self):
        return [t for t in self.tracks if t.is_confirmed() or t.is_tentative()]
