"""Kalman and sparse optical-flow tracking for intermittent FUEL detections."""
import math
import os
import threading
import time

import cv2
import numpy as np


FLOW_SCALE = 0.5
FLOW_FEATURES_PER_TRACK = max(1, int(os.getenv('SCOUT_FLOW_FEATURES_PER_TRACK', '8')))
FLOW_WINDOW_SIZE = int(os.getenv('SCOUT_FLOW_WINDOW_SIZE', '15'))
FLOW_MAX_ITERATIONS = int(os.getenv('SCOUT_FLOW_MAX_ITERATIONS', '20'))
if not 3 <= FLOW_WINDOW_SIZE <= 31 or FLOW_WINDOW_SIZE % 2 == 0:
    raise ValueError('SCOUT_FLOW_WINDOW_SIZE must be an odd value from 3 to 31')
if not 1 <= FLOW_MAX_ITERATIONS <= 40:
    raise ValueError('SCOUT_FLOW_MAX_ITERATIONS must be from 1 to 40')
_EYE4 = np.eye(4, dtype=np.float32)


def _center_box(detection):
    width = max(1.0, float(detection.x2 - detection.x1))
    height = max(1.0, float(detection.y2 - detection.y1))
    return np.asarray(((detection.x1 + detection.x2) * 0.5,
                       (detection.y1 + detection.y2) * 0.5,
                       width, height), dtype=np.float32)


def _box_iou(a, b):
    ax1, ay1 = a[0] - a[2] * 0.5, a[1] - a[3] * 0.5
    ax2, ay2 = a[0] + a[2] * 0.5, a[1] + a[3] * 0.5
    bx1, by1 = b[0] - b[2] * 0.5, b[1] - b[3] * 0.5
    bx2, by2 = b[0] + b[2] * 0.5, b[1] + b[3] * 0.5
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1))
    union = a[2] * a[3] + b[2] * b[3] - intersection
    return intersection / union if union > 0 else 0.0


def _small_median_displacement(good, old):
    """Median 2D displacement without NumPy's general NaN/axis machinery."""
    count = len(good)
    delta = good - old
    if count == 1:
        return delta[0]
    dx = np.sort(delta[:, 0])
    dy = np.sort(delta[:, 1])
    middle = count // 2
    if count & 1:
        return np.asarray((dx[middle], dy[middle]), dtype=np.float32)
    return np.asarray(((dx[middle - 1] + dx[middle]) * 0.5,
                       (dy[middle - 1] + dy[middle]) * 0.5), dtype=np.float32)


class _Track:
    def __init__(self, track_id, detection, now, pts, direct_flow=False):
        self.track_id = track_id
        self.class_id = int(detection.class_id)
        self.filter = None
        if not direct_flow:
            self.filter = cv2.KalmanFilter(8, 4, 0, cv2.CV_32F)
            transition = np.eye(8, dtype=np.float32)
            transition[:4, 4:] = np.eye(4, dtype=np.float32)
            self.filter.transitionMatrix = transition
            measurement = np.zeros((4, 8), dtype=np.float32)
            measurement[:4, :4] = np.eye(4, dtype=np.float32)
            self.filter.measurementMatrix = measurement
            self.filter.processNoiseCov = np.diag(
                (4, 4, 9, 9, 60, 60, 100, 100)).astype(np.float32)
            self.filter.measurementNoiseCov = np.diag(
                (16, 16, 25, 25)).astype(np.float32)
            self.filter.errorCovPost = np.diag(
                (16, 16, 25, 25, 400, 400, 400, 400)).astype(np.float32)
            self.filter.statePost = np.r_[
                _center_box(detection), np.zeros(4, np.float32)].reshape(8, 1)
            self.box = self.filter.statePost.reshape(-1)[:4].copy()
        else:
            self.box = _center_box(detection)
        self.last_update = now
        self.last_detector_time = now
        self.last_visual_time = now
        self.last_seed_time = None
        self.last_range_time = now if getattr(detection, 'distance_m', None) is not None else None
        self.last_detector_pts = pts
        self.last_detection = detection
        self.last_distance_m = getattr(detection, 'distance_m', None)
        self.last_state = 'measured'
        self.flow_points = None
        self.misses = 0

    def predict(self, now):
        dt = max(0.0, min(0.5, now - self.last_update))
        if dt > 0:
            transition = self.filter.transitionMatrix.copy()
            transition[:4, 4:] = np.eye(4, dtype=np.float32) * dt
            self.filter.transitionMatrix = transition
            self.box = self.filter.predict().reshape(-1)[:4].copy()
            self.last_update = now
        return self.box.copy()

    def correct_box(self, box, now):
        self.box = self.filter.correct(
            np.asarray(box, dtype=np.float32).reshape(4, 1)).reshape(-1)[:4].copy()
        self.last_update = now
        return self.box.copy()


class FuelTracker:
    """Reconcile detector anchors with per-frame LK flow and decaying tracks."""
    def __init__(self, max_age_ms=None, flow_grace_ms=None, decay_ms=None):
        max_age_ms = (float(os.getenv('SCOUT_TRACK_MAX_AGE_MS', '500'))
                      if max_age_ms is None else float(max_age_ms))
        flow_grace_ms = (float(os.getenv('SCOUT_TRACK_FLOW_GRACE_MS', '120'))
                         if flow_grace_ms is None else float(flow_grace_ms))
        decay_ms = (float(os.getenv('SCOUT_TRACK_DECAY_MS', '250'))
                    if decay_ms is None else float(decay_ms))
        if not 0 < max_age_ms <= 2000:
            raise ValueError('SCOUT_TRACK_MAX_AGE_MS must be greater than zero and at most 2000')
        if not 0 < flow_grace_ms <= max_age_ms:
            raise ValueError('SCOUT_TRACK_FLOW_GRACE_MS must be positive and no greater than max age')
        if not 0 < decay_ms <= max_age_ms:
            raise ValueError('SCOUT_TRACK_DECAY_MS must be positive and no greater than max age')
        self.max_age_s = max_age_ms / 1000.0
        self.flow_grace_s = flow_grace_ms / 1000.0
        self.decay_s = decay_ms / 1000.0
        self.direct_flow = os.getenv('SCOUT_TRACK_DIRECT_FLOW', '0').lower() in (
            '1', 'true', 'yes')
        self.tracks = {}
        self.next_id = 1
        self.seed_pending = set()
        self.lock = threading.RLock()
        self.anchor_update_lock = threading.Lock()
        self.previous_gray = None
        self.previous_pts = None

    @staticmethod
    def _gray(frame):
        gray = (frame if frame.ndim == 2 else
                cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        width = max(1, round(gray.shape[1] * FLOW_SCALE))
        height = max(1, round(gray.shape[0] * FLOW_SCALE))
        return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _seed_tracks(gray, tracks, now):
        """Detect features once in the union of all track regions."""
        if gray is None or not tracks:
            return
        tracks = [track for track in tracks
                  if track.last_seed_time is None or now - track.last_seed_time >= 0.5]
        if not tracks:
            return
        mask = np.zeros(gray.shape, dtype=np.uint8)
        centers = np.empty((len(tracks), 2), dtype=np.float32)
        radii = np.empty(len(tracks), dtype=np.float32)
        for index, track in enumerate(tracks):
            cx, cy, width, height = track.box
            center = (float(cx) * FLOW_SCALE, float(cy) * FLOW_SCALE)
            radius = max(6, round(min(float(width), float(height)) * FLOW_SCALE * 0.36))
            centers[index] = center
            radii[index] = radius
            cv2.circle(mask, (round(center[0]), round(center[1])), radius, 255, -1)
        maximum = max(FLOW_FEATURES_PER_TRACK * len(tracks) * 2, 32)
        points = cv2.goodFeaturesToTrack(
            gray, maxCorners=maximum, qualityLevel=0.003, minDistance=3,
            mask=mask, blockSize=5, useHarrisDetector=False)
        for track in tracks:
            track.flow_points = None
            track.last_seed_time = now
        if points is None:
            return
        points = points.reshape(-1, 2).astype(np.float32, copy=False)
        delta = points[None, :, :] - centers[:, None, :]
        distance_sq = np.sum(delta * delta, axis=2)
        nearest = np.argmin(distance_sq, axis=0)
        nearest_distance_sq = distance_sq[nearest, np.arange(len(points))]
        within = nearest_distance_sq <= radii[nearest] ** 2
        for index, track in enumerate(tracks):
            selected = np.flatnonzero(within & (nearest == index))
            if selected.size:
                selected = selected[:FLOW_FEATURES_PER_TRACK]
                track.flow_points = points[selected].reshape(-1, 1, 2).copy()

    @staticmethod
    def _find_seed_points(gray, track_specs):
        """Find features for copied track boxes without holding the tracker lock."""
        if gray is None or not track_specs:
            return {}
        mask = np.zeros(gray.shape, dtype=np.uint8)
        centers = np.empty((len(track_specs), 2), dtype=np.float32)
        radii = np.empty(len(track_specs), dtype=np.float32)
        for index, (_, box) in enumerate(track_specs):
            cx, cy, width, height = box
            center = (float(cx) * FLOW_SCALE, float(cy) * FLOW_SCALE)
            radius = max(6, round(min(float(width), float(height)) * FLOW_SCALE * 0.36))
            centers[index] = center
            radii[index] = radius
            cv2.circle(mask, (round(center[0]), round(center[1])), radius, 255, -1)
        maximum = max(FLOW_FEATURES_PER_TRACK * len(track_specs) * 2, 32)
        points = cv2.goodFeaturesToTrack(
            gray, maxCorners=maximum, qualityLevel=0.003, minDistance=3,
            mask=mask, blockSize=5, useHarrisDetector=False)
        result = {track_id: (centers[index], None)
                  for index, (track_id, _) in enumerate(track_specs)}
        if points is None:
            return result
        points = points.reshape(-1, 2).astype(np.float32, copy=False)
        delta = points[None, :, :] - centers[:, None, :]
        distance_sq = np.sum(delta * delta, axis=2)
        nearest = np.argmin(distance_sq, axis=0)
        nearest_distance_sq = distance_sq[nearest, np.arange(len(points))]
        within = nearest_distance_sq <= radii[nearest] ** 2
        for index, (track_id, _) in enumerate(track_specs):
            selected = np.flatnonzero(within & (nearest == index))
            if selected.size:
                selected = selected[:FLOW_FEATURES_PER_TRACK]
                result[track_id] = (
                    centers[index], points[selected].reshape(-1, 1, 2).copy())
        return result

    def _seed_tracks_async(self, gray, track_specs):
        try:
            seeded = self._find_seed_points(gray, track_specs)
            with self.lock:
                for track_id, (seed_center, points) in seeded.items():
                    track = self.tracks.get(track_id)
                    if track is None:
                        continue
                    if points is not None and (track.flow_points is None or
                                               len(track.flow_points) < 2):
                        offset = (track.box[:2] * FLOW_SCALE - seed_center).reshape(1, 1, 2)
                        track.flow_points = points + offset
        finally:
            with self.lock:
                self.seed_pending.difference_update(track_id for track_id, _ in track_specs)

    def reset(self):
        with self.lock:
            self.tracks.clear()
            self.next_id = 1
            self.seed_pending.clear()
            self.previous_gray = None
            self.previous_pts = None

    @staticmethod
    def _cost(track, predicted, detection):
        observed = _center_box(detection)
        delta = float(np.linalg.norm(predicted[:2] - observed[:2]))
        diagonal = math.hypot(float(predicted[2]), float(predicted[3]))
        normalized = delta / max(18.0, diagonal * 0.5)
        if delta > max(80.0, diagonal * 2.5):
            return None
        distance = getattr(detection, 'distance_m', None)
        if (track.last_distance_m is not None and distance is not None and
                abs(float(distance) - track.last_distance_m) >
                max(0.8, track.last_distance_m * 0.55)):
            return None
        return normalized + 0.35 * (1.0 - _box_iou(predicted, observed))

    def update(self, detections, now=None, pts=None, return_snapshot=True):
        """Apply detector anchors. Positions already advanced by flow stay current."""
        now = float(now if now is not None else time.monotonic())
        if self.direct_flow:
            return self._update_direct(detections, now, pts, return_snapshot)
        seed_request = None
        with self.lock:
            live = [track for track in self.tracks.values()
                    if now - track.last_detector_time <= self.max_age_s]
            self.tracks = {track.track_id: track for track in live}
            predicted = {track.track_id: (track.box.copy() if self.direct_flow else
                                           track.predict(now)) for track in live}
            pairs = []
            if live and detections:
                predicted_boxes = np.asarray([predicted[track.track_id]
                                              for track in live], dtype=np.float32)
                observed_boxes = np.asarray([_center_box(detection)
                                             for detection in detections], dtype=np.float32)
                delta = (predicted_boxes[:, None, :2] -
                         observed_boxes[None, :, :2])
                distance = np.sqrt(np.sum(delta * delta, axis=2))
                diagonal = np.hypot(predicted_boxes[:, 2], predicted_boxes[:, 3])
                distance_limit = np.maximum(80.0, diagonal * 2.5)
                normalized = distance / np.maximum(18.0, diagonal[:, None] * 0.5)

                ta = predicted_boxes[:, None, :]
                db = observed_boxes[None, :, :]
                tx1, ty1 = ta[:, :, 0] - ta[:, :, 2] * .5, ta[:, :, 1] - ta[:, :, 3] * .5
                tx2, ty2 = ta[:, :, 0] + ta[:, :, 2] * .5, ta[:, :, 1] + ta[:, :, 3] * .5
                dx1, dy1 = db[:, :, 0] - db[:, :, 2] * .5, db[:, :, 1] - db[:, :, 3] * .5
                dx2, dy2 = db[:, :, 0] + db[:, :, 2] * .5, db[:, :, 1] + db[:, :, 3] * .5
                intersection = (np.maximum(0.0, np.minimum(tx2, dx2) - np.maximum(tx1, dx1)) *
                                np.maximum(0.0, np.minimum(ty2, dy2) - np.maximum(ty1, dy1)))
                union = (predicted_boxes[:, 2] * predicted_boxes[:, 3])[:, None] + \
                        (observed_boxes[:, 2] * observed_boxes[:, 3])[None, :] - intersection
                iou = intersection / np.maximum(1e-6, union)
                costs = normalized + .35 * (1.0 - iou)
                valid = distance <= distance_limit[:, None]
                track_classes = np.asarray([track.class_id for track in live])
                detection_classes = np.asarray([int(d.class_id) for d in detections])
                valid &= track_classes[:, None] == detection_classes[None, :]
                old_ranges = np.asarray([
                    np.nan if track.last_distance_m is None else track.last_distance_m
                    for track in live], dtype=np.float32)
                new_ranges = np.asarray([
                    np.nan if getattr(detection, 'distance_m', None) is None else
                    float(detection.distance_m) for detection in detections], dtype=np.float32)
                has_range = np.isfinite(old_ranges[:, None]) & np.isfinite(new_ranges[None, :])
                range_delta = np.abs(old_ranges[:, None] - new_ranges[None, :])
                range_limit = np.maximum(.8, old_ranges[:, None] * .55)
                valid &= ~has_range | (range_delta <= range_limit)
                flat_indices = np.flatnonzero(valid)
                if flat_indices.size:
                    order = np.argsort(costs.reshape(-1)[flat_indices])
                    pairs = flat_indices[order]
            matched_tracks, matched_detections = set(), set()
            tracks_to_seed = []
            detection_count = len(detections)
            for flat_index in pairs:
                track_index, index = divmod(int(flat_index), detection_count)
                track_id = live[track_index].track_id
                if track_id in matched_tracks or index in matched_detections:
                    continue
                track, detection = self.tracks[track_id], detections[index]
                # Correct position when this detector anchor is at least as
                # recent as the flow frame already processed. A late anchor
                # can still confirm identity/range, but must not rewind a track.
                anchor_is_current = (pts is None or self.previous_pts is None or
                                     pts >= self.previous_pts)
                if anchor_is_current:
                    predicted_box = predicted[track_id]
                    if self.direct_flow:
                        observed_box = _center_box(detection)
                        track.box = observed_box
                        track.filter.statePost[:4, 0] = observed_box
                        track.filter.statePost[4:, 0] = 0
                        track.last_update = now
                    else:
                        track.correct_box(_center_box(detection), now)
                    if track.flow_points is not None and len(track.flow_points):
                        correction = track.box[:2] - predicted_box[:2]
                        if float(np.dot(correction, correction)) <= 144.0:
                            track.flow_points += (correction * FLOW_SCALE).reshape(1, 1, 2)
                        else:
                            track.flow_points = None
                    if track.flow_points is None:
                        tracks_to_seed.append(track)
                track.last_detector_time = now
                track.last_visual_time = now
                track.last_detector_pts = pts
                track.last_detection = detection
                track.last_state = 'measured'
                track.misses = 0
                if getattr(detection, 'distance_m', None) is not None:
                    track.last_distance_m = float(detection.distance_m)
                    track.last_range_time = now
                matched_tracks.add(track_id)
                matched_detections.add(index)

            for index, detection in enumerate(detections):
                if index in matched_detections:
                    continue
                track = _Track(self.next_id, detection, now, pts)
                self.tracks[track.track_id] = track
                tracks_to_seed.append(track)
                self.next_id += 1
            if self.previous_gray is not None and tracks_to_seed:
                if self.direct_flow:
                    specs = []
                    for track in tracks_to_seed:
                        if (track.track_id not in self.seed_pending and
                                (track.last_seed_time is None or
                                 now - track.last_seed_time >= 0.5)):
                            track.last_seed_time = now
                            self.seed_pending.add(track.track_id)
                            specs.append((track.track_id, track.box.copy()))
                    if specs:
                        seed_request = (self.previous_gray, specs)
                else:
                    self._seed_tracks(self.previous_gray, tracks_to_seed, now)
            result = self._snapshot_locked(now) if return_snapshot else None
        if seed_request is not None:
            threading.Thread(target=self._seed_tracks_async, args=seed_request,
                             name='fuel-feature-seed', daemon=True).start()
        return result

    def _update_direct(self, detections, now, pts, return_snapshot):
        """Match anchors outside the flow lock so detection cannot stall 60 Hz LK."""
        seed_request = None
        with self.anchor_update_lock:
            with self.lock:
                live = [track for track in self.tracks.values()
                        if now - track.last_detector_time <= self.max_age_s]
                self.tracks = {track.track_id: track for track in live}
                track_ids = [track.track_id for track in live]
                predicted_boxes = np.asarray([track.box.copy() for track in live],
                                             dtype=np.float32).reshape(-1, 4)
                track_classes = np.asarray([track.class_id for track in live])
                old_ranges = np.asarray([
                    np.nan if track.last_distance_m is None else track.last_distance_m
                    for track in live], dtype=np.float32)

            flat_pairs = np.empty(0, dtype=np.int64)
            observed_boxes = np.asarray([_center_box(detection)
                                         for detection in detections],
                                        dtype=np.float32).reshape(-1, 4)
            if live and detections:
                delta = (predicted_boxes[:, None, :2] -
                         observed_boxes[None, :, :2])
                distance = np.sqrt(np.sum(delta * delta, axis=2))
                diagonal = np.hypot(predicted_boxes[:, 2], predicted_boxes[:, 3])
                valid = distance <= np.maximum(80.0, diagonal[:, None] * 2.5)
                tx1 = predicted_boxes[:, 0] - predicted_boxes[:, 2] * .5
                ty1 = predicted_boxes[:, 1] - predicted_boxes[:, 3] * .5
                tx2 = predicted_boxes[:, 0] + predicted_boxes[:, 2] * .5
                ty2 = predicted_boxes[:, 1] + predicted_boxes[:, 3] * .5
                dx1 = observed_boxes[:, 0] - observed_boxes[:, 2] * .5
                dy1 = observed_boxes[:, 1] - observed_boxes[:, 3] * .5
                dx2 = observed_boxes[:, 0] + observed_boxes[:, 2] * .5
                dy2 = observed_boxes[:, 1] + observed_boxes[:, 3] * .5
                intersection = (
                    np.maximum(0.0, np.minimum(tx2[:, None], dx2[None, :]) -
                               np.maximum(tx1[:, None], dx1[None, :])) *
                    np.maximum(0.0, np.minimum(ty2[:, None], dy2[None, :]) -
                               np.maximum(ty1[:, None], dy1[None, :])))
                union = (predicted_boxes[:, 2] * predicted_boxes[:, 3])[:, None] + \
                        (observed_boxes[:, 2] * observed_boxes[:, 3])[None, :] - intersection
                iou = intersection / np.maximum(1e-6, union)
                costs = (distance / np.maximum(18.0, diagonal[:, None] * .5) +
                         .35 * (1.0 - iou))
                detection_classes = np.asarray([int(d.class_id) for d in detections])
                valid &= track_classes[:, None] == detection_classes[None, :]
                new_ranges = np.asarray([
                    np.nan if getattr(detection, 'distance_m', None) is None else
                    float(detection.distance_m) for detection in detections], dtype=np.float32)
                has_range = np.isfinite(old_ranges[:, None]) & np.isfinite(new_ranges[None, :])
                range_delta = np.abs(old_ranges[:, None] - new_ranges[None, :])
                valid &= ~has_range | (range_delta <= np.maximum(.8, old_ranges[:, None] * .55))
                candidates = np.flatnonzero(valid)
                if candidates.size:
                    flat_pairs = candidates[np.argsort(costs.reshape(-1)[candidates])]

            with self.lock:
                matched_tracks, matched_detections = set(), set()
                tracks_to_seed = []
                detection_count = len(detections)
                anchor_is_current = (pts is None or self.previous_pts is None or
                                     pts >= self.previous_pts)
                for flat_index in flat_pairs:
                    track_index, detection_index = divmod(int(flat_index), detection_count)
                    track_id = track_ids[track_index]
                    if (track_id in matched_tracks or detection_index in matched_detections or
                            track_id not in self.tracks):
                        continue
                    track = self.tracks[track_id]
                    if anchor_is_current:
                        predicted = predicted_boxes[track_index]
                        observed = observed_boxes[detection_index]
                        track.box = observed.copy()
                        track.last_update = now
                        if track.flow_points is not None and len(track.flow_points):
                            correction = track.box[:2] - predicted[:2]
                            if float(np.dot(correction, correction)) <= 144.0:
                                track.flow_points += (correction * FLOW_SCALE).reshape(1, 1, 2)
                            else:
                                track.flow_points = None
                        if track.flow_points is None:
                            tracks_to_seed.append(track)
                    detection = detections[detection_index]
                    track.last_detector_time = now
                    track.last_visual_time = now
                    track.last_detector_pts = pts
                    track.last_detection = detection
                    track.last_state = 'measured'
                    track.misses = 0
                    if getattr(detection, 'distance_m', None) is not None:
                        track.last_distance_m = float(detection.distance_m)
                        track.last_range_time = now
                    matched_tracks.add(track_id)
                    matched_detections.add(detection_index)
                for index, detection in enumerate(detections):
                    if index in matched_detections:
                        continue
                    track = _Track(self.next_id, detection, now, pts,
                                   direct_flow=True)
                    self.tracks[track.track_id] = track
                    tracks_to_seed.append(track)
                    self.next_id += 1
                if self.previous_gray is not None and tracks_to_seed:
                    specs = []
                    for track in tracks_to_seed:
                        if (track.track_id not in self.seed_pending and
                                (track.last_seed_time is None or
                                 now - track.last_seed_time >= 0.5)):
                            track.last_seed_time = now
                            self.seed_pending.add(track.track_id)
                            specs.append((track.track_id, track.box.copy()))
                    if specs:
                        seed_request = (self.previous_gray, specs)
                result = self._snapshot_locked(now) if return_snapshot else None
        if seed_request is not None:
            threading.Thread(target=self._seed_tracks_async, args=seed_request,
                             name='fuel-feature-seed', daemon=True).start()
        return result

    def advance(self, frame, now=None, pts=None):
        """Update all positions from one newly captured frame using sparse LK."""
        now = float(now if now is not None else time.monotonic())
        gray = self._gray(frame)
        with self.lock:
            if (pts is not None and self.previous_pts is not None and
                    pts <= self.previous_pts):
                return self._snapshot_locked(now)
            previous = self.previous_gray
            active = []
            point_groups = []
            for track in list(self.tracks.values()):
                if now - track.last_detector_time > self.max_age_s:
                    continue
                # In direct-flow mode advance replaces track.box rather than
                # mutating it, and holds the tracker lock until the frame is
                # complete. Keep this immutable reference instead of copying
                # the four element array twice before LK.
                prior_box = track.box if self.direct_flow else track.box.copy()
                if self.direct_flow:
                    predicted = prior_box
                else:
                    predicted = track.predict(now)
                if previous is not None:
                    points = track.flow_points
                    center_fallback = points is None or len(points) == 0
                    if center_fallback:
                        points = np.asarray([[[predicted[0] * FLOW_SCALE,
                                               predicted[1] * FLOW_SCALE]]],
                                             dtype=np.float32)
                    points = points.reshape(-1, 1, 2).astype(np.float32, copy=False)
                    active.append((track, prior_box, predicted, points,
                                   center_fallback, len(points)))
                    point_groups.append(points)
                elif track.last_detector_pts == pts:
                    track.last_state = 'measured'
                else:
                    track.last_state = 'predicted'
            if previous is not None and point_groups:
                all_points = np.concatenate(point_groups, axis=0)
                tracked, status, errors = cv2.calcOpticalFlowPyrLK(
                    previous, gray, all_points, None,
                    winSize=(FLOW_WINDOW_SIZE, FLOW_WINDOW_SIZE), maxLevel=0,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                              FLOW_MAX_ITERATIONS, 0.03), minEigThreshold=1e-4)
                if tracked is None or status is None:
                    tracked = np.empty_like(all_points)
                    status = np.zeros((len(all_points), 1), dtype=np.uint8)
                    errors = None
                tracked = tracked.reshape(-1, 2)
                status = status.reshape(-1).astype(bool)
                errors = errors.reshape(-1) if errors is not None else None
                offset = 0
                for track, prior_box, predicted, points, center_fallback, count in active:
                    end = offset + count
                    valid = status[offset:end].copy()
                    if not center_fallback and errors is not None:
                        valid &= errors[offset:end] < 35
                    good = tracked[offset:end][valid]
                    old = points.reshape(-1, 2)[valid]
                    if len(good):
                        delta = _small_median_displacement(good, old) / FLOW_SCALE
                        measured = predicted.copy()
                        measured[:2] = prior_box[:2] + delta
                        if self.direct_flow:
                            track.box = measured
                            track.last_update = now
                        else:
                            track.correct_box(measured, now)
                        track.flow_points = good.reshape(-1, 1, 2).astype(np.float32)
                        track.last_visual_time = now
                        track.last_state = 'flow'
                    else:
                        if self.direct_flow:
                            track.box = predicted
                            track.last_update = now
                        track.flow_points = None
                        track.last_state = 'predicted'
                    offset = end
            self.previous_gray = gray
            self.previous_pts = pts
            return self._snapshot_locked(now)

    def update_ranges(self, measurements, now=None, return_snapshot=True):
        """Attach fresh stereo measurements to live tracks without changing image state."""
        now = float(now if now is not None else time.monotonic())
        with self.lock:
            for track_id, distance_m, disparity_px in measurements:
                track = self.tracks.get(int(track_id))
                if (track is None or distance_m is None or
                        not math.isfinite(float(distance_m)) or float(distance_m) <= 0):
                    continue
                track.last_distance_m = float(distance_m)
                track.last_range_time = now
                track.last_detection.distance_m = float(distance_m)
                track.last_detection.disparity_px = (
                    float(disparity_px) if disparity_px is not None else None)
        return self.snapshot(now) if return_snapshot else None

    def snapshot(self, now=None):
        now = float(now if now is not None else time.monotonic())
        with self.lock:
            return self._snapshot_locked(now)

    def _snapshot_locked(self, now):
        result = []
        expired = []
        for track_id, track in self.tracks.items():
            detection_age = max(0.0, now - track.last_detector_time)
            visual_age = max(0.0, now - track.last_visual_time)
            if detection_age > self.max_age_s or visual_age > self.flow_grace_s:
                if detection_age > self.max_age_s:
                    expired.append(track_id)
                continue
            box = track.box
            box_snapshot = (float(box[0]), float(box[1]),
                            max(1.0, float(box[2])),
                            max(1.0, float(box[3])))
            confidence = float(track.last_detection.confidence) * math.exp(
                -detection_age / self.decay_s)
            range_age = (None if track.last_range_time is None else
                         max(0.0, now - track.last_range_time) * 1000.0)
            result.append({
                'detection': track.last_detection,
                'track_id': track.track_id,
                'state': ('measured' if detection_age < 0.001 else track.last_state),
                'age_ms': round(detection_age * 1000.0, 1),
                'visual_age_ms': round(visual_age * 1000.0, 1),
                'confidence': confidence,
                'box': box_snapshot,
                'distance_m': track.last_distance_m,
                'range_age_ms': round(range_age, 1) if range_age is not None else None,
            })
        for track_id in expired:
            self.tracks.pop(track_id, None)
        return sorted(result, key=lambda item: item['track_id'])
