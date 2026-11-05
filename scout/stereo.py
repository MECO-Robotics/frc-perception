"""CPU stereo-depth helper for rectified side-by-side camera frames."""
import os

import cv2
import numpy as np
from pathlib import Path
import time


class StereoDepth:
    def __init__(self, calibration_path=None, baseline_m=None, focal_px=None,
                 use_environment_defaults=True):
        self.enabled = os.getenv('SCOUT_STEREO_ENABLED', '1').lower() not in ('0', 'false', 'no')
        self.backend = os.getenv('SCOUT_STEREO_BACKEND', 'opencv').lower()
        if self.backend not in ('opencv', 'vpi_cuda'):
            raise ValueError('SCOUT_STEREO_BACKEND must be opencv or vpi_cuda')
        self.vpi = None
        self.vpi_output = None
        self.vpi_output_shape = None
        self.vpi_window = int(os.getenv('SCOUT_STEREO_VPI_WINDOW', '5'))
        if not 3 <= self.vpi_window <= 11 or self.vpi_window % 2 == 0:
            raise ValueError('SCOUT_STEREO_VPI_WINDOW must be an odd value from 3 to 11')
        if self.backend == 'vpi_cuda':
            try:
                import vpi
            except ImportError as exc:
                raise RuntimeError('VPI CUDA stereo requires NVIDIA VPI Python bindings') from exc
            self.vpi = vpi
        default_calibration = (os.getenv(
            'SCOUT_STEREO_CALIBRATION', 'media/calibration/stereo_calibration.yml')
            if use_environment_defaults else None)
        selected_calibration = calibration_path or default_calibration
        self.calibration_path = (Path(selected_calibration)
                                 if selected_calibration else None)
        self.calibration = None
        self.maps = None
        self.rectify = os.getenv('SCOUT_STEREO_RECTIFY', '0').lower() in ('1', 'true', 'yes')
        # An explicitly supplied mechanical baseline is useful when the
        # current stereo calibration is only an approximation. Keep calibrated
        # intrinsics/focal length, but do not let its noisy translation replace
        # the operator-provided baseline.
        baseline = (baseline_m if baseline_m is not None else
                    os.getenv('SCOUT_STEREO_BASELINE_M')
                    if use_environment_defaults else None)
        focal = (focal_px if focal_px is not None else
                 os.getenv('SCOUT_STEREO_FOCAL_PX')
                 if use_environment_defaults else None)
        self.focal_override = focal is not None
        self.disparity_scale = float(os.getenv('SCOUT_STEREO_DISPARITY_SCALE', '1'))
        self.num_disparities = int(os.getenv('SCOUT_STEREO_NUM_DISPARITIES', '128'))
        self.block_size = int(os.getenv('SCOUT_STEREO_BLOCK_SIZE', '5'))
        self.match_scale = float(os.getenv('SCOUT_STEREO_MATCH_SCALE', '0.5'))
        self.vertical_roi_pad = int(os.getenv('SCOUT_STEREO_VERTICAL_ROI_PAD', '0'))
        self.horizontal_roi_enabled = os.getenv(
            'SCOUT_STEREO_HORIZONTAL_ROI', '0').lower() in ('1', 'true', 'yes')
        self.local_rectify_enabled = os.getenv(
            'SCOUT_STEREO_LOCAL_RECTIFY', '0').lower() in ('1', 'true', 'yes')
        self.dense = os.getenv('SCOUT_STEREO_DENSE', '0').lower() in ('1', 'true', 'yes')
        self.adaptive_disparity_width = os.getenv(
            'SCOUT_STEREO_ADAPTIVE_DISPARITY_WIDTH', '0').lower() in ('1', 'true', 'yes')
        self.fuel_diameter_m = float(os.getenv('SCOUT_FUEL_DIAMETER_M', '0'))
        self.size_gate_max_relative_error = float(
            os.getenv('SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR', '0'))
        self.size_gate_duplicate_iou = float(
            os.getenv('SCOUT_STEREO_SIZE_GATE_DUPLICATE_IOU', '0.98'))
        if not 16 <= self.num_disparities <= 256 or self.num_disparities % 16:
            raise ValueError('SCOUT_STEREO_NUM_DISPARITIES must be a multiple of 16 from 16 to 256')
        if not 3 <= self.block_size <= 21 or self.block_size % 2 == 0:
            raise ValueError('SCOUT_STEREO_BLOCK_SIZE must be an odd value from 3 to 21')
        if not 0.25 <= self.match_scale <= 1.0:
            raise ValueError('SCOUT_STEREO_MATCH_SCALE must be between 0.25 and 1.0')
        if self.vertical_roi_pad < 0:
            raise ValueError('SCOUT_STEREO_VERTICAL_ROI_PAD must be nonnegative')
        if self.size_gate_max_relative_error < 0 or self.size_gate_max_relative_error > 1:
            raise ValueError('SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR must be 0 (disabled) or from 0 to 1')
        if self.size_gate_max_relative_error and self.fuel_diameter_m <= 0:
            raise ValueError('SCOUT_FUEL_DIAMETER_M must be positive when the stereo size gate is enabled')
        if not 0 <= self.size_gate_duplicate_iou <= 1:
            raise ValueError('SCOUT_STEREO_SIZE_GATE_DUPLICATE_IOU must be between 0 and 1')
        self.baseline_m = float(baseline) if baseline else None
        self.focal_px = float(focal) if focal else None
        if self.calibration_path is not None and self.calibration_path.is_file():
            storage = cv2.FileStorage(str(self.calibration_path), cv2.FILE_STORAGE_READ)
            values = {name: storage.getNode(name).mat() for name in (
                'left_camera_matrix', 'left_distortion', 'right_camera_matrix',
                'right_distortion', 'rotation', 'translation')}
            storage.release()
            if all(value is not None for value in values.values()):
                self.calibration = values
                calibrated_baseline = float(np.linalg.norm(values['translation']))
                self.baseline_m = (float(baseline) if baseline else calibrated_baseline)
                if not focal:
                    self.focal_px = float(values['left_camera_matrix'][0, 0])
        self.approximate = self.calibration is not None
        self.matcher = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=8 * self.block_size * self.block_size,
            P2=32 * self.block_size * self.block_size,
            uniquenessRatio=8, speckleWindowSize=50, speckleRange=2,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
        self.matcher_cache = {}
        self.match_maps = None
        self.last_num_disparities = self.num_disparities
        self.last_min_disparity = 0
        self.last_size_gate_rejected = 0
        self.last_size_gate_duplicate_rejected = 0

    @property
    def calibrated(self):
        return bool(self.enabled and self.baseline_m and self.focal_px)

    def project_pixel_to_camera(self, x_px, y_px, z_m, width, height):
        """Project a raw left-image pixel into camera-relative XYZ metres."""
        if not self.calibrated or self.calibration is None:
            return None
        values = (x_px, y_px, z_m, width, height)
        if not all(np.isfinite(float(value)) for value in values):
            return None
        if z_m <= 0 or width <= 0 or height <= 0:
            return None
        scale_x, scale_y = width / 1600.0, height / 1200.0
        kl = self.calibration['left_camera_matrix'].copy()
        kl[0, :] *= scale_x
        kl[1, :] *= scale_y
        if self.maps is not None and self.maps[0] == (width, height):
            _, _, _, kl, rotation_left, projection_left = self.maps
        else:
            rotation_left = np.eye(3, dtype=np.float64)
            projection_left = kl
        pixel = np.array([[[float(x_px), float(y_px)]]], dtype=np.float32)
        rectified = cv2.undistortPoints(
            pixel, kl, self.calibration['left_distortion'],
            R=rotation_left, P=projection_left)[0, 0]
        if not np.isfinite(rectified).all():
            return None
        fx, fy = float(projection_left[0, 0]), float(projection_left[1, 1])
        cx, cy = float(projection_left[0, 2]), float(projection_left[1, 2])
        if fx <= 0 or fy <= 0:
            return None
        return ((float(rectified[0]) - cx) * float(z_m) / fx,
                (float(rectified[1]) - cy) * float(z_m) / fy,
                float(z_m))

    def _matcher_for_width(self, num_disparities, min_disparity=0):
        if (num_disparities == self.num_disparities and
                min_disparity == 0):
            return self.matcher
        key = (int(min_disparity), int(num_disparities))
        matcher = self.matcher_cache.get(key)
        if matcher is None:
            block_size = self.block_size
            matcher = cv2.StereoSGBM_create(
                minDisparity=min_disparity, numDisparities=num_disparities,
                blockSize=block_size, P1=8 * block_size * block_size,
                P2=32 * block_size * block_size, uniquenessRatio=8,
                speckleWindowSize=50, speckleRange=2,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)
            self.matcher_cache[key] = matcher
        return matcher

    def _predicted_disparity_window(self, boxes):
        """Narrow SGBM around recent range estimates, retaining an anchor fallback."""
        predictions = []
        if self.calibrated and self.baseline_m and self.focal_px:
            for box in boxes:
                distance = getattr(box, 'distance_m', None)
                if distance is None or not np.isfinite(float(distance)) or float(distance) <= 0:
                    continue
                full_px = (self.baseline_m * self.focal_px /
                           (float(distance) * max(1e-6, self.disparity_scale)))
                predictions.append(full_px * self.match_scale)
        if not predictions or self.backend == 'vpi_cuda':
            return 0, self.num_disparities
        margin = max(8.0, max(predictions) * 0.2)
        low = max(0, int(np.floor((min(predictions) - margin) / 16.0)) * 16)
        minimum_window = min(32, self.num_disparities)
        high = max(low + minimum_window,
                   int(np.ceil((max(predictions) + margin) / 16.0)) * 16)
        width = min(self.num_disparities, max(16, high - low))
        if high - low > width:
            low = max(0, high - width)
        return low, width

    def estimate(self, left, right, boxes):
        if not self.enabled or not boxes:
            return boxes
        profile_start = time.perf_counter()
        left_source, right_source = left, right
        raw_height, raw_width = left.shape[:2]
        can_local_rectify = (self.local_rectify_enabled and not self.dense and
                             self.horizontal_roi_enabled and self.rectify and
                             self.calibration is not None)
        gray_left = gray_right = None
        dense_rectified = False
        if not can_local_rectify:
            if self.dense and self.match_scale < 1.0:
                reduced_size = (max(1, round(raw_width * self.match_scale)),
                                max(1, round(raw_height * self.match_scale)))
                reduced_left = cv2.resize(left, reduced_size,
                                          interpolation=cv2.INTER_AREA)
                reduced_right = cv2.resize(right, reduced_size,
                                           interpolation=cv2.INTER_AREA)
                gray_left = (reduced_left if reduced_left.ndim == 2 else
                             cv2.cvtColor(reduced_left, cv2.COLOR_BGR2GRAY))
                gray_right = (reduced_right if reduced_right.ndim == 2 else
                              cv2.cvtColor(reduced_right, cv2.COLOR_BGR2GRAY))
                dense_rectified = True
            else:
                gray_left = (left if left.ndim == 2 else
                             cv2.cvtColor(left, cv2.COLOR_BGR2GRAY))
                gray_right = (right if right.ndim == 2 else
                              cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))
        projection_left = projection_right = rotation_left = None
        rectified_box_bounds = None
        if self.calibration is not None and self.rectify:
            height, width = raw_height, raw_width
            cache_key = (width, height)
            if self.maps is None or self.maps[0] != cache_key:
                c = self.calibration
                sx = width / 1600.0
                sy = height / 1200.0
                kl, kr = c['left_camera_matrix'].copy(), c['right_camera_matrix'].copy()
                kl[0, :] *= sx; kl[1, :] *= sy
                kr[0, :] *= sx; kr[1, :] *= sy
                r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
                    kl, c['left_distortion'], kr, c['right_distortion'], cache_key,
                    c['rotation'], c['translation'], flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
                ml = cv2.initUndistortRectifyMap(kl, c['left_distortion'], r1, p1,
                                                  cache_key, cv2.CV_32FC1)
                mr = cv2.initUndistortRectifyMap(kr, c['right_distortion'], r2, p2,
                                                  cache_key, cv2.CV_32FC1)
                self.maps = (cache_key, ml, mr, kl, r1, p1)
                if not self.focal_override:
                    self.focal_px = float(p1[0, 0])
            _, ml, mr, kl, rotation_left, projection_left = self.maps
            local_rectify = (not self.dense and self.local_rectify_enabled and
                             self.horizontal_roi_enabled and boxes)
            if boxes:
                # Transform every box corner in one OpenCV call. The result is
                # also reused by range extraction below, avoiding two calls per
                # box when a crowded scene enables local rectification.
                corners = np.empty((len(boxes), 4, 1, 2), dtype=np.float32)
                for index, box in enumerate(boxes):
                    x1 = max(0, min(raw_width - 1, box.x1))
                    y1 = max(0, min(raw_height - 1, box.y1))
                    x2 = max(x1 + 1, min(raw_width, box.x2))
                    y2 = max(y1 + 1, min(raw_height, box.y2))
                    corners[index, :, 0, :] = ((x1, y1), (x2, y1),
                                                (x2, y2), (x1, y2))
                rectified_box_bounds = cv2.undistortPoints(
                    corners.reshape(-1, 1, 2), kl,
                    self.calibration['left_distortion'],
                    R=rotation_left, P=projection_left).reshape(len(boxes), 4, 2)
            local_bounds = []
            if local_rectify:
                for points in rectified_box_bounds:
                    if np.isfinite(points).all():
                        scaled_points = points * self.match_scale
                        local_bounds.append((float(scaled_points[:, 0].min()),
                                             float(scaled_points[:, 1].min()),
                                             float(scaled_points[:, 0].max()),
                                             float(scaled_points[:, 1].max())))
            if local_rectify and local_bounds:
                match_width = max(1, round(raw_width * self.match_scale))
                match_height = max(1, round(raw_height * self.match_scale))
                if self.match_maps is None or self.match_maps[0] != (raw_width, raw_height, match_width, match_height):
                    self.match_maps = ((raw_width, raw_height, match_width, match_height),
                                       tuple(cv2.resize(channel, (match_width, match_height),
                                                        interpolation=cv2.INTER_LINEAR)
                                             for channel in ml),
                                       tuple(cv2.resize(channel, (match_width, match_height),
                                                        interpolation=cv2.INTER_LINEAR)
                                             for channel in mr))
                bounds_x = [v for b in local_bounds for v in (b[0], b[2])]
                bounds_y = [v for b in local_bounds for v in (b[1], b[3])]
                min_disparity, active = self._predicted_disparity_window(boxes)
                if self.adaptive_disparity_width:
                    leftmost = min((b[0] + b[2]) / 2 for b in local_bounds)
                    active = min(active, max(16, (int(np.floor(leftmost)) // 16) * 16))
                search_context = min_disparity + active + 5
                roi_x_offset = max(0, int(np.floor(min(bounds_x))) - search_context)
                roi_x_end = min(match_width, int(np.ceil(max(bounds_x))) + 5)
                roi_y_offset = max(0, int(np.floor(min(bounds_y))) - self.vertical_roi_pad)
                roi_y_end = min(match_height, int(np.ceil(max(bounds_y))) + self.vertical_roi_pad)
                if roi_x_end - roi_x_offset > search_context and roi_y_end > roi_y_offset:
                    map_left = tuple(channel[roi_y_offset:roi_y_end,
                                             roi_x_offset:roi_x_end]
                                     for channel in self.match_maps[1])
                    map_right = tuple(channel[roi_y_offset:roi_y_end,
                                              roi_x_offset:roi_x_end]
                                      for channel in self.match_maps[2])
                    local_left = cv2.remap(left_source, map_left[0], map_left[1],
                                           cv2.INTER_LINEAR)
                    local_right = cv2.remap(right_source, map_right[0], map_right[1],
                                            cv2.INTER_LINEAR)
                    gray_left = (local_left if local_left.ndim == 2 else
                                 cv2.cvtColor(local_left, cv2.COLOR_BGR2GRAY))
                    gray_right = (local_right if local_right.ndim == 2 else
                                  cv2.cvtColor(local_right, cv2.COLOR_BGR2GRAY))
                    self._local_rectified = True
                else:
                    local_rectify = False
                    self._local_rectified = False
            else:
                local_rectify = False
                self._local_rectified = False
            if dense_rectified:
                match_width = gray_left.shape[1]
                match_height = gray_left.shape[0]
                map_key = (raw_width, raw_height, match_width, match_height)
                if self.match_maps is None or self.match_maps[0] != map_key:
                    scaled_maps = []
                    for source_map in (ml, mr):
                        channels = []
                        for index, channel in enumerate(source_map):
                            scaled = cv2.resize(channel, (match_width, match_height),
                                                interpolation=cv2.INTER_LINEAR)
                            scaled *= self.match_scale
                            channels.append(scaled)
                        scaled_maps.append(tuple(channels))
                    self.match_maps = (map_key, tuple(scaled_maps[0]),
                                       tuple(scaled_maps[1]))
                gray_left = cv2.remap(gray_left, *self.match_maps[1], cv2.INTER_LINEAR)
                gray_right = cv2.remap(gray_right, *self.match_maps[2], cv2.INTER_LINEAR)
            elif not local_rectify:
                if gray_left is None:
                    gray_left = (left_source if left_source.ndim == 2 else
                                 cv2.cvtColor(left_source, cv2.COLOR_BGR2GRAY))
                    gray_right = (right_source if right_source.ndim == 2 else
                                  cv2.cvtColor(right_source, cv2.COLOR_BGR2GRAY))
                gray_left = cv2.remap(gray_left, *ml, cv2.INTER_LINEAR)
                gray_right = cv2.remap(gray_right, *mr, cv2.INTER_LINEAR)
        else:
            local_rectify = False
            self._local_rectified = False
            if gray_left is None:
                gray_left = (left_source if left_source.ndim == 2 else
                             cv2.cvtColor(left_source, cv2.COLOR_BGR2GRAY))
                gray_right = (right_source if right_source.ndim == 2 else
                              cv2.cvtColor(right_source, cv2.COLOR_BGR2GRAY))
        self.last_profile = {
            'rectify_and_roi_ms': (time.perf_counter() - profile_start) * 1000.0}
        match_width = max(1, round(gray_left.shape[1] * self.match_scale))
        match_height = max(1, round(gray_left.shape[0] * self.match_scale))
        if local_rectify:
            match_left, match_right = gray_left, gray_right
            match_width = max(1, round(raw_width * self.match_scale))
            match_height = max(1, round(raw_height * self.match_scale))
        elif dense_rectified:
            match_left, match_right = gray_left, gray_right
            match_width, match_height = gray_left.shape[1], gray_left.shape[0]
        else:
            match_left = cv2.resize(gray_left, (match_width, match_height), interpolation=cv2.INTER_AREA)
            match_right = cv2.resize(gray_right, (match_width, match_height), interpolation=cv2.INTER_AREA)
        roi_y_offset = 0 if not local_rectify else roi_y_offset
        if self.vertical_roi_pad and boxes and not local_rectify and not self.dense:
            rectified_y = []
            for box in boxes:
                if projection_left is not None:
                    corners = np.array([[[box.x1, box.y1]], [[box.x2, box.y1]],
                                        [[box.x2, box.y2]], [[box.x1, box.y2]]], dtype=np.float32)
                    points = cv2.undistortPoints(
                        corners, kl, self.calibration['left_distortion'],
                        R=rotation_left, P=projection_left).reshape(-1, 2)
                    finite = np.isfinite(points).all(axis=1)
                    if finite.any():
                        rectified_y.extend(points[finite, 1] * self.match_scale)
                else:
                    rectified_y.extend((box.y1 * self.match_scale, box.y2 * self.match_scale))
            rectified_y = np.asarray(rectified_y, dtype=np.float64)
            rectified_y = rectified_y[np.isfinite(rectified_y)]
            if not rectified_y.size:
                rectified_y = None
        else:
            rectified_y = None
        if rectified_y is not None:
            roi_y_offset = max(0, int(np.floor(min(rectified_y))) - self.vertical_roi_pad)
            roi_y_end = min(match_height, int(np.ceil(max(rectified_y))) + self.vertical_roi_pad)
            match_left = match_left[roi_y_offset:roi_y_end, :]
            match_right = match_right[roi_y_offset:roi_y_end, :]
        rectified_bounds = local_bounds if local_rectify else []
        if ((self.horizontal_roi_enabled or self.adaptive_disparity_width) and
                boxes and not local_rectify and not self.dense):
            if rectified_box_bounds is not None:
                scaled_bounds = rectified_box_bounds * self.match_scale
                finite = np.isfinite(scaled_bounds).all(axis=(1, 2))
                valid_bounds = scaled_bounds[finite]
                if len(valid_bounds):
                    rectified_bounds.extend(np.column_stack((
                        valid_bounds[:, :, 0].min(axis=1),
                        valid_bounds[:, :, 1].min(axis=1),
                        valid_bounds[:, :, 0].max(axis=1),
                        valid_bounds[:, :, 1].max(axis=1))).tolist())
            else:
                raw_bounds = np.asarray([(box.x1, box.y1, box.x2, box.y2)
                                         for box in boxes], dtype=np.float32)
                points = raw_bounds[:, None, :].reshape(-1, 2) * self.match_scale
                points = points.reshape(-1, 2, 2)
                raw_rectified_bounds = np.column_stack((
                    points[:, :, 0].min(axis=1), points[:, :, 1].min(axis=1),
                    points[:, :, 0].max(axis=1), points[:, :, 1].max(axis=1)))
                rectified_bounds.extend(raw_rectified_bounds.tolist())
        min_disparity, active_num_disparities = (
            (0, self.num_disparities) if self.dense else
            self._predicted_disparity_window(boxes))
        if self.adaptive_disparity_width and rectified_bounds:
            leftmost_center = min((bounds[0] + bounds[2]) / 2
                                  for bounds in rectified_bounds)
            edge_limited_width = max(16, (int(np.floor(leftmost_center)) // 16) * 16)
            active_num_disparities = min(self.num_disparities, edge_limited_width)
        self.last_num_disparities = active_num_disparities
        self.last_min_disparity = min_disparity
        matcher = self._matcher_for_width(active_num_disparities, min_disparity)
        if not local_rectify:
            roi_x_offset = 0
        if self.horizontal_roi_enabled and rectified_bounds and not local_rectify and not self.dense:
            rectified_x = [x for bounds in rectified_bounds
                           for x in (bounds[0], bounds[2])]
            # A left pixel may match as far as num_disparities pixels to its
            # left. Keep that search context plus the SGBM block window.
            search_context = min_disparity + active_num_disparities + 5
            crop_start = max(0, int(np.floor(min(rectified_x))) - search_context)
            crop_end = min(match_width,
                           int(np.ceil(max(rectified_x))) + 5)
            if crop_end - crop_start > search_context:
                match_left = match_left[:, crop_start:crop_end]
                match_right = match_right[:, crop_start:crop_end]
                roi_x_offset = crop_start
        matcher_started = time.perf_counter()
        if self.backend == 'vpi_cuda':
            vpi = self.vpi
            output_size = (match_left.shape[1], match_left.shape[0])
            if self.vpi_output is None or self.vpi_output_shape != output_size:
                self.vpi_output = vpi.Image(output_size, vpi.Format.S16)
                self.vpi_output_shape = output_size
            left_image = vpi.asimage(np.ascontiguousarray(match_left), format=vpi.Format.U8)
            right_image = vpi.asimage(np.ascontiguousarray(match_right), format=vpi.Format.U8)
            with vpi.Backend.CUDA:
                disparity_image = vpi.stereodisp(
                    left_image, right_image, backend=vpi.Backend.CUDA,
                    out=self.vpi_output, window=self.vpi_window,
                    maxdisp=self.last_num_disparities, confthreshold=1,
                    uniqueness=-1, includediagonals=True)
            disparity = disparity_image.cpu().astype(np.float32) / 32.0
        else:
            disparity = matcher.compute(match_left, match_right).astype(np.float32) / 16.0
        self.last_profile['matcher_and_transfer_ms'] = (
            time.perf_counter() - matcher_started) * 1000.0
        range_started = time.perf_counter()
        if local_rectify:
            height = max(1, round(raw_height * self.match_scale))
            width = max(1, round(raw_width * self.match_scale))
        else:
            height, width = gray_left.shape
        # Transform all box corners in one OpenCV call. Calling
        # undistortPoints separately for every object dominates range lookup
        # once a crowded view contains dozens of balls.
        raw_boxes = np.asarray([(box.x1, box.y1, box.x2, box.y2)
                                for box in boxes], dtype=np.float32)
        x1 = np.clip(raw_boxes[:, 0], 0, raw_width - 1)
        y1 = np.clip(raw_boxes[:, 1], 0, raw_height - 1)
        x2 = np.maximum(x1 + 1, np.minimum(raw_width, raw_boxes[:, 2]))
        y2 = np.maximum(y1 + 1, np.minimum(raw_height, raw_boxes[:, 3]))
        valid_boxes = np.ones(len(boxes), dtype=bool)
        if projection_left is not None:
            scaled_rectified = rectified_box_bounds * self.match_scale
            valid_boxes = np.isfinite(scaled_rectified).all(axis=(1, 2))
            lower = np.floor(np.nan_to_num(scaled_rectified, nan=0.0).min(axis=1)).astype(np.int32)
            upper = np.ceil(np.nan_to_num(scaled_rectified, nan=0.0).max(axis=1)).astype(np.int32)
            rx1 = np.clip(lower[:, 0], 0, width) - roi_x_offset
            ry1 = np.clip(lower[:, 1], 0, height) - roi_y_offset
            rx2 = np.clip(upper[:, 0], 0, width) - roi_x_offset
            ry2 = np.clip(upper[:, 1], 0, height) - roi_y_offset
        else:
            rx1 = (x1 * self.match_scale).astype(np.int32) - roi_x_offset
            ry1 = (y1 * self.match_scale).astype(np.int32) - roi_y_offset
            rx2 = (x2 * self.match_scale).astype(np.int32) - roi_x_offset
            ry2 = (y2 * self.match_scale).astype(np.int32) - roi_y_offset
        rx1 = np.clip(rx1, 0, disparity.shape[1])
        rx2 = np.clip(rx2, 0, disparity.shape[1])
        ry1 = np.clip(ry1, 0, disparity.shape[0])
        ry2 = np.clip(ry2, 0, disparity.shape[0])
        roi_width = np.maximum(0, rx2 - rx1)
        roi_height = np.maximum(0, ry2 - ry1)
        # Sample a uniform grid per box (exact for balls up to 8 matcher
        # pixels across). This bounds range extraction cost when many objects
        # are visible while preserving the robust median statistic.
        grid_size = 8
        fractions = (np.arange(grid_size, dtype=np.float32) + 0.5) / grid_size
        sample_x = rx1[:, None] + np.minimum(
            (fractions[None, :] * roi_width[:, None]).astype(np.int32),
            np.maximum(0, roi_width[:, None] - 1))
        sample_y = ry1[:, None] + np.minimum(
            (fractions[None, :] * roi_height[:, None]).astype(np.int32),
            np.maximum(0, roi_height[:, None] - 1))
        sample_x = np.clip(sample_x, 0, disparity.shape[1] - 1)
        sample_y = np.clip(sample_y, 0, disparity.shape[0] - 1)
        flat_indices = (sample_y[:, :, None] * disparity.shape[1] +
                        sample_x[:, None, :]).reshape(len(boxes), -1)
        samples = disparity.reshape(-1)[flat_indices]
        accepted = (np.isfinite(samples) &
                    (samples > 3.0 * self.match_scale) &
                    (samples < 120.0 * self.match_scale))
        sample_counts = accepted.sum(axis=1)
        samples[~accepted] = np.inf
        samples.sort(axis=1)
        lower_mid = np.maximum(0, (sample_counts - 1) // 2)
        upper_mid = np.maximum(0, sample_counts // 2)
        median_disparity = (
            np.take_along_axis(samples, lower_mid[:, None], axis=1)[:, 0] +
            np.take_along_axis(samples, upper_mid[:, None], axis=1)[:, 0]) * 0.5
        enough_area = ((np.minimum(roi_width, grid_size) *
                        np.minimum(roi_height, grid_size)) >= 8)
        enough_samples = sample_counts >= 8
        has_depth = valid_boxes & enough_area & enough_samples & np.isfinite(median_disparity)
        for index, box in enumerate(boxes):
            box.disparity_px = (float(median_disparity[index] / self.match_scale)
                                if has_depth[index] else None)
            effective_disparity = (box.disparity_px * self.disparity_scale
                                   if box.disparity_px is not None else None)
            box.distance_m = (self.baseline_m * self.focal_px / effective_disparity
                              if self.calibrated and effective_disparity else None)
        self.last_profile['range_extract_ms'] = (time.perf_counter() - range_started) * 1000.0
        self.last_profile['total_ms'] = (time.perf_counter() - profile_start) * 1000.0
        return boxes

    def filter_fuel_size_inconsistent(self, boxes, frame_width=1600):
        """Drop detections whose stereo Z conflicts with nominal FUEL apparent size.

        This is an optional quality gate for robot-facing 3D positions. A box
        without valid stereo depth cannot pass the gate because it cannot form
        a camera-relative 3D point.
        """
        self.last_size_gate_rejected = 0
        self.last_size_gate_duplicate_rejected = 0
        if not self.size_gate_max_relative_error:
            return boxes
        if self.calibration is None or not self.calibrated:
            self.last_size_gate_rejected = len(boxes)
            return []
        focal_px = (float(self.calibration['left_camera_matrix'][0, 0]) *
                    (float(frame_width) / 1600.0))
        kept = []
        for box in boxes:
            width = max(0, box.x2 - box.x1)
            height = max(0, box.y2 - box.y1)
            size_range_m = (focal_px * self.fuel_diameter_m / np.sqrt(width * height)
                            if width and height else None)
            distance_m = getattr(box, 'distance_m', None)
            relative_error = (abs(size_range_m - distance_m) / distance_m
                              if size_range_m is not None and distance_m else None)
            if (relative_error is not None and
                    relative_error <= self.size_gate_max_relative_error):
                kept.append(box)
        self.last_size_gate_rejected = len(boxes) - len(kept)
        if self.size_gate_duplicate_iou <= 0:
            return kept
        unique = []
        points = {}
        rectification = self.maps

        def camera_point(box):
            if rectification is None or not getattr(box, 'distance_m', None):
                return None
            _, _, _, kl, rotation_left, projection_left = rectification
            center = np.array([[[float(box.x1 + box.x2) / 2,
                                 float(box.y1 + box.y2) / 2]]], dtype=np.float32)
            u, v = cv2.undistortPoints(
                center, kl, self.calibration['left_distortion'],
                R=rotation_left, P=projection_left)[0, 0]
            z = float(box.distance_m)
            return np.array(((u - projection_left[0, 2]) * z / projection_left[0, 0],
                             (v - projection_left[1, 2]) * z / projection_left[1, 1], z))

        for box in sorted(kept, key=lambda item: item.confidence, reverse=True):
            duplicate = False
            for previous in unique:
                if box.class_id != previous.class_id:
                    continue
                x1, y1 = max(box.x1, previous.x1), max(box.y1, previous.y1)
                x2, y2 = min(box.x2, previous.x2), min(box.y2, previous.y2)
                intersection = max(0, x2 - x1) * max(0, y2 - y1)
                area_a = max(0, box.x2 - box.x1) * max(0, box.y2 - box.y1)
                area_b = max(0, previous.x2 - previous.x1) * max(0, previous.y2 - previous.y1)
                union = area_a + area_b - intersection
                if union and intersection / union > self.size_gate_duplicate_iou:
                    point_a = points.setdefault(id(box), camera_point(box))
                    point_b = points.setdefault(id(previous), camera_point(previous))
                    if (point_a is not None and point_b is not None and
                            np.linalg.norm(point_a - point_b) < self.fuel_diameter_m * 0.5):
                        duplicate = True
                        break
            if duplicate:
                self.last_size_gate_duplicate_rejected += 1
            else:
                unique.append(box)
                points[id(box)] = camera_point(box)
        return unique


class CharucoCollector:
    """Collect distinct, usable synchronized stereo board views."""
    MIN_CORNERS = 12
    MIN_COMMON_CORNERS = 10
    MIN_BOARD_AREA = 0.015
    MAX_BOARD_AREA = 0.55
    MAX_BOARD_DIMENSION = 0.88
    MIN_EDGE_MARGIN = 0.01
    MIN_SHARPNESS = 8.0

    def __init__(self, calibration_path=None):
        self.active = False
        self.samples = 0
        self.next_index = 1
        self.last_capture = 0.0
        self.last_preview_frame = 0.0
        self.preview_frames = {'left': None, 'right': None}
        self.last_signature = None
        self.signatures = []
        self.cols, self.rows, self.square_m = 5, 7, 0.03
        self.dictionary_name = 'DICT_4X4_50'
        self.board_type = 'custom'
        self.marker_ratio = 0.70
        self.marker_mm = 21.0
        self.frames_checked = 0
        self.last_reason = 'not started'
        self.last_corners = 0
        self.calibration_state = 'not_run'
        self.calibration_result = None
        self.preview = dict(active=False, left_markers=0, right_markers=0,
                            left_corners=0, right_corners=0, polygon=[], corners=[])
        if calibration_path:
            self.calibration_file = Path(calibration_path)
            self.output = (self.calibration_file.parent /
                           f'{self.calibration_file.stem}_captures')
        else:
            self.output = Path(os.getenv('SCOUT_CALIBRATION_DIR', 'media/calibration'))
            self.calibration_file = self.output / 'stereo_calibration.yml'
        self.output.mkdir(parents=True, exist_ok=True)
        self.lock = __import__('threading').Lock()

    def start(self, cols=5, rows=7, square_mm=30, dictionary_name='DICT_4X4_50', board_type='custom', marker_mm=None):
        dictionaries = {'DICT_4X4_50': cv2.aruco.DICT_4X4_50, 'DICT_4X4_100': cv2.aruco.DICT_4X4_100,
                        'DICT_4X4_250': cv2.aruco.DICT_4X4_250,
                        'DICT_5X5_100': cv2.aruco.DICT_5X5_100, 'DICT_6X6_250': cv2.aruco.DICT_6X6_250}
        if board_type not in ('custom', 'andymark'):
            raise ValueError('Unsupported calibration board')
        if board_type == 'andymark':
            cols, rows, dictionary_name = 15, 11, 'DICT_4X4_250'
        marker_mm = float(marker_mm) if marker_mm is not None else (10 if board_type == 'andymark' else float(square_mm) * 0.70)
        if marker_mm <= 0 or marker_mm >= float(square_mm):
            raise ValueError('Marker width must be greater than zero and smaller than square width')
        if dictionary_name not in dictionaries:
            raise ValueError('Unsupported ChArUco dictionary')
        with self.lock:
            self.cols, self.rows, self.square_m = int(cols), int(rows), float(square_mm) / 1000
            self.dictionary_name, self.dictionary_id = dictionary_name, dictionaries[dictionary_name]
            self.board_type = board_type
            self.marker_mm = marker_mm
            self.marker_ratio = marker_mm / float(square_mm)
            left_indices = {int(p.stem.split('_')[-1]) for p in self.output.glob('left_[0-9][0-9][0-9].png')}
            right_indices = {int(p.stem.split('_')[-1]) for p in self.output.glob('right_[0-9][0-9][0-9].png')}
            paired_indices = left_indices & right_indices
            self.samples = len(paired_indices)
            self.next_index = max(paired_indices, default=0) + 1
            self.last_capture = 0
            self.last_preview_frame = 0
            self.preview_frames = {'left': None, 'right': None}
            self.last_signature = None
            self.signatures = []
            self.frames_checked = 0
            self.last_reason = 'looking for board in both sensors'
            self.last_corners = 0
            self.calibration_state = 'capturing'
            self.calibration_result = None
            self.active = True
            self.preview = dict(active=True, left_markers=0, right_markers=0,
                                left_corners=0, right_corners=0, polygon=[], corners=[])
        return self.status()

    def stop(self):
        with self.lock:
            self.active = False
            self.calibration_state = 'computing'
            self.preview = dict(self.preview, active=False)
        self._compute_calibration()
        return self.status()

    def clear(self):
        with self.lock:
            if self.active or self.calibration_state == 'computing':
                raise RuntimeError('Stop calibration capture before clearing images')
            for pattern in ('left_[0-9][0-9][0-9].png', 'right_[0-9][0-9][0-9].png',
                            'debug_left.png', 'debug_right.png'):
                for filename in self.output.glob(pattern):
                    filename.unlink()
            self.samples = 0
            self.next_index = 1
            self.frames_checked = 0
            self.last_corners = 0
            self.last_reason = 'captures cleared'
            self.calibration_result = None
            self.calibration_state = 'not_run'
            self.preview = dict(active=False, left_markers=0, right_markers=0,
                                left_corners=0, right_corners=0, polygon=[], corners=[])
        return self.status()

    def status(self):
        with self.lock:
            return dict(active=self.active, samples=self.samples, target=15,
                        cols=self.cols, rows=self.rows, square_mm=round(self.square_m * 1000, 2),
                        dictionary=self.dictionary_name, board_type=self.board_type,
                        marker_mm=round(self.marker_mm, 2), directory=str(self.output),
                        calibration_file=str(self.calibration_file),
                        frames_checked=self.frames_checked, last_reason=self.last_reason,
                        last_corners=self.last_corners, calibration_state=self.calibration_state,
                        calibration=self.calibration_result, preview=self.preview)

    def preview_frame(self, side):
        if side not in self.preview_frames:
            raise ValueError('Unknown camera side')
        with self.lock:
            return self.preview_frames[side]

    def _compute_calibration(self):
        left_files = sorted(self.output.glob('left_[0-9][0-9][0-9].png'))
        right_files = sorted(self.output.glob('right_[0-9][0-9][0-9].png'))
        dictionary = cv2.aruco.getPredefinedDictionary(self.dictionary_id)
        board = cv2.aruco.CharucoBoard_create(self.cols, self.rows, self.square_m, self.square_m * self.marker_ratio, dictionary)
        params = cv2.aruco.DetectorParameters_create()
        params.minMarkerPerimeterRate = .005
        params.maxMarkerPerimeterRate = 1.0
        params.adaptiveThreshWinSizeMin = 3
        params.adaptiveThreshWinSizeMax = 51
        params.adaptiveThreshWinSizeStep = 4
        object_points, left_points, right_points, view_names = [], [], [], []
        rejected = {}
        accepted_signatures = []
        def reject(reason):
            rejected[reason] = rejected.get(reason, 0) + 1

        def quality(image, points):
            if image is None or points is None or len(points) < self.MIN_CORNERS:
                return 'too_few_corners'
            height, width = image.shape[:2]
            xy = points.reshape(-1, 2).astype(np.float32)
            x0, y0 = xy.min(axis=0)
            x1, y1 = xy.max(axis=0)
            area = ((x1 - x0) * (y1 - y0)) / (width * height)
            if area < self.MIN_BOARD_AREA:
                return 'board_too_small'
            if area > self.MAX_BOARD_AREA or (x1 - x0) / width > self.MAX_BOARD_DIMENSION or (y1 - y0) / height > self.MAX_BOARD_DIMENSION:
                return 'board_too_close_or_large'
            if x0 / width < self.MIN_EDGE_MARGIN or y0 / height < self.MIN_EDGE_MARGIN or (width - x1) / width < self.MIN_EDGE_MARGIN or (height - y1) / height < self.MIN_EDGE_MARGIN:
                return 'board_clipped_or_near_edge'
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            roi = gray[max(0, int(y0)):min(height, int(np.ceil(y1)) + 1), max(0, int(x0)):min(width, int(np.ceil(x1)) + 1)]
            if roi.size == 0 or cv2.Laplacian(roi, cv2.CV_64F).var() < self.MIN_SHARPNESS:
                return 'blurry'
            return None
        image_size = None
        for left_file, right_file in zip(left_files, right_files):
            views = []
            images = []
            for filename in (left_file, right_file):
                image = cv2.imread(str(filename))
                if image is None:
                    views.append((0, None, None)); images.append(None); continue
                images.append(image)
                image_size = (image.shape[1], image.shape[0])
                # Captures are already high resolution. Detect at native size
                # to avoid multiplying calibration cost across the whole set.
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY, 31, 5)
                detected = cv2.aruco.detectMarkers(binary, dictionary, parameters=params)
                if detected[1] is None:
                    views.append((0, None, None)); continue
                count, corners, ids = cv2.aruco.interpolateCornersCharuco(
                    detected[0], detected[1], binary, board)
                views.append((int(count or 0), corners, ids))
            if len(views) != 2:
                reject('missing_pair')
                continue
            reasons = [quality(images[i], views[i][1]) for i in range(2)]
            if any(reasons):
                reject(next(reason for reason in reasons if reason))
                continue
            left_ids, right_ids = views[0][2].ravel().astype(int), views[1][2].ravel().astype(int)
            common = np.intersect1d(left_ids, right_ids)
            if len(common) < self.MIN_COMMON_CORNERS:
                reject('too_few_stereo_matches')
                continue
            def points(view):
                ids, corners = view[2].ravel().astype(int), view[1].reshape(-1, 2)
                return np.array([corners[np.where(ids == item)[0][0]] for item in common], dtype=np.float32)
            signature_points = views[0][1].reshape(-1, 2).astype(np.float32)
            sx0, sy0 = signature_points.min(axis=0); sx1, sy1 = signature_points.max(axis=0)
            signature = (float((sx0+sx1)/2/image_size[0]), float((sy0+sy1)/2/image_size[1]),
                         float((sx1-sx0)/image_size[0]), float((sy1-sy0)/image_size[1]),
                         float(cv2.minAreaRect(signature_points)[-1]))
            if any(abs(signature[0]-old[0]) < .045 and abs(signature[1]-old[1]) < .045
                   and abs(signature[2]-old[2]) < .10 and abs(signature[3]-old[3]) < .10
                   and min(abs(signature[4]-old[4]), 180-abs(signature[4]-old[4])) < 8
                   for old in accepted_signatures):
                reject('redundant_view')
                continue
            accepted_signatures.append(signature)
            object_points.append(np.array([board.chessboardCorners[item] for item in common], dtype=np.float32))
            left_points.append(points(views[0]))
            right_points.append(points(views[1]))
            view_names.append(left_file.name)
        if len(object_points) < 5 or image_size is None:
            result = dict(state='insufficient', usable_pairs=len(object_points), required_pairs=5,
                          rejected_pairs=rejected)
            with self.lock: self.calibration_state, self.calibration_result = 'insufficient', result
            return
        # Fit the ordinary five-coefficient lens model first. The rational
        # model has six extra distortion terms and can become unstable with a
        # small set of handheld board views. Iteratively remove paired views
        # that have unusually high reprojection error in either camera.
        def fit_camera(image_points):
            rms, camera_matrix, distortion, rotations, translations = cv2.calibrateCamera(
                object_points, image_points, image_size, None, None)
            errors = []
            for objects, observed, rotation_vector, translation_vector in zip(
                    object_points, image_points, rotations, translations):
                projected, _ = cv2.projectPoints(objects, rotation_vector, translation_vector,
                                                  camera_matrix, distortion)
                delta = projected.reshape(-1, 2) - observed.reshape(-1, 2)
                errors.append(float(np.sqrt(np.mean(np.sum(delta * delta, axis=1)))))
            return rms, camera_matrix, distortion, errors

        discarded_views = []
        for _ in range(len(object_points)):
            left_fit = fit_camera(left_points)
            right_fit = fit_camera(right_points)
            pair_errors = np.maximum(left_fit[3], right_fit[3])
            median_error = float(np.median(pair_errors))
            mad = float(np.median(np.abs(pair_errors - median_error)))
            cutoff = max(2.5, median_error + 2.5 * 1.4826 * mad)
            outliers = [index for index, error in enumerate(pair_errors) if error > cutoff]
            if not outliers:
                break
            if len(object_points) - len(outliers) < 5:
                result = dict(state='poor_fit', usable_pairs=len(object_points),
                              rejected_pairs=rejected, discarded_pairs=[view_names[i] for i in outliers],
                              reason='Outlier rejection left fewer than five usable pairs')
                with self.lock: self.calibration_state, self.calibration_result = 'poor_fit', result
                return
            for index in reversed(outliers):
                discarded_views.append(view_names.pop(index))
                object_points.pop(index)
                left_points.pop(index)
                right_points.pop(index)
        left_rms, left_k, left_d = left_fit[:3]
        right_rms, right_k, right_d = right_fit[:3]
        criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-6)
        stereo_rms, left_k, left_d, right_k, right_d, rotation, translation, _, _ = cv2.stereoCalibrate(
            object_points, left_points, right_points, left_k, left_d, right_k, right_d,
            image_size, criteria=criteria, flags=cv2.CALIB_FIX_INTRINSIC)
        if discarded_views:
            rejected['reprojection_outlier'] = len(discarded_views)
        if max(float(left_rms), float(right_rms), float(stereo_rms)) > 2.0:
            result = dict(state='poor_fit', usable_pairs=len(object_points),
                          rejected_pairs=rejected, discarded_pairs=discarded_views,
                          left_rms=round(float(left_rms), 5), right_rms=round(float(right_rms), 5),
                          stereo_rms=round(float(stereo_rms), 5),
                          reason='Reprojection RMS exceeds 2 pixels; calibration was not saved')
            with self.lock: self.calibration_state, self.calibration_result = 'poor_fit', result
            return
        path = self.calibration_file
        path.parent.mkdir(parents=True, exist_ok=True)
        storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_WRITE)
        for name, value in (('left_camera_matrix', left_k), ('left_distortion', left_d),
                            ('right_camera_matrix', right_k), ('right_distortion', right_d),
                            ('rotation', rotation), ('translation', translation)):
            storage.write(name, value)
        storage.release()
        result = dict(state='complete', usable_pairs=len(object_points), image_size=list(image_size),
                      rejected_pairs=rejected, discarded_pairs=discarded_views,
                      left_rms=round(float(left_rms), 5), right_rms=round(float(right_rms), 5),
                      stereo_rms=round(float(stereo_rms), 5), baseline_m=round(float(np.linalg.norm(translation)), 6),
                      calibration_file=str(path))
        with self.lock: self.calibration_state, self.calibration_result = 'complete', result

    def observe(self, left, right):
        with self.lock:
            if not self.active:
                return
            now = time.monotonic()
            update_frames = now - self.last_preview_frame >= 0.35
            if update_frames:
                self.last_preview_frame = now
                frame_pair = (left.copy(), right.copy())
            else:
                frame_pair = None
            should_detect = now - self.last_capture >= 0.75
        if frame_pair is not None:
            encoded_frames = {}
            for side, frame in zip(('left', 'right'), frame_pair):
                height, width = frame.shape[:2]
                if width > 800:
                    frame = cv2.resize(frame, (800, round(height * 800 / width)), interpolation=cv2.INTER_AREA)
                ok, encoded = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
                if ok:
                    encoded_frames[side] = encoded.tobytes()
            with self.lock:
                self.preview_frames.update(encoded_frames)
        if not should_detect:
            return
        with self.lock:
            if not self.active:
                return
            self.frames_checked += 1
            if self.frames_checked == 1 or self.frames_checked % 120 == 0:
                cv2.imwrite(str(self.output / 'debug_left.png'), left)
                cv2.imwrite(str(self.output / 'debug_right.png'), right)
            dictionary_id = self.dictionary_id
            cols, rows, marker_ratio = self.cols, self.rows, self.marker_ratio
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        params = cv2.aruco.DetectorParameters_create()
        # Laptop-displayed boards can be small in a wide-FOV frame. Upscale
        # only for marker detection; retain native frames for calibration.
        params.minMarkerPerimeterRate = 0.01
        params.maxMarkerPerimeterRate = 1.0
        detect_left = cv2.resize(left, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        detect_right = cv2.resize(right, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        detect_left = cv2.adaptiveThreshold(cv2.cvtColor(detect_left, cv2.COLOR_BGR2GRAY),
                                            255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY, 31, 5)
        detect_right = cv2.adaptiveThreshold(cv2.cvtColor(detect_right, cv2.COLOR_BGR2GRAY),
                                             255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                             cv2.THRESH_BINARY, 31, 5)
        detected_l = cv2.aruco.detectMarkers(detect_left, dictionary, parameters=params)
        detected_r = cv2.aruco.detectMarkers(detect_right, dictionary, parameters=params)
        corners_l, ids_l = detected_l[0], detected_l[1]
        corners_r, ids_r = detected_r[0], detected_r[1]
        board = cv2.aruco.CharucoBoard_create(cols, rows, 1.0, marker_ratio, dictionary)
        valid_l, char_l, ids_char_l = (cv2.aruco.interpolateCornersCharuco(corners_l, ids_l, detect_left, board)
                                      if ids_l is not None else (0, None, None))
        valid_r, char_r, ids_char_r = (cv2.aruco.interpolateCornersCharuco(corners_r, ids_r, detect_right, board)
                                      if ids_r is not None else (0, None, None))
        polygon = []
        if corners_l:
            marker_points = np.asarray(corners_l, dtype=np.float32).reshape(-1, 2)
            if marker_points.shape[0] >= 3:
                marker_points = np.ascontiguousarray(marker_points)
                hull = cv2.convexHull(marker_points).reshape(-1, 2) / 2
                polygon = [[round(float(x / left.shape[1]), 5), round(float(y / left.shape[0]), 5)]
                           for x, y in hull]
        charuco_points = []
        if char_l is not None:
            charuco_points = [[round(float(x / detect_left.shape[1]), 5),
                               round(float(y / detect_left.shape[0]), 5)]
                              for x, y in char_l.reshape(-1, 2)]
        preview = dict(active=True, left_markers=len(ids_l) if ids_l is not None else 0,
                       right_markers=len(ids_r) if ids_r is not None else 0,
                       left_corners=int(valid_l or 0), right_corners=int(valid_r or 0),
                       polygon=polygon, corners=charuco_points, updated_at=time.monotonic())
        with self.lock:
            self.preview = preview
            self.last_corners = min(int(valid_l or 0), int(valid_r or 0))
        if ids_l is None or ids_r is None or len(ids_l) < 4 or len(ids_r) < 4:
            with self.lock: self.last_reason = 'board not detected in both sensors'
            return
        if valid_l < self.MIN_CORNERS or valid_r < self.MIN_CORNERS:
            with self.lock:
                self.last_reason = f'need {self.MIN_CORNERS}+ ChArUco corners in each sensor ({valid_l}/{valid_r})'
                self.last_corners = min(valid_l, valid_r)
            return
        with self.lock: self.last_corners = min(valid_l, valid_r)
        def live_quality(image, points):
            xy = points.reshape(-1, 2).astype(np.float32) / 2.0
            h, w = image.shape[:2]
            x0, y0 = xy.min(axis=0); x1, y1 = xy.max(axis=0)
            area = (x1 - x0) * (y1 - y0) / (w * h)
            if area < self.MIN_BOARD_AREA: return 'board too small'
            if area > self.MAX_BOARD_AREA or (x1-x0)/w > self.MAX_BOARD_DIMENSION or (y1-y0)/h > self.MAX_BOARD_DIMENSION:
                return 'board too close/large'
            if x0/w < self.MIN_EDGE_MARGIN or y0/h < self.MIN_EDGE_MARGIN or (w-x1)/w < self.MIN_EDGE_MARGIN or (h-y1)/h < self.MIN_EDGE_MARGIN:
                return 'board clipped/too near edge'
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            roi = gray[max(0,int(y0)):min(h,int(np.ceil(y1))+1), max(0,int(x0)):min(w,int(np.ceil(x1))+1)]
            if roi.size == 0 or cv2.Laplacian(roi, cv2.CV_64F).var() < self.MIN_SHARPNESS:
                return 'board image too blurry'
            return None
        reasons = [live_quality(left, char_l), live_quality(right, char_r)]
        if any(reasons):
            with self.lock: self.last_reason = next(reason for reason in reasons if reason)
            return
        common = np.intersect1d(ids_char_l.ravel(), ids_char_r.ravel())
        if len(common) < self.MIN_COMMON_CORNERS:
            with self.lock: self.last_reason = f'only {len(common)} matching stereo corners; need {self.MIN_COMMON_CORNERS}+'
            return
        center = tuple(np.mean(char_l.reshape(-1, 2), axis=0) / [detect_left.shape[1], detect_left.shape[0]])
        xy = char_l.reshape(-1, 2).astype(np.float32) / 2.0
        x0, y0 = xy.min(axis=0); x1, y1 = xy.max(axis=0)
        angle = float(cv2.minAreaRect(xy)[-1])
        signature = (float(center[0]), float(center[1]),
                     float((x1-x0) / left.shape[1]), float((y1-y0) / left.shape[0]), angle)
        with self.lock:
            if any(abs(signature[0]-old[0]) < .045 and abs(signature[1]-old[1]) < .045
                   and abs(signature[2]-old[2]) < .10 and abs(signature[3]-old[3]) < .10
                   and min(abs(signature[4]-old[4]), 180-abs(signature[4]-old[4])) < 8
                   for old in self.signatures):
                self.last_reason = 'view too similar to an existing capture'
                return
            index = self.next_index
            cv2.imwrite(str(self.output / f'left_{index:03d}.png'), left)
            cv2.imwrite(str(self.output / f'right_{index:03d}.png'), right)
            self.samples = index
            self.next_index = index + 1
            self.last_capture = time.monotonic()
            self.last_signature = signature
            self.signatures.append(signature)
            self.last_reason = 'accepted good stereo pose'
