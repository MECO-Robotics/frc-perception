#!/usr/bin/env python3
"""Compare causal 3D position filters on unique live replay outputs."""
import json
import statistics
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / 'media/frc-fuel-api-unique-block3-position.json'
source = json.loads(INPUT.read_text())
stats = source['stats_before']
width, height = map(int, stats['inference_resolution'].split('x'))
calibration = ROOT / 'media/calibration/stereo_calibration_live_order_candidate.yml'
fs = cv2.FileStorage(str(calibration), cv2.FILE_STORAGE_READ)
kl = fs.getNode('left_camera_matrix').mat()
dl = fs.getNode('left_distortion').mat()
kr = fs.getNode('right_camera_matrix').mat()
dr = fs.getNode('right_distortion').mat()
rotation = fs.getNode('rotation').mat()
translation = fs.getNode('translation').mat()
fs.release()
image_size = (width, height)
_, _, projection_left, _, _, _, _ = cv2.stereoRectify(
    kl, dl, kr, dr, image_size, rotation, translation,
    flags=cv2.CALIB_ZERO_DISPARITY, alpha=0)
baseline = float(stats['stereo_baseline_m'])
focal = float(stats['stereo_focal_px'])
disparity_scale = float(stats.get('stereo_disparity_scale', 1.0))


def median(values):
    return statistics.median(values) if values else None


def axis_summary(points):
    values = np.asarray(points, dtype=float)
    result = {}
    for index, axis in enumerate(('x', 'y', 'z')):
        column = values[:, index].tolist()
        center = median(column)
        deviations = sorted(abs(value - center) for value in column)
        result[axis] = {
            'median_m': round(center, 6),
            'mad_m': round(median([abs(value - center) for value in column]), 6),
            'p95_abs_deviation_m': round(deviations[int(.95 * (len(deviations) - 1))], 6),
        }
    return result


def distance_summary(values):
    center = median(values)
    deviations = sorted(abs(value - center) for value in values)
    return {'n': len(values), 'median_m': round(center, 6),
            'mad_m': round(median(deviations), 6),
            'p95_abs_deviation_m': round(deviations[int(.95 * (len(deviations) - 1))], 6)}


observations = [[], []]
for row in source['unique_frames']:
    if row['count'] != 2 or len(row.get('centers_normalized', [])) != 2:
        observations[0].append(None)
        observations[1].append(None)
        continue
    for target in range(2):
        u_norm, v_norm = row['centers_normalized'][target]
        disparity = row['disparities_px'][target]
        if not disparity:
            observations[target].append(None)
            continue
        z = baseline * focal / (disparity * disparity_scale)
        u, v = u_norm * width, v_norm * height
        point = np.asarray(((u - projection_left[0, 2]) * z / projection_left[0, 0],
                            (v - projection_left[1, 2]) * z / projection_left[1, 1], z))
        observations[target].append(point)


def smooth(method, sequence):
    output = []
    history = []
    ema = None
    for point in sequence:
        if point is None:
            output.append(None)
            continue
        if method.startswith('median'):
            size = int(method[len('median'):])
            history.append(point)
            history = history[-size:]
            output.append(np.median(np.asarray(history), axis=0))
        elif method.startswith('ema'):
            alpha = float(method[len('ema'):])
            ema = point.copy() if ema is None else alpha * point + (1 - alpha) * ema
            output.append(ema.copy())
        else:
            output.append(point.copy())
    return output


methods = ('raw', 'median3', 'median5', 'ema0.25', 'ema0.5')
results = {}
for method in methods:
    smoothed = [smooth(method, sequence) for sequence in observations]
    aligned = [(smoothed[0][i], smoothed[1][i])
               for i in range(len(source['unique_frames']))
               if smoothed[0][i] is not None and smoothed[1][i] is not None]
    target_points = [[smoothed[t][i] for i in range(len(smoothed[t]))
                      if smoothed[t][i] is not None] for t in range(2)]
    centroids = [(pair[0] + pair[1]) / 2 for pair in aligned]
    separations = [float(np.linalg.norm(pair[0] - pair[1])) for pair in aligned]
    results[method] = {
        'valid_two_target_outputs': len(aligned),
        'target_camera_xyz': [axis_summary(points) for points in target_points],
        'cluster_centroid_xyz': axis_summary(centroids),
        'pair_separation_m': distance_summary(separations),
    }

output = {
    'method': ('Use one sample per RTP timestamp from the current live MP4 replay. '
               'Reconstruct camera-relative XYZ from box centers and disparity rounded '
               'to 0.01 px, then compare causal coordinate-wise median/EMA filters. '
               'Missing current outputs remain missing; no position hold is applied.'),
    'input': str(INPUT.relative_to(ROOT)),
    'sampled_unique_frames': len(source['unique_frames']),
    'valid_two_box_frames': sum(row['count'] == 2 for row in source['unique_frames']),
    'approx_sample_rate_hz': round(len(source['unique_frames']) / source['elapsed_s'], 3),
    'latency_and_state': {key: stats.get(key) for key in (
        'capture_fps', 'inference_fps', 'inference_latency_ms',
        'stereo_block_size', 'stereo_num_disparities', 'stereo_horizontal_roi')},
    'filter_results': results,
    'limitations': ('A stationary two-ball scene measures jitter, not absolute position '
                    'accuracy or moving-target response. API disparity is rounded, and '
                    'no tape-measured range reference is available.'),
}
path = ROOT / 'media/frc-fuel-api-position-smoother-analysis.json'
path.write_text(json.dumps(output, indent=2) + '\n')
print(json.dumps({'sampled_unique_frames': output['sampled_unique_frames'],
                  'valid_two_box_frames': output['valid_two_box_frames'],
                  'approx_sample_rate_hz': output['approx_sample_rate_hz'],
                  'filter_results': results,
                  'saved': str(path)}, indent=2))
