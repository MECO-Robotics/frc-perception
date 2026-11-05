#!/usr/bin/env python3
"""Replay fixed and health-triggered detector anchors with cached detections.

This is an exploratory CPU simulation: detector calls are modeled by selecting
cached per-frame boxes, so it measures cadence/position behavior, not inference
latency or GPU load.
"""
import json
import math
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DET = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())
DEPTH = json.loads((ROOT / 'media/frc-fuel-dense-stereo-evaluation.json').read_text())
VIDEOS = {
    'earlier': ROOT / 'media/frc-2026-live-two-fuel-60fps-cfr.mp4',
    'corrected-settings': ROOT / 'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4',
}
TARGETS = ((808.0, 589.0), (902.0, 591.0))
HALF = 0.5
cv2.setNumThreads(1)


def med(a):
    return float(statistics.median(a)) if a else None


def pct(a, p):
    return float(np.percentile(a, p)) if a else None


def get_boxes(row):
    out = [d for d in row['detections'] if d['confidence'] >= .05]
    assigned = [None, None]
    for d in sorted(out, key=lambda x: -x['confidence']):
        cx, cy = (d['x1'] + d['x2']) / 2, (d['y1'] + d['y2']) / 2
        k = min(range(2), key=lambda i: math.dist((cx, cy), TARGETS[i]))
        if math.dist((cx, cy), TARGETS[k]) < 60 and assigned[k] is None:
            assigned[k] = d
    return assigned


def seed(gray, center):
    mask = np.zeros(gray.shape, np.uint8)
    cv2.circle(mask, (round(center[0]), round(center[1])), 10, 255, -1)
    p = cv2.goodFeaturesToTrack(gray, maxCorners=24, qualityLevel=.003,
                                minDistance=2, mask=mask, blockSize=5)
    return p


def fb_track(prev, curr, points):
    if points is None or len(points) < 3:
        return None, None, 0.0, None
    nxt, st, err = cv2.calcOpticalFlowPyrLK(
        prev, curr, points, None, winSize=(15, 15), maxLevel=1,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, .03),
        minEigThreshold=1e-4)
    if nxt is None:
        return None, None, 0.0, None
    back, bst, _ = cv2.calcOpticalFlowPyrLK(
        curr, prev, nxt, None, winSize=(15, 15), maxLevel=1,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, .03),
        minEigThreshold=1e-4)
    if back is None:
        return None, None, 0.0, None
    old = points.reshape(-1, 2)
    new = nxt.reshape(-1, 2)
    fwdback = np.linalg.norm(back.reshape(-1, 2) - old, axis=1)
    ok = (st.reshape(-1).astype(bool) & bst.reshape(-1).astype(bool) &
          (err.reshape(-1) < 35) & (fwdback < 1.5))
    survival = float(ok.mean())
    if ok.sum() < 3:
        return None, None, survival, float(np.median(fwdback))
    return new[ok], np.median(new[ok] - old[ok], axis=0), survival, float(np.median(fwdback[ok]))


def scenario(scene, policy):
    dc = next(x for x in DET['results'] if x['scene'] == scene)['per_frame']
    dr = next(x for x in DEPTH['results'] if x['scene'] == scene)['frames']
    cap = cv2.VideoCapture(str(VIDEOS[scene]))
    states = [None, None]
    prev = None
    errs, range_errs, disparity_errs = [], [], []
    coverage = [0, 0, 0]
    detections = 0
    forced = 0
    trigger_reasons = {'invalid_or_survival': 0, 'forward_backward': 0,
                       'stereo_disparity_consistency': 0}
    trigger_survival, trigger_fb = [], []
    process_s = 0.0
    frame = 0
    while True:
        ok, side = cap.read()
        if not ok:
            break
        t0 = time.perf_counter()
        left = cv2.resize(cv2.cvtColor(side[:, :1600], cv2.COLOR_BGR2GRAY),
                          (800, 600), interpolation=cv2.INTER_AREA)
        right = cv2.resize(cv2.cvtColor(side[:, 1600:], cv2.COLOR_BGR2GRAY),
                           (800, 600), interpolation=cv2.INTER_AREA)
        should_anchor = frame % policy['stride'] == 0
        # Advance tracks and assess health before deciding on a forced anchor.
        health = []
        for k, state in enumerate(states):
            if state is None or prev is None:
                health.append((0.0, 99.0, None, None))
                continue
            lp, ld, ls, lf = fb_track(prev[0], left, state['lp'])
            rp, rd, rs, rf = fb_track(prev[1], right, state['rp'])
            survival = min(ls, rs)
            fb = max(lf if lf is not None else 99, rf if rf is not None else 99)
            health.append((survival, fb, lp, rp))
            if lp is None or rp is None:
                state['valid'] = False
            else:
                state['center_l'] += ld
                state['center_r'] += rd
                state['lp'], state['rp'] = lp.reshape(-1, 1, 2), rp.reshape(-1, 1, 2)
                state['survival'], state['fb'] = survival, fb
                # Stereo quality gate from tracked centers: epipolar row agreement,
                # positive disparity, and disparity stability against anchor.
                disp = float(state['center_l'][0] - state['center_r'][0])
                rowerr = abs(float(state['center_l'][1] - state['center_r'][1]))
                dq = disp > 0 and rowerr <= 2.0 and abs(disp - state['anchor_disp']) <= 2.0
                state['disp_quality'] = dq
                state['valid'] = True
        if policy['adaptive'] and not should_anchor:
            for state in states:
                if state is not None and (not state.get('valid', False) or
                    state.get('survival', 0) < .5 or state.get('fb', 99) > 1.5 or
                    not state.get('disp_quality', False)):
                    should_anchor = True
                    forced += 1
                    if (not state.get('valid', False) or
                        state.get('survival', 0) < .5):
                        trigger_reasons['invalid_or_survival'] += 1
                    elif state.get('fb', 99) > 1.5:
                        trigger_reasons['forward_backward'] += 1
                    else:
                        trigger_reasons['stereo_disparity_consistency'] += 1
                    trigger_survival.append(state.get('survival', 0))
                    trigger_fb.append(state.get('fb', 99))
                    break
        assigned = get_boxes(dc[frame]) if should_anchor else None
        if should_anchor:
            detections += 1
            for k, b in enumerate(assigned):
                if b is None:
                    states[k] = None
                    continue
                cx, cy = (b['x1'] + b['x2']) / 4, (b['y1'] + b['y2']) / 4
                dp = dr[frame]['disparities_px'][k]
                if dp is None:
                    states[k] = None
                    continue
                dsmall = dp * HALF
                cr = (cx - dsmall, cy)
                states[k] = {'center_l': np.array([cx, cy], np.float32),
                             'center_r': np.array(cr, np.float32),
                             'lp': seed(left, (cx, cy)), 'rp': seed(right, cr),
                             'anchor_disp': dsmall, 'valid': True,
                             'survival': 1.0, 'fb': 0.0, 'disp_quality': True}
        count = 0
        for k, state in enumerate(states):
            if state is None or not state.get('valid', False):
                continue
            count += 1
            bx = get_boxes(dc[frame])[k]
            if bx:
                target = ((bx['x1'] + bx['x2']) / 4, (bx['y1'] + bx['y2']) / 4)
                errs.append(math.dist(state['center_l'], target) * 2)
            disp = float(state['center_l'][0] - state['center_r'][0]) / HALF
            ref = dr[frame]['disparities_px'][k]
            if ref is not None:
                disparity_errs.append(abs(disp - ref))
                if disp > 0:
                    z = .0635 * 1094.95 / disp
                    rz = dr[frame]['ranges_m'][k]
                    if rz is not None:
                        range_errs.append(abs(z - rz))
        coverage[count] += 1
        prev = (left, right)
        frame += 1
        process_s += time.perf_counter() - t0
    cap.release()
    return {'scene': scene, 'policy': policy['name'], 'frames': frame,
            'detector_calls_simulated': detections,
            'simulated_detector_rate_hz': detections / (frame / 60),
            'forced_calls': forced,
            'forced_call_trigger_reasons': trigger_reasons,
            'range_output_coverage_0_1_2': coverage,
            'center_error_px_median_p95_vs_cached_boxes': [med(errs), pct(errs, 95)],
            'disparity_abs_error_px_median_p95_vs_cached_dense_sgbm': [med(disparity_errs), pct(disparity_errs, 95)],
            'range_abs_error_m_median_p95_vs_cached_dense_sgbm': [med(range_errs), pct(range_errs, 95)],
            'trigger_track_survival_median': med(trigger_survival),
            'trigger_fb_error_px_median': med(trigger_fb),
            'cpu_replay_processing_seconds_excluding_decode': round(process_s, 3),
            'cpu_processing_fps_excluding_decode': round(frame / process_s, 2)}


policies = [
    {'name': 'fixed_stride_3', 'stride': 3, 'adaptive': False},
    {'name': 'fixed_stride_7', 'stride': 7, 'adaptive': False},
    {'name': 'adaptive_max_stride_7_health_and_disparity', 'stride': 7, 'adaptive': True},
]
results = [scenario(scene, pol) for scene in VIDEOS for pol in policies]
out = {'method': 'Cached detector outputs stand in for detector calls. At each selected anchor, use cached box and cached dense SGBM disparity, then LK-track independent center patches in half-resolution left/right images. Adaptive trigger if either eye has <50% FB-consistent features, median forward-backward error >1.5px, or tracked stereo disparity has invalid sign, >2px epipolar row error, or >2px change from anchor. All center/range/disparity errors compare to cached same-frame outputs, not ground truth.',
       'limitations': 'Only stationary two-ball clips; cached detector and SGBM references are not independent truth. The trigger uses center-disparity stability and epipolar consistency, not a calibrated matcher confidence. The replay processing time includes image conversion, LK and metrics, excludes decode, concurrent inference, NetworkTables and production threading; it cannot be translated into actual GPU load or inference savings.',
       'results': results}
path = ROOT / 'media/frc-fuel-adaptive-anchor-policy-evaluation.json'
path.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(results, indent=2))
print('saved', path)
