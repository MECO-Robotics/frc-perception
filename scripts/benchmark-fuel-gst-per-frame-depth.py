#!/usr/bin/env python3
"""Measure per-frame stereo range on Jetson hardware-decoded FUEL replay."""
import argparse
from collections import deque
import json
import os
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault('SCOUT_STEREO_ENABLED', '1')
os.environ.setdefault('SCOUT_STEREO_RECTIFY', '1')
os.environ.setdefault('SCOUT_STEREO_LOCAL_RECTIFY', '1')
os.environ.setdefault('SCOUT_STEREO_HORIZONTAL_ROI', '1')
os.environ.setdefault('SCOUT_STEREO_CALIBRATION',
                       'media/calibration/stereo_calibration_live_order_candidate.yml')
os.environ.setdefault('SCOUT_STEREO_BASELINE_M', '0.0635')
os.environ.setdefault('SCOUT_STEREO_BLOCK_SIZE', '3')
os.environ.setdefault('SCOUT_STEREO_MATCH_SCALE', '0.5')
os.environ.setdefault('SCOUT_STEREO_NUM_DISPARITIES', '64')
os.environ.setdefault('SCOUT_STEREO_VERTICAL_ROI_PAD', '24')

sys.path.insert(0, str(ROOT))
from scout.gst_runtime import Gst, GstVideo
import cv2
import numpy as np
cv2.setNumThreads(max(1, int(os.getenv('SCOUT_CV2_THREADS', '1'))))
from scout.detectors import Detection
from scout.stereo import StereoDepth
from scout.tracking import FuelTracker

CLIPS = {
    'earlier': ROOT / 'media/frc-2026-live-two-fuel-60fps-cfr.mp4',
    'corrected-settings': ROOT / 'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4',
}
DATA = json.loads((ROOT / 'media/frc-fuel-dense-count-evaluation.json').read_text())


def summarize(values):
    if not values:
        return None
    return {'median_ms': round(float(statistics.median(values)), 3),
            'p95_ms': round(float(np.percentile(values, 95)), 3),
            'mean_ms': round(float(statistics.mean(values)), 3),
            'max_ms': round(float(max(values)), 3)}


def summarize_pixels(values):
    if not values:
        return None
    return {'median_px': round(float(statistics.median(values)), 3),
            'p95_px': round(float(np.percentile(values, 95)), 3),
            'mean_px': round(float(statistics.mean(values)), 3),
            'max_px': round(float(max(values)), 3)}


def pull(sink):
    sample = sink.emit('try-pull-sample', 5 * Gst.SECOND)
    if sample is None:
        return None
    info = GstVideo.VideoInfo()
    if not info.from_caps(sample.get_caps()):
        raise RuntimeError('Unable to read GStreamer sample caps')
    buffer = sample.get_buffer()
    ok, mapped = buffer.map(Gst.MapFlags.READ)
    if not ok:
        raise RuntimeError('Unable to map GStreamer sample')
    try:
        if info.finfo.name == 'BGR':
            frame = np.ndarray((info.height, info.width, 3), dtype=np.uint8,
                               buffer=mapped.data,
                               strides=(info.stride[0], 3, 1)).copy()
        elif info.finfo.name == 'GRAY8':
            frame = np.ndarray((info.height, info.width), dtype=np.uint8,
                               buffer=mapped.data,
                               strides=(info.stride[0], 1)).copy()
        else:
            raise RuntimeError(f'Unexpected GStreamer format {info.finfo.name}')
        return buffer.pts, frame
    finally:
        buffer.unmap(mapped)


def run(scene, max_frames, synthetic_boxes=0, overlap_stereo=False):
    video = CLIPS[scene]
    rows = next(row['per_frame'] for row in DATA['results']
                if row['scene'] == scene)
    description = (
        f'filesrc location="{video}" ! qtdemux name=demux '
        'demux.video_0 ! queue ! h264parse ! nvv4l2decoder ! '
        # Do not pace frames to their media timestamps: replay wait time would
        # cap the measured rate at the recording FPS rather than the pipeline's
        # processing capacity.
        'identity sync=false single-segment=true ! nvvidconv ! '
        'video/x-raw,format=BGRx,width=1280,height=480 ! tee name=views '
        'views. ! queue max-size-buffers=4 max-size-bytes=0 max-size-time=0 ! '
        'videocrop right=640 ! videoconvert ! video/x-raw,format=GRAY8 ! '
        'appsink name=left sync=false max-buffers=4 drop=false enable-last-sample=false '
        'views. ! queue max-size-buffers=4 max-size-bytes=0 max-size-time=0 ! '
        'videocrop left=640 ! videoconvert ! video/x-raw,format=GRAY8 ! '
        'appsink name=right sync=false max-buffers=4 drop=false enable-last-sample=false')
    pipeline = Gst.parse_launch(description)
    left_sink, right_sink = pipeline.get_by_name('left'), pipeline.get_by_name('right')
    stereo, tracker = StereoDepth(), FuelTracker()
    depth_executor = ThreadPoolExecutor(max_workers=1) if overlap_stereo else None
    last_depth_boxes = []
    latencies, stereo_profiles, valid_counts = [], [], []
    left_pull_ms, right_pull_ms = [], []
    tracker_ms, tracker_advance_ms, tracker_anchor_ms = [], [], []
    anchor_position_errors = []
    depths_by_track = {}
    processed = paired = all_fresh = 0
    left_condition, right_condition = threading.Condition(), threading.Condition()
    left_state = {'items': deque(), 'done': False, 'sequence': 0, 'replaced': 0}
    right_state = {'frames': {}, 'done': False, 'replaced': 0}
    reader_errors = []

    def reader(sink, side):
        pull_times = left_pull_ms if side == 'left' else right_pull_ms
        while True:
            stage = time.perf_counter()
            item = pull(sink)
            pull_times.append((time.perf_counter() - stage) * 1000)
            if item is None:
                with (left_condition if side == 'left' else right_condition):
                    (left_state if side == 'left' else right_state)['done'] = True
                    (left_condition if side == 'left' else right_condition).notify_all()
                return
            pts, frame = item
            if side == 'left':
                with left_condition:
                    if max_frames is not None and left_state['sequence'] >= max_frames:
                        left_state['done'] = True
                        left_condition.notify_all()
                        return
                    left_condition.wait_for(lambda: len(left_state['items']) < 4 or
                                            left_state['done'])
                    if left_state['done']:
                        return
                    left_state['items'].append((pts, left_state['sequence'], frame))
                    left_state['sequence'] += 1
                    left_condition.notify_all()
            else:
                with right_condition:
                    frames = right_state['frames']
                    right_condition.wait_for(lambda: len(frames) < 8 or
                                             right_state['done'])
                    if right_state['done']:
                        return
                    frames[pts] = frame
                    right_condition.notify_all()

    def start_reader(sink, side):
        thread = threading.Thread(target=reader, args=(sink, side),
                                  name=f'{side}-capture', daemon=True)
        thread.start()
        return thread

    pipeline.set_state(Gst.State.PLAYING)
    started = time.perf_counter()
    left_thread = start_reader(left_sink, 'left')
    right_thread = start_reader(right_sink, 'right')
    try:
        while True:
            with left_condition:
                left_condition.wait_for(lambda: left_state['items'] or
                                        left_state['done'], timeout=5)
                item = left_state['items'].popleft() if left_state['items'] else None
                left_condition.notify_all()
                if item is None and left_state['done']:
                    break
            left_pts, frame_id, left = item
            with right_condition:
                right_condition.wait_for(lambda: left_pts in right_state['frames'] or
                                         right_state['done'], timeout=5)
                right = right_state['frames'].pop(left_pts, None)
                for old_pts in [key for key in right_state['frames'] if key < left_pts]:
                    del right_state['frames'][old_pts]
                right_condition.notify_all()
            if right is None:
                paired_this_frame = False
            else:
                paired_this_frame = True
            frame_started = time.perf_counter()
            pts = float(left_pts) / float(Gst.SECOND)
            now = time.monotonic()
            depth_future = None
            depth_boxes = []
            if overlap_stereo and right is not None and last_depth_boxes:
                depth_boxes = last_depth_boxes
                depth_future = depth_executor.submit(
                    stereo.estimate, left, right, depth_boxes)
            stage = time.perf_counter()
            tracker_stage = stage
            tracks = tracker.advance(left, now, pts)
            tracker_advance_ms.append((time.perf_counter() - tracker_stage) * 1000)
            if frame_id % 3 == 0 and frame_id < len(rows):
                tracker_stage = time.perf_counter()
                if synthetic_boxes:
                    detections = []
                    for index in range(synthetic_boxes):
                        cx, cy = 32 + (index % 10) * 64, 24 + (index // 10) * 48
                        detections.append(Detection(cx - 16, cy - 16, cx + 16,
                                                    cy + 16, 0, 0.9))
                else:
                    detections = [Detection(round(int(r['x1']) * .4),
                                            round(int(r['y1']) * .4),
                                            round(int(r['x2']) * .4),
                                            round(int(r['y2']) * .4),
                                            0, float(r['confidence']))
                                  for r in rows[frame_id]['detections']
                                  if float(r['confidence']) >= 0.05]
                    candidates = []
                    for track_index, track in enumerate(tracks):
                        cx, cy = track['box'][:2]
                        for detection_index, detection in enumerate(detections):
                            dx = (detection.x1 + detection.x2) * 0.5
                            dy = (detection.y1 + detection.y2) * 0.5
                            candidates.append((float(np.hypot(cx - dx, cy - dy)),
                                               track_index, detection_index))
                    used_tracks, used_detections = set(), set()
                    for error, track_index, detection_index in sorted(candidates):
                        if (track_index in used_tracks or
                                detection_index in used_detections):
                            continue
                        anchor_position_errors.append(error)
                        used_tracks.add(track_index)
                        used_detections.add(detection_index)
                tracker.update(detections, now, pts, return_snapshot=False)
                tracks = tracker.snapshot(now)
                tracker_anchor_ms.append((time.perf_counter() - tracker_stage) * 1000)
            else:
                tracker_anchor_ms.append(0.0)
            tracker_ms.append((time.perf_counter() - stage) * 1000)
            boxes = []
            height, width = left.shape[:2]
            for track in tracks:
                x, y, w, h = track['box']
                boxes.append(SimpleNamespace(
                    x1=max(0, int(round(x - w * .5))),
                    y1=max(0, int(round(y - h * .5))),
                    x2=min(width, int(round(x + w * .5))),
                    y2=min(height, int(round(y + h * .5))),
                    class_id=0, confidence=float(track['confidence']),
                    distance_m=track['distance_m'],
                    disparity_px=getattr(track['detection'], 'disparity_px', None),
                    track_id=track['track_id']))
            if depth_future is not None:
                depth_future.result()
                stereo_profiles.append(dict(stereo.last_profile))
            elif boxes and right is not None:
                depth_boxes = boxes
                stereo.estimate(left, right, depth_boxes)
                stereo_profiles.append(dict(stereo.last_profile))
            if depth_boxes and right is not None:
                paired += 1
                measurements = [(box.track_id, box.distance_m, box.disparity_px)
                                for box in depth_boxes if box.distance_m is not None]
                tracker.update_ranges(measurements, now=time.monotonic(),
                                      return_snapshot=False)
                valid = [box for box in depth_boxes if box.distance_m is not None and
                         box.disparity_px is not None]
                valid_counts.append(len(valid))
                if len(valid) == len(depth_boxes):
                    all_fresh += 1
                for box in valid:
                    depths_by_track.setdefault(str(box.track_id), []).append(
                        float(box.distance_m))
            else:
                valid_counts.append(0)
            if overlap_stereo:
                last_depth_boxes = boxes
            latencies.append((time.perf_counter() - frame_started) * 1000)
            processed += 1
    finally:
        pipeline.set_state(Gst.State.NULL)
        with left_condition:
            left_state['done'] = True
            left_condition.notify_all()
        with right_condition:
            right_state['done'] = True
            right_condition.notify_all()
        left_thread.join(timeout=2)
        right_thread.join(timeout=2)
        if depth_executor is not None:
            depth_executor.shutdown(wait=True)
    elapsed = time.perf_counter() - started
    # Per-frame work intervals isolate tracker/stereo cost. Unpaced wall time
    # additionally includes hardware decode, sample transfer, and scheduling.
    active_processing_s = sum(latencies) / 1000.0
    return {
        'scene': scene, 'video': str(video.relative_to(ROOT)),
        'synthetic_box_count': synthetic_boxes or None,
        'stereo_overlapped_with_tracker': overlap_stereo,
        'frames': processed, 'elapsed_s': round(elapsed, 3),
        'output_hz': round(processed / elapsed, 2) if elapsed else None,
        'active_processing_s': round(active_processing_s, 3),
        'active_processing_hz': round(processed / active_processing_s, 2)
            if active_processing_s else None,
        'stereo_pair_fraction': round(paired / processed, 5) if processed else None,
        'fresh_depth_for_every_track_fraction': round(all_fresh / processed, 5)
            if processed else None,
        'valid_depths_per_frame_histogram': {
            str(k): valid_counts.count(k) for k in sorted(set(valid_counts))},
        'frame_processing_ms': summarize(latencies),
        'left_sample_pull_ms': summarize(left_pull_ms),
        'right_sample_pull_ms': summarize(right_pull_ms),
        'tracker_ms': summarize(tracker_ms),
        'tracker_advance_ms': summarize(tracker_advance_ms),
        'tracker_anchor_update_ms': summarize(tracker_anchor_ms),
        'detector_anchor_position_error_px': summarize_pixels(anchor_position_errors),
        'captured_frames': left_state['sequence'],
        'flow_frames_replaced': left_state['replaced'],
        'right_frames_dropped_from_pair_cache': right_state['replaced'],
        'stereo_stage_ms': {
            key: summarize([p[key] for p in stereo_profiles if key in p])
            for key in sorted({key for p in stereo_profiles for key in p})},
        'range_by_track_m': {
            track_id: {'n': len(values),
                       'median': round(float(statistics.median(values)), 4),
                       'mad': round(float(statistics.median(
                           [abs(value - statistics.median(values)) for value in values])), 4)}
            for track_id, values in sorted(depths_by_track.items())},
        'stereo_configuration': {
            'backend': stereo.backend, 'local_rectify': stereo.local_rectify_enabled,
            'match_scale': stereo.match_scale,
            'num_disparities': stereo.num_disparities,
            'last_min_disparity': stereo.last_min_disparity,
            'last_num_disparities': stereo.last_num_disparities,
            'block_size': stereo.block_size},
        'limitations': [
            'Replay is unpaced; output rate excludes media timestamp waits and includes hardware decode and sample delivery.',
            'Active processing rate sums per-frame tracker/stereo work intervals; concurrent reader/decode work and inter-frame scheduling are not included.',
            'Jetson hardware-decoded replay, not a physical USB camera run.',
            'No concurrent TensorRT inference or NetworkTables publisher in this benchmark.',
            'Range consistency is not absolute accuracy because these captures lack tape-measured truth.']}


parser = argparse.ArgumentParser()
parser.add_argument('--scene', choices=tuple(CLIPS), default='corrected-settings')
parser.add_argument('--max-frames', type=int)
parser.add_argument('--synthetic-boxes', type=int, default=0,
                    help='Stress object scaling with this many evenly spaced boxes; 1..100.')
parser.add_argument('--overlap-stereo', action='store_true',
                    help='Overlap disparity with tracker updates, as in the live pipeline.')
parser.add_argument('--output', type=Path,
                    default=ROOT / 'media/frc-fuel-gst-per-frame-depth.json')
args = parser.parse_args()
if args.synthetic_boxes < 0 or args.synthetic_boxes > 100:
    parser.error('--synthetic-boxes must be 0 (use saved detections) or 1..100')
result = run(args.scene, args.max_frames, args.synthetic_boxes, args.overlap_stereo)
method = ('Unpaced Jetson GStreamer hardware decode; '
          'update LK/Kalman every frame and compute fresh dense stereo depth '
          f'for each current track; {args.synthetic_boxes} synthetic anchors every '
          'third frame.' if args.synthetic_boxes else
          'Unpaced Jetson GStreamer hardware decode; update '
          'LK/Kalman every frame, replay detector anchors every third frame, '
          'compute fresh stereo depth for each current track every frame.')
if args.overlap_stereo:
    method += ' Stereo disparity is overlapped with current-frame tracker updates.'
payload = {'method': method,
           'result': result}
args.output.write_text(json.dumps(payload, indent=2) + '\n')
print(args.output.read_text())
