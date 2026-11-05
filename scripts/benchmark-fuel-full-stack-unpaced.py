#!/usr/bin/env python3
"""Benchmark the production stereo perception path without replay pacing."""
import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLIPS = (
    ROOT / 'media/frc-2026-fuel-camera-settings-fixed-60fps.mp4',
    ROOT / 'media/frc-2026-live-two-fuel-60fps-cfr.mp4',
)


def set_profile_defaults(stereo_mode, disparities, engine, inference_width, flow_width,
                         pair_wait_ms, right_cache_frames, vertical_pad,
                         size_gate_error, inference_hz, stereo_backend,
                         stream_enabled, nt_stale_ms, flow_grace_ms):
    values = {
        'SCOUT_REPLAY_SYNC': '0',
        'SCOUT_STREAM_ENABLED': '1' if stream_enabled else '0',
        'SCOUT_INFERENCE_ALWAYS': '1',
        'SCOUT_INFERENCE_MAX_FPS': str(inference_hz),
        'SCOUT_ENGINE': engine,
        'SCOUT_INFERENCE_WIDTH': str(inference_width),
        'SCOUT_FLOW_WIDTH': str(flow_width),
        'SCOUT_CONFIDENCE': '0.05',
        'SCOUT_NMS_IOU': '0.15',
        'SCOUT_MAX_BOX_ASPECT_RATIO': '2.0',
        'SCOUT_STEREO_ENABLED': '1',
        'SCOUT_STEREO_BACKEND': stereo_backend,
        'SCOUT_STEREO_CALIBRATION': 'media/calibration/stereo_calibration_live_order_candidate.yml',
        'SCOUT_STEREO_DISPARITY_SCALE': '1.0',
        'SCOUT_STEREO_RECTIFY': '1',
        'SCOUT_STEREO_DENSE': '1' if stereo_mode == 'dense' else '0',
        'SCOUT_STEREO_HORIZONTAL_ROI': '1',
        'SCOUT_STEREO_LOCAL_RECTIFY': '1',
        'SCOUT_STEREO_MATCH_SCALE': '0.25',
        'SCOUT_STEREO_NUM_DISPARITIES': str(disparities),
        'SCOUT_STEREO_BLOCK_SIZE': '3',
        'SCOUT_STEREO_PAIR_WAIT_MS': str(pair_wait_ms),
        'SCOUT_FLOW_RIGHT_CACHE_FRAMES': str(right_cache_frames),
        'SCOUT_STEREO_VERTICAL_ROI_PAD': str(vertical_pad),
        'SCOUT_STEREO_SIZE_GATE_MAX_REL_ERROR': str(size_gate_error),
        'SCOUT_STEREO_SIZE_GATE_DUPLICATE_IOU': '0.90',
        'SCOUT_FUEL_DIAMETER_M': '0.15',
        'SCOUT_TRACK_DIRECT_FLOW': '1',
        'SCOUT_TRACK_FLOW_GRACE_MS': str(flow_grace_ms),
        # Publish to an in-process NT4 server so serialization and NT writes
        # stay in the timed path without sending benchmark data to the robot.
        'FRC_NT_ENABLED': '1',
        'FRC_NT_MODE': 'server',
        'FRC_NT_TABLE': '/FRC/Perception/v1',
        'FRC_CLUSTER_RADIUS_M': '0.25',
        'FRC_NT_STALE_MS': str(nt_stale_ms),
    }
    for key, value in values.items():
        os.environ[key] = value


def percentile(values, pct):
    if not values:
        return None
    ordered = sorted(values)
    return round(ordered[min(len(ordered) - 1,
                             int((len(ordered) - 1) * pct / 100))], 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cameras', type=int, choices=(1, 2), default=1,
                        help='number of stereo camera pipelines to run simultaneously')
    parser.add_argument('--source-1', type=str, default=str(CLIPS[0]),
                        help='first side-by-side stereo video or camera source')
    parser.add_argument('--source-2', type=str, default=str(CLIPS[1]),
                        help='second side-by-side stereo video or camera source')
    parser.add_argument('--stereo-mode', choices=('dense', 'roi'), default='dense',
                        help='dense full-frame matching or track-region matching')
    parser.add_argument('--stereo-backend', choices=('opencv', 'vpi_cuda'), default='opencv',
                        help='stereo matcher backend; vpi_cuda requires JetPack VPI 2')
    parser.add_argument('--disparities', type=int, choices=(16, 32, 64), default=32,
                        help='SGBM horizontal search width')
    parser.add_argument('--cv2-threads', type=int, choices=(1, 2, 4, 8), default=8,
                        help='OpenCV worker threads shared by the camera pipelines')
    parser.add_argument('--engine', default='models/rebuilt_fuel_yolo26n_1280x960_fp16.engine')
    parser.add_argument('--inference-width', type=int, choices=(1280, 2560), default=2560,
                        help='total side-by-side source width passed through inference resize')
    parser.add_argument('--flow-width', type=int, choices=(480, 640, 800, 960, 1280), default=1280,
                        help='total side-by-side width used by optical flow and stereo')
    parser.add_argument('--pair-wait-ms', type=float, choices=(2, 5, 10, 20), default=2,
                        help='maximum wait for the matching right-eye frame')
    parser.add_argument('--right-cache-frames', type=int, choices=(4, 8, 16), default=4,
                        help='right-eye frame cache capacity')
    parser.add_argument('--vertical-pad', type=int, choices=(4, 8, 16, 24), default=24,
                        help='stereo ROI vertical padding in matcher pixels')
    parser.add_argument('--size-gate-error', type=float, choices=(0.0, 0.55, 0.75, 1.0), default=0.55,
                        help='maximum relative size-vs-stereo range error; 0 disables the gate')
    parser.add_argument('--inference-hz', type=float, choices=(0, 5, 10, 12, 15, 20), default=0,
                        help='cap detector anchor rate; 0 leaves it unlimited')
    parser.add_argument('--stream-enabled', action='store_true',
                        help='include the VP9 preview encode path')
    parser.add_argument('--nt-stale-ms', type=int, choices=(250, 500), default=250,
                        help='maximum age for carrying a prior range into predicted 3D positions')
    parser.add_argument('--flow-grace-ms', type=int, choices=(120, 250, 500), default=120,
                        help='maximum visual-flow age before removing a predicted track')
    parser.add_argument('--warmup-seconds', type=float, default=3.0)
    parser.add_argument('--measure-seconds', type=float, default=10.0)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'media/frc-fuel-full-stack-unpaced.json')
    parser.add_argument('--nt-output', type=Path,
                        default=ROOT / 'media/frc-fuel-full-stack-nt4.jsonl',
                        help='JSONL file with each published NT pieces/clusters snapshot')
    parser.add_argument('--inference-output', type=Path,
                        default=ROOT / 'media/frc-fuel-full-stack-inference.jsonl',
                        help='JSONL file with raw detector and tracker-anchor counts')
    args = parser.parse_args()
    if args.warmup_seconds < 0 or args.measure_seconds <= 0:
        parser.error('warmup must be nonnegative and measure must be positive')
    def normalize_source(source):
        if source.startswith(('tcp://', 'mjpeg-tcp://', 'rtsp://')):
            return source
        return str(Path(source).expanduser().resolve())

    clips = (normalize_source(args.source_1), normalize_source(args.source_2))
    for clip in clips[:args.cameras]:
        if not clip.startswith(('tcp://', 'mjpeg-tcp://', 'rtsp://')) and not Path(clip).exists():
            parser.error(f'source does not exist: {clip}')

    set_profile_defaults(args.stereo_mode, args.disparities,
                         args.engine, args.inference_width, args.flow_width,
                         args.pair_wait_ms, args.right_cache_frames,
                         args.vertical_pad, args.size_gate_error,
                         args.inference_hz, args.stereo_backend,
                         args.stream_enabled, args.nt_stale_ms,
                         args.flow_grace_ms)
    import cv2
    cv2.setNumThreads(args.cv2_threads)
    sys.path.insert(0, str(ROOT))
    from scout.pipeline import VideoPipeline

    cores = []
    nt_piece_counts = []
    nt_snapshots = []
    inference_snapshots = []
    fresh_range_counts = []
    nt_range_ages = []
    try:
        for index, clip in enumerate(clips[:args.cameras], start=1):
            os.environ['FRC_NT_TABLE'] = f'/FRC/Perception/benchmark/camera{index}'
            core = VideoPipeline(str(clip), mode='tensorrt',
                                 engine=os.environ['SCOUT_ENGINE'],
                                 source_id=f'stereo-unpaced-benchmark-{index}',
                                 source_name=f'Unpaced stereo replay benchmark {index}')
            cores.append(core)
            piece_counts = []
            nt_piece_counts.append(piece_counts)
            snapshots = []
            nt_snapshots.append(snapshots)
            inference_rows = []
            inference_snapshots.append(inference_rows)
            tracker_update = core.fuel_tracker.update

            def record_tracker_update(detections, *update_args,
                                      _update=tracker_update, _rows=inference_rows,
                                      _tracker=core.fuel_tracker,
                                      **update_kwargs):
                result = _update(detections, *update_args, **update_kwargs)
                if not _rows or 'tracker_anchor_count' in _rows[-1]:
                    _rows.append({'timestamp_monotonic': time.monotonic(),
                                  'raw_detection_count': None})
                _rows[-1]['tracker_anchor_count'] = len(detections)
                _rows[-1]['tracker_pts_seconds'] = (
                    float(update_args[1]) if len(update_args) > 1 else None)
                with _tracker.lock:
                    _rows[-1]['tracks_stored_after_update'] = len(_tracker.tracks)
                    _rows[-1]['tracks_within_flow_grace_after_update'] = sum(
                        time.monotonic() - track.last_visual_time <= _tracker.flow_grace_s
                        for track in _tracker.tracks.values())
                return result

            core.fuel_tracker.update = record_tracker_update
            range_ages = []
            nt_range_ages.append(range_ages)
            range_counts = []
            fresh_range_counts.append(range_counts)
            update_ranges = core.fuel_tracker.update_ranges

            def record_range_count(measurements, *range_args,
                                   _update_ranges=update_ranges,
                                   _range_counts=range_counts, **range_kwargs):
                result = _update_ranges(measurements, *range_args, **range_kwargs)
                _range_counts.append({
                    'timestamp': time.monotonic(),
                    'measurements': tuple((int(track_id), float(distance),
                                           None if disparity is None else float(disparity))
                                          for track_id, distance, disparity in measurements),
                })
                return result

            core.fuel_tracker.update_ranges = record_range_count
            if core.nt_publisher is not None:
                publish = core.nt_publisher.publish

                def record_piece_count(*publish_args, _publish=publish,
                                       _piece_counts=piece_counts,
                                       _snapshots=snapshots,
                                       _publisher=core.nt_publisher,
                                       _tracker=core.fuel_tracker,
                                       _core=core,
                                       _range_ages=range_ages,
                                       **publish_kwargs):
                    result = _publish(*publish_args, **publish_kwargs)
                    _piece_counts.append(int(result['pieces']))
                    tracks = publish_args[0]
                    pts_seconds = float(publish_args[1])
                    publish_monotonic = time.monotonic()
                    pts_key_ms = int(round(pts_seconds * 1000.0))
                    with _core.capture_times_lock:
                        capture_monotonic = _core.capture_times.get(pts_key_ms)
                    end_to_end_ms = (
                        (publish_monotonic - capture_monotonic) * 1000.0
                        if capture_monotonic is not None else None)
                    frame_width, frame_height, project_pixel = publish_args[2:5]
                    track_states = {}
                    with_distance = fresh_range = projectable = 0
                    for track in tracks:
                        state = str(track.get('state', 'unknown'))
                        track_states[state] = track_states.get(state, 0) + 1
                        distance = track.get('distance_m')
                        age = track.get('range_age_ms')
                        if distance is None:
                            continue
                        with_distance += 1
                        if age is None or age > _publisher.stale_ms:
                            continue
                        try:
                            xyz = project_pixel(track['box'][0], track['box'][1],
                                                float(distance), frame_width,
                                                frame_height)
                            if (xyz is not None and
                                    all(math.isfinite(float(value)) for value in xyz)):
                                projectable += 1
                                fresh_range += 1
                        except (TypeError, ValueError, OverflowError):
                            pass
                    now = publish_monotonic
                    with _tracker.lock:
                        internal_tracks = list(_tracker.tracks.values())
                        eligible = [track for track in internal_tracks
                                    if now - track.last_detector_time <= _tracker.max_age_s]
                        visual_fresh = [track for track in eligible
                                        if now - track.last_visual_time <= _tracker.flow_grace_s]
                    _snapshots.append({
                        'timestamp_monotonic': publish_monotonic,
                        'pts_seconds': pts_seconds,
                        'capture_to_nt_publish_latency_ms': (
                            round(end_to_end_ms, 2) if end_to_end_ms is not None else None),
                        'frame': int(result['frame']),
                        'status': result['status'],
                        'pieces_count': int(result['pieces']),
                        'clusters_count': int(result['clusters']),
                        'publish_input': {
                            'tracks': len(tracks),
                            'track_states': track_states,
                            'tracks_with_distance': with_distance,
                            'fresh_range_projectable_tracks': fresh_range,
                        },
                        'tracker_internal': {
                            'stored_tracks': len(internal_tracks),
                            'within_detector_max_age': len(eligible),
                            'within_visual_flow_grace': len(visual_fresh),
                            'visual_flow_grace_ms': round(_tracker.flow_grace_s * 1000, 1),
                        },
                        'pieces': json.loads(_publisher.pieces_entry.getString('[]')),
                        'clusters': json.loads(_publisher.clusters_entry.getString('[]')),
                    })
                    if result['pieces']:
                        for track in publish_args[0]:
                            age = track.get('range_age_ms')
                            if (track.get('distance_m') is not None and age is not None and
                                    age <= args.nt_stale_ms):
                                _range_ages.append(float(age))
                    return result

                core.nt_publisher.publish = record_piece_count
            core.start()
            detector_infer = core.detector.infer

            def record_detector_infer(frame, _infer=detector_infer,
                                      _rows=inference_rows):
                result = _infer(frame)
                detections = result[0] if isinstance(result, tuple) else result
                _rows.append({
                    'timestamp_monotonic': time.monotonic(),
                    'raw_detection_count': len(detections),
                    'raw_confidence_max': max(
                        (float(item.confidence) for item in detections), default=None),
                    'raw_confidence_min': min(
                        (float(item.confidence) for item in detections), default=None),
                })
                return result

            core.detector.infer = record_detector_infer
        if args.warmup_seconds:
            time.sleep(args.warmup_seconds)

        before = []
        nt_before = []
        nt_count_before = []
        range_count_before = []
        stereo_frames_before = []
        missing_pairs_before = []
        for core, piece_counts, range_counts in zip(cores, nt_piece_counts,
                                                    fresh_range_counts):
            with core.lock:
                before.append(core.counters.copy())
                nt_before.append(int(core.stats.get('nt_frame', 0)))
                stereo_frames_before.append(int(core.stats.get('stereo_flow_frames', 0)))
                missing_pairs_before.append(int(core.stats.get('stereo_flow_missing_pairs', 0)))
            nt_count_before.append(len(piece_counts))
            range_count_before.append(len(range_counts))
        range_age_count_before = [len(ages) for ages in nt_range_ages]
        start = time.perf_counter()
        end = start + args.measure_seconds
        samples = [[] for _ in cores]
        while time.perf_counter() < end:
            time.sleep(min(0.1, max(0, end - time.perf_counter())))
            for core, camera_samples in zip(cores, samples):
                stats = core.snapshot()
                camera_samples.append({
                    'inference_latency_ms': stats.get('inference_latency_ms', 0),
                    'detector_total_ms': stats.get('detector_total_ms', 0),
                    'stereo_map_ms': stats.get('stereo_map_ms', 0),
                    'stereo_rectify_ms': stats.get('stereo_rectify_ms', 0),
                    'stereo_match_ms': stats.get('stereo_match_ms', 0),
                    'stereo_extract_ms': stats.get('stereo_extract_ms', 0),
                    'stereo_flow_box_build_ms': stats.get('stereo_flow_box_build_ms', 0),
                    'stereo_flow_tracker_ms': stats.get('stereo_flow_tracker_ms', 0),
                    'tracked_instances': stats.get('tracked_instances', 0),
                    'stereo_flow_fresh_tracks': stats.get('stereo_flow_fresh_tracks', 0),
                })
        elapsed = time.perf_counter() - start
        cameras = []
        for index, (core, previous, previous_nt, camera_samples, piece_counts,
                    piece_start, range_counts, range_start, range_ages,
                    range_age_start, stereo_before,
                    missing_before, snapshots) in enumerate(
                zip(cores, before, nt_before, samples, nt_piece_counts,
                    nt_count_before, fresh_range_counts, range_count_before,
                    nt_range_ages, range_age_count_before,
                    stereo_frames_before, missing_pairs_before,
                    nt_snapshots), start=1):
            with core.lock:
                after = core.counters.copy()
                nt_after = int(core.stats.get('nt_frame', 0))
                final_stats = dict(core.stats)
            counts = {key: after[key] - previous[key] for key in previous}
            measured_piece_counts = piece_counts[piece_start:]
            measured_nt_snapshots = snapshots[piece_start:]
            cluster_counts = [snapshot['clusters_count']
                              for snapshot in measured_nt_snapshots]
            end_to_end_latencies = [
                snapshot['capture_to_nt_publish_latency_ms']
                for snapshot in measured_nt_snapshots
                if snapshot['capture_to_nt_publish_latency_ms'] is not None]
            measured_range_ages = range_ages[range_age_start:]
            measured_range_counts = range_counts[range_start:]
            valid_range_counts = [len(frame['measurements'])
                                  for frame in measured_range_counts]
            complete_depth_times = [frame['timestamp'] for frame in measured_range_counts
                                    if len(frame['measurements']) >= 2]
            complete_depth_gaps_ms = [
                (later - earlier) * 1000.0
                for earlier, later in zip(complete_depth_times,
                                          complete_depth_times[1:])]
            depth_by_track = {}
            for frame in measured_range_counts:
                for track_id, distance, _ in frame['measurements']:
                    depth_by_track.setdefault(str(track_id), []).append(distance)
            cameras.append({
                'camera': index,
                'source': str(clips[index - 1]),
                'rates_hz': {key: round(value / elapsed, 2) if elapsed else None
                             for key, value in counts.items()},
                'processed_frames': counts,
                'local_nt4_write_frames': nt_after - previous_nt,
                'stereo_pair_stats': {
                    'frames_with_stereo_depth': int(final_stats.get('stereo_flow_frames', 0)) - stereo_before,
                    'missing_right_eye_pairs': int(final_stats.get('stereo_flow_missing_pairs', 0)) - missing_before,
                },
                'nt4_3d_payloads': {
                    'frames_with_at_least_one_piece': sum(count >= 1 for count in measured_piece_counts),
                    'frames_with_at_least_two_pieces': sum(count >= 2 for count in measured_piece_counts),
                    'frames_per_second_with_at_least_two_pieces': round(
                        sum(count >= 2 for count in measured_piece_counts) / elapsed, 2)
                        if elapsed else None,
                    'piece_count_histogram': {
                        str(count): measured_piece_counts.count(count)
                        for count in sorted(set(measured_piece_counts))},
                    'cluster_count_histogram': {
                        str(count): cluster_counts.count(count)
                        for count in sorted(set(cluster_counts))},
                    'cluster_count_median': statistics.median(cluster_counts)
                        if cluster_counts else None,
                    'cluster_count_min': min(cluster_counts) if cluster_counts else None,
                    'cluster_count_max': max(cluster_counts) if cluster_counts else None,
                    'published_range_age_ms': {
                        'p50': percentile(measured_range_ages, 50),
                        'p95': percentile(measured_range_ages, 95),
                        'max': round(max(measured_range_ages), 2)
                            if measured_range_ages else None,
                    },
                },
                'capture_to_nt_publish_latency_ms': {
                    'p50': percentile(end_to_end_latencies, 50),
                    'p95': percentile(end_to_end_latencies, 95),
                    'max': round(max(end_to_end_latencies), 2)
                        if end_to_end_latencies else None,
                    'samples': len(end_to_end_latencies),
                },
                'fresh_stereo_measurements': {
                    'frames_with_two_valid_ranges': sum(count >= 2 for count in valid_range_counts),
                    'frames_per_second_with_two_valid_ranges': round(
                        sum(count >= 2 for count in valid_range_counts) / elapsed, 2)
                        if elapsed else None,
                    'valid_range_count_histogram': {
                        str(count): valid_range_counts.count(count)
                        for count in sorted(set(valid_range_counts))},
                    'two_range_update_gap_ms': {
                        'p95': percentile(complete_depth_gaps_ms, 95),
                        'max': round(max(complete_depth_gaps_ms), 2)
                            if complete_depth_gaps_ms else None,
                    },
                    'range_by_track_m': {
                        track_id: {
                            'samples': len(values),
                            'median': round(float(statistics.median(values)), 4),
                            'mad': round(float(statistics.median(
                                [abs(value - statistics.median(values))
                                 for value in values])), 4),
                        }
                        for track_id, values in sorted(depth_by_track.items())
                    },
                },
                'sampled_latest_stage_latency_ms_p50_p95': {
                    key: {'p50': percentile([row[key] for row in camera_samples], 50),
                          'p95': percentile([row[key] for row in camera_samples], 95)}
                    for key in ('inference_latency_ms', 'detector_total_ms',
                                'stereo_map_ms', 'stereo_rectify_ms',
                                'stereo_match_ms', 'stereo_extract_ms',
                                'stereo_flow_box_build_ms', 'stereo_flow_tracker_ms')},
                'tracks_and_depth': {
                    'tracked_instances_last_sample': camera_samples[-1]['tracked_instances'] if camera_samples else 0,
                    'fresh_stereo_ranges_last_sample': camera_samples[-1]['stereo_flow_fresh_tracks'] if camera_samples else 0,
                },
                'pipeline': {key: final_stats.get(key) for key in (
                    'detector', 'precision', 'input_resolution', 'inference_resolution',
                    'tracking_resolution', 'stereo_backend', 'stereo_calibrated',
                    'stereo_approximate', 'stereo_match_scale', 'stereo_num_disparities',
                    'nt_enabled', 'nt_status', 'error')},
            })
        aggregate_counts = {
            key: sum(camera['processed_frames'][key] for camera in cameras)
            for key in cameras[0]['processed_frames']}
        aggregate_rates = {key: round(value / elapsed, 2) if elapsed else None
                           for key, value in aggregate_counts.items()}
        live_camera_inputs = all(str(source).startswith(('/dev/video', 'mjpeg-tcp://'))
                                 for source in clips[:args.cameras])
        duplicated_live_source = (
            live_camera_inputs and args.cameras > 1 and
            (len({str(source) for source in clips[:args.cameras]}) == 1 or
             all(str(source).startswith('mjpeg-tcp://')
                 for source in clips[:args.cameras])))
        if duplicated_live_source:
            method = ('Production perception pipeline duplicated across two concurrent '
                      'pipelines reading the same live USB stereo camera. Includes camera '
                      'capture, hardware decode, resize/split/transfer, TensorRT, '
                      'optical flow/Kalman, stereo range, clustering, NT4 writes, and '
                      'capture-to-NT publish latency.')
        else:
            method = (f'{args.cameras} production VideoPipeline instance(s) running concurrently on input sources; '
                      'includes hardware decode/capture, resize/split/transfer, TensorRT, optical flow/Kalman, '
                      f'stereo range ({args.stereo_mode}), clustering, NT4 writes, and capture-to-NT publish latency.')
        result = {
            'method': method,
            'camera_count': args.cameras,
            'stream_enabled': args.stream_enabled,
            'nt_stale_ms': args.nt_stale_ms,
            'flow_grace_ms': args.flow_grace_ms,
            'engine': os.environ['SCOUT_ENGINE'],
            'stereo_mode': args.stereo_mode,
            'stereo_backend': args.stereo_backend,
            'stereo_disparities': args.disparities,
            'cv2_threads': args.cv2_threads,
            'inference_width': args.inference_width,
            'flow_width': args.flow_width,
            'stereo_pair_wait_ms': args.pair_wait_ms,
            'right_frame_cache_size': args.right_cache_frames,
            'stereo_vertical_pad': args.vertical_pad,
            'stereo_size_gate_relative_error': args.size_gate_error,
            'inference_cap_hz': args.inference_hz or None,
            'real_usb_camera_reference_hz': 60,
            'warmup_seconds': args.warmup_seconds,
            'measurement_seconds': round(elapsed, 3),
            'aggregate_rates_hz': aggregate_rates,
            'aggregate_processed_frames': aggregate_counts,
            'flow_rate_fraction_of_60hz_reference': round(
                aggregate_rates['flow'] / (args.cameras * 60), 3)
                if aggregate_rates['flow'] is not None else None,
            'per_camera': cameras,
            'limitations': [
                'Replay is unpaced, so the rate measures maximum processing throughput without media timestamp waits; it is not a claim of real-time capture cadence.',
                'The pipeline uses leaky latest-frame queues as it does for live input; capture can exceed inference/flow rates and counters show delivered rates.',
                ('One live USB source is fanned out to both pipelines; this tests duplicated processing load, '
                 'not two independent cameras, distinct scenes, or the aggregate bandwidth of two USB devices.'
                 if duplicated_live_source else
                 'The two inputs are two recorded captures, not two live USB camera devices.'),
                ('VP9 preview encoding is enabled, but this benchmark has no connected WebRTC client; '
                 'NT4 publishing is local to this process.' if args.stream_enabled else
                 'Preview encoding and WebRTC transport are disabled; NT4 publishing is local to this benchmark process.'),
                'Synthetic input disparity and object placement do not provide metric stereo ground truth.',
            ],
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        args.nt_output.parent.mkdir(parents=True, exist_ok=True)
        with args.nt_output.open('w') as nt_file:
            for camera_index, snapshots in enumerate(nt_snapshots, start=1):
                for snapshot in snapshots:
                    nt_file.write(json.dumps({'camera': camera_index, **snapshot},
                                              separators=(',', ':')) + '\n')
        args.inference_output.parent.mkdir(parents=True, exist_ok=True)
        with args.inference_output.open('w') as inference_file:
            for camera_index, rows in enumerate(inference_snapshots, start=1):
                for row in rows:
                    inference_file.write(json.dumps({'camera': camera_index, **row},
                                                    separators=(',', ':')) + '\n')
    finally:
        for core in reversed(cores):
            core.stop()


if __name__ == '__main__':
    main()
             
