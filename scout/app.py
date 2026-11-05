import asyncio
import logging
import os
import re
import subprocess
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.responses import FileResponse, Response
import cv2
from pydantic import BaseModel, Field
from .pipeline import VideoPipeline
from .samples import find_sample, load_samples
from .webrtc import WebRTCPeer

ROOT = Path(__file__).parent
LOG = logging.getLogger(__name__)


class ConfidenceUpdate(BaseModel):
    confidence: float = Field(ge=0.1, le=0.95)


class EngineSelection(BaseModel):
    engine: str = Field(min_length=1, max_length=100)


class SourceSelection(BaseModel):
    source_id: str = Field(min_length=1, max_length=80)


class CameraControlUpdate(BaseModel):
    control: str = Field(pattern='^[a-zA-Z0-9_]{1,64}$')
    value: int = Field(ge=-100000, le=100000)


def read_camera_controls(device):
    result = subprocess.run(['v4l2-ctl', '-d', device, '--list-ctrls-menus'],
                            capture_output=True, text=True, timeout=3, check=True)
    controls = {}
    for line in result.stdout.splitlines():
        match = re.match(r'^\s*([a-zA-Z0-9_]+)\s+.*?\((int|bool|menu|integer|integer64|bitmask)\)\s*:\s*(.*)$', line)
        if match:
            name, kind, details = match.groups()
            value = re.search(r'\bvalue=(-?\d+)', details)
            if not value:
                continue
            minimum = re.search(r'\bmin=(-?\d+)', details)
            maximum = re.search(r'\bmax=(-?\d+)', details)
            step = re.search(r'\bstep=(\d+)', details)
            default = re.search(r'\bdefault=(-?\d+)', details)
            flags = re.search(r'\bflags=([^\s]+)', details)
            controls[name] = {'type': kind, 'value': int(value.group(1)),
                              'default': int(default.group(1)) if default else None,
                              'min': int(minimum.group(1)) if minimum else 0,
                              'max': int(maximum.group(1)) if maximum else 1,
                              'step': max(1, int(step.group(1))) if step else 1,
                              'disabled': bool(flags and 'inactive' in flags.group(1)),
                              'menu': {}}
        elif line[:1].isspace() and ':' in line and controls:
            menu = re.match(r'^\s+(\d+):\s*(.*)$', line)
            if menu:
                controls[next(reversed(controls))]['menu'][menu.group(1)] = menu.group(2).strip()
    return controls


def create_app(source, detector_mode=None, engine_path=None, source_2=None):
    cv2.setNumThreads(max(0, int(os.getenv('SCOUT_OPENCV_THREADS', '1'))))
    samples = load_samples()
    catalog = {sample['id']: sample for sample in samples}
    initial_sample = find_sample(source)
    core = VideoPipeline(source, detector_mode or os.getenv('SCOUT_DETECTOR', 'motion'),
                         engine_path or os.getenv('SCOUT_ENGINE', 'models/yolo11n_640_fp16.engine'),
                         initial_sample['id'] if initial_sample else None,
                         initial_sample['name'] if initial_sample else None)
    source_2 = source_2 or os.getenv('SCOUT_SOURCE_2')
    core_2 = None
    if source_2:
        stereo_enabled = os.getenv('SCOUT_STEREO_ENABLED', '1').lower() not in (
            '0', 'false', 'no')
        calibration_2 = os.getenv('SCOUT_STEREO_CALIBRATION_2')
        if stereo_enabled and not calibration_2:
            raise ValueError(
                'SCOUT_STEREO_CALIBRATION_2 is required when a second stereo '
                'camera is configured; explicitly select its calibration file')
        table_2 = os.getenv('FRC_NT_TABLE_2')
        if not table_2:
            table_2 = os.getenv('FRC_NT_TABLE', '/FRC/Perception/v1').rstrip('/') + '/camera2'
        core_2 = VideoPipeline(
            source_2, detector_mode or os.getenv('SCOUT_DETECTOR', 'motion'),
            engine_path or os.getenv('SCOUT_ENGINE', 'models/yolo11n_640_fp16.engine'),
            source_id='stereo-camera2', source_name='FRC Perception Camera 2',
            nt_table=table_2,
            stereo_calibration=calibration_2,
            stereo_baseline_m=os.getenv('SCOUT_STEREO_BASELINE_M_2'),
            stereo_focal_px=os.getenv('SCOUT_STEREO_FOCAL_PX_2'),
            stereo_use_global_config=False)
        if stereo_enabled and core_2.stereo.calibration is not None and (
                not core_2.stereo.baseline_m or not core_2.stereo.focal_px):
            raise ValueError(
                'Camera 2 calibration must contain valid stereo intrinsics and '
                'extrinsics, including a baseline and focal length')

    @asynccontextmanager
    async def lifespan(app):
        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(None, core.start)
            app.state.pipeline = core
            if core_2 is not None:
                await loop.run_in_executor(None, core_2.start)
                app.state.pipeline_2 = core_2
            yield
        finally:
            if core_2 is not None:
                await loop.run_in_executor(None, core_2.stop)
            await loop.run_in_executor(None, core.stop)

    app = FastAPI(title='SCOUT Edge Vision', lifespan=lifespan)

    @app.get('/')
    async def index():
        return FileResponse(ROOT / 'static/webrtc.html', headers={'Cache-Control': 'no-store'})

    @app.get('/calibration')
    async def calibration():
        return FileResponse(ROOT / 'static/calibration.html', headers={'Cache-Control': 'no-store'})

    @app.get('/calibration/charuco.png')
    async def charuco_board(cols: int = Query(5, ge=3, le=20), rows: int = Query(7, ge=3, le=20),
                            dictionary: str = Query('DICT_4X4_50', max_length=32),
                            board_type: str = Query('custom', max_length=16),
                            square_px: int = Query(113, ge=10, le=1000),
                            square_mm: float = Query(30, gt=0),
                            marker_mm: float = Query(10, gt=0)):
        dictionaries = {
            name: getattr(cv2.aruco, name) for name in (
                'DICT_4X4_50', 'DICT_4X4_100', 'DICT_4X4_250', 'DICT_5X5_100', 'DICT_6X6_250')
        }
        if board_type not in ('custom', 'andymark'):
            raise HTTPException(400, 'Unsupported calibration board')
        if board_type == 'andymark':
            cols, rows, dictionary = 15, 11, 'DICT_4X4_250'
        if dictionary not in dictionaries:
            raise HTTPException(400, 'Unsupported ArUco dictionary')
        aruco = cv2.aruco.getPredefinedDictionary(dictionaries[dictionary])
        if marker_mm >= square_mm:
            raise HTTPException(400, 'Marker width must be smaller than square width')
        marker_ratio = marker_mm / square_mm
        board = cv2.aruco.CharucoBoard_create(cols, rows, 1.0, marker_ratio, aruco)
        margin = max(12, square_px // 4)
        image = board.draw((cols * square_px + 2 * margin, rows * square_px + 2 * margin),
                           marginSize=margin, borderBits=1)
        ok, encoded = cv2.imencode('.png', image)
        if not ok:
            raise HTTPException(500, 'Unable to render ChArUco board')
        return Response(content=encoded.tobytes(), media_type='image/png',
                        headers={'Cache-Control': 'no-store'})

    @app.get('/api/stats')
    async def stats(camera: int = Query(1, ge=1, le=2)):
        pipeline = core if camera == 1 else core_2
        if pipeline is None:
            raise HTTPException(404, 'Second camera is not configured')
        return pipeline.snapshot()

    @app.get('/api/cameras')
    async def camera_stats():
        result = [{'camera': 1, 'source': core.source, 'stats': core.snapshot()}]
        if core_2 is not None:
            result.append({'camera': 2, 'source': core_2.source,
                           'stats': core_2.snapshot()})
        return {'cameras': result}

    @app.get('/api/detections')
    async def detections(camera: int = Query(1, ge=1, le=2)):
        pipeline = core if camera == 1 else core_2
        if pipeline is None:
            raise HTTPException(404, 'Second camera is not configured')
        with pipeline.lock:
            payload = pipeline.detections
        if payload is None:
            return dict(status='waiting', boxes=[], tracks=[], age_s=None)
        age = max(0.0, time.monotonic() - payload['completed_at'])
        return dict(status='ready' if age < 0.5 else 'stale',
                    age_s=round(age, 3), rtp_timestamp=payload['rtp_timestamp'],
                    boxes=payload['boxes'], tracks=payload.get('tracks', payload['boxes']))

    def calibration_pipeline(camera):
        pipeline = core if camera == 1 else core_2
        if pipeline is None:
            raise HTTPException(404, 'Second camera is not configured')
        return pipeline

    @app.get('/api/calibration/status')
    async def calibration_status(camera: int = Query(1, ge=1, le=2)):
        return calibration_pipeline(camera).calibrator.status()

    @app.get('/api/calibration/frame/{side}.jpg')
    async def calibration_frame(side: str, camera: int = Query(1, ge=1, le=2)):
        if side not in ('left', 'right'):
            raise HTTPException(404, 'Unknown camera side')
        frame = calibration_pipeline(camera).calibrator.preview_frame(side)
        if frame is None:
            raise HTTPException(404, 'Camera preview is not available yet')
        return Response(content=frame, media_type='image/jpeg',
                        headers={'Cache-Control': 'no-store'})

    @app.post('/api/calibration/start')
    async def calibration_start(cols: int = 5, rows: int = 7, square_mm: float = 30,
                                dictionary: str = 'DICT_4X4_50', board_type: str = 'custom',
                                marker_mm: Optional[float] = None,
                                camera: int = Query(1, ge=1, le=2)):
        try:
            return calibration_pipeline(camera).calibrator.start(
                cols, rows, square_mm, dictionary, board_type, marker_mm)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc

    @app.post('/api/calibration/stop')
    async def calibration_stop(camera: int = Query(1, ge=1, le=2)):
        return calibration_pipeline(camera).calibrator.stop()

    @app.post('/api/calibration/clear')
    async def calibration_clear(camera: int = Query(1, ge=1, le=2)):
        try:
            return calibration_pipeline(camera).calibrator.clear()
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.get('/api/sources')
    async def sources():
        return dict(sources=[dict(id=item['id'], name=item['name']) for item in samples],
                    active_source=core.source_id)

    @app.post('/api/source')
    async def set_source(selection: SourceSelection):
        sample = catalog.get(selection.source_id)
        if sample is None:
            raise HTTPException(404, 'Unknown video sample')
        try:
            return await asyncio.get_running_loop().run_in_executor(
                None, core.switch_source, sample['path'], sample['id'], sample['name'])
        except RuntimeError as exc:
            raise HTTPException(409, str(exc)) from exc

    @app.post('/api/pause')
    async def toggle_pause():
        paused = not core.paused.is_set()
        try:
            return await asyncio.get_running_loop().run_in_executor(None, core.set_paused, paused)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.post('/api/confidence')
    async def set_confidence(update: ConfidenceUpdate):
        try:
            return await asyncio.get_running_loop().run_in_executor(
                None, core.set_confidence, update.confidence)
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(503, str(exc)) from exc

    def engine_catalog():
        model_dir = Path.cwd() / 'models'
        return {path.name: path for path in model_dir.glob('yolo*_fp16.engine')
                if path.is_file()}

    @app.get('/api/engines')
    async def engines():
        options = []
        for filename, path in sorted(engine_catalog().items()):
            stem = path.stem[:-5] if path.stem.endswith('_fp16') else path.stem
            size = re.search(r'_(\d+(?:x\d+)?)$', stem)
            label = stem[:size.start()] if size else stem
            if size:
                dims = size.group(1).replace('x', '×')
                label += f' · {dims}'
            options.append({'id': filename, 'label': label})
        return {'engines': options, 'active_engine': Path(core.engine_path).name}

    @app.post('/api/engine')
    async def set_engine(selection: EngineSelection):
        engine = engine_catalog().get(selection.engine)
        if engine is None:
            raise HTTPException(404, 'Unknown YOLO engine')
        try:
            return await asyncio.get_running_loop().run_in_executor(
                None, core.set_engine, str(engine))
        except (RuntimeError, ValueError) as exc:
            raise HTTPException(503, str(exc)) from exc

    def camera_device():
        source = core.source
        if source.isdigit():
            return '/dev/video' + source
        if source.startswith('/dev/video'):
            return source
        return None

    @app.get('/api/camera-controls')
    async def camera_controls():
        device = camera_device()
        if not device:
            return {'available': False, 'controls': {}, 'reason': 'Select a live camera source'}
        try:
            controls = await asyncio.get_running_loop().run_in_executor(
                None, read_camera_controls, device)
        except (OSError, subprocess.SubprocessError):
            return {'available': False, 'controls': {}, 'reason': 'Camera controls unavailable'}
        if 'exposure_absolute' in controls:
            fps = max(1, int(os.getenv('SCOUT_CAMERA_FPS', '60')))
            hardware_max = controls['exposure_absolute']['max']
            controls['exposure_absolute']['max'] = min(hardware_max, max(1, 10000 // fps))
            controls['exposure_absolute']['hardware_max'] = hardware_max
        return {'available': bool(controls), 'controls': controls,
                'reason': '' if controls else 'Camera exposes no adjustable controls'}

    @app.post('/api/camera-controls')
    async def set_camera_control(update: CameraControlUpdate):
        device = camera_device()
        if not device:
            raise HTTPException(409, 'Select a live camera source')
        if update.control == 'exposure_absolute':
            fps = max(1, int(os.getenv('SCOUT_CAMERA_FPS', '60')))
            exposure_limit = max(1, 10000 // fps)
            if update.value > exposure_limit:
                raise HTTPException(422, f'Exposure is limited to {exposure_limit} at {fps} FPS')
        try:
            def apply_control():
                listed = subprocess.run(['v4l2-ctl', '-d', device, '--list-ctrls'],
                                        capture_output=True, text=True, timeout=3, check=True)
                if not re.search(rf'^\s*{re.escape(update.control)}\s', listed.stdout, re.MULTILINE):
                    raise ValueError('Unsupported camera control')
                subprocess.run(['v4l2-ctl', '-d', device,
                                f'--set-ctrl={update.control}={update.value}'],
                               capture_output=True, text=True, timeout=3, check=True)
            await asyncio.get_running_loop().run_in_executor(None, apply_control)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise HTTPException(503, 'Unable to set camera control') from exc
        return {'control': update.control, 'value': update.value}

    @app.post('/api/camera-controls/reset')
    async def reset_camera_controls():
        device = camera_device()
        if not device:
            raise HTTPException(409, 'Select a live camera source')

        def reset_controls():
            controls = read_camera_controls(device)
            automatic = [(name, control['default']) for name, control in controls.items()
                         if control['default'] is not None and
                         (name.endswith('_auto') or name == 'exposure_auto')]
            for name, value in automatic:
                subprocess.run(['v4l2-ctl', '-d', device, f'--set-ctrl={name}={value}'],
                               capture_output=True, text=True, timeout=3, check=True)
            controls = read_camera_controls(device)
            reset = []
            for name, control in controls.items():
                value = control['default']
                if value is None or control['disabled'] or name in dict(automatic):
                    continue
                subprocess.run(['v4l2-ctl', '-d', device, f'--set-ctrl={name}={value}'],
                               capture_output=True, text=True, timeout=3, check=True)
                reset.append(name)
            return reset

        try:
            reset = await asyncio.get_running_loop().run_in_executor(None, reset_controls)
        except (OSError, subprocess.SubprocessError) as exc:
            raise HTTPException(503, 'Unable to reset camera controls') from exc
        return {'reset': reset, 'controls': await asyncio.get_running_loop().run_in_executor(
            None, read_camera_controls, device)}

    @app.post('/api/camera/reconnect')
    async def reconnect_camera():
        try:
            return await asyncio.get_running_loop().run_in_executor(None, core.reconnect_camera)
        except RuntimeError as exc:
            raise HTTPException(503, str(exc)) from exc

    @app.get('/stream')
    async def retired_mjpeg():
        raise HTTPException(410, 'MJPEG has been replaced by WebRTC. Open the Scout home page.')

    @app.websocket('/ws')
    async def websocket(socket: WebSocket):
        await socket.accept()
        if not core.stream_enabled:
            await socket.close(code=1008, reason='WebRTC output is disabled')
            return
        peer = None
        tasks = []
        try:
            if len(core.peers) >= 8:
                await socket.close(code=1013, reason='Viewer limit reached')
                return
            loop = asyncio.get_running_loop()
            peer = WebRTCPeer(core, loop)
            pending_ice = []
            remote_description_set = False

            async def receive():
                nonlocal remote_description_set
                async for message in socket.iter_json():
                    kind = message.get('type')
                    if kind == 'answer':
                        sdp = message.get('sdp', '')
                        if not isinstance(sdp, str) or len(sdp) > 65536:
                            raise ValueError('Invalid SDP size')
                        await loop.run_in_executor(None, peer.answer, sdp)
                        remote_description_set = True
                        for index, candidate in pending_ice:
                            peer.ice(index, candidate, socket.client.host)
                        pending_ice.clear()
                    elif kind == 'ice':
                        candidate = message.get('candidate', '')
                        index = message.get('sdpMLineIndex', 0)
                        if not isinstance(candidate, str) or len(candidate) > 4096 or not isinstance(index, int) or not 0 <= index < 16:
                            raise ValueError('Invalid ICE candidate')
                        # Browsers mask host ICE addresses with mDNS names. The
                        # signaling socket gives us the actual peer address on
                        # the same LAN/Tailscale route, so resolve that name
                        # before passing the candidate to libnice.
                        if remote_description_set:
                            peer.ice(index, candidate, socket.client.host)
                        elif len(pending_ice) < 64:
                            pending_ice.append((index, candidate))

            async def send():
                last_detection = None
                last_stats = 0
                last_charuco = 0
                while not peer.closed:
                    try:
                        message = await asyncio.wait_for(peer.signals.get(), timeout=.005)
                        await socket.send_json(message)
                    except asyncio.TimeoutError:
                        pass
                    detection = core.detections
                    if detection is not None and detection is not last_detection:
                        await socket.send_json({k: v for k, v in detection.items() if k != 'completed_at'})
                        last_detection = detection
                    now = loop.time()
                    if now-last_charuco >= .2:
                        calibration = core.calibrator.status()
                        await socket.send_json(dict(type='charuco', **calibration['preview']))
                        last_charuco = now
                    if now-last_stats >= 1:
                        await socket.send_json(dict(type='stats', **core.snapshot()))
                        last_stats = now

            tasks = [asyncio.create_task(receive()), asyncio.create_task(send())]
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                task.result()
        except WebSocketDisconnect:
            pass
        except (ValueError, RuntimeError) as exc:
            LOG.exception('WebRTC signaling failed')
            try:
                await socket.send_json(dict(type='error', message=str(exc)))
                await socket.close(code=1011)
            except RuntimeError:
                pass
        finally:
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            if peer:
                await asyncio.get_running_loop().run_in_executor(None, peer.close)

    return app
