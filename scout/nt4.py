"""NetworkTables v1 publisher for FRC FUEL perception tracks."""
import json
import logging
import math
import os
import time

import numpy as np

LOG = logging.getLogger(__name__)


class NT4Publisher:
    """Publish atomic piece/cluster snapshots for an NT4 robot client."""

    def __init__(self, table_path=None):
        try:
            import ntcore
        except ImportError as exc:
            raise RuntimeError('FRC_NT_ENABLED requires pyntcore/ntcore') from exc

        self.ntcore = ntcore
        self.instance = ntcore.NetworkTableInstance.getDefault()
        self.table_path = (table_path or os.getenv(
            'FRC_NT_TABLE', '/FRC/Perception/v1')).strip('/')
        self.team = os.getenv('FRC_TEAM_NUMBER', '').strip()
        self.server = os.getenv('FRC_NT_SERVER', '').strip()
        self.mode = os.getenv('FRC_NT_MODE', 'client').strip().lower()
        self.cluster_radius_m = float(os.getenv('FRC_CLUSTER_RADIUS_M', '0.25'))
        self.stale_ms = float(os.getenv('FRC_NT_STALE_MS', '250'))
        if not self.table_path:
            raise ValueError('FRC_NT_TABLE must not be empty')
        if self.mode not in ('client', 'server'):
            raise ValueError('FRC_NT_MODE must be client or server')
        if self.cluster_radius_m <= 0:
            raise ValueError('FRC_CLUSTER_RADIUS_M must be positive')
        if self.stale_ms <= 0:
            raise ValueError('FRC_NT_STALE_MS must be positive')

        self.table = self.instance.getTable(self.table_path)
        self.status_entry = self.table.getEntry('status')
        self.frame_entry = self.table.getEntry('frame')
        self.timestamp_entry = self.table.getEntry('timestamp')
        self.clock_entry = self.table.getEntry('timestampClock')
        self.pieces_entry = self.table.getEntry('pieces')
        self.clusters_entry = self.table.getEntry('clusters')
        self.calibrated_entry = self.table.getEntry('calibrated')
        self.frame = 0
        self.started_at = time.monotonic()

        if self.mode == 'server':
            self.instance.startServer()
        else:
            if self.server:
                self.instance.setServer(self.server)
            elif self.team:
                team = int(self.team)
                if not 0 < team <= 25599:
                    raise ValueError('FRC_TEAM_NUMBER must be from 1 through 25599')
                self.instance.setServerTeam(team)
            else:
                raise ValueError('Set FRC_NT_SERVER or FRC_TEAM_NUMBER for NT client mode')
            self.instance.startClient4('scout-frc-perception')

        self.status_entry.setString('starting')
        self.clock_entry.setString('gstreamer_pts_seconds')
        self.pieces_entry.setString('[]')
        self.clusters_entry.setString('[]')
        self.calibrated_entry.setBoolean(False)

    @staticmethod
    def _cluster(pieces, radius):
        valid = [piece for piece in pieces if piece['z_m'] is not None]
        if not valid:
            return []
        xyz = np.asarray([(piece['x_m'], piece['y_m'], piece['z_m'])
                          for piece in valid], dtype=np.float32)
        delta = xyz[:, None, :] - xyz[None, :, :]
        linked = np.sum(delta * delta, axis=2) <= radius * radius
        remaining = np.ones(len(valid), dtype=bool)
        clusters = []
        for seed in range(len(valid)):
            if not remaining[seed]:
                continue
            remaining[seed] = False
            component = [seed]
            frontier = [seed]
            while frontier:
                current = frontier.pop()
                neighbors = np.flatnonzero(linked[current] & remaining)
                remaining[neighbors] = False
                component.extend(int(index) for index in neighbors)
                frontier.extend(int(index) for index in neighbors)
            members = [valid[index] for index in component]
            count = len(members)
            clusters.append({
                'id': len(clusters),
                'x_m': sum(item['x_m'] for item in members) / count,
                'y_m': sum(item['y_m'] for item in members) / count,
                'z_m': sum(item['z_m'] for item in members) / count,
                'count': count,
                'confidence': sum(item['confidence'] for item in members) / count,
                'piece_ids': [item['id'] for item in members],
            })
        return clusters

    def publish(self, tracks, pts_seconds, frame_width, frame_height,
                project_pixel, calibrated, approximate=False, pipeline_error=None):
        """Publish one processed camera frame; project_pixel returns camera XYZ."""
        self.frame += 1
        now = time.monotonic()
        pieces = []
        for track in tracks:
            cx, cy, _, _ = track['box']
            range_age = track['range_age_ms']
            distance = track['distance_m']
            xyz = None
            quality = 'invalid'
            if (distance is not None and range_age is not None and
                    range_age <= self.stale_ms and math.isfinite(float(distance)) and
                    float(distance) > 0):
                xyz = project_pixel(cx, cy, float(distance), frame_width, frame_height)
                if xyz is not None and all(math.isfinite(float(value)) for value in xyz):
                    quality = 'approximate' if approximate else 'fresh'
                else:
                    xyz = None
            # NT's `pieces` topic is the robot-facing 3D contract. Keep
            # image-only/predicted tracks internal; publishing null XYZ entries
            # makes them look like gamepiece locations to consumers.
            if xyz is None:
                continue
            pieces.append({
                'id': int(track['track_id']),
                'x_m': round(float(xyz[0]), 4) if xyz is not None else None,
                'y_m': round(float(xyz[1]), 4) if xyz is not None else None,
                'z_m': round(float(xyz[2]), 4) if xyz is not None else None,
                'confidence': round(float(track['confidence']), 4),
                'depth_quality': quality,
                'track_state': track['state'],
                'position_age_ms': round(float(track.get('visual_age_ms',
                                                         track['age_ms'])), 1),
                'range_age_ms': round(float(range_age), 1) if range_age is not None else None,
            })

        clusters = self._cluster(pieces, self.cluster_radius_m)
        connected = bool(self.instance.isConnected())
        fresh_depth = any(piece['z_m'] is not None for piece in pieces)
        status = ('error' if pipeline_error else
                  'ready' if connected and fresh_depth else 'degraded')
        self.status_entry.setString(status)
        self.frame_entry.setInteger(self.frame)
        self.timestamp_entry.setDouble(float(pts_seconds))
        self.pieces_entry.setString(json.dumps(pieces, separators=(',', ':')))
        self.clusters_entry.setString(json.dumps(clusters, separators=(',', ':')))
        self.calibrated_entry.setBoolean(bool(calibrated))
        return {'connected': connected, 'status': status,
                'pieces': len(pieces), 'clusters': len(clusters), 'frame': self.frame}

    def close(self):
        self.status_entry.setString('stopped')
        self.instance.flush()
        self.instance.stopClient() if self.mode == 'client' else self.instance.stopServer()
