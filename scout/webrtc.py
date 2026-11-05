"""Per-viewer SRTP transport; all viewers share the upstream encoder and YOLO."""
import asyncio
import logging
import os
import threading

from .gst_runtime import Gst, GstWebRTC, GstSdp

LOG = logging.getLogger(__name__)


class WebRTCPeer:
    MAX_QUEUED_BYTES = 256 * 1024

    def __init__(self, core, loop):
        self.core, self.loop = core, loop
        self.signals = asyncio.Queue(maxsize=64)
        self.lock = threading.RLock()
        self.closed = False
        self.dropped = 0
        self.pipeline = Gst.parse_launch(
            'appsrc name=source is-live=true format=time block=false max-bytes=262144 ! '
            'webrtcbin name=rtc bundle-policy=max-bundle')
        self.source = self.pipeline.get_by_name('source')
        self.rtc = self.pipeline.get_by_name('rtc')
        self.source.set_property('caps', core.rtp_caps)
        # Preserve the capture clock so RTP timestamps also identify overlay frames.
        self.pipeline.use_clock(core.clock)
        self.pipeline.set_start_time(Gst.CLOCK_TIME_NONE)
        self.pipeline.set_base_time(core.origin)
        self.rtc.connect('on-negotiation-needed', self._negotiate)
        self.rtc.connect('on-ice-candidate', self._ice)
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message::error', self._error)
        with core.lock:
            core.peers.add(self)
        self.pipeline.set_state(Gst.State.PLAYING)

    def _send(self, message):
        def enqueue():
            if self.closed:
                return
            try:
                self.signals.put_nowait(message)
            except asyncio.QueueFull:
                self.close()
        if not self.loop.is_closed():
            self.loop.call_soon_threadsafe(enqueue)

    def _negotiate(self, element):
        if self.closed:
            return
        promise = Gst.Promise.new_with_change_func(self._offer_created, None, None)
        self.rtc.emit('create-offer', None, promise)

    def _offer_created(self, promise, *unused):
        if self.closed:
            return
        reply = promise.get_reply()
        offer = reply.get_value('offer') if reply else None
        if offer is None:
            self._send(dict(type='error', message='Unable to create WebRTC offer'))
            return
        self.rtc.emit('set-local-description', offer, Gst.Promise.new())
        self._send(dict(type='offer', sdp=offer.sdp.as_text()))

    def _ice(self, element, index, candidate):
        parts = candidate.split()
        address = os.getenv('SCOUT_ICE_ADDRESS', '')
        if len(parts) > 7 and parts[7] == 'host':
            if not address or not parts[4].startswith('fd7a:') or parts[2] != 'UDP':
                return
            # Advertise only the reachable Tailscale IPv4 UDP candidate.
            parts[4] = address
            candidate = ' '.join(parts)
        self._send(dict(type='ice', sdpMLineIndex=index, candidate=candidate))

    def _error(self, bus, message):
        error, debug = message.parse_error()
        LOG.error('WebRTC: %s (%s)', error, debug)
        self._send(dict(type='error', message=str(error)))

    def answer(self, sdp):
        status, message = GstSdp.SDPMessage.new()
        if status != GstSdp.SDPResult.OK:
            raise ValueError('Unable to allocate SDP')
        if GstSdp.sdp_message_parse_buffer(sdp.encode(), message) != GstSdp.SDPResult.OK:
            raise ValueError('Invalid SDP answer')
        description = GstWebRTC.WebRTCSessionDescription.new(GstWebRTC.WebRTCSDPType.ANSWER, message)
        promise = Gst.Promise.new()
        self.rtc.emit('set-remote-description', description, promise)
        promise.wait()
        reply = promise.get_reply()
        if reply and reply.has_field('error'):
            raise ValueError(str(reply.get_value('error')))
        LOG.info('Applied WebRTC remote description')

    def ice(self, index, candidate, peer_address):
        parts = candidate.split()
        if len(parts) >= 8 and parts[7] == 'host':
            parts[4] = peer_address
            candidate = ' '.join(parts)
        self.rtc.emit('add-ice-candidate', index, candidate)

    def push(self, buffer):
        with self.lock:
            if self.closed:
                return
            if self.source.get_property('current-level-bytes') >= self.MAX_QUEUED_BYTES:
                self.dropped += 1
                return
            self.source.emit('push-buffer', buffer)

    def close(self):
        with self.lock:
            if self.closed:
                return
            self.closed = True
        with self.core.lock:
            self.core.peers.discard(self)
            if not self.core.peers:
                self.core.detections = None
        # Drop a pending frame as soon as the last viewer leaves. The worker
        # also checks the peer set immediately before inference, covering a
        # disconnect that races with a queued frame.
        self.core.slot.clear()
        self.pipeline.set_state(Gst.State.NULL)
        self.pipeline.get_bus().remove_signal_watch()
