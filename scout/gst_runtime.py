"""Load system GStreamer, including optional user-local Jetson plugins."""
import os
import ctypes
from pathlib import Path

plugins = Path(__file__).resolve().parent.parent / '.gst/usr/lib/aarch64-linux-gnu/gstreamer-1.0'
if plugins.is_dir():
    os.environ['GST_PLUGIN_PATH'] = str(plugins) + (':' + os.environ['GST_PLUGIN_PATH'] if os.environ.get('GST_PLUGIN_PATH') else '')

# Reserve this library's TLS before OpenCV/TensorRT load on glibc-based Jetson.
ctypes.CDLL('libGLdispatch.so.0', mode=ctypes.RTLD_GLOBAL)

import gi
for namespace in ('Gst', 'GstVideo', 'GstWebRTC', 'GstSdp', 'GstPbutils'):
    gi.require_version(namespace, '1.0')
from gi.repository import Gst, GstVideo, GstWebRTC, GstSdp, GstPbutils, GLib
Gst.init(None)
