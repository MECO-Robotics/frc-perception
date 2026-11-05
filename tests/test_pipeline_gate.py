import threading
import unittest

from scout.pipeline import VideoPipeline


class InferenceGateTests(unittest.TestCase):
    def setUp(self):
        self.pipeline = VideoPipeline.__new__(VideoPipeline)
        self.pipeline.lock = threading.RLock()
        self.pipeline.paused = threading.Event()
        self.pipeline.switching = threading.Event()
        self.pipeline.peers = set()

    def test_inference_is_idle_without_viewers(self):
        self.assertFalse(self.pipeline._inference_enabled())

    def test_inference_starts_for_connected_viewer(self):
        self.pipeline.peers.add(object())
        self.assertTrue(self.pipeline._inference_enabled())

    def test_inference_stops_when_viewer_disconnects_or_pipeline_pauses(self):
        self.pipeline.peers.add(object())
        self.pipeline.paused.set()
        self.assertFalse(self.pipeline._inference_enabled())
        self.pipeline.paused.clear()
        self.pipeline.peers.clear()
        self.assertFalse(self.pipeline._inference_enabled())


if __name__ == '__main__':
    unittest.main()
