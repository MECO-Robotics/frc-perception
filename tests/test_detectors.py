import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import numpy as np
from scout.detectors import TensorRTDetector, create_detector, letterbox_geometry


class YoloTests(unittest.TestCase):
    def setUp(self):
        self.detector = TensorRTDetector.__new__(TensorRTDetector)
        self.detector.confidence = .25
        self.detector.iou = .45
        self.detector.class_confidences = {}
        self.detector.class_names = [str(i) for i in range(80)]
        self.detector.max_box_aspect_ratio = 0
        self.detector.last_shape_rejected = 0

    def output(self, predictions):
        rows = np.zeros((len(predictions), 84), dtype=np.float32)
        for row, (box, cid, score) in zip(rows, predictions):
            row[:4] = box
            row[4 + cid] = score
        return rows.T[None]

    def decode(self, predictions):
        return self.detector._decode(self.output(predictions), 1, 0, 0, 100, 100)

    def test_raw_batched_output_and_letterbox_coordinates(self):
        output = self.output([((50, 50, 20, 20), 0, .9)])
        for value in (output, output[0].T):
            result = self.detector._decode(value, .5, 0, 25, 200, 100)
            self.assertEqual(len(result), 1)
            self.assertEqual((result[0].x1, result[0].y1, result[0].x2, result[0].y2), (80, 30, 120, 70))
            self.assertEqual(result[0].class_id, 0)

    def test_yolo26_end_to_end_output_and_letterbox_coordinates(self):
        self.detector.last_candidate_count = 0
        output = np.array([[[10, 15, 30, 35, .9, 0],
                            [20, 20, 40, 40, .2, 1]]], dtype=np.float32)
        result = self.detector._decode(output, .5, 0, 5, 100, 100)
        self.assertEqual(len(result), 1)
        self.assertEqual((result[0].x1, result[0].y1, result[0].x2, result[0].y2),
                         (20, 20, 60, 60))
        self.assertEqual(result[0].class_id, 0)
        self.assertAlmostEqual(result[0].confidence, .9)
        self.assertEqual(self.detector.last_candidate_count, 1)

    def test_yolo26_end_to_end_applies_configured_classwise_nms(self):
        output = np.array([[[10, 10, 30, 30, .9, 0],
                            [11, 11, 31, 31, .8, 0],
                            [32, 10, 52, 30, .7, 0],
                            [10, 10, 30, 30, .85, 1]]], dtype=np.float32)
        result = self.detector._decode(output, 1, 0, 0, 100, 100)
        self.assertEqual([(d.class_id, round(d.confidence, 2)) for d in result],
                         [(0, .9), (1, .85), (0, .7)])
        self.assertEqual(self.detector.last_candidate_count, 4)

    def test_rectangular_native_letterbox_geometry(self):
        geometry = letterbox_geometry(1920, 1080, 1920, 1088)
        self.assertEqual(geometry, (1.0, 1920, 1080, 0, 4))
        scale, width, height, ox, oy = geometry
        result = self.detector._decode(
            np.array([[[100, 104, 300, 404, .9, 0]]], dtype=np.float32),
            scale, ox, oy, 1920, 1080)
        self.assertEqual((result[0].x1, result[0].y1, result[0].x2, result[0].y2),
                         (100, 100, 300, 400))
        self.assertEqual((width, height), (1920, 1080))

    def test_class_specific_confidence_applies_to_yolo26_outputs(self):
        self.detector.class_confidences = {0: .40, 2: .60}
        output = np.array([[[0, 0, 20, 20, .50, 0],
                            [30, 30, 50, 50, .50, 2]]], dtype=np.float32)
        result = self.detector._decode(output, 1, 0, 0, 100, 100)
        self.assertEqual([d.class_id for d in result], [0])

    def test_class_specific_confidence_applies_to_raw_yolo_outputs(self):
        self.detector.class_confidences = {0: .40, 2: .60}
        result = self.decode([((20, 20, 10, 10), 0, .50),
                              ((60, 60, 10, 10), 2, .50)])
        self.assertEqual([d.class_id for d in result], [0])

    def test_nms_keeps_different_classes(self):
        result = self.decode([((50, 50, 20, 20), 0, .9),
                              ((50, 50, 20, 20), 0, .8),
                              ((50, 50, 20, 20), 2, .85)])
        self.assertEqual([d.class_id for d in result], [0, 2])

    def test_empty_low_confidence_invalid_and_offscreen(self):
        self.assertEqual(self.decode([]), [])
        self.assertEqual(self.decode([((50, 50, 20, 20), 0, .1),
                                      ((-50, -50, 20, 20), 0, .9),
                                      ((50, 50, -20, 20), 0, .9),
                                      ((float('nan'), 50, 20, 20), 0, .9)]), [])

    def test_clips_to_frame(self):
        result = self.decode([((0, 0, 20, 20), 0, .9)])[0]
        self.assertEqual((result.x1, result.y1, result.x2, result.y2), (0, 0, 10, 10))

    def test_candidate_limit_keeps_highest_scores_before_nms(self):
        self.detector.max_candidates = 2
        result = self.decode([((10, 10, 4, 4), 0, .6),
                              ((30, 30, 4, 4), 0, .95),
                              ((70, 70, 4, 4), 0, .8)])
        self.assertEqual({round(d.confidence, 2) for d in result}, {.95, .8})
        self.assertEqual(self.detector.last_candidate_count, 2)

    def test_rejects_unsupported_output(self):
        with self.assertRaises(ValueError):
            self.detector._decode(np.zeros((2, 84, 10)), 1, 0, 0, 100, 100)

    def test_context_is_current_in_worker_and_popped_on_error(self):
        local = threading.local()
        context = Mock()
        context.push.side_effect = lambda: setattr(local, 'active', True)
        context.pop.side_effect = lambda: setattr(local, 'active', False)
        self.detector.cuda_context = context
        self.detector.lock = threading.Lock()
        def infer(frame):
            self.assertTrue(local.active)
            raise RuntimeError('inference failed')
        self.detector._infer = infer
        def worker():
            with self.assertRaises(RuntimeError):
                self.detector.infer(None)
            self.assertFalse(local.active)
        with ThreadPoolExecutor(max_workers=1) as executor:
            executor.submit(worker).result()
        context.push.assert_called_once()
        context.pop.assert_called_once()

    def test_unknown_mode_is_not_silently_motion(self):
        with self.assertRaises(ValueError):
            create_detector('yolo-typo')


if __name__ == '__main__':
    unittest.main()
