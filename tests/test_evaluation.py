import unittest

from scout.detectors import Detection
from scout.evaluation import box_iou, frame_counts, metrics


class EvaluationTests(unittest.TestCase):
    def test_iou_and_greedy_one_to_one_class_aware_matching(self):
        self.assertAlmostEqual(box_iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)
        labels = [dict(class_id=0, bbox=[0, 0, 10, 10]),
                  dict(class_id=2, bbox=[0, 0, 10, 10])]
        detections = [Detection(0, 0, 10, 10, 0, .9),
                      Detection(0, 0, 10, 10, 0, .8),
                      Detection(0, 0, 10, 10, 3, .7)]
        result = metrics(frame_counts(labels, detections))
        self.assertEqual(result[0], dict(tp=1, fp=1, fn=0, precision=.5, recall=1.))
        self.assertEqual(result[2], dict(tp=0, fp=0, fn=1, precision=None, recall=0.))
        self.assertEqual(result[3], dict(tp=0, fp=1, fn=0, precision=0., recall=None))


if __name__ == '__main__':
    unittest.main()
