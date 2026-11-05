"""Small, dependency-light COCO box metrics for the SAR tuning set."""
from collections import defaultdict


def box_iou(left, right):
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0., x2 - x1) * max(0., y2 - y1)
    left_area = max(0., left[2] - left[0]) * max(0., left[3] - left[1])
    right_area = max(0., right[2] - right[0]) * max(0., right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union else 0.


def frame_counts(annotations, detections, iou_threshold=.5):
    """Return per-class TP/FP/FN with one-to-one, class-aware matching."""
    counts = defaultdict(lambda: dict(tp=0, fp=0, fn=0))
    by_class = defaultdict(list)
    for annotation in annotations:
        by_class[int(annotation['class_id'])].append(annotation['bbox'])
    detection_classes = defaultdict(list)
    for detection in detections:
        detection_classes[int(detection.class_id)].append(detection)
    for class_id in set(by_class) | set(detection_classes):
        ground_truth = by_class[class_id]
        candidates = sorted(detection_classes[class_id], key=lambda item: item.confidence, reverse=True)
        matched = set()
        for candidate in candidates:
            overlaps = [(box_iou((candidate.x1, candidate.y1, candidate.x2, candidate.y2), box), index)
                        for index, box in enumerate(ground_truth) if index not in matched]
            if overlaps:
                overlap, index = max(overlaps)
            else:
                overlap, index = 0., None
            if overlap >= iou_threshold:
                counts[class_id]['tp'] += 1
                matched.add(index)
            else:
                counts[class_id]['fp'] += 1
        counts[class_id]['fn'] += len(ground_truth) - len(matched)
    return counts


def merge_counts(total, frame):
    for class_id, values in frame.items():
        target = total.setdefault(class_id, dict(tp=0, fp=0, fn=0))
        for name in ('tp', 'fp', 'fn'):
            target[name] += values[name]
    return total


def metrics(counts):
    result = {}
    for class_id, values in sorted(counts.items()):
        tp, fp, fn = values['tp'], values['fp'], values['fn']
        result[class_id] = dict(tp=tp, fp=fp, fn=fn,
                                precision=tp / (tp + fp) if tp + fp else None,
                                recall=tp / (tp + fn) if tp + fn else None)
    tp = sum(value['tp'] for value in counts.values())
    fp = sum(value['fp'] for value in counts.values())
    fn = sum(value['fn'] for value in counts.values())
    result['overall'] = dict(tp=tp, fp=fp, fn=fn,
                             precision=tp / (tp + fp) if tp + fp else None,
                             recall=tp / (tp + fn) if tp + fn else None)
    return result
