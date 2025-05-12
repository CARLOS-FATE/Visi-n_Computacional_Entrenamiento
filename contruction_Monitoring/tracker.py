# tracker.py
import cv2
import numpy as np
from collections import defaultdict
from norfair import Detection, Tracker

class ObjectTracker:
    def __init__(self, distance_threshold=30):
        self.tracker = Tracker(
            distance_function=self._euclidean_distance,
            distance_threshold=distance_threshold
        )
        self.id_to_class = {}
        self.id_to_box = {}

    def _euclidean_distance(self, detection, tracked_object):
        return np.linalg.norm(detection.points - tracked_object.estimate)

    def _convert_to_norfair_detections(self, detections):
        norfair_detections = []
        self.current_info = {}  # temporal para esta llamada

        for det in detections:
            x1, y1, x2, y2 = det["box"]
            class_id = det["class_id"]
            confidence = det["confidence"]

            center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
            d = Detection(points=center, scores=np.array([confidence]))
            d.data = {"box": [x1, y1, x2, y2], "class_id": class_id}
            norfair_detections.append(d)

        return norfair_detections

    def update(self, detections):
        norfair_detections = self._convert_to_norfair_detections(detections)
        tracked = self.tracker.update(detections=norfair_detections)

        results = []
        for obj in tracked:
            if obj.last_detection:
                det_data = obj.last_detection.data
                results.append({
                    "id": obj.id,
                    "box": det_data["box"],
                    "class_id": det_data["class_id"]
                })

        return results
