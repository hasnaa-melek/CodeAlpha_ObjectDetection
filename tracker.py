import numpy as np


def calculate_iou(box_a, box_b):
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    a_x2 = ax + aw
    a_y2 = ay + ah
    b_x2 = bx + bw
    b_y2 = by + bh

    intersection_x1 = max(ax, bx)
    intersection_y1 = max(ay, by)
    intersection_x2 = min(a_x2, b_x2)
    intersection_y2 = min(a_y2, b_y2)

    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)

    intersection_area = intersection_width * intersection_height

    area_a = aw * ah
    area_b = bw * bh

    union_area = area_a + area_b - intersection_area

    if union_area <= 0:
        return 0.0

    return intersection_area / union_area


class Track:
    def __init__(self, track_id, detection):
        self.track_id = track_id
        self.class_id = detection["class_id"]
        self.class_name = detection["class_name"]
        self.confidence = detection["confidence"]
        self.box = detection["box"]
        self.missed_frames = 0

    def update(self, detection):
        self.class_id = detection["class_id"]
        self.class_name = detection["class_name"]
        self.confidence = detection["confidence"]
        self.box = detection["box"]
        self.missed_frames = 0


class SortTracker:
    def __init__(self, iou_threshold=0.30, max_missed_frames=10):
        self.iou_threshold = iou_threshold
        self.max_missed_frames = max_missed_frames

        self.tracks = []
        self.next_track_id = 1

    def update(self, detections):
        if not self.tracks:
            for detection in detections:
                self.tracks.append(
                    Track(self.next_track_id, detection)
                )
                self.next_track_id += 1

            return self.get_tracks()

        matched_tracks = set()
        matched_detections = set()

        possible_matches = []

        for track_index, track in enumerate(self.tracks):
            for detection_index, detection in enumerate(detections):
                if track.class_id != detection["class_id"]:
                    continue

                iou = calculate_iou(
                    track.box,
                    detection["box"]
                )

                if iou >= self.iou_threshold:
                    possible_matches.append(
                        (iou, track_index, detection_index)
                    )

        possible_matches.sort(reverse=True)

        for iou, track_index, detection_index in possible_matches:
            if track_index in matched_tracks:
                continue

            if detection_index in matched_detections:
                continue

            self.tracks[track_index].update(
                detections[detection_index]
            )

            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        for track_index, track in enumerate(self.tracks):
            if track_index not in matched_tracks:
                track.missed_frames += 1

        for detection_index, detection in enumerate(detections):
            if detection_index not in matched_detections:
                self.tracks.append(
                    Track(self.next_track_id, detection)
                )
                self.next_track_id += 1

        self.tracks = [
            track
            for track in self.tracks
            if track.missed_frames <= self.max_missed_frames
        ]

        return self.get_tracks()

    def get_tracks(self):
        results = []

        for track in self.tracks:
            results.append({
                "track_id": track.track_id,
                "class_id": track.class_id,
                "class_name": track.class_name,
                "confidence": track.confidence,
                "box": track.box
            })

        return results