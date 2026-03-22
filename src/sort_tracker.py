"""
SORT (Simple Online and Realtime Tracker) + adapter matching the old DeepSort API.

Based on abewley/sort (GPL-3.0): Kalman bbox + IoU association, no appearance model.
"""
from __future__ import annotations

import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


def linear_assignment(cost_matrix: np.ndarray) -> np.ndarray:
    try:
        import lap

        _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
        return np.array([[y[i], i] for i in x if i >= 0])
    except ImportError:
        x, y = linear_sum_assignment(cost_matrix)
        return np.array(list(zip(x, y)))


def iou_batch(bb_test: np.ndarray, bb_gt: np.ndarray) -> np.ndarray:
    """IoU between rows of detections (bb_test) and rows of trackers (bb_gt); xyxy."""
    bb_gt = np.expand_dims(bb_gt, 0)
    bb_test = np.expand_dims(bb_test, 1)

    xx1 = np.maximum(bb_test[..., 0], bb_gt[..., 0])
    yy1 = np.maximum(bb_test[..., 1], bb_gt[..., 1])
    xx2 = np.minimum(bb_test[..., 2], bb_gt[..., 2])
    yy2 = np.minimum(bb_test[..., 3], bb_gt[..., 3])
    w = np.maximum(0.0, xx2 - xx1)
    h = np.maximum(0.0, yy2 - yy1)
    wh = w * h
    o = wh / (
        (bb_test[..., 2] - bb_test[..., 0]) * (bb_test[..., 3] - bb_test[..., 1])
        + (bb_gt[..., 2] - bb_gt[..., 0]) * (bb_gt[..., 3] - bb_gt[..., 1])
        - wh
        + 1e-8
    )
    return o


def convert_bbox_to_z(bbox: np.ndarray) -> np.ndarray:
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.0
    y = bbox[1] + h / 2.0
    s = w * h
    r = w / float(h + 1e-8)
    return np.array([x, y, s, r]).reshape((4, 1))


def convert_x_to_bbox(x: np.ndarray, score=None) -> np.ndarray:
    w = np.sqrt(np.maximum(x[2, 0] * x[3, 0], 0))
    h = x[2, 0] / (w + 1e-8)
    if score is None:
        return np.array([x[0, 0] - w / 2.0, x[1, 0] - h / 2.0, x[0, 0] + w / 2.0, x[1, 0] + h / 2.0]).reshape(
            (1, 4)
        )
    return np.array(
        [x[0, 0] - w / 2.0, x[1, 0] - h / 2.0, x[0, 0] + w / 2.0, x[1, 0] + h / 2.0, score]
    ).reshape((1, 5))


class KalmanBoxTracker:
    count = 0

    def __init__(self, bbox: np.ndarray):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array(
            [
                [1, 0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 1, 0],
                [0, 0, 1, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1, 0],
                [0, 0, 0, 0, 0, 0, 1],
            ]
        )
        self.kf.H = np.array(
            [
                [1, 0, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0, 0],
            ]
        )

        self.kf.R[2:, 2:] *= 10.0
        self.kf.P[4:, 4:] *= 1000.0
        self.kf.P *= 10.0
        self.kf.Q[-1, -1] *= 0.01
        self.kf.Q[4:, 4:] *= 0.01

        self.kf.x[:4] = convert_bbox_to_z(bbox[:4])
        self.time_since_update = 0
        self.id = KalmanBoxTracker.count
        KalmanBoxTracker.count += 1
        self.history = []
        self.hits = 0
        self.hit_streak = 0
        self.age = 0

    def update(self, bbox: np.ndarray) -> None:
        self.time_since_update = 0
        self.history = []
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox[:4]))

    def predict(self) -> np.ndarray:
        if (self.kf.x[6, 0] + self.kf.x[2, 0]) <= 0:
            self.kf.x[6, 0] *= 0.0
        self.kf.predict()
        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1
        self.history.append(convert_x_to_bbox(self.kf.x))
        return self.history[-1]

    def get_state(self) -> np.ndarray:
        return convert_x_to_bbox(self.kf.x)


def associate_detections_to_trackers(
    detections: np.ndarray, trackers: np.ndarray, iou_threshold: float = 0.3
):
    if len(trackers) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.arange(len(detections)),
            np.empty((0, 5), dtype=int),
        )

    if len(detections) == 0:
        return (
            np.empty((0, 2), dtype=int),
            np.empty((0,), dtype=int),
            np.arange(len(trackers)),
        )

    iou_matrix = iou_batch(detections, trackers)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.size and a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_detections = []
    for d in range(len(detections)):
        if len(matched_indices) == 0 or d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    unmatched_trackers = []
    for t in range(len(trackers)):
        if len(matched_indices) == 0 or t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)

    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)

    return (
        matches,
        np.array(unmatched_detections),
        np.array(unmatched_trackers),
    )


class Sort:
    """Classic SORT (bbox Kalman + IoU data association)."""

    def __init__(self, max_age: int = 1, min_hits: int = 3, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: list[KalmanBoxTracker] = []
        self.frame_count = 0

    def update(self, dets: np.ndarray = None) -> np.ndarray:
        if dets is None:
            dets = np.empty((0, 5))
        self.frame_count += 1
        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)
        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(
            dets[:, :4], trks[:, :4], self.iou_threshold
        )

        for m in matched:
            self.trackers[m[1]].update(dets[m[0], :])

        for i in unmatched_dets:
            trk = KalmanBoxTracker(dets[i, :])
            self.trackers.append(trk)

        # Original MOT code uses time_since_update < 1 (only matched-this-frame).
        # For video overlay we keep predicted boxes until max_age (like typical SORT demos).
        ret = []
        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            visible = (trk.hits >= self.min_hits or self.frame_count <= self.min_hits) and (
                trk.time_since_update <= self.max_age
            )
            if visible:
                ret.append(np.concatenate((d, [trk.id + 1])).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 5))


class SortTrack:
    """Minimal track object compatible with drawing (DeepSort-like API)."""

    __slots__ = ("track_id", "_ltrb")

    def __init__(self, track_id: int, ltrb):
        self.track_id = int(track_id)
        self._ltrb = [float(x) for x in ltrb]

    def is_confirmed(self) -> bool:
        return True

    def to_ltrb(self):
        return self._ltrb


class SortTrackerAdapter:
    """
    Wraps SORT with ``update_tracks(raw_detections, frame=...)`` like DeepSort.

    ``raw_detections``: list of ``([x, y, w, h], conf, class_name)``.
    """

    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3):
        self._sort = Sort(
            max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold
        )

    def update_tracks(self, raw_detections, embeds=None, frame=None):
        _ = embeds, frame
        if not raw_detections:
            dets = np.empty((0, 5))
        else:
            rows = []
            for det in raw_detections:
                l, t, w, h = det[0]
                conf = float(det[1])
                rows.append([l, t, l + w, t + h, conf])
            dets = np.asarray(rows, dtype=np.float64)
        arr = self._sort.update(dets)
        if arr.size == 0:
            return []
        out = []
        for row in np.atleast_2d(arr):
            x1, y1, x2, y2, tid = row[0], row[1], row[2], row[3], row[4]
            out.append(SortTrack(int(tid), [x1, y1, x2, y2]))
        return out
