import numpy as np
from PIL import Image

from ghats.edge import decode, letterbox, nms, to_detections


def test_letterbox_centres_wide_frame_on_grey():
    x, rev = letterbox(Image.new("RGB", (1280, 720), (255, 0, 0)))
    assert x.shape == (3, 640, 640) and x.dtype == np.float32
    assert rev.tolist() == [0.5, 0, 140]
    assert np.allclose(x[:, 0, 0], 114 / 255)  # padding above
    assert np.allclose(x[:, 320, 320], [1, 0, 0])  # image


def test_nms_drops_overlaps_and_keeps_score_order():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 10, 10], [20, 20, 30, 30]], dtype=np.float32)
    keep = nms(boxes, np.array([0.5, 0.9, 0.7]), iou=0.5)
    assert keep.tolist() == [1, 2]


def test_decode_is_per_class_and_filters_low_scores():
    boxes = np.array([[0, 0, 10, 10], [0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32)
    scores = np.array([[0.9, 0.0], [0.0, 0.8], [0.005, 0.0]], dtype=np.float32)
    rows = decode(scores, boxes, min_conf=0.01)
    # Same box, different classes: both kept. The third is below the floor.
    assert rows[:, 0].tolist() == [0, 1]
    assert np.allclose(rows[:, 5], [0.9, 0.8])
    assert decode(np.zeros((3, 2), np.float32), boxes).shape == (0, 6)


def test_to_detections_undoes_letterbox_and_drops_padding_boxes():
    rev = np.array([0.5, 0, 140], dtype=np.float32)  # 1280x720 frame
    rows = np.array(
        [
            [0, 0, 140, 640, 500, 0.9],  # the whole frame
            [0, 100, 0, 200, 100, 0.2],  # wholly in the top padding
        ],
        dtype=np.float32,
    )
    dets = to_detections(rows, rev, 1280, 720)
    assert len(dets) == 1
    assert dets[0].category == "animal" and dets[0].bbox == [0.0, 0.0, 1.0, 1.0]
