import pytest

from ghats.filters import (
    background_suppressed_score,
    iou,
    kth_frame_score,
    max_score,
    spatial_agreement_score,
)

LEFT = [0.0, 0.0, 0.4, 0.4]
RIGHT = [0.6, 0.6, 1.0, 1.0]


def test_iou():
    assert iou(LEFT, LEFT) == pytest.approx(1.0)
    assert iou(LEFT, RIGHT) == 0.0
    assert iou([0, 0, 0.2, 0.2], [0.1, 0, 0.3, 0.2]) == pytest.approx(1 / 3)


def test_max_score():
    assert max_score([[(0.2, LEFT)], [], [(0.9, RIGHT), (0.1, LEFT)]]) == 0.9
    assert max_score([[], []]) == 0.0


def test_kth_frame_score_needs_k_frames():
    frames = [[(0.9, LEFT)], [(0.3, LEFT)], []]
    assert kth_frame_score(frames, k=2) == 0.3
    assert kth_frame_score(frames, k=3) == 0.0


def test_kth_frame_score_short_event_falls_back_to_max():
    assert kth_frame_score([[(0.7, LEFT)]], k=2) == 0.7
    assert kth_frame_score([], k=2) == 0.0


def test_spatial_agreement_needs_same_place():
    same_place = [[(0.9, LEFT)], [(0.5, LEFT)]]
    elsewhere = [[(0.9, LEFT)], [(0.5, RIGHT)]]
    assert spatial_agreement_score(same_place, k=2) == 0.5
    assert spatial_agreement_score(elsewhere, k=2) == 0.0


def test_background_suppression_drops_known_spots():
    frames = [[(0.9, LEFT), (0.4, RIGHT)]]
    assert background_suppressed_score(frames, other_event_boxes=[LEFT]) == 0.4
    assert background_suppressed_score(frames, other_event_boxes=[]) == 0.9
    assert background_suppressed_score(frames, other_event_boxes=[LEFT], min_seen=2) == 0.9
