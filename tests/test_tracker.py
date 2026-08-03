#!/usr/bin/env python3
"""Unit tests for PersonTracker state machine."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from person_following.tracker import PersonTracker
from person_following.utils import bbox_iou


class FakeDet:
    """Minimal stub that behaves like a PersonDetection for the tracker."""
    def __init__(self, cx, cy, w, h, x, y, z, conf=0.8):
        self.bbox_center_x = cx
        self.bbox_center_y = cy
        self.bbox_width = w
        self.bbox_height = h
        self.x = x
        self.y = y
        self.z = z
        self.confidence = conf


def make_det(cx=0.5, cy=0.5, w=0.2, h=0.4, x=0.0, y=0.0, z=1.5, conf=0.8):
    return FakeDet(cx, cy, w, h, x, y, z, conf)


def test_acquiring_picks_closest():
    tracker = PersonTracker(initial_target_policy='closest')
    dets = [
        make_det(z=3.0),
        make_det(z=1.2),  # closest
        make_det(z=2.0),
    ]
    target, state, tid = tracker.update(dets)
    assert state == PersonTracker.TRACKING
    assert target.z == 1.2
    assert tid == 1
    print("  acquiring → closest OK")


def test_tracking_maintains_id():
    tracker = PersonTracker(iou_threshold=0.3)
    # acquire
    det_a = make_det(cx=0.5, cy=0.5, z=1.5)
    target, state, tid = tracker.update([det_a])
    assert tid == 1

    # slight movement — same person
    det_b = make_det(cx=0.52, cy=0.48, z=1.6)
    target, state, tid = tracker.update([det_b])
    assert state == PersonTracker.TRACKING
    assert tid == 1, "Should keep same tracking ID"
    print("  tracking maintains ID OK")


def test_lost_after_disappeared():
    tracker = PersonTracker(max_disappeared=3)
    # acquire
    det_a = make_det(cx=0.5, cy=0.5, z=1.5)
    tracker.update([det_a])
    assert tracker.state == PersonTracker.TRACKING

    # empty frames
    for _ in range(3):
        target, state, tid = tracker.update([])

    assert state == PersonTracker.LOST
    print("  lost after N empty frames OK")


def test_reacquire_after_brief_loss():
    tracker = PersonTracker(max_disappeared=10, reacquire_timeout=99.0)
    # acquire
    det_a = make_det(cx=0.5, cy=0.5, z=1.5)
    tracker.update([det_a])

    # lose for a few frames
    for _ in range(5):
        tracker.update([])
    assert tracker.state == PersonTracker.TRACKING  # still within tolerance

    # same person reappears at similar position
    det_b = make_det(cx=0.51, cy=0.49, z=1.55)
    target, state, tid = tracker.update([det_b])
    assert state == PersonTracker.TRACKING
    assert tid == 1
    print("  re-acquire after brief loss OK")


def test_iou_edge_cases():
    """Verify IoU computation with known shapes."""
    # Identical boxes
    a = make_det(cx=0.5, cy=0.5, w=0.2, h=0.4)
    b = make_det(cx=0.5, cy=0.5, w=0.2, h=0.4)
    assert bbox_iou(a, b) == 1.0

    # Disjoint boxes
    a = make_det(cx=0.2, cy=0.2, w=0.1, h=0.1)
    b = make_det(cx=0.8, cy=0.8, w=0.1, h=0.1)
    assert bbox_iou(a, b) == 0.0

    # 50% overlap
    a = make_det(cx=0.5, cy=0.5, w=0.4, h=0.4)
    b = make_det(cx=0.7, cy=0.5, w=0.4, h=0.4)
    iou = bbox_iou(a, b)
    assert 0.25 < iou < 0.35, f"Expected ~0.29 IoU, got {iou:.3f}"
    print("  IoU edge cases OK")


if __name__ == '__main__':
    test_acquiring_picks_closest()
    test_tracking_maintains_id()
    test_lost_after_disappeared()
    test_reacquire_after_brief_loss()
    test_iou_edge_cases()
    print("All tracker tests passed.")
