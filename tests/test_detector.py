#!/usr/bin/env python3
"""Unit tests for the PersonDetector class (offline — no ROS required)."""

import os
import sys
import numpy as np
import cv2

# Add package to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from person_following.detector import PersonDetector


def test_cpu_fallback():
    """Smoke-test: detector initialises and produces output on CPU."""
    # Build paths relative to this file
    pkg_dir = os.path.join(os.path.dirname(__file__), '..')
    model = os.path.join(pkg_dir, 'models', 'MobileNetSSD_deploy.caffemodel')
    proto = os.path.join(pkg_dir, 'models', 'MobileNetSSD_deploy.prototxt')

    if not os.path.exists(model):
        print("SKIP: model file not found. Run download_model.sh first.")
        return

    detector = PersonDetector(model, proto, use_gpu=False)
    assert detector is not None

    # Create a synthetic dark image (should find zero people)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    results = detector.detect(img)
    # On a blank image we expect zero or very-low-confidence detections
    assert isinstance(results, list)
    print(f"  CPU inference on blank image: {len(results)} detections "
          f"({detector.last_inference_ms:.1f} ms)")


def test_draw_detections_no_crash():
    """draw_detections should not raise."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    fake = [{'bbox_center_x': 0.5, 'bbox_center_y': 0.5,
             'bbox_width': 0.3, 'bbox_height': 0.4, 'confidence': 0.9}]
    PersonDetector.draw_detections(img, fake)
    # Should have drawn something (red bounding box)
    assert np.any(img > 0), "No pixels modified by draw_detections"
    print("  draw_detections OK")


if __name__ == '__main__':
    test_cpu_fallback()
    test_draw_detections_no_crash()
    print("All detector tests passed.")
