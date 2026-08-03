#!/usr/bin/env python3
"""
Shared utilities for the LIMO person following system.

Provides: depth sampling from bounding boxes, 3D projection from camera
intrinsics, velocity ramping with acceleration limits, and bounding box
IoU computation.
"""

import math
import numpy as np


# ---------------------------------------------------------------------------
# Depth sampling
# ---------------------------------------------------------------------------

def sample_depth_in_bbox(depth_image, bbox_center_x, bbox_center_y,
                         bbox_width, bbox_height,
                         depth_scale=0.001, center_fraction=0.3):
    """Extract a robust depth estimate from inside a bounding box.

    Takes the *median* depth value from the central `center_fraction` region
    of the bounding box.  Median guards against zero-depth holes and background
    bleed at box edges.

    Parameters
    ----------
    depth_image : np.ndarray (H, W), uint16
        Raw depth frame (RealSense delivers millimetre values).
    bbox_center_x, bbox_center_y : float
        Normalised [0, 1] bounding-box centre.
    bbox_width, bbox_height : float
        Normalised [0, 1] bounding-box dimensions.
    depth_scale : float
        Multiplier to convert raw depth to metres (default 0.001 for mm→m).
    center_fraction : float
        Fraction of the bbox width/height to sample from the centre.

    Returns
    -------
    depth_m : float or None
        Median depth in metres, or None if all valid samples were zero/unobtainable.
    """
    h, w = depth_image.shape[:2]

    # -- pixel-space bounding box -------------------------------------------
    x1 = int(max(0, (bbox_center_x - bbox_width / 2) * w))
    x2 = int(min(w, (bbox_center_x + bbox_width / 2) * w))
    y1 = int(max(0, (bbox_center_y - bbox_height / 2) * h))
    y2 = int(min(h, (bbox_center_y + bbox_height / 2) * h))

    if x2 <= x1 or y2 <= y1:
        return None

    # -- shrink to centre region --------------------------------------------
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    half_w = int((x2 - x1) * center_fraction / 2)
    half_h = int((y2 - y1) * center_fraction / 2)

    sx1, sx2 = max(x1, cx - half_w), min(x2, cx + half_w)
    sy1, sy2 = max(y1, cy - half_h), min(y2, cy + half_h)

    if sx2 <= sx1 or sy2 <= sy1:
        sx1, sx2, sy1, sy2 = x1, x2, y1, y2  # fall back to full bbox

    roi = depth_image[sy1:sy2, sx1:sx2].astype(np.float32)

    # discard zeros (depth holes) and implausible values
    valid = roi[roi > 0]
    if len(valid) == 0:
        return None

    # Use median — robust to outliers at edges
    median_raw = float(np.median(valid))
    if median_raw <= 0:
        return None

    return median_raw * depth_scale


# ---------------------------------------------------------------------------
# 3D projection
# ---------------------------------------------------------------------------

def pixel_to_3d(u, v, depth_m, camera_info):
    """Project a pixel coordinate + depth to a 3D point in the camera frame.

    Standard pinhole model:
        X = (u - cx) * Z / fx
        Y = (v - cy) * Z / fy
        Z = depth

    Parameters
    ----------
    u, v : float
        Pixel coordinates (can be sub-pixel).
    depth_m : float
        Depth in metres.
    camera_info : sensor_msgs/CameraInfo
        ROS camera-info message containing the intrinsic matrix K.

    Returns
    -------
    (x, y, z) : tuple of float
        3D coordinates in the camera optical frame (metres).
        x = right, y = down, z = forward.
    """
    if depth_m <= 0:
        return None

    # K matrix: [fx, 0, cx; 0, fy, cy; 0, 0, 1]
    fx = camera_info.K[0]
    fy = camera_info.K[4]
    cx = camera_info.K[2]
    cy = camera_info.K[5]

    x = (u - cx) * depth_m / fx
    y = (v - cy) * depth_m / fy
    z = depth_m

    return (x, y, z)


# ---------------------------------------------------------------------------
# Velocity ramping (acceleration limiting)
# ---------------------------------------------------------------------------

def ramp_velocity(target, current, max_accel, dt):
    """Limit velocity change to respect a maximum acceleration.

    Parameters
    ----------
    target : float
        Desired velocity (m/s or rad/s).
    current : float
        Current (last-commanded) velocity.
    max_accel : float
        Maximum allowed change per second (m/s² or rad/s²).
    dt : float
        Time step in seconds.

    Returns
    -------
    float : Ramped velocity.
    """
    delta = target - current
    max_delta = max_accel * dt
    if abs(delta) <= max_delta:
        return target
    return current + math.copysign(max_delta, delta)


# ---------------------------------------------------------------------------
# Bounding-box IoU
# ---------------------------------------------------------------------------

def bbox_iou(a, b):
    """Intersection-over-Union for two axis-aligned bounding boxes.

    Each box is a dict or object with fields:
        bbox_center_x, bbox_center_y, bbox_width, bbox_height
    (all normalised [0, 1]).

    Returns
    -------
    float : IoU in [0, 1].
    """

    def _to_corners(box):
        half_w = box.bbox_width / 2.0
        half_h = box.bbox_height / 2.0
        return (
            box.bbox_center_x - half_w,   # x1
            box.bbox_center_y - half_h,   # y1
            box.bbox_center_x + half_w,   # x2
            box.bbox_center_y + half_h,   # y2
        )

    ax1, ay1, ax2, ay2 = _to_corners(a)
    bx1, by1, bx2, by2 = _to_corners(b)

    # intersection
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h

    if inter_area == 0.0:
        return 0.0

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter_area

    return inter_area / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Clamp helper
# ---------------------------------------------------------------------------

def clamp(value, low, high):
    """Clamp *value* to [low, high]."""
    return max(low, min(high, value))
