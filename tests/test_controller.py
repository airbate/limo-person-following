#!/usr/bin/env python3
"""Unit tests for the controller's control law (offline).

We import ramp_velocity and clamp from utils, then manually exercise
the same P-controller logic that controller_node.py implements.
"""

import os
import sys
import math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from person_following.utils import ramp_velocity, clamp


# Default controller parameters
TARGET_DIST = 1.2
MIN_DIST = 0.6
MAX_DIST = 3.5
DEAD_DIST = 0.15
DEAD_ANGLE = 0.08
KP_LINEAR = 0.6
KP_ANGULAR = 1.2
MAX_LINEAR = 0.5
MAX_ANGULAR = 1.0
MAX_LINEAR_ACC = 0.3
MAX_ANGULAR_ACC = 0.8


def compute_velocity(distance, angle, dt, last_linear=0.0, last_angular=0.0):
    """Replicate the controller logic for testing."""
    # Emergency zones
    if distance < MIN_DIST:
        return (-0.15, 0.0)
    if distance > MAX_DIST:
        return (0.0, 0.0)

    # Linear
    err_lin = distance - TARGET_DIST
    if abs(err_lin) < DEAD_DIST:
        target_lin = 0.0
    else:
        target_lin = clamp(KP_LINEAR * err_lin, -0.2, MAX_LINEAR)

    # Angular
    if abs(angle) < DEAD_ANGLE:
        target_ang = 0.0
    else:
        target_ang = clamp(KP_ANGULAR * angle, -MAX_ANGULAR, MAX_ANGULAR)

    # Cross-coupling
    if abs(angle) > DEAD_ANGLE:
        target_lin *= max(0.3, 1.0 - abs(angle))
    target_ang *= min(1.0, distance / TARGET_DIST)

    # Ramp
    lin = ramp_velocity(target_lin, last_linear, MAX_LINEAR_ACC, dt)
    ang = ramp_velocity(target_ang, last_angular, MAX_ANGULAR_ACC, dt)

    return (lin, ang)


def test_dead_zone():
    """At target distance and centred, command should be zero."""
    lin, ang = compute_velocity(TARGET_DIST, 0.0, 0.05)
    assert lin == 0.0, f"Expected 0 linear in dead zone, got {lin:.3f}"
    assert ang == 0.0, f"Expected 0 angular in dead zone, got {ang:.3f}"
    print("  dead zone OK")


def test_forward():
    """If the target is far away, we should move forward."""
    lin, ang = compute_velocity(2.5, 0.0, 0.05)
    assert lin > 0.0, f"Expected forward when far, got {lin:.3f}"
    assert abs(ang) < 0.01
    print("  forward movement OK")


def test_turn_right():
    """Person offset to the right → positive angular (counter-clockwise)."""
    # x positive = right → positive angle
    angle = math.atan2(0.5, 1.5)  # person at (0.5, 1.5) — right side
    lin, ang = compute_velocity(1.5, angle, 0.05)
    assert ang > 0.0, f"Expected positive angular for right offset, got {ang:.3f}"
    print("  turn right OK")


def test_too_close_backs_up():
    """Within MIN_DIST we back away."""
    lin, ang = compute_velocity(0.3, 0.0, 0.05)
    assert lin < 0.0, f"Expected backwards when too close, got {lin:.3f}"
    assert ang == 0.0
    print("  too-close back-up OK")


def test_too_far_stops():
    """Beyond MAX_DIST we stop."""
    lin, ang = compute_velocity(10.0, 0.0, 0.05)
    assert lin == 0.0
    assert ang == 0.0
    print("  too-far stop OK")


def test_ramp_limits_acceleration():
    """ramp_velocity should limit the change per step."""
    result = ramp_velocity(1.0, 0.0, 0.3, 0.1)  # max 0.03 change per step
    assert result == 0.03, f"Ramp should limit to 0.03, got {result:.3f}"
    print("  ramp acceleration OK")


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(15, 0, 10) == 10
    print("  clamp OK")


if __name__ == '__main__':
    test_dead_zone()
    test_forward()
    test_turn_right()
    test_too_close_backs_up()
    test_too_far_stops()
    test_ramp_limits_acceleration()
    test_clamp()
    print("All controller tests passed.")
