#!/usr/bin/env python3
"""
person_controller node — Stage 3 of the person-following pipeline.

Subscribes to /target_person and converts the target's 3D position into
velocity commands (/cmd_vel) using a decoupled P-controller with dead
zones, safety bounds, acceleration limiting, and a communication watchdog.
"""

import math
import threading
import rospy
from geometry_msgs.msg import Twist

from limo_person_following.msg import TargetPerson, PersonDetections
from person_following.utils import ramp_velocity, clamp


class ControllerNode:
    """ROS node that converts target-person positions to /cmd_vel."""

    def __init__(self):
        rospy.init_node('person_controller', log_level=rospy.INFO)

        self._load_params()

        # ---- state --------------------------------------------------------
        self._last_target = None
        self._last_target_time = rospy.Time.now()
        self._last_linear = 0.0
        self._last_angular = 0.0
        self._last_cmd_time = None
        self._lock = threading.Lock()

        # ---- publisher ----------------------------------------------------
        self._cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # ---- subscribers --------------------------------------------------
        self._target_sub = rospy.Subscriber(
            '/target_person', TargetPerson, self._target_callback, queue_size=10)

        # Emergency-stop subscription: any close detection triggers a halt
        # regardless of which person the tracker is locked onto.
        self._det_sub = rospy.Subscriber(
            '/person_detections', PersonDetections,
            self._detections_callback, queue_size=5)

        # ---- control timer (fixed rate) -----------------------------------
        self._control_timer = rospy.Timer(
            rospy.Duration(1.0 / self.control_rate), self._control_loop)

        rospy.loginfo("person_controller node started "
                      "(target_dist=%.1f m, max_linear=%.1f m/s)",
                      self.target_dist, self.max_linear)

    # ------------------------------------------------------------------
    # Parameter loading
    # ------------------------------------------------------------------

    def _load_params(self):
        self.target_dist = rospy.get_param('~controller/target_distance', 1.2)
        self.min_dist = rospy.get_param('~controller/min_distance', 0.6)
        self.max_dist = rospy.get_param('~controller/max_distance', 3.5)
        self.dead_dist = rospy.get_param('~controller/dead_zone_distance', 0.15)
        self.dead_angle = rospy.get_param('~controller/dead_zone_angle', 0.08)
        self.kp_linear = rospy.get_param('~controller/kp_linear', 0.6)
        self.kp_angular = rospy.get_param('~controller/kp_angular', 1.2)
        self.max_linear = rospy.get_param('~controller/max_linear_speed', 0.5)
        self.max_angular = rospy.get_param('~controller/max_angular_speed', 1.0)
        self.max_linear_acc = rospy.get_param('~controller/max_linear_accel', 0.3)
        self.max_angular_acc = rospy.get_param('~controller/max_angular_accel', 0.8)
        self.back_up_speed = rospy.get_param('~controller/back_up_speed', 0.15)
        self.use_ang_dist_factor = rospy.get_param(
            '~controller/angular_distance_factor', True)
        self.use_lin_ang_factor = rospy.get_param(
            '~controller/linear_angular_factor', True)
        self.watchdog_timeout = rospy.get_param('~controller/watchdog_timeout', 1.0)
        self.control_rate = rospy.get_param('~controller/control_rate', 20.0)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def _target_callback(self, msg):
        with self._lock:
            self._last_target = msg
            self._last_target_time = rospy.Time.now()

    def _detections_callback(self, msg):
        """Check all detections for emergency proximity."""
        for det in msg.detections:
            dist = math.sqrt(det.x ** 2 + det.z ** 2)
            if dist < self.min_dist:
                # Someone is dangerously close – emergency stop
                rospy.logwarn_throttle(
                    2.0,
                    "EMERGENCY STOP: person at %.2f m (min=%.2f m)",
                    dist, self.min_dist)
                self._publish_cmd(0.0, 0.0)
                return

    # ------------------------------------------------------------------
    # Control loop
    # ------------------------------------------------------------------

    def _control_loop(self, _event):
        """Compute and publish velocity at fixed rate."""
        now = rospy.Time.now()

        with self._lock:
            target = self._last_target
            last_time = self._last_target_time

        # ---- watchdog -----------------------------------------------------
        if (target is None or not target.is_tracking or
                (now - last_time).to_sec() > self.watchdog_timeout):
            # No valid target – stop
            if self._last_linear != 0.0 or self._last_angular != 0.0:
                self._publish_cmd(0.0, 0.0)
                rospy.logdebug("Controller: watchdog stop")
            return

        # ---- extract distance & angle ------------------------------------
        distance = target.distance
        angle = target.angle

        # ---- emergency bounds ---------------------------------------------
        if distance < self.min_dist:
            # Too close – back away slowly, no turning
            rospy.logwarn_throttle(2.0, "Too close (%.2f m), backing up", distance)
            self._publish_cmd(-self.back_up_speed, 0.0)
            return

        if distance > self.max_dist:
            # Too far to track reliably – stop
            rospy.logwarn_throttle(5.0, "Target too far (%.2f m), stopping", distance)
            self._publish_cmd(0.0, 0.0)
            return

        # ---- linear velocity ----------------------------------------------
        error_linear = distance - self.target_dist
        if abs(error_linear) < self.dead_dist:
            target_linear = 0.0
        else:
            target_linear = self.kp_linear * error_linear
            # Don't reverse towards the target; negative speed only from
            # the emergency back-up path above.
            target_linear = clamp(target_linear, -0.2, self.max_linear)

        # ---- angular velocity ---------------------------------------------
        if abs(angle) < self.dead_angle:
            target_angular = 0.0
        else:
            target_angular = self.kp_angular * angle
            target_angular = clamp(target_angular,
                                   -self.max_angular, self.max_angular)

        # ---- cross-coupling ------------------------------------------------
        # Slow down linear speed when the angular error is large
        if self.use_lin_ang_factor and abs(angle) > self.dead_angle:
            ang_factor = max(0.3, 1.0 - abs(angle))
            target_linear *= ang_factor

        # Fine-tune angular speed based on distance (gentle steering up close)
        if self.use_ang_dist_factor:
            dist_factor = min(1.0, distance / self.target_dist)
            target_angular *= dist_factor

        # ---- acceleration limiting ----------------------------------------
        dt = 1.0 / self.control_rate
        linear_cmd = ramp_velocity(
            target_linear, self._last_linear, self.max_linear_acc, dt)
        angular_cmd = ramp_velocity(
            target_angular, self._last_angular, self.max_angular_acc, dt)

        # ---- publish ------------------------------------------------------
        self._publish_cmd(linear_cmd, angular_cmd)

    def _publish_cmd(self, linear, angular):
        msg = Twist()
        msg.linear.x = linear
        msg.angular.z = angular
        self._cmd_pub.publish(msg)
        self._last_linear = linear
        self._last_angular = angular
        self._last_cmd_time = rospy.Time.now()


# ======================================================================
# Main
# ======================================================================

def main():
    try:
        ControllerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
