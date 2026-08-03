#!/usr/bin/env python3
"""
person_tracker node — Stage 2 of the person-following pipeline.

Wraps the PersonTracker state machine in a ROS node that subscribes
to /person_detections and publishes /target_person.
"""

import math
import rospy
from std_msgs.msg import String
from visualization_msgs.msg import Marker

from limo_person_following.msg import PersonDetections, TargetPerson
from person_following.tracker import PersonTracker


class TrackerNode:
    """ROS wrapper around PersonTracker."""

    def __init__(self):
        rospy.init_node('person_tracker', log_level=rospy.INFO)

        # ---- parameters ---------------------------------------------------
        iou_thresh = rospy.get_param('~tracker/iou_match_threshold', 0.3)
        max_disappeared = rospy.get_param('~tracker/max_disappeared_frames', 15)
        reacquire = rospy.get_param('~tracker/reacquire_timeout', 3.0)
        policy = rospy.get_param('~tracker/initial_target_policy', 'closest')
        publish_markers = rospy.get_param('~debug/publish_markers', True)

        # ---- tracker ------------------------------------------------------
        self._tracker = PersonTracker(
            iou_threshold=iou_thresh,
            max_disappeared=max_disappeared,
            reacquire_timeout=reacquire,
            initial_target_policy=policy,
        )
        # Wire ROS logger into the pure-Python tracker
        self._tracker.info = rospy.loginfo
        self._tracker.warn = rospy.logwarn

        # ---- publishers ---------------------------------------------------
        self._target_pub = rospy.Publisher(
            '/target_person', TargetPerson, queue_size=10)
        self._status_pub = rospy.Publisher(
            '/tracking_status', String, queue_size=10)

        self._marker_pub = None
        if publish_markers:
            self._marker_pub = rospy.Publisher(
                '/target_marker', Marker, queue_size=10)

        # ---- subscriber ---------------------------------------------------
        self._sub = rospy.Subscriber(
            '/person_detections', PersonDetections,
            self._callback, queue_size=10)

        self._last_status = None

        rospy.loginfo("person_tracker node started (iou=%.2f, max_lost=%d)",
                      iou_thresh, max_disappeared)

    def _callback(self, msg):
        detections = msg.detections
        target, state_str, track_id = self._tracker.update(
            detections, now=rospy.Time.now().to_sec())

        # -- publish target ------------------------------------------------
        tmsg = TargetPerson()
        tmsg.header.stamp = msg.header.stamp
        tmsg.header.frame_id = msg.header.frame_id
        tmsg.tracking_id = track_id

        if target is not None:
            tmsg.is_tracking = (state_str == PersonTracker.TRACKING)
            tmsg.position.x = target.x
            tmsg.position.y = target.y
            tmsg.position.z = target.z

            tmsg.distance = math.sqrt(target.x ** 2 + target.z ** 2)
            tmsg.angle = math.atan2(target.x, target.z)
            tmsg.confidence = target.confidence
        else:
            tmsg.is_tracking = False

        self._target_pub.publish(tmsg)

        # -- publish status ------------------------------------------------
        if state_str != self._last_status:
            self._status_pub.publish(String(data=state_str))
            self._last_status = state_str

        # -- publish marker ------------------------------------------------
        if self._marker_pub is not None and target is not None:
            self._publish_marker(target, msg.header, track_id)

    def _publish_marker(self, det, header, track_id):
        marker = Marker()
        marker.header.stamp = header.stamp
        marker.header.frame_id = header.frame_id
        marker.ns = "person_following"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = det.x
        marker.pose.position.y = det.y
        marker.pose.position.z = det.z
        marker.pose.orientation.w = 1.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.15
        marker.color.r = 1.0
        marker.color.g = 0.2
        marker.color.b = 0.2
        marker.color.a = 0.9
        marker.lifetime = rospy.Duration(2.0)
        self._marker_pub.publish(marker)


def main():
    try:
        TrackerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass


if __name__ == '__main__':
    main()
