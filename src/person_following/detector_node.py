#!/usr/bin/env python3
"""
person_detector node — Stage 1 of the person-following pipeline.

Subscribes to synchronised RGB + depth camera streams, runs MobileNet-SSD
person detection at a fixed rate, samples depth inside each bounding box,
projects 2D→3D using camera intrinsics, and publishes the result as
PersonDetections messages.
"""

import math
import rospy
import numpy as np
import cv2
from cv_bridge import CvBridge, CvBridgeError
import message_filters
from sensor_msgs.msg import Image, CameraInfo

from limo_person_following.msg import PersonDetection, PersonDetections
from person_following.detector import PersonDetector
from person_following.utils import sample_depth_in_bbox, pixel_to_3d


class DetectorNode:
    """ROS node that detects people in RGB-D frames."""

    def __init__(self):
        rospy.init_node('person_detector', log_level=rospy.INFO)

        # ---- parameters ---------------------------------------------------
        self._load_params()

        # ---- CV bridge ----------------------------------------------------
        self._bridge = CvBridge()

        # ---- cached frames ------------------------------------------------
        self._latest_bgr = None
        self._latest_depth = None
        self._latest_depth_stamp = None
        self._camera_info = None
        self._frame_lock = None  # will be a threading.Lock

        # ---- detector ----------------------------------------------------
        try:
            self._detector = PersonDetector(
                model_path=self.model_path,
                config_path=self.config_path,
                confidence_threshold=self.conf_thresh,
                person_class_id=self.person_class_id,
                input_size=(self.input_w, self.input_h),
                scale_factor=self.scale_factor,
                mean_values=tuple(self.mean_values),
                use_gpu=self.use_gpu,
            )
            rospy.loginfo("PersonDetector initialised (GPU=%s)", self.use_gpu)
        except Exception as exc:
            rospy.logerr("Failed to load detection model: %s", exc)
            raise

        # ---- publishers ---------------------------------------------------
        self._detections_pub = rospy.Publisher(
            '/person_detections', PersonDetections, queue_size=10)

        self._anno_pub = None
        if self.publish_anno:
            self._anno_pub = rospy.Publisher(
                '/person_detections_image', Image, queue_size=1)

        # ---- subscribers --------------------------------------------------
        rgb_sub = message_filters.Subscriber(self.rgb_topic, Image)
        depth_sub = message_filters.Subscriber(self.depth_topic, Image)

        self._sync = message_filters.ApproximateTimeSynchronizer(
            [rgb_sub, depth_sub], queue_size=5, slop=self.sync_slop)
        self._sync.registerCallback(self._image_callback)

        self._cinfo_sub = rospy.Subscriber(
            self.cinfo_topic, CameraInfo, self._cinfo_callback, queue_size=1)

        # ---- detection timer ----------------------------------------------
        self._detection_timer = rospy.Timer(
            rospy.Duration(1.0 / self.detection_rate),
            self._detection_loop)

        # ---- frame cache lock ---------------------------------------------
        import threading
        self._frame_lock = threading.Lock()

        rospy.loginfo("person_detector node started (%.1f Hz detection)",
                      self.detection_rate)

    # ------------------------------------------------------------------
    # Parameter loading
    # ------------------------------------------------------------------

    def _load_params(self):
        """Read all parameters from the ROS param server with defaults."""
        # -- detection --
        self.model_path = rospy.get_param('~detector/model_path', '')
        self.config_path = rospy.get_param('~detector/config_path', '')
        self.conf_thresh = rospy.get_param('~detector/confidence_threshold', 0.5)
        self.person_class_id = rospy.get_param('~detector/person_class_id', 15)
        self.input_w = rospy.get_param('~detector/input_width', 300)
        self.input_h = rospy.get_param('~detector/input_height', 300)
        self.detection_rate = rospy.get_param('~detector/detection_rate', 8.0)
        self.use_gpu = rospy.get_param('~detector/use_gpu', True)
        self.depth_scale = rospy.get_param('~detector/depth_scale', 0.001)
        self.depth_frac = rospy.get_param('~detector/depth_bbox_fraction', 0.3)
        self.scale_factor = rospy.get_param('~detector/scale_factor', 0.007843)
        self.mean_values = rospy.get_param('~detector/mean_values',
                                           [127.5, 127.5, 127.5])

        # -- height filter --
        self.height_min = rospy.get_param('~controller/height_min', -1.5)
        self.height_max = rospy.get_param('~controller/height_max', 2.0)

        # -- camera --
        self.rgb_topic = rospy.get_param('~camera/rgb_topic',
                                         '/camera/color/image_raw')
        self.depth_topic = rospy.get_param('~camera/depth_topic',
                                           '/camera/aligned_depth_to_color/image_raw')
        self.cinfo_topic = rospy.get_param('~camera/camera_info_topic',
                                           '/camera/color/camera_info')
        self.sync_slop = rospy.get_param('~camera/sync_slop', 0.1)

        # -- debug --
        self.publish_anno = rospy.get_param('~debug/publish_annotated_image', True)

    # ------------------------------------------------------------------
    # Subscriber callbacks
    # ------------------------------------------------------------------

    def _image_callback(self, rgb_msg, depth_msg):
        """Cache the latest synchronised RGB+D frame pair."""
        with self._frame_lock:
            self._latest_bgr = rgb_msg
            self._latest_depth = depth_msg

    def _cinfo_callback(self, msg):
        self._camera_info = msg

    # ------------------------------------------------------------------
    # Detection loop (timer-driven)
    # ------------------------------------------------------------------

    def _detection_loop(self, _event):
        """Run detection on the most recent cached frame pair."""
        # -- grab frames under lock -----------------------------------------
        with self._frame_lock:
            rgb_msg = self._latest_bgr
            depth_msg = self._latest_depth

        if rgb_msg is None or depth_msg is None:
            return  # no frames yet

        if self._camera_info is None:
            rospy.logwarn_throttle(10, "Waiting for camera_info...")
            return

        # -- decode images --------------------------------------------------
        try:
            bgr = self._bridge.imgmsg_to_cv2(rgb_msg, desired_encoding='bgr8')
            depth = self._bridge.imgmsg_to_cv2(
                depth_msg, desired_encoding='passthrough')
        except CvBridgeError as exc:
            rospy.logerr("cv_bridge error: %s", exc)
            return

        h, w = bgr.shape[:2]

        # -- run detection --------------------------------------------------
        raw_detections = self._detector.detect(bgr)

        # -- 2D → 3D -------------------------------------------------------
        person_detections = []
        for d in raw_detections:
            # estimate depth
            depth_m = sample_depth_in_bbox(
                depth, d['bbox_center_x'], d['bbox_center_y'],
                d['bbox_width'], d['bbox_height'],
                depth_scale=self.depth_scale,
                center_fraction=self.depth_frac,
            )

            if depth_m is None or depth_m <= 0.1:
                continue  # no usable depth

            # pixel centre of the bounding box
            u = d['bbox_center_x'] * w
            v = d['bbox_center_y'] * h
            pt_3d = pixel_to_3d(u, v, depth_m, self._camera_info)
            if pt_3d is None:
                continue

            x, y, z = pt_3d

            # height sanity filter – ignore detections at implausible heights
            if y < self.height_min or y > self.height_max:
                continue

            # build ROS message
            pd = PersonDetection()
            pd.x = x
            pd.y = y
            pd.z = z
            pd.confidence = d['confidence']
            pd.bbox_center_x = d['bbox_center_x']
            pd.bbox_center_y = d['bbox_center_y']
            pd.bbox_width = d['bbox_width']
            pd.bbox_height = d['bbox_height']
            person_detections.append(pd)

        # -- publish --------------------------------------------------------
        detections_msg = PersonDetections()
        detections_msg.header.stamp = rgb_msg.header.stamp
        detections_msg.header.frame_id = self._camera_info.header.frame_id
        detections_msg.detections = person_detections
        self._detections_pub.publish(detections_msg)

        # -- debug: annotated image -----------------------------------------
        if self._anno_pub is not None and self.publish_anno:
            self._publish_annotated(bgr, raw_detections, person_detections,
                                    rgb_msg.header)

        # -- log ------------------------------------------------------------
        rospy.logdebug("Detector: %d raw, %d with depth (%.1f ms)",
                       len(raw_detections), len(person_detections),
                       self._detector.last_inference_ms)

    def _publish_annotated(self, bgr, raw_dets, person_dets, header):
        """Draw bounding boxes and publish the annotated image."""
        vis = bgr.copy()

        # draw raw detections in yellow
        PersonDetector.draw_detections(vis, raw_dets, colour=(0, 255, 255))

        # overlay 3D position text for those with valid depth
        h, w = vis.shape[:2]
        for pd in person_dets:
            cx = int(pd.bbox_center_x * w)
            cy = int(pd.bbox_center_y * h)
            label = f"d={math.sqrt(pd.x**2+pd.z**2):.1f}m"
            cv2.putText(vis, label, (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        try:
            anno_msg = self._bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            anno_msg.header = header
            self._anno_pub.publish(anno_msg)
        except CvBridgeError as exc:
            rospy.logerr("Failed to publish annotated image: %s", exc)


# ======================================================================
# Main
# ======================================================================

def main():
    try:
        DetectorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except Exception as exc:
        rospy.logfatal("person_detector crashed: %s", exc)
        raise


if __name__ == '__main__':
    main()
