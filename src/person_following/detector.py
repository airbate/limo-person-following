#!/usr/bin/env python3
"""
Reusable MobileNet-SSD person detector.

Wraps OpenCV's DNN module so the ROS node (detector_node.py) stays focused
on message passing and frame synchronisation.
"""

import time
import numpy as np
import cv2


class PersonDetector:
    """MobileNet-SSD person detector using OpenCV DNN.

    Parameters
    ----------
    model_path : str
        Path to the .caffemodel weights file.
    config_path : str
        Path to the deploy.prototxt architecture file.
    confidence_threshold : float
        Minimum confidence to accept a detection (default 0.5).
    person_class_id : int
        COCO class ID for "person" (default 15).
    input_size : tuple (width, height)
        Blob input dimensions (default (300, 300)).
    scale_factor : float
        Normalisation factor applied to pixel values.
    mean_values : tuple
        Mean subtraction values (BGR order).
    use_gpu : bool
        Use CUDA backend if available.
    """

    def __init__(self, model_path, config_path,
                 confidence_threshold=0.5, person_class_id=15,
                 input_size=(300, 300), scale_factor=0.007843,
                 mean_values=(127.5, 127.5, 127.5), use_gpu=True):

        self.confidence_threshold = confidence_threshold
        self.person_class_id = person_class_id
        self.input_width, self.input_height = input_size
        self.scale_factor = scale_factor
        self.mean_values = mean_values

        self._net = cv2.dnn.readNetFromCaffe(config_path, model_path)
        self._use_gpu = use_gpu

        if use_gpu:
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        else:
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

        self._last_inference_ms = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, bgr_image):
        """Run person detection on a BGR image.

        Parameters
        ----------
        bgr_image : np.ndarray (H, W, 3), uint8

        Returns
        -------
        list of dict
            Each dict: {'bbox_center_x', 'bbox_center_y', 'bbox_width',
                         'bbox_height', 'confidence'}  (normalised [0,1]).
        """
        h, w = bgr_image.shape[:2]
        t0 = time.time()

        blob = cv2.dnn.blobFromImage(
            bgr_image, self.scale_factor,
            (self.input_width, self.input_height),
            self.mean_values, swapRB=False, crop=False,
        )
        self._net.setInput(blob)
        detections = self._net.forward()

        self._last_inference_ms = (time.time() - t0) * 1000.0

        results = []
        for i in range(detections.shape[2]):
            confidence = float(detections[0, 0, i, 2])
            if confidence < self.confidence_threshold:
                continue
            class_id = int(detections[0, 0, i, 1])
            if class_id != self.person_class_id:
                continue

            # Normalised coordinates [0, 1]
            x1 = detections[0, 0, i, 3]   # left
            y1 = detections[0, 0, i, 4]   # top
            x2 = detections[0, 0, i, 5]   # right
            y2 = detections[0, 0, i, 6]   # bottom

            bbox_w = x2 - x1
            bbox_h = y2 - y1
            cx = x1 + bbox_w / 2.0
            cy = y1 + bbox_h / 2.0

            results.append({
                'bbox_center_x': float(np.clip(cx, 0.0, 1.0)),
                'bbox_center_y': float(np.clip(cy, 0.0, 1.0)),
                'bbox_width': float(np.clip(bbox_w, 0.0, 1.0)),
                'bbox_height': float(np.clip(bbox_h, 0.0, 1.0)),
                'confidence': confidence,
            })

        return results

    @property
    def last_inference_ms(self):
        """Duration of the most recent forward pass in milliseconds."""
        return self._last_inference_ms

    # ------------------------------------------------------------------
    # Visualisation helper (used by detector_node for debug image)
    # ------------------------------------------------------------------

    @staticmethod
    def draw_detections(bgr_image, detections, colour=(0, 255, 0)):
        """Draw bounding boxes and confidence labels on an image (in-place)."""
        h, w = bgr_image.shape[:2]
        for d in detections:
            half_w = int(d['bbox_width'] * w / 2)
            half_h = int(d['bbox_height'] * h / 2)
            cx = int(d['bbox_center_x'] * w)
            cy = int(d['bbox_center_y'] * h)
            x1, y1 = cx - half_w, cy - half_h
            x2, y2 = cx + half_w, cy + half_h
            cv2.rectangle(bgr_image, (x1, y1), (x2, y2), colour, 2)
            label = f"{d['confidence']:.2f}"
            cv2.putText(bgr_image, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)
        return bgr_image
