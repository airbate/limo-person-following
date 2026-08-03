#!/usr/bin/env python3
"""
Person target tracker — pure-Python state machine (no ROS dependency).

Implements a three-state machine (ACQUIRING / TRACKING / LOST) that
selects and maintains a single target person across detection frames
using bounding-box IoU matching.
"""

import time
from person_following.utils import bbox_iou


class PersonTracker:
    """Stateful tracker that selects and follows one target person.

    Parameters
    ----------
    iou_threshold : float
        Minimum IoU to consider two bounding boxes the same person.
    max_disappeared : int
        Consecutive frames without a match before transitioning to LOST.
    reacquire_timeout : float
        Seconds to attempt re-acquisition before starting fresh.
    initial_target_policy : str
        'closest' (min Z) or 'largest_bbox'.
    """

    ACQUIRING = 'ACQUIRING'
    TRACKING = 'TRACKING'
    LOST = 'LOST'

    def __init__(self, iou_threshold=0.3, max_disappeared=15,
                 reacquire_timeout=3.0, initial_target_policy='closest'):
        self.iou_threshold = iou_threshold
        self.max_disappeared = max_disappeared
        self.reacquire_timeout = reacquire_timeout
        self.initial_target_policy = initial_target_policy

        # internal state
        self._state = self.ACQUIRING
        self._target_bbox = None      # last known bbox of tracked person
        self._target_det = None       # last full detection
        self._disappeared = 0
        self._lost_since = None       # time.time() when LOST was entered
        self._tracking_id = 0

        # optional: external logger (set by ROS node)
        self.info = _default_info
        self.warn = _default_warn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def state(self):
        return self._state

    def update(self, detections, now=None):
        """Process a new detection array and return the current target.

        Parameters
        ----------
        detections : list of detection objects with
            .bbox_center_x, .bbox_center_y, .bbox_width, .bbox_height, .z
        now : float or None
            Current time in seconds (e.g. time.time()).  Used for the
            re-acquire timeout.  If None, ``time.time()`` is called.

        Returns
        -------
        (target_detection, state_str, tracking_id)
            target_detection is the matched detection or None.
        """
        if now is None:
            now = time.time()

        if not detections:
            return self._handle_empty(now)

        if self._state == self.ACQUIRING:
            return self._handle_acquiring(detections)
        elif self._state == self.TRACKING:
            return self._handle_tracking(detections, now)
        else:  # LOST
            return self._handle_lost(detections, now)

    # ------------------------------------------------------------------
    # State handlers
    # ------------------------------------------------------------------

    def _handle_empty(self, now):
        """No detections this frame."""
        if self._state == self.TRACKING:
            self._disappeared += 1
            if self._disappeared >= self.max_disappeared:
                self._enter_lost(now)
                return (None, self.LOST, self._tracking_id)
        return (self._target_det, self._state, self._tracking_id)

    def _handle_acquiring(self, detections):
        """Select the initial target from available detections."""
        if self.initial_target_policy == 'closest':
            best = min(detections, key=lambda d: d.z)
        else:
            best = max(detections, key=lambda d: d.bbox_width * d.bbox_height)

        self._target_det = best
        self._target_bbox = best
        self._disappeared = 0
        self._tracking_id += 1
        self._state = self.TRACKING

        self.info("Tracker: ACQUIRED target id=%d (z=%.2f m, conf=%.2f)",
                  self._tracking_id, best.z, best.confidence)
        return (best, self.TRACKING, self._tracking_id)

    def _handle_tracking(self, detections, now):
        """Match current detections against the tracked target."""
        best_iou = 0.0
        best_idx = -1

        for i, d in enumerate(detections):
            if self._target_bbox is not None:
                iou = bbox_iou(self._target_bbox, d)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i

        if best_iou >= self.iou_threshold:
            # match found
            self._target_det = detections[best_idx]
            self._target_bbox = detections[best_idx]
            self._disappeared = 0
            return (self._target_det, self.TRACKING, self._tracking_id)

        # no match
        self._disappeared += 1
        if self._disappeared >= self.max_disappeared:
            self._enter_lost(now)
            self.warn("Tracker: LOST target id=%d after %d frames",
                      self._tracking_id, self._disappeared)
            return (None, self.LOST, self._tracking_id)

        # still within tolerance – keep last known position
        return (self._target_det, self.TRACKING, self._tracking_id)

    def _handle_lost(self, detections, now):
        """Try to re-acquire the lost target, or start fresh."""
        if self._lost_since is not None:
            elapsed = now - self._lost_since
            if elapsed > self.reacquire_timeout:
                # re-acquire timeout: pick a new target
                self._state = self.ACQUIRING
                self.info("Tracker: re-acquire timeout, picking new target")
                return self._handle_acquiring(detections)

        # still within re-acquire window – match against last bbox
        best_iou = 0.0
        best_det = None
        for d in detections:
            iou = bbox_iou(self._target_bbox, d) if self._target_bbox else 0.0
            if iou > best_iou:
                best_iou = iou
                best_det = d

        if best_iou >= self.iou_threshold:
            self._target_det = best_det
            self._target_bbox = best_det
            self._disappeared = 0
            self._state = self.TRACKING
            self.info("Tracker: RE-ACQUIRED target id=%d", self._tracking_id)
            return (best_det, self.TRACKING, self._tracking_id)

        return (None, self.LOST, self._tracking_id)

    def _enter_lost(self, now):
        self._state = self.LOST
        self._lost_since = now


# ------------------------------------------------------------------
# Default loggers (ROS-free)
# ------------------------------------------------------------------

def _default_info(fmt, *args):
    pass  # silent by default in pure-Python mode

def _default_warn(fmt, *args):
    pass
