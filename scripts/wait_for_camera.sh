#!/bin/bash
# Wait for camera topics to become available before starting person following
set -e

echo "Waiting for camera topics..."

# Wait for RGB image topic
echo "  Waiting for /camera/color/image_raw..."
while ! rostopic list 2>/dev/null | grep -q "/camera/color/image_raw"; do
    sleep 1
done
echo "  /camera/color/image_raw is available."

# Wait for depth image topic
echo "  Waiting for /camera/aligned_depth_to_color/image_raw..."
while ! rostopic list 2>/dev/null | grep -q "/camera/aligned_depth_to_color/image_raw"; do
    sleep 1
done
echo "  /camera/aligned_depth_to_color/image_raw is available."

# Wait for camera info topic
echo "  Waiting for /camera/color/camera_info..."
while ! rostopic list 2>/dev/null | grep -q "/camera/color/camera_info"; do
    sleep 1
done
echo "  /camera/color/camera_info is available."

echo "All camera topics are available. Ready."
