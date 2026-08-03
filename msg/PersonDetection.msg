# Single person detection with 3D position and bounding box

float64 x          # 3D X in camera optical frame (m, positive = right)
float64 y          # 3D Y in camera optical frame (m, positive = down)
float64 z          # 3D Z in camera optical frame (m, positive = forward)
float64 confidence # Detection confidence [0.0, 1.0]

# Normalized bounding box [0.0, 1.0] relative to image dimensions
float64 bbox_center_x
float64 bbox_center_y
float64 bbox_width
float64 bbox_height
