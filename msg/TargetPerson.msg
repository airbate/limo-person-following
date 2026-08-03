# Currently tracked target person

std_msgs/Header header
geometry_msgs/Point position  # 3D position in camera optical frame
float64 distance              # Horizontal distance from camera (m)
float64 angle                 # Horizontal angle from camera center (rad)
bool is_tracking              # True if we have a valid target
float64 confidence            # Detection confidence of current match
int32 tracking_id             # Incremented when a new target is acquired
