# This message describes one detected person.

std_msgs/Header header

# x coordinate of the face center, scaling similar to y while respecting aspect ratio (eg. from -1.33 to +1.33)
float64 x

# y coordinate of the face center center, from -1 (bottom) to +1 (top)
# (seems most natural for visual servoing)
float64 y

float64 size

# Direction vector to the person
geometry_msgs/Vector3 direction

# Position of the person in camera coordinates
geometry_msgs/Point position
