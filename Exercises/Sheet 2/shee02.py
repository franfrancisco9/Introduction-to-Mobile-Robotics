import math
import numpy as np
from matplotlib import pyplot as plt

scan = np.loadtxt('laserscan.dat')
angle = np.linspace(-math.pi/2, math.pi/2,
np.shape(scan)[0], endpoint='true')

x = scan * np.cos(angle)
y = scan * np.sin(angle)

# A robot is located at x = 1.0 m, y = 0.5 m, θ = π/4
# Its laser range finder is mounted
# on the robot at x = 0.2 m, y = 0.0 m, θ = π with respect to the robot’s frame of
# reference
# (a) Use Python to plot all laser end-points in the frame of reference of the laser
# range finder.


#plt.plot(x, y, '.k', markersize=3)
#plt.gca().set_aspect('equal')
#plt.show()

# (b) The provided scan exhibits an unexpected property. Identify it an suggest an
# explanation.



# (c) Use homogeneous transformation matrices in Python to compute and plot the
# center of the robot, the center of the laser range finder, and all laser end-points
# in world coordinates.

robot_pose_world_frame = (1.0, 0.5, math.pi/4)
laser_pose_robot_frame = (0.2, 0.0, math.pi)

matrix_robot_to_world = np.array(
    [[math.cos(robot_pose_world_frame[2]), -math.sin(robot_pose_world_frame[2]), robot_pose_world_frame[0]],
     [math.sin(robot_pose_world_frame[2]), math.cos(robot_pose_world_frame[2]), robot_pose_world_frame[1]],
        [0, 0, 1]])

matrix_laser_to_robot = np.array(	
    [[math.cos(laser_pose_robot_frame[2]), -math.sin(laser_pose_robot_frame[2]), laser_pose_robot_frame[0]],
        [math.sin(laser_pose_robot_frame[2]), math.cos(laser_pose_robot_frame[2]), laser_pose_robot_frame[1]],
        [0, 0, 1]])

matrix_laser_to_world = np.matmul(matrix_robot_to_world, matrix_laser_to_robot)

# Plot the laser points in the world coordinate frame
# convert the laser points to the world coordinate frame
laser_points_world_frame = np.zeros((np.shape(scan)[0], 2))
for i in range(np.shape(scan)[0]):
    laser_points_world_frame[i] = np.matmul(matrix_laser_to_world, np.array([scan[i]*math.cos(angle[i]), scan[i]*math.sin(angle[i]), 1]))[:2]

plt.plot(laser_points_world_frame[:, 0], laser_points_world_frame[:, 1], '.k', markersize=3)

# plot robot in world frame as a blue dot
plt.plot(robot_pose_world_frame[0], robot_pose_world_frame[1], '+b')

# plot the laser in world fram as a red dot
laser_pose_world_frame = np.matmul(matrix_laser_to_world, np.array([0, 0, 1]))[:2]
plt.plot(laser_pose_world_frame[0], laser_pose_world_frame[1], '+r')
plt.show()
