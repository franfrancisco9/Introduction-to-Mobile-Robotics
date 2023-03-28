'''
Exercise 1: Locomotion
A robot equipped with a differential drive starts at position x = 1.0 m, y = 2.0 m
and with heading θ = π/2.
It has to move to the position x = 1.5 m, y = 2.0 m, θ = π/2(all angles in radians).
The movement of the vehicle is described by steering
commands (vl = speed of left wheel, vr = speed of right wheel, t = driving time).
(a) What is the minimal number of steering commands (vl,vr,t) needed to guide
the vehicle to the desired target location?
(b) What is the length of the shortest trajectory under this constraint?
(c) Which sequence of steering commands guides the robot on the shortest trajec-
tory to the desired location if an arbitrary number of steering commands can
be used? The maximum velocity of each wheel is v and the distance between
both wheels is l.
(d) What is the length of this trajectory?
Note: the length of a trajectory refers to the traveled distance along the trajectory.

Exercise 2: Differential Drive Implementation
Write a function in Python that implements the forward kinematics for the differ-
ential drive as explained in the lecture.
(a) The function header should look like
def diffdrive(x, y, theta, v l, v r, t, l):
return x n, y n, theta n
where x, y, and θ is the pose of the robot, vl and vr are the speed of the left
and right wheel, t is the driving time, and l is the distance between the wheels
of the robot. The output of the function is the new pose of the robot xn, yn,
and θn.
'''
import sys
import numpy as np
import matplotlib.pyplot as plt
# import animation
from matplotlib import animation


# define a plot trajectory function that gets a list of commands (vl, vr, t)
# the pose0 (x0, y0, theta0), the distance between the wheels l and
# the desired pose (x, y, theta)
def plot_trajectory(pose_desired, pose0, commands, l):
    # plot trajectory and in yellow show the path between commands
    # use an arrow to show the orientation of the robot
    plt.figure()
    plt.plot(pose_desired[0], pose_desired[1], 'ro')
    plt.plot(pose0[0], pose0[1], 'go')
    plt.arrow(pose0[0], pose0[1], 0.05 * np.cos(pose0[2]), 0.05 * np.sin(pose0[2]), color='g')
    plt.arrow(pose_desired[0], pose_desired[1], 0.05 * np.cos(pose_desired[2]), 0.05 * np.sin(pose_desired[2]), color='r')
    plt.axis('equal')
    plt.grid()
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Trajectory')

    # plot the path between commands using 25 steps
    x = pose0[0]
    y = pose0[1]
    theta = pose0[2]
    for command in commands:
        for i in range(25):
            x, y, theta = diffdrive(x, y, theta, command[0], command[1], command[2] / 25, l)
            plt.plot(x, y, 'y.')
            # plot the orientation of the robot
            plt.arrow(x, y, 0.05 * np.cos(theta), 0.05 * np.sin(theta), color='y')
    # plot final point in blue and orientation
    plt.plot(x, y, 'b.')
    plt.arrow(x, y, 0.05 * np.cos(theta), 0.05 * np.sin(theta), color='b')
    
        
    plt.show()




def diffdrive(x, y, theta, vl, vr, t, l):
    # differential drive forward kinematics
    # x, y, theta: current pose
    # vl, vr: wheel velocities
    # t: time
    # l: distance between wheels
    # returns: new pose

    w = (vr - vl) / l
    # if w is zero, the robot is moving straight
    if w == 0:
        x_n = x + vl * t * np.cos(theta)
        y_n = y + vl * t * np.sin(theta)
        theta_n = theta
        return x_n, y_n, theta_n

    r = l * (vr + vl) / (2 * (vr - vl))
    # rotation around z-axis
    rotation_matrix = np.array([[np.cos(w * t), -np.sin(w * t), 0],
                                [np.sin(w * t), np.cos(w * t), 0],
                                [0, 0, 1]])
    # icc is [x-r * sin(theta), y + r * cos(theta)]
    icc = np.array([[x - r * np.sin(theta)],
                    [y + r * np.cos(theta)]])
    # vector to multiply with rotation matrix
    vector = np.array([[x - icc[0, 0]],
                          [y - icc[1, 0]],
                            [theta]])
    # vector after rotation
    vector_rotated = np.dot(rotation_matrix, vector)
    # new pose
    xn = vector_rotated[0, 0] + icc[0, 0]
    yn = vector_rotated[1, 0] + icc[1, 0]
    thetan = vector_rotated[2, 0] + w * t

    return xn, yn, thetan



def exercise():
    # Exercise 2
    # initial pose
    x0 = 1.0
    y0 = 2.0
    theta0 = np.pi / 2

    # desired pose
    x = 1.5
    y = 2.0
    theta = np.pi / 2

    l = 0.05
    # first command
    # lets assume we do not know l
    # R = l * (vr + vl) / (2 * (vr - vl))
    # We want R = 0.25 and circle to the right
    # R * 2 * (vr - vl) = l * (vr + vl)
    # vr (R * 2 - l) = vl (R * 2 + l)
    # vr = vl * (R * 2 + l) / (R * 2 - l)

    # assume vr = -1
    vr = -1
    vl = vr * (0.25 * 2 + l) / (0.25 * 2 - l)

    # with w = (vr - vl) / l and w = 2 * pi / t 
    w = (vr - vl) / l
    t = 2 * np.pi / w
    command1 = [vl, vr, t/2]
    # second command
    # we need to turn around the same point until our pose is the same as the desired pose
    # first we need to calculate the pose obtained after the first command
    pose = diffdrive(x0, y0, theta0, vl, vr, t, l)
    # now we need to calculate the angle between the current pose and the desired pose
    
    diff_angle = pose[2] - theta

    # to turn in same point we must have vr = -vl
    # w = (vr - vl) / l = rad / s
    # we need to turn diff_angle in t seconds
    # we choose vr = 1
    vr = 1
    vl = -vr
    w = (vr - vl) / l
    t = diff_angle / w
    command2 = [vl, vr, t/2]

    # calculate the lenght of the trajectory
    # first command
    # we calculate half of the circumference of the circle
    # with R = 0.25
    l1 = 0.25 * np.pi
    # second command
    # we calculate half of the circumference of the circle of r = l/2
    l2 = l / 2 * np.pi

    print('Length of the trajectory: ', l1 + l2)

    # plot trajectory
    plot_trajectory([x, y, theta], [x0, y0, theta0], [command1, command2], l)

    # If we have infinite commands we can use very small changes to try and go in a straight line to the derised pose:
    # we can use n commands of only vl with vr = 0 alternated with vr = 1 and vl = 0
    # the time for each command will be t = l * pi
    # the number n of commands will be n = l / (2 * pi)
    # max velocity:
    v = 100
    vr = -v
    vl = -vr
    w = (vr - vl) / l
    t = np.pi / 2 / np.abs(w)
    command1 = [vl, vr, t]
    vr = v
    vl = v
    t = 0.5 / v
    command2 = [vl, vr, t]

    vr = v
    vl = -vr
    w = (vr - vl) / l
    t = np.pi / 2 / np.abs(w)
    command3 = [vl, vr, t]

    commands = [command1, command2, command3]
    # calculate the lenght of the trajectory
    total_lenght = 0.5 + l * np.pi
    print('Length of the trajectory: ', total_lenght)
    # plot trajectory
    plot_trajectory([x, y, theta], [x0, y0, theta0], commands, l)

    # Exercise 2 (b)
    l = 0.5
    # After reaching position x = 1.5 m, y = 2.0 m, and θ = π/2
    #  the robot executes
    # the following sequence of steering commands:
    # (a) c1 = (vl = 0.3 m/s,vr = 0.3 m/s,t = 3s)
    # (b) c2 = (vl = 0.1 m/s,vr = −0.1 m/s,t = 1 s)
    # (c) c3 = (vl = 0.2 m/s,vr = 0 m/s,t = 2 s)
    # Use the function to compute the position of the robot after the execution of
    # each command in the sequence (the distance l between the wheels of the robot
    # is 0.5 m).
    
    # initial pose
    x0 = 1.5
    y0 = 2.0
    theta0 = np.pi / 2

    command1 = [0.3, 0.3, 3]
    command2 = [0.1, -0.1, 1]
    command3 = [0.2, 0, 2]

    commands = [command1, command2, command3]
    # caluclate the final pose
    pose = [x0, y0, theta0]
    for command in commands:
        pose = diffdrive(pose[0], pose[1], pose[2], command[0], command[1], command[2], l)
    print('Final pose: ', pose)

    plot_trajectory(pose, [x0, y0, theta0], commands, l)

# define calculate_commands called in main
def calculate_commands(x, y, theta, x0, y0, theta0, l):
    # calculate the commands to go from pose (x0, y0, theta0) to (x, y, theta)
    # x, y, theta: desired pose
    # x0, y0, theta0: initial pose
    # l: distance between wheels
    # returns: commands

    # first command
    # lets assume we do not know l
    # R = l * (vr + vl) / (2 * (vr - vl))
    # We want R = 0.25 and circle to the right
    # R * 2 * (vr - vl) = l * (vr + vl)
    # vr (R * 2 - l) = vl (R * 2 + l)
    # vr = vl * (R * 2 + l) / (R * 2 - l)

    # calulate R as the distance between the initial pose and the desired pose
    R = np.sqrt((x - x0)**2 + (y - y0)**2) / 2

    # calculate vr and vl as the velocities of the wheels
    vr = 1000
    vl = vr * (R * 2 + l) / (R * 2 - l)

    # with w = (vr - vl) / l and w = 2 * pi / t 
    w = (vr - vl) / l
    t = 2 * np.pi / w
    command1 = [vl, vr, t/2]
    # second command
    # we need to turn around the same point until our pose is the same as the desired pose
    # first we need to calculate the pose obtained after the first command
    pose = diffdrive(x0, y0, theta0, vl, vr, t, l)
    # now we need to calculate the angle between the current pose and the desired pose
    
    diff_angle = pose[2] - theta

    # to turn in same point we must have vr = -vl
    # w = (vr - vl) / l = rad / s
    # we need to turn diff_angle in t seconds
    # we choose vr = 1
    vr = 1
    vl = -vr
    w = (vr - vl) / l
    t = diff_angle * 2 * np.pi  / w
    command2 = [vl, vr, t/2]

    return [command1, command2]


# define animate_robot called in main
def animate_robot(x, y, theta, x0, y0, theta0, commands, l):
    # animate the robot moving from pose (x0, y0, theta0) to (x, y, theta)
    # x, y, theta: desired pose
    # x0, y0, theta0: initial pose
    # commands: list of commands
    # l: distance between wheels

    # plot trajectory
    plot_trajectory([x, y, theta], [x0, y0, theta0], commands, l)

    # create figure
    fig = plt.figure(1)
    ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                         xlim=(-1, 3), ylim=(-1, 3))
    ax.grid()

    # create robot
    robot, = ax.plot([], [], 'o-', lw=2)
    robot_tail, = ax.plot([], [], 'o-', lw=2)
    robot_tail.set_data([x0, x0], [y0, y0])
    robot.set_data([x0, x0], [y0, y0])

    # create desired pose
    desired_pose, = ax.plot([], [], 'o-', lw=2)
    desired_pose.set_data([x], [y])

    # create initial pose
    initial_pose, = ax.plot([], [], 'o-', lw=2)
    initial_pose.set_data([x0], [y0])

    # create trajectory
    trajectory, = ax.plot([], [], 'o-', lw=2)
    trajectory.set_data([x0, x0], [y0, y0])

    # create time text
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)

    







def main():
    # Create a loop where the user can interact with the robot
    # The user can choose between the following options:
    # 1. Move the robot to a desired pose and see the trajectory appear on the screen in real time the robot should follow the trajectory
    # 2. Exit
    # Start code here
    # figure for the plot
    # enter the loop
    while True:
        print('1. Move the robot to a desired pose and see the trajectory appear on the screen in real time the robot should follow the trajectory')
        print('2. Exit')
        option = input('Choose an option: ')
        if option == '1':
            # ask for the desired pose
            x = float(input('x: '))
            y = float(input('y: '))
            theta = float(input('theta: '))
            # initial pose
            x0 = 1.0
            y0 = 2.0
            theta0 = np.pi / 2
            # calculate the commands
            commands = calculate_commands(x, y, theta, x0, y0, theta0, l = 0.4)
            # plot the trajectory
            #plot_trajectory([x, y, theta], [x0, y0, theta0], commands, 0.05)
            # animate the robot and allow the user to send commands to the robot

            animate_robot(x, y, theta, x0, y0, theta0, commands, l = 0.4)
        elif option == '2':
            break
        else:
            print('Invalid option')
    


    # End code here

    
if __name__ == '__main__':
    # take as argument wheter to run main or exercise
    if len(sys.argv) > 1:
        if sys.argv[1] == 'exercise':
            exercise()
        else:
            main()
    else:
        # explain how to run the code
        print('To run the exercise run: python3 robot.py exercise')
        print('To run the main run: python3 robot.py')
