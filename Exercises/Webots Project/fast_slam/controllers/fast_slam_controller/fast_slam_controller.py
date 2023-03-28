"""pf_controller controller."""

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
import math
import sys
import copy

from controller import Robot
from controller import Supervisor
from controller import Keyboard

from misc_tools import *

MAX_SPEED = 12.3


################## SIMULATOR HELPERS ##########################################

# Normalizes the angle theta in range (-pi, pi)
def normalize_angle(theta):
    if (theta > np.pi):
        return theta - 2*np.pi
    if (theta < -np.pi):
        return theta + 2*np.pi
    return theta

def velFromKeyboard(keyboard):
    turn_base = 3.0
    linear_base = 6.0
    vel_left = 0.0
    vel_right = 0.0
    key = keyboard.getKey()
    while (key != -1):
        if (key==Keyboard.UP):
            vel_left += linear_base
            vel_right += linear_base
        if (key==Keyboard.DOWN):
            vel_left += -linear_base
            vel_right += -linear_base
        if (key==Keyboard.LEFT):
            vel_left += -turn_base
            vel_right += turn_base
        if (key==Keyboard.RIGHT):
            vel_left += turn_base
            vel_right += -turn_base
        key = keyboard.getKey()
    return vel_left, vel_right


def get_curr_pose(trans_field, rot_field):
    values = trans_field.getSFVec3f()
    rob_theta = np.sign(rot_field.getSFRotation()[2])*rot_field.getSFRotation()[3]
    rob_x = values[0]
    rob_y = values[1]
    return [rob_x, rob_y, rob_theta]


def get_pose_delta(last_pose, curr_pose):
    trans_delta = np.sqrt((last_pose[0]-curr_pose[0])**2 + (last_pose[1]-curr_pose[1])**2)
    theta_delta = abs(normalize_angle(last_pose[2]-curr_pose[2]))
    return trans_delta, theta_delta


# Returns the odometry measurement between two poses
# according to the odometry-based motion model.
def get_odometry(last_pose, curr_pose):
    x = last_pose[0]
    y = last_pose[1]
    x_bar = curr_pose[0]
    y_bar = curr_pose[1]
    delta_trans = np.sqrt((x_bar - x) ** 2 + (y_bar - y) ** 2)
    delta_rot = normalize_angle(last_pose[2] - curr_pose[2])
    delta_rot1 = delta_rot / 2.0
    delta_rot2 = delta_rot / 2.0

    if (delta_trans > 0.01):
        delta_rot1 = normalize_angle(math.atan2(y_bar - y, x_bar - x) - last_pose[2])
        delta_rot2 = normalize_angle(curr_pose[2] - last_pose[2] - delta_rot1)

    return [delta_rot1, delta_rot2, delta_trans]

def get_sensor_reading(landmarks, pose):
    px = pose[0]
    py = pose[1]
    ptheta = pose[2]
    
    lm_ids = []
    ranges = []
    bearings = []
    for i in range(len(landmarks)):
        lx_clean = landmarks[i+1][0]
        ly_clean = landmarks[i+1][1]
        noise = np.random.normal(loc=0.0, scale=0.01, size=2)
        lx = lx_clean + noise[0]
        ly = ly_clean + noise[1]
        id = i+1
    
        # calculate expected range measurement
        meas_range = np.sqrt( (lx - px )**2 + (ly - py )**2 )
        meas_bearing = math.atan2(ly - py, lx - px) - ptheta
        
        if meas_range > 5.0:
            continue
        
        lm_ids.append(id)
        ranges.append(meas_range)
        bearings.append(meas_bearing)

    return {'id':lm_ids,'range':ranges,'bearing':bearings}

def initialize_particles(num_particles, num_landmarks, init_pose):
    #initialize particle at pose [0,0,0] with an empty map

    particles = []

    for i in range(num_particles):
        particle = dict()

        #initialize pose: at the beginning, robot is certain it is at [0,0,0]
        particle['x'] = init_pose[0]
        particle['y'] = init_pose[1]
        particle['theta'] = init_pose[2]

        #initial weight
        particle['weight'] = 1.0 / num_particles
        
        #particle history aka all visited poses
        particle['history'] = []

        #initialize landmarks of the particle
        landmarks = dict()

        for i in range(num_landmarks):
            landmark = dict()

            #initialize the landmark mean and covariance 
            landmark['mu'] = [0,0]
            landmark['sigma'] = np.zeros([2,2])
            landmark['observed'] = False

            landmarks[i+1] = landmark

        #add landmarks to particle
        particle['landmarks'] = landmarks

        #add particle to set
        particles.append(particle)

    return particles


################### MEASUREMENT MODEL #########################################


def measurement_model(particle, landmark):
    #Compute the expected measurement for a landmark
    #and the Jacobian with respect to the landmark.

    px = particle['x']
    py = particle['y']
    ptheta = particle['theta']

    lx = landmark['mu'][0]
    ly = landmark['mu'][1]

    #calculate expected range measurement
    meas_range_exp = np.sqrt( (lx - px)**2 + (ly - py)**2 )
    meas_bearing_exp = math.atan2(ly - py, lx - px) - ptheta


    h = np.array([meas_range_exp, meas_bearing_exp])

    # Compute the Jacobian H of the measurement function h 
    #wrt the landmark location
    
    H = np.zeros((2,2)) # Jacobian of h
    x = lx - px 
    y = ly - py
    H[0,0] = x / meas_range_exp
    H[0,1] = y / meas_range_exp
    H[1,0] = -y / (x**2 + y**2)
    H[1,1] = x / (x**2 + y**2)

    return h, H


def eval_sensor_model(sensor_data, particles):
    #Correct landmark poses with a measurement and
    #calculate particle weight

    #sensor noise
    Q_t = np.array([[1.0, 0],\
                    [0, 0.1]])

    #measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']
    bearings = sensor_data['bearing']

    #update landmarks and calculate weight for each particle
    for particle in particles:

        landmarks = particle['landmarks']

        px = particle['x']
        py = particle['y']
        ptheta = particle['theta'] 

        #loop over observed landmarks 
        for i in range(len(ids)):

            #current landmark
            lm_id = ids[i]
            landmark = landmarks[lm_id]
            
            #measured range and bearing to current landmark
            meas_range = ranges[i]
            meas_bearing = bearings[i]

            particle['weight'] = 1.0

            if not landmark['observed']:
                # landmark is observed for the first time
                
                # initialize landmark mean and covariance. You can use the
                # provided function 'measurement_model' above
                # initialize mean using h from 
                # measurement_model and the measured range and bearing
                # initialize covariance using H from
                # measurement_model and the sensor noise Q_t
                # set
                #
                _, H =  measurement_model(particle, landmark)

                lx = px + meas_range * np.cos(meas_bearing + ptheta)
                ly = py + meas_range * np.sin(meas_bearing + ptheta)
                mean = np.array([lx, ly])

                covariance = np.dot(np.linalg.inv(H), np.dot( Q_t , np.linalg.inv(H).T))
                particle['weight'] = 1.0
                landmark['mu'] = mean
                landmark['sigma'] = covariance
                landmark['observed'] = True

            else:
                # landmark was observed before

                # update landmark mean and covariance. You can use the
                # provided function 'measurement_model' above. 
                # calculate particle weight: particle['weight'] = ...
                # update landmarkx
                z_hat, H =  measurement_model(particle, landmark)
                z = np.array([meas_range, meas_bearing]) 
                Q = np.dot(H , np.dot( landmark['sigma'] , H.T)) + Q_t
                K =  np.dot(landmark['sigma'],np.dot(H.T ,np.linalg.inv(Q)))
                #test_mu = landmark['mu']
                mean = np.array(landmark['mu'] + np.dot(K, (z - z_hat)))
                #print(mean)
                covariance = np.dot((np.eye(2) -  np.dot(K , H)) , landmark['sigma'])

                landmark['mu'] = mean
                landmark['sigma'] = covariance

                # calculate particle weight: particle['weight'] = ... 
                det = np.linalg.det(2 * np.pi * (Q)) ** (-1/2)
                weight_particle = det * np.exp(-1/2 * np.dot((z - z_hat).T, np.dot(np.linalg.inv(Q), (z - z_hat))))
                # pprint the likelihood
                #print("likelihood ", weight_particle)
                
                #print("weight_particle: ", weight_particle)
                # if weight_particle == 0:
                #     print("z: ", z)
                #     print("z_hat: ", z_hat)
                #     print("Q: ", Q)
                #     print("px: ", px)
                #     print("py: ", py)
                #     print("mu: ", landmark['mu'])
                #     print("test_mu: ", test_mu)
                #     weight_particle = 1
                particle['weight'] *= weight_particle #* particle['weight']
                #print(particle['weight'])

    #print("Particle weights before normalizing: ", [p['weight'] for p in particles])
    #normalize weights
    normalizer = sum([p['weight'] for p in particles])
    #print("normalizer", normalizer)
    for particle in particles:
        particle['weight'] = particle['weight'] / normalizer
   

###################### MOTION MODEL ###########################################
def sample_motion_model(odometry, particles):
    # Samples new particle positions, based on old positions, the odometry
    # measurements and the motion noise 
    # (odometry based motion model)

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.05, 0.05]

    # standard deviations of motion noise
    sigma_delta_rot1 = noise[0] * abs(delta_rot1) + noise[1] * delta_trans
    
    sigma_delta_trans = noise[2] * delta_trans + \
                        noise[3] * (abs(delta_rot1) + abs(delta_rot2))
    
    sigma_delta_rot2 = noise[0] * abs(delta_rot2) + noise[1] * delta_trans

    # "move" each particle according to the odometry measurements plus sampled noise
    # to generate new particle set 
    
    for particle in particles:
        new_particle = dict()

        #sample noisy motions
        noisy_delta_rot1 = delta_rot1 + np.random.normal(0, sigma_delta_rot1)
        noisy_delta_trans = delta_trans + np.random.normal(0, sigma_delta_trans)
        noisy_delta_rot2 = delta_rot2 + np.random.normal(0, sigma_delta_rot2)

        #calculate new particle pose
        particle['x'] = particle['x'] + \
                    noisy_delta_trans * np.cos(particle['theta'] + noisy_delta_rot1)
        
        particle['y'] = particle['y'] + \
                    noisy_delta_trans * np.sin(particle['theta'] + noisy_delta_rot1)
        
        particle['theta'] = particle['theta'] + \
                    noisy_delta_rot1 + noisy_delta_rot2
        

    



##################### RESAMPLING ##############################################

def resample_particles(particles):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle 
    # weights.
    
    # distance between pointers
    step = 1.0/len(particles)
    
    c = [0.0 for i in range(len(particles))]
    # random start of first pointer
    u = [0.0 for i in range(len(particles))]
    u[0] = np.random.uniform(0, step)
    
    weights = [p['weight'] for p in particles]

    # print("particles_weights: ", [p['weight'] for p in particles])
    # where we are along the weights
    c[0] = weights[0]

    for i in range(1, len(particles)):
        c[i] = c[i-1] + weights[i]
    # index of weight container and corresponding particle 
    i = 0
    
    new_particles = [dict() for i in range(len(particles))]
    
    #loop over all particle weights
    for j in range(len(particles)):

        #go through the weights until you find the particle 
        #to which the pointer points 
        while u[j] > c[i]:
            i = i + 1

        #copy the particle to the new set
        new_particles[j] = copy.deepcopy(particles[i])
        new_particles[j]['weight'] = 1.0/len(particles)

        
        #increase the threshold
        if j < len(particles) - 1:
            u[j + 1] = u[j] + step
        

    # print("new_particles_weights: ", [p['weight'] for p in particles])
    return new_particles



####################### MAIN ##################################################
def main():
    # create the Robot instance.
    robot = Supervisor()
    robot_node = robot.getFromDef("Pioneer3dx")

    # robot pose translation and rotation objects
    trans_field = robot_node.getField("translation")
    rot_field = robot_node.getField("rotation")
    
    # get the time step of the current world.
    timestep = int(robot.getBasicTimeStep())

    # init keyboard readings
    keyboard = Keyboard()
    keyboard.enable(10)
    
    # get wheel motor controllers
    leftMotor = robot.getDevice('left wheel')
    rightMotor = robot.getDevice('right wheel')
    leftMotor.setPosition(float('inf'))
    rightMotor.setPosition(float('inf'))

    # get wheel encoder sensors
    leftSensor = robot.getDevice('left wheel sensor')
    rightSensor = robot.getDevice('right wheel sensor')
    leftSensor.enable(60)
    rightSensor.enable(60)

    # initialize wheel velocities
    leftMotor.setVelocity(0.0)
    rightMotor.setVelocity(0.0)
    
    # create list of landmarks
    n_obs = 9
    landmarks = {}
    for i in range(n_obs):
        obs_name = "Obs_"+str(i+1)
        obs_node = robot.getFromDef(obs_name)
        tr_field = obs_node.getField("translation")
        x = tr_field.getSFVec3f()[0]
        y = tr_field.getSFVec3f()[1]
        landmarks[i+1] = [x, y]
    n_walls = 10
    walls = {}
    for i in range(n_walls):
        wall_name = "wall("+str(i) +")"
        wall_node = robot.getFromDef(wall_name)
        tr_field = wall_node.getField("translation")
        wall_size_field = wall_node.getField("size")
        wall_orientation_field = wall_node.getField("rotation")
        #print(wall_orientation_field)
        x = tr_field.getSFVec3f()[0]
        y = tr_field.getSFVec3f()[1]
        size_x = wall_size_field.getSFVec3f()[0]
        size_z = wall_size_field.getSFVec3f()[1]
        size_y = wall_size_field.getSFVec3f()[2]
        # get rotation angles x, y, z
        orientation_x = wall_orientation_field.getSFRotation()[0]
        orientation_y = wall_orientation_field.getSFRotation()[1]
        orientation_z = wall_orientation_field.getSFRotation()[2]
        walls[i] = [[x, y], [size_x, size_y, size_z], [orientation_x, orientation_y, orientation_z]]

    # get map limits
    ground_node = robot.getFromDef("RectangleArena")
    floor_size_field = ground_node.getField("floorSize")
    fs_x = floor_size_field.getSFVec2f()[0]
    fs_y = floor_size_field.getSFVec2f()[1]
    map_limits = [-fs_x/2.0, fs_x/2.0, -fs_y/2.0, fs_y/2.0]

    # init particles and weights
    num_particles = 100
    num_landmarks = len(landmarks)
    
    # last pose used for odometry calculations
    last_pose = get_curr_pose(trans_field, rot_field)
    particles = initialize_particles(num_particles, num_landmarks, last_pose)

    # translation threshold for odometry calculation
    trans_thr = 0.1

    gt_poses = [last_pose]
    while robot.step(timestep) != -1:
        # key controls
        vel_left, vel_right = velFromKeyboard(keyboard)
        leftMotor.setVelocity(vel_left)
        rightMotor.setVelocity(vel_right)

        # read robot pose and compute difference to last used pose
        curr_pose = get_curr_pose(trans_field, rot_field)
        trans_delta, theta_delta = get_pose_delta(last_pose, curr_pose)

        # skip until translation change is big enough
        if (trans_delta < trans_thr):
            continue
        
        
        # compute odometry
        odom_raw = get_odometry(last_pose, curr_pose)
        last_pose = curr_pose
        gt_poses.append(curr_pose)
        odom_dict = dict()
        odom_dict['r1'] = odom_raw[0]
        odom_dict['r2'] = odom_raw[1]
        odom_dict['t'] = odom_raw[2]
        
        #predict particles by sampling from motion model with odometry info
        sample_motion_model(odom_dict, particles)

        #evaluate sensor model to update landmarks and calculate particle weights
        sensor_reading = get_sensor_reading(landmarks, curr_pose)
        eval_sensor_model(sensor_reading, particles)

        out_of_bounds = 0.00001
        for particle in particles:
            # check if particle is out of bounds of [-fs_x/2.0, fs_x/2.0, -fs_y/2.0, fs_y/2.0] (map_limits)
            # if so, remove particle from particle set
            if particle['x'] < map_limits[0] or particle['x'] > map_limits[1] or particle['y'] < map_limits[2] or particle['y'] > map_limits[3]:
                particle['weight'] = out_of_bounds
                # go to next particle
                continue
            # check if particles is whitin 0.305m of a landmark
            # if so, remove particle from particle set
            for i in range(1, len(landmarks) + 1):
                # get lx and ly from landmark
                landmark = landmarks[i]
                lx = landmark[0]
                ly = landmark[1]
                if np.sqrt((particle['x'] - lx)**2 + (particle['y'] - ly)**2) < 0.305:
                    particle['weight'] = out_of_bounds
                    # go to next particle
                    continue
            # check if particles is in a wall
            # if so, remove particle from particle set
             # walls positions and size
            wx = []
            wy = []
            ww = []
            wh = []
            ox = []
            oy = []
            # wasll is a dict where wall[i] = [[x,y],[w,h],[orientationx, orientationy]]
            for i in range(len(walls)):
                wx.append(walls[i][0][0])
                wy.append(walls[i][0][1])
                ww.append(walls[i][1][0])
                wh.append(walls[i][1][1])
                ox.append(walls[i][2][0])
                oy.append(walls[i][2][1])
            for i in range(0, len(walls)):
                sign = 1
                if walls[i][2][0] < 0 and walls[i][2][1] < 0:
                    sign = -1
                angle=sign*np.arctan2(oy[i],ox[i])/np.pi*180 + 180 -(180-sign*np.arctan2(oy[i],ox[i])/np.pi*180)

            

                # check if particle is in wall takin in account the orientation of the wall
                # calculate the wall limits and check if particle is in the wall by considering the lines that form the wall
                # create a linespace between the limits of the wall
                wall_x_min = wx[i] - ww[i]/2*np.cos(angle*np.pi/180) - wh[i]/2*np.sin(angle*np.pi/180)
                wall_x_max = wx[i] + ww[i]/2*np.cos(angle*np.pi/180) + wh[i]/2*np.sin(angle*np.pi/180)
                wall_y_min = wy[i] - ww[i]/2*np.sin(angle*np.pi/180) + wh[i]/2*np.cos(angle*np.pi/180)
                wall_y_max = wy[i] + ww[i]/2*np.sin(angle*np.pi/180) - wh[i]/2*np.cos(angle*np.pi/180)

                # considering the equation of line y = mx + b check if the particle is in the wall
                # if the particle is in the wall, remove the particle from the particle set
                # find the slope of the wall of the wall from the orientation the small side and the big side
                m = (wall_y_max - wall_y_min)/(wall_x_max - wall_x_min)
                # find the b of the wall
                b = wall_y_min - m*wall_x_min
                # check if the particle is in the wall
                if particle['y'] < m*particle['x'] + b + 0.305 and particle['y'] > m*particle['x'] + b - 0.305:
                    particle['weight'] = out_of_bounds
                    # go to next particle
                    continue

        # renormalize weights
        normalizer = sum([p['weight'] for p in particles])
        #print("normalizer", normalizer)
        for particle in particles:
            particle['weight'] = particle['weight'] / normalizer
   
        #print("Particle weights: ", [p['weight'] for p in particles])
        #plot filter state
        plot_state(particles, landmarks, map_limits, gt_poses, walls)

        #calculate new set of equally weighted particles
        #comment in after implementing resample_particles
        particles = resample_particles(particles)
    
    
    plt.show('hold')


if __name__ == "__main__":
    main()