import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data
import math

#add random seed for generating comparable pseudo random numbers
np.random.seed(123)

#plot preferences, interactive plotting mode
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()

def plot_state(particles, landmarks, map_limits):
    # Visualizes the state of the particle filter.
    #
    # Displays the particle cloud, mean position and landmarks.
    
    xs = []
    ys = []
    #print(particles)
    for p in particles:
        #print(p)
        xs.append(p["x"])
        ys.append(p["y"])
   

    # landmark positions
    lx=[]
    ly=[]

    for i in range (len(landmarks)):
        lx.append(landmarks[i+1][0])
        ly.append(landmarks[i+1][1])

    # mean pose as current estimate
    estimated_pose = mean_pose(particles)

    # plot filter state
    plt.clf()
    plt.plot(xs, ys, 'r.')
    plt.plot(lx, ly, 'bo',markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy',scale_units='xy')
    plt.axis(map_limits)

    plt.pause(0.01)

def initialize_particles(num_particles, map_limits):
    # randomly initialize the particles inside the map limits

    particles = []

    for i in range(num_particles):
        particle = dict()

        # draw x,y and theta coordinate from uniform distribution
        # inside map limits
        particle['x'] = np.random.uniform(map_limits[0], map_limits[1])
        particle['y'] = np.random.uniform(map_limits[2], map_limits[3])
        particle['theta'] = np.random.uniform(-np.pi, np.pi)

        particles.append(particle)

    return particles

def mean_pose(particles):
    # calculate the mean pose of a particle set.
    #
    # for x and y, the mean position is the mean of the particle coordinates
    #
    # for theta, we cannot simply average the angles because of the wraparound 
    # (jump from -pi to pi). Therefore, we generate unit vectors from the 
    # angles and calculate the angle of their average 

    # save x and y coordinates of particles
    xs = []
    ys = []

    # save unit vectors corresponding to particle orientations 
    vxs_theta = []
    vys_theta = []

    for particle in particles:
        xs.append(particle['x'])
        ys.append(particle['y'])

        #make unit vector from particle orientation
        vxs_theta.append(np.cos(particle['theta']))
        vys_theta.append(np.sin(particle['theta']))

    #calculate average coordinates
    mean_x = np.mean(xs)
    mean_y = np.mean(ys)
    mean_theta = np.arctan2(np.mean(vys_theta), np.mean(vxs_theta))

    return [mean_x, mean_y, mean_theta]

def normal_numpy(mean, var):
    return np.random.normal(mean, np.sqrt(var))

def motion_model(x, y, theta, dr1, dr2, dt, alpha1, alpha2, alpha3, alpha4, normal):
    # noise using using the normal function that comes in string normal
    # normal can be 'normal_sum', 'normal_rejection', 'normal_box_muller', 'normal_numpy'

    # get the function from the string
    normal = globals()[normal]
    dr1_hat = dr1 - normal(0, alpha1 * dr1 ** 2 + alpha2 * dr2 ** 2)
    dr2_hat = dr2 - normal(0, alpha3 * dr1 ** 2 + alpha4 * dr2 ** 2)
    dt_hat = dt - normal(0, alpha1 * dr1 ** 2 + alpha2 * dr2 ** 2 + alpha3 * dr1 ** 2 + alpha4 * dr2 ** 2)

    # motion model
    x_hat = x + (dr1_hat + dr2_hat) / 2 * np.cos(theta + (dt_hat / 2))
    y_hat = y + (dr1_hat + dr2_hat) / 2 * np.sin(theta + (dt_hat / 2))
    theta_hat = theta + dt_hat
    return x_hat, y_hat, theta_hat

def sample_motion_model(odometry, particles):
    # Samples new particle positions, based on old positions, the odometry
    # measurements and the motion noise 
    # (odometry-based sensor model)

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # the motion noise parameters: [alpha1, alpha2, alpha3, alpha4]
    noise = [0.1, 0.1, 0.05, 0.05]

    # generate new particle set after motion update
    new_particles = []
    
    '''your code here'''
    '''***        ***'''

    # odometry-based motion model
    # The function samples new particle positions based
    # on the old positions, the odometry measurements delta_rot1, delta_trans and delta_rot2 and the motion noise.
    # for each particle
    
    for particle in particles:
        # use function motion_model to sample new particle position
        # the function motion_model takes as input the old particle position (x, y, dr1, dr2, dt, alpha1, alpha2, alpha3, alpha4, normal)
        # and returns the new particle position (x_hat, y_hat, theta_hat)

        coords = (motion_model(particle['x'], particle['y'], particle['theta'], delta_rot1, delta_rot2, delta_trans, noise[0], noise[1], noise[2], noise[3], 'normal_numpy'))
        new_particle = dict()
        new_particle['x'] = coords[0]
        new_particle['y'] = coords[1]
        new_particle['theta'] = coords[2]
        new_particles.append(new_particle)


    return new_particles

def likelihood(m, x, d, sigma):
    return 1 / (2 * math.pi * sigma**2) * math.exp(-1 / (2 * sigma**2) * (np.linalg.norm(m - x) - d)**2)

def eval_sensor_model(sensor_data, particles, landmarks):
    # Computes the observation likelihood of all particles, given the
    # particle and landmark positions and sensor measurements
    # (probabilistic sensor models slide 33)
    #
    # The employed sensor model is range only.

    sigma_r = 0.2

    #measured landmark ids and ranges
    ids = sensor_data['id']
    ranges = sensor_data['range']

    weights = []

    #rate each particle
    for particle in particles:

        all_meas_likelihood = 1.0 #for combining multiple measurements 

        #loop for each observed landmark
        for i in range(len(ids)):

            lm_id = ids[i]
            meas_range = ranges[i]

            lx = landmarks[lm_id][0]
            ly = landmarks[lm_id][1]
            px = particle['x']
            py = particle['y']

            #calculate expected range measurement
            meas_range_exp = np.sqrt( (lx - px)**2 + (ly - py)**2 )

            #evaluate sensor model (probability density function of normal distribution)
            meas_likelihood = scipy.stats.norm.pdf(meas_range, meas_range_exp, sigma_r)
            
            #combine (independent) measurements
            all_meas_likelihood = all_meas_likelihood * meas_likelihood

        weights.append(all_meas_likelihood)

    #normalize weights
    normalizer = sum(weights)
    weights = weights / normalizer

    return weights

def resample_particles(particles, weights):
    # Returns a new set of particles obtained by performing
    # stochastic universal sampling, according to the particle 
    # weights.

    new_particles = []

    # distance between pointers
    step = 1.0/len(particles)
    
    # random start of first pointer
    u = np.random.uniform(0,step)
    
    # where we are along the weights
    c = weights[0]

    # index of weight container and corresponding particle 
    i = 0

    new_particles = []

    #loop over all particle weights
    for particle in particles:

        #go through the weights until you find the particle 
        #to which the pointer points 
        while u > c:

            i = i + 1
            c = c + weights[i]

        #add that particle
        new_particles.append(particles[i]) 

        #increase the threshold
        u = u + step

    return new_particles


def main():
    # implementation of a particle filter for robot pose estimation

    print("Reading landmark positions")
    landmarks = read_world("../data/world.dat")

    print("Reading sensor data")
    sensor_readings = read_sensor_data("../data/sensor_data.dat")

    #initialize the particles
    map_limits = [-1, 12, 0, 10]
    particles = initialize_particles(1000, map_limits)

    #run particle filter
    for timestep in range(len(sensor_readings)//2):

        #plot the current state
        plot_state(particles, landmarks, map_limits)

        #predict particles by sampling from motion model with odometry info
        new_particles = sample_motion_model(sensor_readings[timestep,'odometry'], particles)

        #calculate importance weights according to sensor model
        weights = eval_sensor_model(sensor_readings[timestep, 'sensor'], new_particles, landmarks)

        #resample new particle set according to their importance weights
        particles = resample_particles(new_particles, weights)

    plt.show('hold')

if __name__ == "__main__":
    main()



    

