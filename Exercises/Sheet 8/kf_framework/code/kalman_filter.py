import numpy as np
import scipy.stats
import matplotlib.pyplot as plt
from read_data import read_world, read_sensor_data
from matplotlib.patches import Ellipse

#plot preferences, interactive plotting mode
fig = plt.figure()
plt.axis([-1, 12, 0, 10])
plt.ion()
plt.show()

def plot_state(mu, sigma, landmarks, map_limits):
    # Visualizes the state of the kalman filter.
    #
    # Displays the mean and standard deviation of the belief,
    # the state covariance sigma and the position of the 
    # landmarks.

    # landmark positions
    lx=[]
    ly=[]

    for i in range (len(landmarks)):
        lx.append(landmarks[i+1][0])
        ly.append(landmarks[i+1][1])

    # mean of belief as current estimate
    estimated_pose = mu

    #calculate and plot covariance ellipse
    covariance = sigma[0:2,0:2]
    eigenvals, eigenvecs = np.linalg.eig(covariance)

    #get largest eigenvalue and eigenvector
    max_ind = np.argmax(eigenvals)
    max_eigvec = eigenvecs[:,max_ind]
    max_eigval = eigenvals[max_ind]

    #get smallest eigenvalue and eigenvector
    min_ind = 0
    if max_ind == 0:
        min_ind = 1

    min_eigvec = eigenvecs[:,min_ind]
    min_eigval = eigenvals[min_ind]

    #chi-square value for sigma confidence interval
    chisquare_scale = 2.2789  

    #calculate width and height of confidence ellipse
    width = 2 * np.sqrt(chisquare_scale*max_eigval)
    height = 2 * np.sqrt(chisquare_scale*min_eigval)
    angle = np.arctan2(max_eigvec[1],max_eigvec[0])

    #generate covariance ellipse
    ell = Ellipse(xy=[estimated_pose[0],estimated_pose[1]], width=width, height=height, angle=angle/np.pi*180)
    ell.set_alpha(0.25)

    # plot filter state and covariance
    plt.clf()
    plt.gca().add_artist(ell)
    plt.plot(lx, ly, 'bo',markersize=10)
    plt.quiver(estimated_pose[0], estimated_pose[1], np.cos(estimated_pose[2]), np.sin(estimated_pose[2]), angles='xy',scale_units='xy')
    plt.axis(map_limits)
    
    plt.pause(0.01)


def prediction_step(odometry, mu, sigma):
    # Updates the belief, i.e., mu and sigma, according to the motion 
    # model
    # 
    # mu: 3x1 vector representing the mean (x,y,theta) of the 
    #     belief distribution
    # sigma: 3x3 covariance matrix of belief distribution 
    
    x = mu[0]
    y = mu[1]
    theta = mu[2]

    delta_rot1 = odometry['r1']
    delta_trans = odometry['t']
    delta_rot2 = odometry['r2']

    # Compute the new mean of the belief distribution
    mu_prime = np.array([
        x + delta_trans * np.cos(theta + delta_rot1),
        y + delta_trans * np.sin(theta + delta_rot1),
        theta + delta_rot1 + delta_rot2
    ])
    mu = mu_prime

    # Compute the Jacobian of the motion model
    F = np.array([
        [1, 0, -delta_trans * np.sin(theta + delta_rot1)],
        [0, 1, delta_trans * np.cos(theta + delta_rot1)],
        [0, 0, 1]
    ])

    # Compute the noise covariance matrix of the motion model
    Q = np.array([
        [0.2, 0, 0],
        [0, 0.2, 0],
        [0, 0, 0.2]
    ])

    # Compute the new covariance of the belief distribution
    sigma_prime = F @ sigma @ F.T + Q
    sigma = sigma_prime

    return mu, sigma


def correction_step(sensor_data, mu, sigma, landmarks):
    # Updates the belief, i.e., mu and sigma, according to the sensor 
    # model.
    #
    # The employed sensor model is range-only, i.e., it can measure 
    # the range to a landmark, but not the bearing.
    #
    # sensor_data: a list of measurements, one for each landmark
    # mu: mean vector of the belief distribution
    # sigma: covariance matrix of the belief distribution
    # landmarks: a dictionary of landmark positions
    
    # number of landmarks
    n = len(landmarks)

    # measurement noise covariance
    R = np.eye(n) * 0.1

    # current estimate of robot pose
    x = mu[0]
    y = mu[1]
    theta = mu[2]

    # initialize matrices for the Kalman filter update
    S = np.zeros((n, n))
    K = np.zeros((3, n))
    z_hat = np.zeros((n, 1))
    
    # Iterate over each landmark to compute the update
    for i, landmark in enumerate(landmarks.values()):
        # compute expected measurement, i.e., distance to the landmark
        dx = landmark[0] - x
        dy = landmark[1] - y
        z_hat[i] = np.sqrt(dx**2 + dy**2)
        
        # Jacobian of the measurement model wrt the state
        H = np.array([
            [-dx/z_hat[i], -dy/z_hat[i], 0],
            [dy/z_hat[i]**2, -dx/z_hat[i]**2, -1]
        ])
        H = H.T
        
        # Compute the innovation covariance
        S[i][i] = (H.T.dot(sigma).dot(H) + R[i][i])[0][0]

        
        # Compute the Kalman gain
        K[:, i] = (sigma.dot(H).dot(1/S[i][i]))[:, 0]
        
    # Extract the relevant values from the dictionary
    sensor_value = sensor_data['range']
    sensor_bearing = sensor_data['bearing']

    # Perform the subtraction on the extracted values
    innovation = [sensor_value - z_hat[0], sensor_bearing - z_hat[1]]
    
    # Transpose the innovation matrix to make its dimensions compatible with K
    innovation = np.array(innovation).T

    ## Select the first 3 rows of K and the first 2 columns of innovation
    K_subset = K[:, :2]
    innovation_subset = innovation[:3, :]

    # Perform the dot product
    result_subset = K_subset.dot(innovation_subset).flatten()[:3]

    # Perform the element-wise addition
    mu += result_subset

    
    # Update the covariance of the belief distribution
    sigma = sigma - K.dot(S).dot(K.T)
    
    return mu, sigma


def main():
    # implementation of an extended Kalman filter for robot pose estimation

    print ("Reading landmark positions")
    landmarks = read_world("../data/world.dat")

    print ("Reading sensor data")
    sensor_readings = read_sensor_data("../data/sensor_data.dat")

    #initialize belief
    mu = [0.0, 0.0, 0.0]
    sigma = np.array([[1.0, 0.0, 0.0],\
                      [0.0, 1.0, 0.0],\
                      [0.0, 0.0, 1.0]])

    map_limits = [-1, 12, -1, 10]

    #run kalman filter
    for timestep in range(len(sensor_readings)//2):

        #plot the current state
        plot_state(mu, sigma, landmarks, map_limits)

        #perform prediction step
        mu, sigma = prediction_step(sensor_readings[timestep,'odometry'], mu, sigma)

        #perform correction step
        mu, sigma = correction_step(sensor_readings[timestep, 'sensor'], mu, sigma, landmarks)

    plt.show('hold')

if __name__ == "__main__":
    main()

