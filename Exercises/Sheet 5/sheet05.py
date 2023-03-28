# Exercise 1: Sampling
# Implement three functions in Python which generate samples of a normal distri-
# bution N(; 2). The input parameters of these functions should be the mean 
# and the variance 2 of the normal distribution. As only source of randomness, use
# samples of a uniform distribution.
#  In the rst function, generate the normal distributed samples by summing up
# 12 uniform distributed samples, as explained in the lecture.
#  In the second function, use rejection sampling.
#  In the third function, use the Box-Muller transformation method. The Box-
# Muller method allows to generate samples from a standard normal distribution
# using two uniformly distributed samples u1, u2 2 [0; 1] via the following equa-
# tion:
# x = cos(2u1)
# p
# 􀀀2 log u2:
# Compare the execution times of the three functions using Python's built-in function
# timeit. Also, compare the execution times of your own functions to the built-in
# function numpy.random.normal.

import numpy as np
import matplotlib.pyplot as plt
import timeit

mean = 0
var = 10

def normal_sum(mean, var):
    # we want var = var so we subtract the mean
    return mean + np.sqrt(var) * np.sum(np.random.rand(12)) - mean

def normal_rejection(mean, var):
    while True:
        u1 = np.random.rand()
        u2 = np.random.rand()
        z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        if u2 <= np.exp(-z ** 2 / 2):
            return mean + np.sqrt(var) * z

def normal_box_muller(mean, var):
    u1 = np.random.rand()
    u2 = np.random.rand()
    z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
    return mean + np.sqrt(var) * z

def normal_numpy(mean, var):
    return np.random.normal(mean, np.sqrt(var))

def exercise1():
    n = 100000
    time_sum = timeit.timeit('normal_sum(mean, var)', setup='from __main__ import normal_sum, mean, var', number=n)
    time_rejection = timeit.timeit('normal_rejection(mean, var)', setup='from __main__ import normal_rejection, mean, var', number=n)
    time_box_muller = timeit.timeit('normal_box_muller(mean, var)', setup='from __main__ import normal_box_muller, mean, var', number=n)
    time_numpy = timeit.timeit('normal_numpy(mean, var)', setup='from __main__ import normal_numpy, mean, var', number=n)
    print('Time sum: ', time_sum)
    print('Time rejection: ', time_rejection)
    print('Time box muller: ', time_box_muller)
    print('Time numpy: ', time_numpy)

    x = np.linspace(-10, 10, 1000)
    # use mean and var from main
    y = 1 / np.sqrt(2 * np.pi * var) * np.exp(- (x - mean) ** 2 / (2 * var))
    plt.plot(x, y, label='True distribution')
    plt.hist([normal_sum(mean, var) for i in range(n)], bins=100, density=True, label='Sum')
    plt.hist([normal_rejection(mean, var) for i in range(n)], bins=100, density=True, label='Rejection')
    plt.hist([normal_box_muller(mean, var) for i in range(n)], bins=100, density=True, label='Box-Muller')
    plt.hist([normal_numpy(mean, var) for i in range(n)], bins=100, density=True, label='Numpy')
    plt.legend()
    plt.show()

# Exercise 2: Odometry-based Motion Model
# Implement the odometry-based motion model in Python. Your function should
# take the following three arguments:
# The current state of the robot (x; y; theta) in the world frame before moving.
# The odometry measurements (dr1, dr2, dt) of the robot.
# The noise parameters (alpha1, alpha2, alpha3, alpha4) of the odometry model.
# The function should return the new state of the robot (x; y; theta) in the world
# As we do not expect the odometry measurements to be perfect, you will have
# to take the measurement error into account when implementing your function.
# Use the sampling methods you implemented in Exercise 1 to draw normally
# distributed random numbers for the noise in the motion model.

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
# Evaluate your motion model 5000 times for the following values
# x = 2.0, y = 4.0, theta = 0.0
# dr1 = pi/2, dr2 = 0.0, dt = 1.0
# alpha1 = 0.1, alpha2 = 0.1, alpha3 = 0.01, alpha4 = 0.01
# Plot the resulting (x; y) positions for each of the 5000 evaluations in a single
# plot.
def exercise2():
    x = 2.0
    y = 4.0
    theta = 0.0
    dr1 = np.pi / 2
    dr2 = 0.0
    dt = 1.0
    alpha1 = 0.1
    alpha2 = 0.1
    alpha3 = 0.01
    alpha4 = 0.01
    n = 5000
    x_hat = []
    y_hat = []
    # for each normal function plot a subplot in the same figure
    for normal in ['normal_sum', 'normal_rejection', 'normal_box_muller', 'normal_numpy']:
        for i in range(n):
            x_hat_, y_hat_, _ = motion_model(x, y, theta, dr1, dr2, dt, alpha1, alpha2, alpha3, alpha4, normal)
            x_hat.append(x_hat_)
            y_hat.append(y_hat_)
        plt.plot(x_hat, y_hat, '.', label=normal)
        x_hat = []
        y_hat = []


    plt.legend()

    # add title and labels
    plt.title('Motion model')
    plt.xlabel('x')
    plt.ylabel('y')

    
    
    plt.show()



if __name__ == "__main__":
    exercise1()
    exercise2()