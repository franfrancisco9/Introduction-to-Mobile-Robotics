import numpy as np
import matplotlib.pyplot as plt
import math


# Exercise 1: Distance-Only Sensor

# In this exercise, you try to locate your friend using her cell phone signals. Suppose
# that on a map, the university campus is located at m0 = (10; 8)T , and your friend's
# home is situated at m1 = (6; 3)T . You have access to the data received by two cell
# towers, which are located at the positions x0 = (12; 4)T and x1 = (5; 7)T , respec-
# tively. The distance between your friend's cell phone and the towers can be computed
# from the intensities of your friend's cell phone signals. These distance measurements
# are disturbed by independent zero-mean Gaussian noise with variances 2
# 0 = 1 for
# tower 0 and 2
# 1 = 1:5 for tower 1. You receive the distance measurements d0 = 3:9
# and d1 = 4:5 from the two towers.

m0 = np.array([10, 8])
m1 = np.array([6, 3])

x0 = np.array([12, 4])
x1 = np.array([5, 7])

sigma0 = 1
sigma1 = 1.5

d0 = 3.9
d1 = 4.5

def likelihood(m, x, d, sigma):
    return 1 / (2 * math.pi * sigma**2) * math.exp(-1 / (2 * sigma**2) * (np.linalg.norm(m - x) - d)**2)

# (a) Is your friend more likely to be at home or at the university? Explain your
# calculations.

# calculate the likelihood for each point using the sigmas and distances from the exercise
def calculate_likelihood():
    # calculate the likelihood for each point using the sigmas and distances from the exercise
    likelihood_m0 = likelihood(m0, x0, d0, sigma0) * likelihood(m0, x1, d1, sigma1)
    likelihood_m1 = likelihood(m1, x0, d0, sigma0) * likelihood(m1, x1, d1, sigma1)

    print("Likelihood of m0: {}".format(likelihood_m0))
    print("Likelihood of m1: {}".format(likelihood_m1))

    if likelihood_m0 > likelihood_m1:
        print("University is more likely")
    else:
        print("House is more likely")

# (b) Implement a function in Python which generates a 3D-plot of the likelihood
# p(zjm) over all locations m in the vicinity of the towers. Furthermore, mark
# m0, m1, x0 and x1 in the plot. Is the likelihood function which you plotted a
# probability density function? Give a reason for your answer.


# generate a 3D-plot of the likelihood p(zjm) over all locations m in the vicinity of the towers
def plot_likelihood():
    # generate a grid of points
    x = np.linspace(0, 15, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)

    # calculate the likelihood for each point using the sigmas and distances from the exercise
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = likelihood(np.array([X[i, j], Y[i, j]]), x0, d0, sigma0) * likelihood(np.array([X[i, j], Y[i, j]]), x1, d1, sigma1)
    


    # plot the likelihood
    fig = plt.figure()

    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('p(z|m)')
    ax.set_title('Likelihood')

    # mark m0, m1, x0 and x1 in the plot
    ax.scatter(m0[0], m0[1], likelihood(m0, x0, d0, sigma0) * likelihood(m0, x1, d1, sigma1), color='red', marker='x')
    ax.scatter(m1[0], m1[1], likelihood(m1, x0, d0, sigma0) * likelihood(m1, x1, d1, sigma1), color='red', marker='x')
    ax.scatter(x0[0], x0[1], 0, color='red', marker='x')
    ax.scatter(x1[0], x1[1], 0, color='red', marker='x')

    plt.show()


# (c) Now, suppose you have prior knowledge about your friend's habits which sug-
# gests that your friend currently is at home with probability P(at home) = 0:7,
# at the university with P(at university) = 0:3, and at any other place with
# P(other) = 0. Use this prior knowledge to recalculate the likelihoods of a).

p_home = 0.7
p_university = 0.3
p_other = 0.0

def calculate_likelihood_with_prior():
    # calculate the likelihood for each point using the sigmas and distances from the exercise
    p_z = likelihood(m0, x0, d0, sigma0) * likelihood(m0, x1, d1, sigma1) * p_university + likelihood(m1, x0, d0, sigma0) * likelihood(m1, x1, d1, sigma1) * p_home
    likelihood_m0 = likelihood(m0, x0, d0, sigma0) * likelihood(m0, x1, d1, sigma1) * p_university / p_z
    likelihood_m1 = likelihood(m1, x0, d0, sigma0) * likelihood(m1, x1, d1, sigma1) * p_home / p_z

    print("Likelihood of m0: {}".format(likelihood_m0))
    print("Likelihood of m1: {}".format(likelihood_m1))

    if likelihood_m0 > likelihood_m1:
        print("University is more likely")
    else:
        print("House is more likely")


def plot_likelihood_with_prior():
    # generate a grid of points
    x = np.linspace(0, 15, 100)
    y = np.linspace(0, 10, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)

    # calculate the likelihood for each point using the sigmas and distances from the exercise
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = likelihood(np.array([X[i, j], Y[i, j]]), x0, d0, sigma0) * likelihood(np.array([X[i, j], Y[i, j]]), x1, d1, sigma1) * p_university
    

    # plot the likelihood
    fig = plt.figure()

    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, Z, cmap='viridis', linewidth=0)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('p(z|m)')
    ax.set_title('Likelihood')

    # mark m0, m1, x0 and x1 in the plot
    ax.scatter(m0[0], m0[1], likelihood(m0, x0, d0, sigma0) * likelihood(m0, x1, d1, sigma1), color='red', marker='x')
    ax.scatter(m1[0], m1[1], likelihood(m1, x0, d0, sigma0) * likelihood(m1, x1, d1, sigma1), color='red', marker='x')
    ax.scatter(x0[0], x0[1], 0, color='green', marker='x')
    ax.scatter(x1[0], x1[1], 0, color='green', marker='x')

    plt.show()

calculate_likelihood()
plot_likelihood()
calculate_likelihood_with_prior()
plot_likelihood_with_prior()