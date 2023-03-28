
import math
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Ellipse, Rectangle
import numpy as np

def angle_diff(angle1, angle2):
    return np.arctan2(np.sin(angle1-angle2), np.cos(angle1-angle2))

def error_ellipse(position, sigma):

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
    error_ellipse = Ellipse(xy=[position[0],position[1]], width=width, height=height, angle=angle/np.pi*180)
    error_ellipse.set_alpha(0.25)

    return error_ellipse

def plot_state(particles, landmarks, map_limits, gt_poses, walls):
    # Visualizes the state of the particle filter.
    #
    # Displays the particle cloud, mean position and 
    # estimated mean landmark positions and covariances.

    draw_mean_landmark_poses = False

    #map_limits = [-1, 12, 0, 10]
    
    #particle positions
    xs = []
    ys = []

    # particles_wight 0

    xs0 = []
    ys0 = []

    #landmark mean positions
    lxs = []
    lys = []

    for particle in particles:
        if particle['weight'] < 0.0001:
            xs0.append(particle['x'])
            ys0.append(particle['y'])
        else:
            xs.append(particle['x'])
            ys.append(particle['y'])
        
        for i in range(len(landmarks)):
            landmark = particle['landmarks'][i+1]
            lxs.append(landmark['mu'][0])
            lys.append(landmark['mu'][1])

    # ground truth landmark positions
    lx=[]
    ly=[]

    for i in range (len(landmarks)):
        lx.append(landmarks[i+1][0])
        ly.append(landmarks[i+1][1])

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
        
    # best particle
    estimated = best_particle(particles)
    if estimated != None:
        robot_x = estimated['x']
        robot_y = estimated['y']
        robot_theta = estimated['theta']
    else:
        robot_x = 0
        robot_y = 0
        robot_theta = 0
    # estimated traveled path of best particle
    if estimated != None:
        hist = estimated['history']
    else:
        hist = []
    hx = []
    hy = []

    for pos in hist:
        hx.append(pos[0])
        hy.append(pos[1])
    
    # ground truth path
    hx_gt = []
    hy_gt = []
    
    for pos in gt_poses:
        hx_gt.append(pos[0])
        hy_gt.append(pos[1])

    # plot filter state
    plt.clf()

    #particles
    plt.plot(xs, ys, 'r.')

    #particles weight 0
    plt.plot(xs0, ys0, 'g.')
    
    if draw_mean_landmark_poses:
        # estimated mean landmark positions of each particle
        plt.plot(lxs, lys, 'b.')

    # estimated traveled path of best particle
    plt.plot(hx, hy, 'r-')
    
    # ground truth path
    plt.plot(hx_gt, hy_gt, 'b-')
    
    # true landmark positions
    plt.plot(lx, ly, 'b+',markersize=10)

    # for each wall plot a rectangle where the center is the center of the wall
    # and the orientation is the orientation of the wall~
    # change the sign of the orientation depending on the side of the wall
    for i in range(len(walls)):
        sign = 1
        if walls[i][2][0] < 0 and walls[i][2][1] < 0:
            sign = -1
        plt.gca().add_patch(Rectangle((wx[i], wy[i]), ww[i], wh[i], angle=sign*np.arctan2(oy[i],ox[i])/np.pi*180 + 180 -(180-sign*np.arctan2(oy[i],ox[i])/np.pi*180)))

    

    # draw error ellipse of estimated landmark positions of best particle 
    for i in range(len(landmarks)):
        landmark = estimated['landmarks'][i+1]

        ellipse = error_ellipse(landmark['mu'], landmark['sigma'])
        plt.gca().add_artist(ellipse)

    # draw pose of best particle
    plt.quiver(robot_x, robot_y, np.cos(robot_theta), np.sin(robot_theta), angles='xy',scale_units='xy')
    
    plt.axis(map_limits)
    plt.pause(0.01)

def best_particle(particles):
    #find particle with highest weight 

    highest_weight = 0

    best_particle = None
    
    for particle in particles:
        if particle['weight'] > highest_weight:
            best_particle = particle
            highest_weight = particle['weight']

    return best_particle
