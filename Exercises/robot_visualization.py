# Program Description:
# This program uses the package 
# The robot starts in position (0, 0, 0) (x, y, theta)
# The robot moves according to the motion model
# seen in the sheet 5 exercise 2
# The robot moves 5000 times
# We see the robot moving in fast speed to each position
# When it gets there it leaves a dot in that position

import numpy as np
import matplotlib.pyplot as plt
import timeit
import math
import random
import time
import sys
import os

# add sheet 05 exercise 2
sys.path.append("./Sheet 5")
from sheet05 import motion_model
# import the unity package
from unityagents import UnityEnvironment

# create robot class
class Robot:
    def __init__(self, x, y, theta):
        self.x = x
        self.y = y
        self.theta = theta

     

# create robot
robot = Robot(0, 0, 0)
