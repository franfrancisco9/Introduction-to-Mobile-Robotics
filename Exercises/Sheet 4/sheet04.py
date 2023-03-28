import numpy

# Exercise 1: Bayes Rule
# Suppose you are a witness to a nighttime hit-and-run accident involving a taxi in
# Athens. All taxi cars in Athens are blue or green. You swear under oath that
# the taxi was blue. Extensive testing shows that, under the dim lighting conditions,
# discrimination between blue and green is 75% reliable.
# (a) Given your statement as a witness and given that 9 out of 10 Athenian taxis
# are green, what is the probability of the taxi being blue?

# P(blue|witness) = P(witness|blue) * P(blue) / P(witness)
# P(witness|blue) = 0.75
# P(witness_blue ! green) = 0.25
# P(blue) = 0.1
# P(witness) = P(witness|blue) * P(blue) + P(witness_blue|green) * P(green)
#            = 0.75 * 0.1 + 0.25 * 0.9
#            = 0.325
# P(blue|witness) = 0.75 * 0.1 / 0.325
#                 = 0.23076923076923078

# (b) Is there a signicant change if 7 out of 10 Athenian taxis are green?
# CONSIDER WITNESS AS WITNESS CHOSSING BLUE TAXI
# P(blue|witness) = P(witness|blue) * P(blue) / P(witness)
# P(witness|blue) = 0.75
# P(blue) = 0.3
# P(witness) = P(witness|blue) * P(blue) + P(witness|green) * P(green)
#            =  0.75 * 0.3 + 0.25 * 0.7
#            = 0.4
# P(blue|witness) = 0.75 * 0.3 / 0.4
#                 = 0.5625
# calculate the difference
# P(blue|witness) = 0.2727272727272727 - 0.1111111111111111
#                 = 0.1616161616161616

# (c) Suppose now that there is a second witness who swears that the taxi is green.
# Unfortunately he is color blind, so he has only a 50% chance of being right.
# How would this change the estimate from (b)?

# Second witness has 50% chance of being right
# caculate the probability of the taxi being blue given witness and witness2
# P(blue|witness,witness2) = P(blue|witness) * P(witness2|blue)  / P(witness2|witness)
# P(blue|witness) = 0.3913043478260869
# P(witness2|blue) = 0.5
# P(blue) = 0.3
# P(green|witness) = 0.7272727272727273
# P(witness2|witness) = P(witness2|blue) * P(blue|witness) + P(witness2|green) * P(green|witness)
#                     = 0.5 * 0.2727272727272727 + 0.5 * 0.7272727272727273
#                     = 0.5
# P(blue|witness,witness2) = 0.56 * 0.5 / 0.5
#                          = 0.56
# There is no change in the estimate from (b)

# Exercise 2: Bayes Filter
# A vacuum cleaning robot is equipped with a cleaning unit to clean the 
# oor. Fur-
# thermore, the robot has a sensor to detect whether the 
# oor is clean or dirty. Neither
# the cleaning unit nor the sensor are perfect.
# From previous experience you know that the robot succeeds in cleaning a dirty 
# oor
# with a probability of
# p(xt+1 = clean j xt = dirty; ut+1 = vacuum-clean) = 0:7;
# where xt+1 is the state of the 
# oor after having vacuum-cleaned, ut+1 is the control
# command, and xt is the state of the 
# oor before performing the action.
# The probability that the sensor indicates that the 
# oor is clean although it is dirty
# is given by p(z = clean j x = dirty) = 0:3, and the probability that the sensor
# correctly detects a clean 
# oor is given by p(z = clean j x = clean) = 0:9.
# Unfortunately, you have no knowledge about the current state of the 
# oor. However,
# after cleaning the 
# oor the sensor of the robot indicates that the 
# oor is clean.
# (a) Compute the probability that the 
# oor is still dirty after the robot has vacuum-
# cleaned it. Use an appropriate prior distribution and justify your choice.

# Use the gaussian distribution as the prior distribution
# P(xt = dirty) = 0.5
# P(xt = clean) = 0.5
# P(xt+1 = clean j xt = dirty; ut+1 = vacuum-clean) = 0.7
# P(xt+1 = dirty j xt = dirty; ut+1 = vacuum-clean) = 0.3
# P(xt+1 = clean j xt = clean; ut+1 = vacuum-clean) = 1
# P(xt+1 = dirty j xt = clean; ut+1 = vacuum-clean) = 0
# P(z = clean j x = dirty) = 0.3
# P(z = clean j x = clean) = 0.9
# P(z = dirty j x = dirty) = 0.7
# P(z = dirty j x = clean) = 0.1
# P(xt+1 = dirty j z = clean) = P(z = clean j x = dirty) * P(xt = dirty) / P(z = clean)
#                            = 0.3 * 0.5 / (0.3 * 0.5 + 0.9 * 0.5)
#                            = 0.25
# P(xt+1 = clean j z = clean) = P(z = clean j x = clean) * P(xt = clean) / P(z = clean)
#                             = 0.9 * 0.5 / (0.3 * 0.5 + 0.9 * 0.5)
#                             = 0.75
# P(xt+1 = dirty j z = clean) = 0.25
# P(xt+1 = clean j z = clean) = 0.75
# P(xt+1 = dirty j z = clean, ut+1 = vacuum-clean) = P(xt+1 = dirty j z = clean) * P(ut+1 = vacuum-clean j xt+1 = dirty)
#                                                 = 0.25 * 0.3
#                                                 = 0.075


# (b) Which prior gives you a lower bound for that probability?
