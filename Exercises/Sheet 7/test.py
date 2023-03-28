import matplotlib.pyplot as plt
import numpy as np

def is_outlier(points1, points2, thresh=3.5):
    """
    Returns a boolean array with True if points are outliers and False 
    otherwise.

    Parameters:
    -----------
        points : An numobservations by numdimensions array of observations
        thresh : The modified z-score to use as a threshold. Observations with
            a modified z-score (based on the median absolute deviation) greater
            than this value will be classified as outliers.

    Returns:
    --------
        mask : A numobservations-length boolean array.

    References:
    ----------
        Boris Iglewicz and David Hoaglin (1993), "Volume 16: How to Detect and
        Handle Outliers", The ASQC Basic References in Quality Control:
        Statistical Techniques, Edward F. Mykytka, Ph.D., Editor. 
    """
    if len(points1.shape) == 1:
        points1 = points1[:,None]
    median1 = np.median(points1, axis=0)
    diff1 = np.sum((points1 - median1)**2, axis=-1)
    diff1 = np.sqrt(diff1)
    med_abs_deviation1 = np.median(diff1)

    modified_z_score1 = 0.6745 * diff1 / med_abs_deviation1

    if len(points2.shape) == 1:
        points2 = points2[:,None]
    median2 = np.median(points2, axis=0)
    diff2 = np.sum((points2 - median2)**2, axis=-1)
    diff2 = np.sqrt(diff2)
    med_abs_deviation2 = np.median(diff2)

    modified_z_score2 = 0.6745 * diff2 / med_abs_deviation2

    i = 0
    points1_filtered = np.array([])
    points2_filtered = np.array([])

    for (element1, element2) in zip(modified_z_score1, modified_z_score2):
        if (element1 < thresh) and (element2 < thresh):
            points1_filtered = np.append(points1_filtered, points1[i])
            points2_filtered = np.append(points2_filtered, points2[i])

    return (points1_filtered, points2_filtered)

def plots(param1_blip, param2_blip, param1_injections, param2_injections, param1_name, param2_name, xlims = None, ylims = None):

    param1_blip_filtered, param2_blip_filtered = is_outlier(param1_blip, param2_blip)

    plt.scatter(param1_blip_filtered, param2_blip_filtered, label = "Blip", s = 50, color = "red", alpha = 0.1)

    param1_injections_filtered, param2_injections_filtered = is_outlier(param1_injections, param2_injections)

    plt.scatter(param1_injections_filtered, param2_injections_filtered, label = "Injections", s = 50, color = "blue", alpha = 0.1)

    plt.legend()
    plt.title(param1_name + " vs " + param2_name)
    if xlims != None:
        plt.xlim(xlims)
    if ylims != None:
        plt.ylim(ylims)
    plt.savefig("plots/" + param1_name + "vs" + param2_name + ".png")
    plt.clf()

