"""
Script used to transform data. The current transformations that are available are:
 * HSV
 * Local Binary Pattern (LBP)
"""
import numpy as np

from math import sqrt, pow
from skimage.color import rgb2hsv, rgb2gray, label2rgb
from skimage.feature import local_binary_pattern


def transform_hsv(img):
    """
    Transforms image from the RGB domain into the HSV domain.
    """
    # Convert rgb to hsv
    hsv_img = rgb2hsv(img)
    return hsv_img

def get_energy(img):
    """

    """
    height, width = img.shape[0], img.shape[1]

    hsv_img = transform_hsv(img)
    h, s, v = hsv_img[:, :, 0], hsv_img[:, :, 1], hsv_img[:, :, 2]

    pixels = (height - 2) * (width - 2)
    energy_h = [0] * height
    energy_s = [0] * height
    energy_v = [0] * height
    for i in range(len(energy_h)):
        energy_h[i] = [0] * width
        energy_s[i] = [0] * width
        energy_v[i] = [0] * width

    # energy of h channel
    for j in range(1, height -1):
        for k in range(1, width -1):
            # horizontally movement
            x1 = h[j][k-1]
            x2 = h[j][k+1]
            # vertical movement
            y1 = h[j-1][k]
            y2 = h[j+1][k]
            energy = sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))
            energy_h[j][k] = energy

    # energy for s channel
    for j in range(1, height -1):
        for k in range(1, width -1):
            # horizontally movement
            x1 = s[j][k-1]
            x2 = s[j][k+1]
            # vertical movement
            y1 = s[j-1][k]
            y2 = h[j+1][k]
            energy = sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))
            energy_s[j][k] = energy

    # energy for the v channel
    for j in range(1, height -1):
        for k in range(1, width -1):
            # horizontally movement
            x1 = v[j][k-1]
            x2 = v[j][k+1]
            # vertical movement
            y1 = v[j-1][k]
            y2 = v[j+1][k]
            energy = sqrt(pow((x1 - x2), 2) + pow((y1 - y2), 2))
            energy_v[j][k] = energy

    energy_h = np.array(energy_h, dtype=np.float32)
    energy_s = np.array(energy_s)
    energy_v = np.array(energy_v)
    out = np.stack([energy_h, energy_s, energy_v], axis=2)

    return out


def transform_lbp(img, n_points= 10, radius=3,  method='default'):
    """
    Transforms image from the RGB domain into the Local Binary Pattern. If the image is RGB, it is transformed into
    grayscale (w, d), otherwise it will be processed directly.
    """
    # transform to 2-D if more than one channel exists
    if len(img.shape) > 2:
        img = rgb2gray(img)
    lbp =  local_binary_pattern(img,
                                P=radius,
                                R=n_points,
                                method=method)

    # get histogram from lbp
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    return lbp, hist

def transform_lbp_test(img, n_points= 10, radius=3,  method='default'):
    """
    Transforms image from the RGB domain into the Local Binary Pattern. If the image is RGB, it is transformed into
    grayscale (w, d), otherwise it will be processed directly.
    """
    # transform to 2-D if more than one channel exists
    if len(img.shape) > 2:
        img = rgb2gray(img)
    lbp =  local_binary_pattern(img,
                                P=radius,
                                R=n_points,
                                method=method)

    # get histogram from lbp
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)

    # test
    mask = np.logical_or.reduce([lbp >= 2.0])
    final_img = label2rgb(mask, img, bg_label=0, alpha=0.5)
    return final_img, hist

