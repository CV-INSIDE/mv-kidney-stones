"""
Script used to transform data. The current transformations that are available are:
 * HSV
 * Local Binary Pattern (LBP)
"""
import numpy as np

from skimage.color import rgb2hsv, rgb2gray
from skimage.feature import local_binary_pattern


def transform_hsv(img):
    """
    Transforms image from the RGB domain into the HSV domain.
    """
    # Convert rgb to hsv
    hsv_img = rgb2hsv(img)
    return hsv_img

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