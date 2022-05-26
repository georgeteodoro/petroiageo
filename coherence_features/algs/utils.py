import numpy as np
import scipy.ndimage
import scipy.signal

def moving_window(data, window, func):
    """
    Applies <func> over all the volume <data> using windows of
    size <window>.
    """
    wrapped = lambda region: func(region.reshape(window))
    return scipy.ndimage.generic_filter(data, wrapped, window)


