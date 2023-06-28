from algs.utils import moving_window
from algs.coherence import gersztenkorn
from scipy.ndimage import sobel
import numpy as np


def gersz_sobel(data, window, n_cpu=1):
    coh = moving_window(data, window, gersztenkorn, n_cpu)
    return np.sqrt(sum(sobel(coh, axis) ** 2 for axis in range(3)))
