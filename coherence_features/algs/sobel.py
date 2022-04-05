from algs.utils import moving_window
from algs.coherence import gersztenkorn
from scipy.ndimage import sobel

def gersz_sobel(data, window):
    coh = moving_window(data, window, gersztenkorn)
    return np.sqrt(sum(sobel(coh,axis)**2 for axis in range(3)))
