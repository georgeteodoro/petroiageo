import numpy as np
import scipy.ndimage

def rms(windowDepth, cube):
    windowSize = (1, 1, windowDepth)
    
    weightType = 'sine'
    weights = getWeights(weightType, windowDepth)
    
    return param_moving_window(cube, windowSize, rmsOf, weights)

def getWeights(weightType, maxN):
    if weightType == 'sine':
        return sineWeights(maxN)

def sineWeights(maxN):
    """
    https://en.wikipedia.org/wiki/Window_function#:~:text=of%20the%20window.-,Sine%20window,-%5Bedit%5D
    """
    weights = [np.sin([(np.pi*idx)/maxN])[0] for idx in range(1, maxN+1)]
    
    return weights

def param_moving_window(data, window, func, funcParam):
    wrapped = lambda region: func(region.reshape(window), funcParam)
    return scipy.ndimage.generic_filter(data, wrapped, window)

def rmsOf(amps, weights=None):
    if weights == None:
        weights = np.ones(amps.shape[2])
    
    sum = 0
    for idx, data in enumerate(amps.ravel()):
        sum += weights[idx]*(data**2)

    rms = np.sqrt(sum/amps.shape[2])

    return rms