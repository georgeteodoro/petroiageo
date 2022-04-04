import sys
import numpy as np
import matplotlib as mtpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import scipy.ndimage
from scipy.signal import hilbert

#Done: Common Functions, RMS, Complex Attrs

#==================================Common Functions=============================

def synthCube(x, y, z, type):
    if type == 'random':
        randomCube = np.random.randn(x, y, z)
    elif type == 'gaussianFilter':
        randomCube = np.random.normal(0, 1, (x, y, z))
        randomCube = scipy.ndimage.gaussian_filter(randomCube, 3)
    return randomCube

def saveNormalizedCubeImg(cube, fileName, show=False):
    """
    Normalize and save an image of the cube.
    If show == True, shows the normalized cube in exec time.
    fileName should not have a extension. It will be saved as .png
    cube: ndarray
    """
    normalizedCube = normalize(cube)
    saveCube(normalizedCube, fileName, show)

def normalize(cube):
    normCube = cube.copy()

    minValue = np.amin(normCube)
    if minValue < 0:
        normCube = normCube -minValue
    # else:
    #     normCube = normCube +minValue

    normCube = normCube / np.amax(normCube)

    return normCube

def saveCube(cube, fileName, show=False):
    #From https://stackoverflow.com/a/45971363/16264901
    x = np.arange(cube.shape[0])[:, None, None]
    y = np.arange(cube.shape[1])[None, :, None]
    z = np.arange(cube.shape[2])[None, None, :]
    x, y, z = np.broadcast_arrays(x, y, z)

    c = np.tile(cube.ravel()[:, None], [1, 3])
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.scatter(x.ravel(),
           y.ravel(),
           z.ravel(),
           c=c)
    plt.savefig(f'{fileName}.png', bbox_inches='tight')
    if show:
        plt.show()

def moving_window(data, window, func):
    wrapped = lambda region: func(region.reshape(window))
    return scipy.ndimage.generic_filter(data, wrapped, window)

def param_moving_window(data, window, func, funcParam):
    wrapped = lambda region: func(region.reshape(window), funcParam)
    return scipy.ndimage.generic_filter(data, wrapped, window)

#==================================\Common Functions============================

#======================================RMS======================================

def applyRmsForWindowDepths(windowDepths, cube, saveImgs, weightType:str):
    windowTraceNumX = 1
    windowTraceNumY = 1

    for windowDepth in windowDepths:
        
        windowSize = (windowTraceNumX, windowTraceNumY, windowDepth)
        applyRmsWithWindow(windowSize, cube, saveImgs, weightType)

def applyRmsWithWindow(windowSize, cube, saveImgs, weightType):
    stCube = rmsCubeOf(cube, windowSize, weightType)
    np.save(f'rmsCube-{windowSize[2]}', stCube)

    if saveImgs:
        saveNormalizedCubeImg(stCube, f'rms-{windowSize[2]}')

def rmsCubeOf(cube, windowSize, weightType):
    weights = getWeights(weightType, windowSize[2])
    return param_moving_window(cube, windowSize, rmsOf, weights)

def getWeights(weightType, maxN):
    if weightType == 'sine':
        return sineWeights(maxN)

def sineWeights(maxN):
    """
    https://en.wikipedia.org/wiki/Window_function#:~:text=of%20the%20window.-,Sine%20window,-%5Bedit%5D
    """
    weights = []
    for idx in range(1, maxN+1):
        weights.append(np.sin([(np.pi*idx)/maxN])[0])
    
    return weights

def rmsOf(amps, weights=None):
    if weights == None:
        weights = np.ones(amps.shape[2])
    
    sum = 0
    for idx, data in enumerate(amps.ravel()):
        sum += weights[idx]*(data**2)

    rms = np.sqrt(sum/amps.shape[2])

    return rms


#=====================================\RMS=====================================

#=====================================Complex Attrs============================
def generateComplexAttrsOf(cube, saveImgs):
    """
    Generate and save the envelope, instantaneous phase and instantaneous frequency of cube
    """
    analiticCube = analiticOf(cube)

    envelopeCube = envelopeOf(analiticCube)
    np.save('envelope', envelopeCube)
    if saveImgs:
        saveNormalizedCubeImg(envelopeCube, 'envelopeCube')
    
    envelopeCube = None

    instantaneousPhaseCube = instantaneousPhaseOf(analiticCube)
    np.save('instPhase', instantaneousPhaseCube)
    if saveImgs:
        saveNormalizedCubeImg(instantaneousPhaseCube, 'instantaneousPhaseCube')
    
    instantaneousFrequencyCube = instantaneousFrequencyOf(instantaneousPhaseCube)
    np.save('instFreq', instantaneousFrequencyCube)

    if saveImgs:
        saveNormalizedCubeImg(instantaneousFrequencyCube, 'instantaneousFrequencyCube')
    
    instantaneousFrequencyCube = None
    instantaneousPhaseCube = None
    

    analiticCube = None

def analiticOf(cube):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html#:~:text=analytic_signal%20%3D%20hilbert(signal)
    https://www.delftstack.com/pt/howto/numpy/imaginary-numbers-numpy/#use-o-par%C3%A2metro-dtype-para-armazenar-n%C3%BAmeros-imagin%C3%A1rios-em-numpy-arrays
    """
    analiticCube = np.zeros(cube.shape, dtype=complex)
    for x in range(cube.shape[0]):
        for y in range(cube.shape[1]):
            analytic_signal = hilbert(cube[x,y])
            analiticCube[x,y] = analytic_signal
    return analiticCube

def envelopeOf(analiticCube):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html#:~:text=amplitude_envelope%20%3D%20np.abs(analytic_signal)
    """
    envelopeCube = np.zeros(analiticCube.shape)
    for x in range(analiticCube.shape[0]):
        for y in range(analiticCube.shape[1]):
            envelope_signal = np.abs(analiticCube[x,y])
            envelopeCube[x,y] = envelope_signal

    return envelopeCube

def instantaneousPhaseOf(analiticCube):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html#:~:text=np.abs(analytic_signal)%0A%3E%3E%3E-,instantaneous_phase,-%3D%20np.unwrap(np
    """
    phaseCube = np.zeros(analiticCube.shape)
    for x in range(analiticCube.shape[0]):
        for y in range(analiticCube.shape[1]):
            instantaneous_phase = np.unwrap(np.angle(analiticCube[x,y]))
            phaseCube[x,y] = instantaneous_phase

    return phaseCube


def instantaneousFrequencyOf(phaseCube):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html#:~:text=np.angle(analytic_signal))%0A%3E%3E%3E-,instantaneous_frequency,-%3D%20(np.diff(instantaneous_phase
    """
    freqCube = np.zeros(phaseCube.shape)
    fs = phaseCube.shape[2]
    for x in range(phaseCube.shape[0]):
        for y in range(phaseCube.shape[1]):
            
            instantaneous_frequency = (np.diff(phaseCube[x,y]) / (2.0*np.pi) * fs)

            #Add a number to complete instantaneous_frequency size
            lastInstFreqNumArray = np.array([instantaneous_frequency[-1]])
            instantaneous_frequency = np.concatenate((instantaneous_frequency, lastInstFreqNumArray), axis=0)

            freqCube[x,y] = instantaneous_frequency

    return freqCube

#=====================================\Complex Attrs===========================

if __name__ == "__main__":
    """
    The first arg can be the path to a .npy cube
    """
    cube = 0

    if len(sys.argv) == 1:
        #Generates a cube
        CUBE_TYPES = ['random', 'gaussianFilter']
        CUBE_TYPE = CUBE_TYPES[0]
        cubeX = 25
        cubeY = 25
        cubeZ = 25
        cube = synthCube(cubeX, cubeY, cubeZ, CUBE_TYPE)
    else:
        cube = np.load(sys.argv[1], allow_pickle=True)
    
    SAVE_IMGS = True
    RMS_WEIGHTS_TYPES = ['sine']
    RMS_WEIGHTS_TYPE = RMS_WEIGHTS_TYPES[0]

    saveNormalizedCubeImg(cube, 'originalCube')

    applyRmsForWindowDepths([5], cube, SAVE_IMGS, RMS_WEIGHTS_TYPE)
    generateComplexAttrsOf(cube, SAVE_IMGS)