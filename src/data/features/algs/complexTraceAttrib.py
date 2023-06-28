import numpy as np
from scipy.signal import hilbert


def analiticOf(cube):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html#:~:text=analytic_signal%20%3D%20hilbert(signal)
    https://www.delftstack.com/pt/howto/numpy/imaginary-numbers-numpy/#use-o-par%C3%A2metro-dtype-para-armazenar-n%C3%BAmeros-imagin%C3%A1rios-em-numpy-arrays
    """
    analiticCube = np.zeros(cube.shape, dtype=complex)
    for x in range(cube.shape[0]):
        for y in range(cube.shape[1]):
            analytic_signal = hilbert(cube[x, y])
            analiticCube[x, y] = analytic_signal
    return analiticCube


def envelopeOf(analiticCube):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html#:~:text=amplitude_envelope%20%3D%20np.abs(analytic_signal)
    """
    envelopeCube = np.zeros(analiticCube.shape)
    for x in range(analiticCube.shape[0]):
        for y in range(analiticCube.shape[1]):
            envelope_signal = np.abs(analiticCube[x, y])
            envelopeCube[x, y] = envelope_signal

    return envelopeCube


def instantaneousFrequencyOf(analiticCube):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html#:~:text=np.angle(analytic_signal))%0A%3E%3E%3E-,instantaneous_frequency,-%3D%20(np.diff(instantaneous_phase
    """
    phaseCube = instantaneousPhaseOf(analiticCube)

    freqCube = np.zeros(phaseCube.shape)
    fs = phaseCube.shape[2]
    for x in range(phaseCube.shape[0]):
        for y in range(phaseCube.shape[1]):
            phaseCubeTrace = phaseCube[x, y]
            instantaneous_frequency = (
                np.diff(phaseCubeTrace) / (2.0 * np.pi) * fs
            )

            # Add a number to complete instantaneous_frequency size
            lastInstFreqNumArray = np.array([instantaneous_frequency[-1]])
            instantaneous_frequency = np.concatenate(
                (instantaneous_frequency, lastInstFreqNumArray), axis=0
            )

            freqCube[x, y] = instantaneous_frequency

    return freqCube


def instantaneousPhaseOf(analiticCube):
    """
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.hilbert.html#:~:text=np.abs(analytic_signal)%0A%3E%3E%3E-,instantaneous_phase,-%3D%20np.unwrap(np
    """
    phaseCube = np.zeros(analiticCube.shape)
    for x in range(analiticCube.shape[0]):
        for y in range(analiticCube.shape[1]):
            analiticCubeTrace = analiticCube[x, y]
            instantaneous_phase = np.unwrap(np.angle(analiticCubeTrace))
            phaseCube[x, y] = instantaneous_phase

    return phaseCube
