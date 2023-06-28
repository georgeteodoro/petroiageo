import numpy as np
import scipy.ndimage


def gradients(seismic, sigma):
    grads = []
    for axis in range(3):
        grad = scipy.ndimage.gaussian_filter1d(
            seismic, sigma, axis=axis, order=1
        )
        grads.append(grad[..., np.newaxis])
    return np.concatenate(grads, axis=3)


def moving_window4d(grad, window, func):
    half_window = [(x // 2, x // 2) for x in window] + [(0, 0)]
    padded = np.pad(grad, half_window, mode="reflect")
    out = np.empty(grad.shape[:3], dtype=float)
    for i, j, k in np.ndindex(out.shape):
        region = padded[
            i : i + window[0], j : j + window[1], k : k + window[2], :
        ]
        out[i, j, k] = func(region)
    return out


def gst_coherence_calc(region):
    region = region.reshape(-1, 3)
    gst = region.T.dot(region)
    eigs = np.sort(np.linalg.eigvalsh(gst))[::-1]
    return (eigs[0] - eigs[1]) / (eigs[0] + eigs[1])


def gst_coherence(seismic, window, sigma=1):
    grad = gradients(seismic, sigma)
    return moving_window4d(grad, window, gst_coherence_calc)
