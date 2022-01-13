import numpy as np
import scipy.ndimage
import scipy.signal

files = ["NEAR"]
windows = [(3,3,7), (3,3,9), (3,3,11), (5,5,7), (5,5,9), (5,5,11)]
algs = [False, True, True]

def moving_window(data, window, func):
    wrapped = lambda region: func(region.reshape(window))
    return scipy.ndimage.generic_filter(data, wrapped, window)

def marfurt_semblance2(region):
    region = region.reshape(-1, region.shape[-1])
    ntraces, nsamples = region.shape
    cov = region.dot(region.T)
    sembl = cov.sum() / cov.diagonal().sum()
    return sembl / ntraces

def gersztenkorn_eigenstructure(region):
    region = region.reshape(-1, region.shape[-1])
    cov = region.dot(region.T)
    vals = np.linalg.eigvalsh(cov)
    return vals.max() / vals.sum()

def flatten(data, surface, window):
    surface = scipy.ndimage.gaussian_filter(surface.astype(float), 1)
    ni, nj, nk = data.shape
    ik = np.arange(nk)
    out_ik = np.arange(window) - window // 2
    out = np.zeros((ni, nj, window))
    for i, j in np.ndindex(ni, nj):
        trace = data[i,j,:]
        k = surface[i, j]
        shifted = np.interp(out_ik + k, ik, trace)
        out[i,j,:] = shifted
    return out

def unflatten(data, surface, orig_shape):
    out = np.zeros(orig_shape)
    surface = np.clip(surface, 0, orig_shape[-1] - 1)
    win = data.shape[-1] // 2
    for i, j in np.ndindex(orig_shape[0], orig_shape[1]):
        k = int(surface[i,j])
        outmin, outmax = max(0, k - win), min(orig_shape[-1], k + win + 1)
        inmin, inmax = outmin - (k - win), k + win + 1 - outmax
        inmax = data.shape[-1] - abs(inmax)
        out[i, j, outmin:outmax] = data[i, j, inmin:inmax]
    return out

def dip_corrected(seismic, window, func):
    surface = data.load_horizon()
    flat = flatten(seismic, surface, seismic.shape[-1])
    sembl = moving_window(flat, window, func)
    return unflatten(sembl, surface, seismic.shape)

def gradients(seismic, sigma):
    grads = []
    for axis in range(3):
        grad = scipy.ndimage.gaussian_filter1d(seismic, sigma, axis=axis, order=1)
        grads.append(grad[..., np.newaxis])
    return np.concatenate(grads, axis=3)

def moving_window4d(grad, window, func):
    half_window = [(x // 2, x // 2) for x in window] + [(0, 0)]
    padded = np.pad(grad, half_window, mode='reflect')
    out = np.empty(grad.shape[:3], dtype=float)
    for i, j, k in np.ndindex(out.shape):
        region = padded[i:i+window[0], j:j+window[1], k:k+window[2], :]
        out[i,j,k] = func(region)
    return out

def gst_coherence_calc(region):
    region = region.reshape(-1, 3)
    gst = region.T.dot(region)
    eigs = np.sort(np.linalg.eigvalsh(gst))[::-1]
    return (eigs[0] - eigs[1]) / (eigs[0] + eigs[1])
        
def gst_coherence(seismic, window, sigma=1):
    grad = gradients(seismic, sigma)
    return moving_window4d(grad, window, gst_coherence_calc)

name = lambda w: "-".join(str(x) for x in w)
for f in files:
    seismic = np.load("data/"+f+".npy")
    for w in windows:
        if algs[0]:
            coh = moving_window(seismic, w, marfurt_semblance2)
            np.save("results/"+f+"_marfurt_"+name(w)+".npy", coh)
        if algs[1]:
            coh = moving_window(seismic, w, gersztenkorn_eigenstructure)
            np.save("results/"+f+"_gersztenkorn_"+name(w)+".npy", coh)
        if algs[2]:
            coh = gst_coherence(seismic, w, sigma=1)
            np.save("results/"+f+"_gst_"+name(w)+".npy", coh)
