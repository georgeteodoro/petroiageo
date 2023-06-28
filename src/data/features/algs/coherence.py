from algs.utils import moving_window
import numpy as np


def marfurt_semblance(region):
    region = region.reshape(-1, region.shape[-1])
    ntraces, nsamples = region.shape
    cov = region.dot(region.T)
    sembl = cov.sum() / cov.diagonal().sum()
    return sembl / ntraces


def gersztenkorn(region):
    region = region.reshape(-1, region.shape[-1])
    cov = region.dot(region.T)
    vals = np.linalg.eigvalsh(cov)
    return vals.max() / vals.sum()
