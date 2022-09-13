import numpy as np
import scipy.ndimage
import scipy.signal
from joblib import Parallel, delayed


def sliceit(data, window, n_cpu):
    side = window[0]
    length = data.shape[0]
    step = length / n_cpu
    rows = [(round(step*i), round(step*(i+1))) for i in range(n_cpu)]
    ids = [(max(0,a-side), min(b+side,length)) for a,b in rows]
    slices = [data[a:b,:] for a,b in ids]
    cols = [
        (int((i!=0))*side, min(int((i!=0))*side + rows[i][1] - rows[i][0], ids[i][1] - ids[i][0]))
        for i in range(n_cpu)
    ]
    return slices, rows, cols


def moving_window(data, window, func, n_cpu=1):
    """
    Applies <func> over all the volume <data> using windows of
    size <window>.
    """
    wrapped = lambda region: func(region.reshape(window))
    if n_cpu == 1:
        return scipy.ndimage.generic_filter(data, wrapped, window)
    
    result = np.zeros(data.shape)
    slices, rows, cols = sliceit(data, window, n_cpu)
    runit = lambda x: scipy.ndimage.generic_filter(x, wrapped, window)
    result_transposed = Parallel(n_jobs=n_cpu, verbose=0)(
        delayed(runit)(slices[i]) for i in range(n_cpu)
    )

    for i in range(n_cpu):
        result[rows[i][0]:rows[i][1],:] = result_transposed[i][cols[i][0]:cols[i][1],:]
        
    return result