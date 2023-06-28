import numpy as np
import scipy.ndimage
import scipy.signal
from joblib import Parallel, delayed
from timeit import default_timer as timer
from datetime import datetime


def rows_range_for_cpus(n_cpu, step):
    return [(round(step * i), round(step * (i + 1))) for i in range(n_cpu)]


def rows_range_for_applying_window(window_size, length, rows_range):
    return [
        (max(0, min_id - window_size), min(max_id + window_size, length))
        for min_id, max_id in rows_range
    ]


def data_slices(data, ranges):
    return [data[min_id:max_id, :] for min_id, max_id in ranges]


def slice_it(data, window, n_cpu):
    side = window[0]
    length = data.shape[0]
    step = length / n_cpu
    rows = rows_range_for_cpus(n_cpu, step)
    ids = rows_range_for_applying_window(side, length, rows)
    slices = data_slices(data, ids)
    cols = [
        (
            int((i != 0)) * side,
            min(
                int((i != 0)) * side + rows[i][1] - rows[i][0],
                ids[i][1] - ids[i][0],
            ),
        )
        for i in range(n_cpu)
    ]
    return slices, rows, cols


def moving_window(data, window, func, n_cpu=1):
    """
    Applies <func> over all the volume <data> using windows of
    size <window>.
    """
    print(f"n_cpus: {n_cpu}")
    wrapped = lambda region: func(region.reshape(window))
    if n_cpu == 1:
        return scipy.ndimage.generic_filter(data, wrapped, window)

    # Aloca espaço para o resultado agregado
    result = np.zeros(data.shape)

    slices, rows, cols = slice_it(data, window, n_cpu)
    runit = lambda x: scipy.ndimage.generic_filter(x, wrapped, window)

    print(
        f'Começando execução paralela {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}'
    )
    start = timer()
    result_transposed = Parallel(n_jobs=n_cpu, verbose=10)(
        delayed(runit)(slices[i]) for i in range(n_cpu)
    )
    end = timer()
    print(f"Execução paralela levou {end-start} seconds")

    print(f'Começando agregação {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}')
    start = timer()
    for i in range(n_cpu):
        result[rows[i][0] : rows[i][1], :] = result_transposed[i][
            cols[i][0] : cols[i][1], :
        ]

    end = timer()
    print(f"Agregação levou {end-start} seconds")
    return result
