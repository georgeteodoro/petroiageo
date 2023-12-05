import numpy as np


def propagate(it, best_features, config):
    '''
    Propagates the wavefront a single ring. Initial data have no 'expanded' data.
    '''

    window_size = self._config.alg["window"]
    disp_cube_shape = (
        window_size * 2 + 1,
        window_size * 2 + 1,
        window_size * 2 + 1,
    )

    #
