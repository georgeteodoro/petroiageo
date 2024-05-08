import numpy as np
from abc import ABC, abstractmethod


class FeatureDataBase(ABC):
    '''
    Abstract feature_data class. Base for different feature_data backends.
    Feature data should be just a float value for a 3D hypercube space.
    '''
    def __init__(self):
        self._feature = None

    @abstractmethod
    def filter_coords(self, coords):
        '''
        Should return a 1D array with the feature values only, on the same
        order as the input 'coords'.
        '''
        raise Exception("[FeatureDataBase][filter_coords] "
                        "Abstract method not implemented.")
