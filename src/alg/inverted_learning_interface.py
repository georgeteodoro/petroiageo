from abc import ABC, abstractmethod

import config_parser


class CompatibilityCheckable(ABC):
    """
    Maybe using a flag array? (e.g., with opencv: FLAG_A | FLAG_B | ...)
    """

    def __init__(self):
        pass

    @abstractmethod
    def compatible(self, to_compare):
        pass


class AbstractSeismicDataLoader(CompatibilityCheckable, ABC):

    @abstractmethod
    def load(self):
        pass


class AbstractPorosityDataLoader(CompatibilityCheckable, ABC):

    @abstractmethod
    def load(self):
        pass


class AbstractExpandAlg(CompatibilityCheckable, ABC):

    @abstractmethod
    def expand_points(self, porosity_data, it):
        pass


class AbstractFeatureSelectionAlg(CompatibilityCheckable, ABC):

    @abstractmethod
    def feature_selection(self, features_dict, porosity_data, it):
        pass


class AbstractApplyAlg(CompatibilityCheckable, ABC):

    @abstractmethod
    def perform_prediction(self):
        pass


class BaseInvertedLearning:

    def __init__(self, seismic_data_loader: AbstractSeismicDataLoader,
                 porosity_data_loader: AbstractPorosityDataLoader,
                 expand_alg: AbstractExpandAlg,
                 feature_selection_alg: AbstractFeatureSelectionAlg,
                 apply_alg: AbstractApplyAlg, config: config_parser.Config):
        self._config = config

        # Set strategy objects up
        self._seismic_data_loader = seismic_data_loader
        self._porosity_data_loader = porosity_data_loader
        self._expand_alg = expand_alg
        self._feature_selection_alg = feature_selection_alg
        self._apply_alg = apply_alg

        # Assert strategies compatibility
        assert self._seismic_data_loader.compatible(self._porosity_data_loader)

    def run(self):
        # Retrieve config parameters
        starting_it = self._config.alg['starting_it']
        max_iteration = starting_it + self._config.alg['num_its'] + 1

        # Load seismic data
        print('loading seismic')
        features_dict = self._seismic_data_loader.load()

        # Load porosity data
        print('loading porosity')
        porosity_data = self._porosity_data_loader.load()

        # Perform the required iterations
        for it in range(starting_it, max_iteration):
            self._expand_alg.expand_points(porosity_data, it)

            best_features_set = self._feature_selection_alg.feature_selection(
                features_dict, porosity_data, it)

            return

            self._apply_alg.perform_prediction(best_features_set,
                                               features_dict, porosity_data)
