from abc import ABC, abstractmethod
from enum import Enum, auto

import config_parser


class CompatibilityCheckable(ABC):
    """
    Maybe using a flag array? (e.g., with opencv: FLAG_A | FLAG_B | ...)
    """

    class _AlgType(Enum):
        SEISMIC_LOADER = auto()
        POROSITY_LOADER = auto()
        EXPAND_ALG = auto()
        FEATURE_SELECTION_ALG = auto()
        APPLY_ALG = auto()

    def __init__(self):
        self._alg_type = None
        self._using_h5 = False

    def compatible(self, to_compare):
        """
        Compare a single object or a list of objects.
        A '_single_compatible()' template method is required from 
        the subclasses.
        """

        compatible = False

        if (type(to_compare) == list) or (type(to_compare) == tuple):
            for comp in to_compare:
                compatible |= self._single_compatible(comp)
        else:
            compatible |= self._single_compatible(to_compare)

        return compatible

    @abstractmethod
    def _single_compatible(self, to_compare):
        pass


class AbstractSeismicDataLoader(CompatibilityCheckable, ABC):

    def __init__(self):
        # Compatibility flags:
        super().__init__()
        self._alg_type = CompatibilityCheckable._AlgType.SEISMIC_LOADER

    @abstractmethod
    def load(self):
        pass


class AbstractPorosityDataLoader(CompatibilityCheckable, ABC):

    def __init__(self):
        # Compatibility flags:
        super().__init__()
        self._alg_type = CompatibilityCheckable._AlgType.POROSITY_LOADER

    @abstractmethod
    def load(self):
        pass


class AbstractExpandAlg(CompatibilityCheckable, ABC):

    def __init__(self):
        # Compatibility flags:
        super().__init__()
        self._alg_type = CompatibilityCheckable._AlgType.EXPAND_ALG

    @abstractmethod
    def expand_points(self, porosity_data, it):
        pass


class AbstractFeatureSelectionAlg(CompatibilityCheckable, ABC):

    def __init__(self):
        # Compatibility flags:
        super().__init__()
        self._alg_type = CompatibilityCheckable._AlgType.FEATURE_SELECTION_ALG

    @abstractmethod
    def feature_selection(self, features_dict, porosity_data, it):
        pass


class AbstractApplyAlg(CompatibilityCheckable, ABC):

    def __init__(self):
        # Compatibility flags:
        super().__init__()
        self._alg_type = CompatibilityCheckable._AlgType.APPLY_ALG

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

        # Simple func to return a list 'l' except value 'v'
        def _except_l(l, v):
            return [x for x in l if x != v]

        # Assert strategies compatibility.
        # Each algorithm is tested against all others
        all_algs = [
            seismic_data_loader, porosity_data_loader, expand_alg,
            feature_selection_alg, apply_alg
        ]
        assert seismic_data_loader.compatible(
            _except_l(all_algs, seismic_data_loader))
        assert porosity_data_loader.compatible(
            _except_l(all_algs, porosity_data_loader))
        assert expand_alg.compatible(_except_l(all_algs, expand_alg))
        assert feature_selection_alg.compatible(
            _except_l(all_algs, feature_selection_alg))
        assert apply_alg.compatible(_except_l(all_algs, apply_alg))

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
                                               features_dict, porosity_data,
                                               it)
