import h5py
from typing import Dict

from inverted_learning_interface import AbstractFeatureSelectionAlg
import petro_dist4_hdf5
import petro5_hdf5
import profiling
import config_parser


class H5FeatureSelectionAlg(AbstractFeatureSelectionAlg):
    """
    Feature selection algorithm for HDF5 files and MPI.
    Although MPI support is there, it is possible to run it without 'mpirun'.
    """

    def __init__(self, config: config_parser.Config):
        self._config = config

        # Compatibility flags:
        super().__init__()
        self._using_h5 = True

    @staticmethod
    def _generate_seismic_features_names(window, base_features) -> list:
        """
        Generate a list of the product of base_features and possible
        displacements.
        """
        all_features = []

        for f in base_features:
            for i in range(-window, window + 1):
                for j in range(-window, window + 1):
                    for k in range(-window, window + 1):
                        all_features.append((f, i, j, k))

        return all_features

    def feature_selection(
        self,
        features_dict_h5: Dict[str, h5py.Dataset],
        porosity_data_h5: h5py.Dataset,
        it: int,
    ):
        # Retrieve config parameters
        num_features = self._config.get_param("num_features")
        base_features = self._config.features_files_names
        assert len(base_features) > 0
        # TODO: Centralize this behavior of removing other files
        base_features = [f for f in base_features if f != ".gitkeep"]
        if num_features != 0:
            base_features = base_features[:num_features]
        window_size = self._config.get_param("window")
        mpi_size = self._config.get_param("mpi_size")
        mpi_rank = self._config.get_param("mpi_rank")
        mpi_manager_rank = self._config.get_param("mpi_manager_rank")

        # Calculate remaining variables
        all_features = self._generate_seismic_features_names(
            window_size, base_features)
        displacement_cube_shape = (
            window_size * 2 + 1,
            window_size * 2 + 1,
            window_size * 2 + 1,
        )

        msg = ""
        if mpi_size == 1:
            best_features_set, rmse_err, mae_err = petro5_hdf5.get_features_sets(
                porosity_data_h5,
                features_dict_h5,
                all_features,
                displacement_cube_shape,
                it,
                self._config,
            )
            msg += "[manager]"
        else:
            best_features_set, rmse_err, mae_err = petro_dist4_hdf5.get_features_sets(
                porosity_data_h5,
                features_dict_h5,
                all_features,
                displacement_cube_shape,
                it,
                self._config,
            )
            if mpi_rank == mpi_manager_rank:
                msg += "[manager]"

        if len(msg) > 0:  # Is a manager
            msg += f"[get_features_sets][it{it}][best-features-and-error]"
            msg += f"{best_features_set} with error: RMSE {rmse_err}"
            msg += f" MAE {mae_err}"
            print(msg)

        return best_features_set

    def _single_compatible(self, to_compare):
        # Check if to_compare have h5 support
        compatible = to_compare._using_h5

        return compatible
