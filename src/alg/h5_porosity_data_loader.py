import h5py

from inverted_learning_interface import AbstractPorosityDataLoader
import config_parser
from datasets_names import POROSITY_DSET_NAME


class H5PorosityDataLoader(AbstractPorosityDataLoader):
    """
    Porosity hypercube data loader for HDF5 files.
    Porosity hypercube must have the same shape and chunking of features. This
    is not checked, and if untrue can break the application.
    If running on MPI, the MPI communicator should be local to all processes
    within the same node. Local files between distributed nodes are (obviously)
    different, and will break the file opening with a synchronization error.

    No compute intensive tasks are done outside 'load()', allowing better
    performance profiling.

    Only the File object is stored internally, for later closure.
    The File object is closed on '__del__', thus encapsulating anything
    h5-related to this class.
    """

    def __init__(self, config: config_parser.Config):
        self._config = config
        self._porosity_cube_file = None

        # Compatibility flags:
        super().__init__()
        self._using_h5 = True

    def load(self) -> h5py.Dataset:
        # For MPI_FILE_OPEN, used by hdf5 with mpi, all files must be opened
        # with the same access/mode: existing file with write permission
        # However, only one process should update this porosity_data_h5
        # structure.
        write_str = "r+"

        self._porosity_cube_file = h5py.File(
            self._config.starting_porosity_cube_path,
            write_str,
            driver="mpio",
            comm=self._config.get_param("mpi_local_comm"),
        )
        porosity_cube_dset = self._porosity_cube_file[POROSITY_DSET_NAME]

        return porosity_cube_dset

    def _single_compatible(self, to_compare):
        # Check if to_compare have h5 support
        compatible = to_compare._using_h5

        return compatible

    def __del__(self):
        self._porosity_cube_file.close()
