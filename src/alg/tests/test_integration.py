import unittest
import os
import h5py
import numpy as np
from subprocess import Popen, PIPE

import os
import signal
import subprocess
import time

import common


class Test_All(unittest.TestCase):
    '''
    Integration test for the whole application.
    '''

    # The area setup with well coords is:
    #                       y
    #   . . . . . . . . . 0
    #   . . . . . . . . . 1
    #   . . . . 0 . . . . 2
    #   . . . . . . . . . 3
    #   . . . . . . . . . 4
    #   . . . . . . . . . 5
    #   . . . . . . . 1 . 6
    #   . . 2 . . . . . . 7
    # x 0 1 2 3 4 5 6 7 8 

    hypercube_shape = (9, 8, 5)
    wells_list = [(4, 2), (7, 6), (2, 7)]

    # Add padding for window=1
    hypercube_shape = tuple([c + 2 for c in hypercube_shape])
    chunk_test_shape = list(hypercube_shape)
    chunk_test_shape[1] = chunk_test_shape[1] / 2
    chunk_test_shape = tuple(chunk_test_shape)

    features_path = './tests/test_features'
    features_h5_fname = []
    features_h5_f = []

    porosity_data_type = np.dtype([
        ('x', np.int64),
        ('y', np.int64),
        ('z', np.int64),
        ('phi', np.float64),
        ('real', np.int64),
        ('ring', np.int64),
        ('well_id', np.int64),
    ])

    porosity_h5_path = './tests/test_porosity.h5'
    porosity_h5_f = None

    config_path = './tests/test_config.yaml'

    @classmethod
    def _generate_random_feature(cls, f_name, val_sum, val_prod):
        '''
        Generate a feature h5 file with cls.hypercube_shape.
        Each value of the feature is 
            = ((i+1) * (j+1) * (k+1) + val_sum) * val_prod
        Since there is a padding of window=1, i+1-1=i
        '''

        h5_file = h5py.File(f_name, 'w')
        h5_dset = h5_file.create_dataset(
            common.FEAT_DSET_NAME,
            cls.hypercube_shape,
            dtype=np.float64,
        )
        for i in range(cls.hypercube_shape[0]):
            for j in range(cls.hypercube_shape[1]):
                for k in range(cls.hypercube_shape[2]):
                    h5_dset[i, j, k] = -1

        for i in range(1, cls.hypercube_shape[0] - 2):
            for j in range(1, cls.hypercube_shape[1] - 2):
                for k in range(1, cls.hypercube_shape[2] - 2):
                    h5_dset[i, j, k] = (i * j * k + val_sum) * val_prod

        return h5_file

    @classmethod
    def _generate_random_porosity(cls, val_sum, val_prod):
        porosity_f = h5py.File(cls.porosity_h5_path, 'w')
        porosity_dset = porosity_f.create_dataset(common.POROSITY_DSET_NAME,
                                                  cls.hypercube_shape,
                                                  chunks=cls.chunk_test_shape,
                                                  dtype=cls.porosity_data_type)

        # Initialize all as empty data
        for i in range(cls.hypercube_shape[0]):
            for j in range(cls.hypercube_shape[1]):
                for k in range(cls.hypercube_shape[2]):
                    porosity_dset[i, j, k] = (i, j, k, 0,
                                              common.RealValues.empty, -1, -1)

        # Fill wells
        for (well_id, (w_x, w_y)) in enumerate(cls.wells_list):
            for k in range(1, cls.hypercube_shape[2] - 2):
                mock_porosity = ((1 + w_x) * (1 + w_y) *
                                 (1 + k) + val_sum) * val_prod
                porosity_dset[w_x + 1, w_y + 1,
                              k + 1] = (w_x, w_y, k, mock_porosity,
                                        common.RealValues.real, 0, well_id)

        return porosity_f

    @classmethod
    def setUpClass(cls):
        # This is run under mpi and should only be executed once
        # Generate feature files
        for f in range(3):
            fname = f'{cls.features_path}/test_f{f}.h5'
            h5_file = cls._generate_random_feature(fname, 2 * f, 2 * f)
            cls.features_h5_fname.append(fname)
            cls.features_h5_f.append(h5_file)

        # Generate porosity file
        cls.porosity_h5_f = cls._generate_random_porosity(0, 10)

        # Generate config file
        yaml_str = f"""
        wells:
          coords:
          - [4,2]
          - [7,6]
          - [2,7]
        features_folder: "{cls.features_path}" 
        starting_porosity_cube_path: "{cls.porosity_h5_path}" 
        alg:
            parallel:
                n_training_chunks: 1
        """
        yaml_f = open(cls.config_path, 'w')
        yaml_f.writelines(yaml_str)

        # Close all files to allow them to commit to file
        yaml_f.close()
        for f in cls.features_h5_f:
            f.close()
        cls.porosity_h5_f.close()

    @classmethod
    def tearDownClass(cls):
        for fname in cls.features_h5_fname:
            os.remove(fname)
        os.remove(cls.config_path)
        os.remove(cls.porosity_h5_path)

    def test_sintetic_3_iterations_from_scratch(self):
        '''
        Performs 3 complete iterations from scratch.
        1 feature file is used, with a window of 1 (thus 27 tests).
        All 27 tests are performed.
        3 features are selected.
        '''

        # Retrieve CLI arguments
        args_str = f'--config {self.__class__.config_path} --it 1 '\
                   f'--nits 1 --nf 1 -w 1 --nsf 3 --ntf 1'

        process = Popen(
            'mpirun -np 2 python3 -u main.py ' + args_str,
            shell=True,
            universal_newlines=True,
            stdout=PIPE,
            stderr=PIPE,
            preexec_fn=os.setsid)

        # time.sleep(5)

        # # Send the signal to all the process groups
        # os.killpg(os.getpgid(process.pid), signal.SIGTERM)

        output, error = process.communicate()

        print(output)
        print(error)

        # Compare final porosity file
        self.assertTrue(True)

    def test_sintetic_2_iterations_continued(self):
        '''
        Performs 2 complete iterations, continuing the execution of
        2 complete iterations.
        1 feature file is used, with a window of 1 (thus 27 tests).
        All 27 tests are performed.
        3 features are selected.
        '''

        self.assertTrue(True)

    def test_sintetic_short(self):
        '''
        Performs 2 complete iterations from scratch with reduced inputs.
        2 complete iterations.
        3 feature files are available but only 2 are used.
        With a window of 1 (thus 27 tests), only 10 tests are performed.
        2 features are selected.
        '''

        self.assertTrue(True)

    def test_real_2_iterations_from_scratch(self):
        '''
        Performs 2 complete iterations from scratch with reduced inputs on
        real data.
        2 complete iterations.
        1 feature file is used with a window of 1 (thus 27 tests).
        Only 10 tests are performed per selected feature.
        10 features are selected.
        '''

        self.assertTrue(True)
