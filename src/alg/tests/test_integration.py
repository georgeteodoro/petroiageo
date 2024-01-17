import unittest
import os
import h5py
import numpy as np
from subprocess import Popen, PIPE
import pytest

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
    # All coords are with padding
    #                       y
    #   . . . . . . . . . 1
    #   . . . . . . . . . 2
    #   . . . . 0 . . . . 3
    #   . . . . . . . . . 4
    #   . . . . . . . . . 5
    #   . . . . . . . . . 6
    #   . . . . . . . 1 . 7
    #   . . 2 . . . . . . 8
    # x 1 2 3 4 5 6 7 8 9

    # it 1 :
    #                       y
    #   . . . . . . . . . 1
    #   . . . 0 0 0 . . . 2
    #   . . . 0 x 0 . . . 3
    #   . . . 0 0 0 . . . 4
    #   . . . . . . . . . 5
    #   . . . . . . 1 1 1 6
    #   . 2 2 2 . . 1 x 1 7
    #   . 2 x 2 . . 1 1 1 8
    # x 1 2 3 4 5 6 7 8 9

    # it 2:
    #                       y
    #   . . 0 0 0 0 0 . . 1
    #   . . 0 0 0 0 0 . . 2
    #   . . 0 0 x 0 0 . . 3
    #   . . 0 0 0 0 0 . . 4
    #   . . 0 0 0 0 0 1 1 5
    #   2 2 2 2 2 1 1 1 1 6
    #   2 2 2 2 2 1 1 x 1 7
    #   2 2 x 2 2 1 1 1 1 8
    # x 1 2 3 4 5 6 7 8 9

    # it 3:
    #                       y
    #   . 0 0 0 0 0 0 0 . 1
    #   . 0 0 0 0 0 0 0 . 2
    #   . 0 0 0 x 0 0 0 . 3
    #   . 0 0 0 0 0 0 0 1 4
    #   2 0 0 0 0 0 0 1 1 5
    #   2 2 2 2 2 1 1 1 1 6
    #   2 2 2 2 2 1 1 x 1 7
    #   2 2 x 2 2 1 1 1 1 8
    # x 1 2 3 4 5 6 7 8 9

    # it 4(with padded coords):
    #                       y
    #   0 0 0 0 0 0 0 0 0 1
    #   0 0 0 0 0 0 0 0 0 2
    #   0 0 0 0 x 0 0 0 0 3
    #   0 0 0 0 0 0 0 0 1 4
    #   2 0 0 0 0 0 0 1 1 5
    #   2 2 2 2 2 1 1 1 1 6
    #   2 2 2 2 2 1 1 x 1 7
    #   2 2 x 2 2 1 1 1 1 8
    # x 1 2 3 4 5 6 7 8 9

    hypercube_shape = (9, 8, 5)
    wells_list = [(2, 7), (4, 2), (7, 6)]

    # Add padding for window=1
    window = 1
    hypercube_shape_padded = tuple([c + (2 * 1) for c in hypercube_shape])

    chunk_test_shape = list(hypercube_shape_padded)
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

    config_path = './tests/test_config.yaml'

    # =========================================================================
    # === Tests ===============================================================
    # =========================================================================

    def test_sintetic_base(self):
        '''
        Performs 1 iteration from scratch.
        1 feature file is used, with a window of 1 (thus 27 trials).
        Only 1/27 trial is performed.
        3 features are selected.
        '''

        # Retrieve CLI arguments
        args_str = f'--config {self.__class__.config_path} --it 1 '\
                   f'--nits 1 --nf 1 -w 1 --nsf 3 --ntf 1'

        process = Popen('mpirun -np 2 python3 -u main.py ' + args_str,
                        shell=True,
                        universal_newlines=True,
                        stdout=PIPE,
                        stderr=PIPE,
                        preexec_fn=os.setsid)

        # time.sleep(2)

        # # Send the signal to all the process groups
        # os.killpg(os.getpgid(process.pid), signal.SIGTERM)

        output, error = process.communicate()

        print(output)
        print(error)
        assert len(error) == 0

        porosity_h5_file = h5py.File(self.__class__.porosity_h5_path, 'r')
        porosity_dset = porosity_h5_file[common.POROSITY_DSET_NAME]

        # print(porosity_dset[porosity_dset['real'] != common.RealValues.empty])
        # print(porosity_dset[porosity_dset['well_id'] == 1])

        # Validate propagation (see diagrams on the TestClass beginning)
        depth = self.__class__.hypercube_shape[2]
        assert sum(sum(sum(porosity_dset['well_id'] == 0))) == depth * 9
        assert sum(sum(sum(porosity_dset['well_id'] == 1))) == depth * 9
        assert sum(sum(sum(porosity_dset['well_id'] == 2))) == depth * 6
        porosity_h5_file.close()

    def test_sintetic_all_iterations_from_scratch(self):
        '''
        Performs 3 complete iterations from scratch.
        1 feature file is used, with a window of 1 (thus 27 trials).
        All 27 trials are performed.
        3 features are selected.
        '''

        # Retrieve CLI arguments
        args_str = f'--config {self.__class__.config_path} --it 1 '\
                   f'--nits 4 --nf 1 -w 1 --nsf 3'

        process = Popen('mpirun -np 2 python3 -u main.py ' + args_str,
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
        assert len(error) == 0

        # Validate propagation (see diagrams on the TestClass beginning)
        porosity_h5_file = h5py.File(self.__class__.porosity_h5_path, 'r')
        porosity_dset = porosity_h5_file[common.POROSITY_DSET_NAME]

        depth = self.__class__.hypercube_shape[2]
        assert sum(sum(sum(porosity_dset['well_id'] == 0))) == depth * 41
        assert sum(sum(sum(porosity_dset['well_id'] == 1))) == depth * 15
        assert sum(sum(sum(porosity_dset['well_id'] == 2))) == depth * 16
        porosity_h5_file.close()

    def test_sintetic_2_iterations_continued(self):
        '''
        Performs 2 complete iterations, continuing the execution of
        2 complete iterations.
        1 feature file is used, with a window of 1 (thus 27 trials).
        All 27 trials are performed.
        3 features are selected.
        '''

        # Retrieve CLI arguments
        args1_str = f'--config {self.__class__.config_path} --it 1 '\
                   f'--nits 2 --nf 1 -w 1 --nsf 3'
        args2_str = f'--config {self.__class__.config_path} --it 3 '\
                   f'--nits 2 --nf 1 -w 1 --nsf 3'

        process = Popen('mpirun -np 2 python3 -u main.py ' + args1_str,
                        shell=True,
                        universal_newlines=True,
                        stdout=PIPE,
                        stderr=PIPE,
                        preexec_fn=os.setsid)

        output, error = process.communicate()
        print(output)
        print(error)

        # Validate propagation (see diagrams on the TestClass beginning)
        porosity_h5_file = h5py.File(self.__class__.porosity_h5_path, 'r')
        porosity_dset = porosity_h5_file[common.POROSITY_DSET_NAME]

        depth = self.__class__.hypercube_shape[2]

        assert sum(sum(sum(porosity_dset['well_id'] == 0))) == depth * 25
        assert sum(sum(sum(porosity_dset['well_id'] == 1))) == depth * 14
        assert sum(sum(sum(porosity_dset['well_id'] == 2))) == depth * 15
        porosity_h5_file.close()

        process = Popen('mpirun -np 2 python3 -u main.py ' + args2_str,
                        shell=True,
                        universal_newlines=True,
                        stdout=PIPE,
                        stderr=PIPE,
                        preexec_fn=os.setsid)
        output, error = process.communicate()
        print(output)
        print(error)
        assert len(error) == 0

        # Validate propagation (see diagrams on the TestClass beginning)
        porosity_h5_file = h5py.File(self.__class__.porosity_h5_path, 'r')
        porosity_dset = porosity_h5_file[common.POROSITY_DSET_NAME]

        depth = self.__class__.hypercube_shape[2]
        assert sum(sum(sum(porosity_dset['well_id'] == 0))) == depth * 41
        assert sum(sum(sum(porosity_dset['well_id'] == 1))) == depth * 15
        assert sum(sum(sum(porosity_dset['well_id'] == 2))) == depth * 16
        porosity_h5_file.close()

    def test_sintetic_sampling(self):
        '''
        Performs 3 iteration from scratch, followed by the 4th iteration 
        with sampling.
        1 feature file is used, with a window of 1 (thus 27 trials).
        Only 1/27 trial is performed.
        3 features are selected.
        '''

        # Retrieve CLI arguments
        args_str = f'--config {self.__class__.config_path} --it 1 '\
                   f'--nits 3 --nf 1 -w 1 --nsf 3 --ntf 1'

        # Perform first
        process = Popen('mpirun -np 2 python3 -u main.py ' + args_str,
                        shell=True,
                        universal_newlines=True,
                        stdout=PIPE,
                        stderr=PIPE,
                        preexec_fn=os.setsid)

        output, error = process.communicate()

        print(output)
        print(error)
        assert len(error) == 0

        # ===========================================================

        # Generate custom config file with sampling
        yaml_str = f"""
        wells:
          coords:
          - [4,2]
          - [7,6]
          - [2,7]
          window: 1
        features_folder: "{self.__class__.features_path}" 
        starting_porosity_cube_path: "{self.__class__.porosity_h5_path}" 
        alg:
          parallel:
            n_training_chunks: 1
          sampling:
            sampler: v1
            max_points: 10
            seed: 0
        """
        yaml_f = open(self.__class__.config_path, 'w')
        yaml_f.writelines(yaml_str)

        # Close all files to allow them to commit to file
        yaml_f.close()

        # Coords chosen for training when sampling with the above configs.
        # These are hand-filled, and should the shape or rng seed change,
        # these will also be different.
        sampled_coords = [(1, 5, 1), (1, 6, 1), (2, 7, 1), (3, 8, 4), (4, 1, 2),
                          (5, 3, 2), (5, 7, 2), (6, 2, 1), (7, 4, 1),
                          (7, 6, 2), (7, 8, 1), (6, 6, 2), (8, 2, 4),
                          (8, 7, 1), (9, 4, 2)]

        # Update porosity file. All points which should not be visited since
        # they were not sampled are set phi=NaN. This breaks the execution if
        # any value outside the sampling is visited.
        porosity_h5_file = h5py.File(self.__class__.porosity_h5_path, 'r+')
        porosity_dset = porosity_h5_file[common.POROSITY_DSET_NAME]
        for x in range(0, self.__class__.hypercube_shape[0]):
            for y in range(0, self.__class__.hypercube_shape[1]):
                for z in range(0, self.__class__.hypercube_shape[2]):
                    if (x, y, z) not in sampled_coords:
                        porosity_dset[x, y, z, 'phi'] = np.nan
        porosity_h5_file.close()

        # Retrieve CLI arguments
        args_str = f'--config {self.__class__.config_path} --it 4 '\
                   f'--nits 1 --nf 1 -w 1 --nsf 3 --ntf 1 --no-abort'

        process = Popen('mpirun -np 2 python3 -u main.py ' + args_str,
                        shell=True,
                        universal_newlines=True,
                        stdout=PIPE,
                        stderr=PIPE,
                        preexec_fn=os.setsid)

        time.sleep(2)

        # Send the signal to all the process groups
        os.killpg(os.getpgid(process.pid), signal.SIGTERM)

        output, error = process.communicate()

        print(output)
        print(error)
        assert len(error) == 0

        porosity_h5_file = h5py.File(self.__class__.porosity_h5_path, 'r')
        porosity_dset = porosity_h5_file[common.POROSITY_DSET_NAME]

        # print(porosity_dset[porosity_dset['real'] != common.RealValues.empty])
        # print(porosity_dset[porosity_dset['well_id'] == 1])

        # Validate propagation (see diagrams on the TestClass beginning)
        depth = self.__class__.hypercube_shape[2]
        assert sum(sum(sum(porosity_dset['well_id'] == 0))) == depth * 41
        assert sum(sum(sum(porosity_dset['well_id'] == 1))) == depth * 15
        assert sum(sum(sum(porosity_dset['well_id'] == 2))) == depth * 16
        porosity_h5_file.close()

    # =========================================================================
    # === Setup/teardown ======================================================
    # =========================================================================

    @classmethod
    def setUpClass(cls):
        '''
        Only features and config files are created.
        They aren't updated, so 1 creation per session is enough.
        '''

        # Generate feature files
        for f in range(3):
            fname = f'{cls.features_path}/test_f{f}.h5'
            h5_file = cls._generate_random_feature(fname, 2 * f, 2 * f)
            cls.features_h5_fname.append(fname)
            cls.features_h5_f.append(h5_file)
            h5_file.close()

        # Generate config file
        yaml_str = f"""
        wells:
          coords:
          - [4,2]
          - [7,6]
          - [2,7]
          window: 1
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

    @classmethod
    def tearDownClass(cls):
        for fname in cls.features_h5_fname:
            print(fname)
            os.remove(fname)
        os.remove(cls.config_path)

    @pytest.fixture(scope="function", autouse=True)
    def all_tests_prep(request):
        '''
        All test should use a brand new porosity file.
        '''
        Test_All.setup_porosity()
        yield
        Test_All.teardown_porosity()

    @classmethod
    def setup_porosity(cls):
        # Generate porosity file
        porosity_h5_f = cls._generate_random_porosity(0, 10)

        for f in cls.features_h5_f:
            f.close()
        porosity_h5_f.close()

    @classmethod
    def teardown_porosity(cls):
        os.remove(cls.porosity_h5_path)
        # pass

    # =========================================================================
    # === Helper functions ====================================================
    # =========================================================================

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
            cls.hypercube_shape_padded,
            dtype=np.float64,
        )
        for i in range(cls.hypercube_shape_padded[0]):
            for j in range(cls.hypercube_shape_padded[1]):
                for k in range(cls.hypercube_shape_padded[2]):
                    h5_dset[i, j, k] = -1

        for i in range(1, cls.hypercube_shape[0] + 1):
            for j in range(1, cls.hypercube_shape[1] + 1):
                for k in range(1, cls.hypercube_shape[2] + 1):
                    h5_dset[i, j, k] = (i * j * k + val_sum) * val_prod

        return h5_file

    @classmethod
    def _generate_random_porosity(cls, val_sum, val_prod):
        porosity_f = h5py.File(cls.porosity_h5_path, 'w')
        porosity_dset = porosity_f.create_dataset(common.POROSITY_DSET_NAME,
                                                  cls.hypercube_shape_padded,
                                                  chunks=cls.chunk_test_shape,
                                                  dtype=cls.porosity_data_type)

        # Initialize all as empty data
        for i in range(cls.hypercube_shape_padded[0]):
            for j in range(cls.hypercube_shape_padded[1]):
                for k in range(cls.hypercube_shape_padded[2]):
                    porosity_dset[i, j, k] = (i, j, k, 0,
                                              common.RealValues.empty, -1, -1)

        # Fill wells
        for (well_id, (w_x, w_y)) in enumerate(cls.wells_list):
            for k in range(cls.window, cls.hypercube_shape[2] + cls.window):
                mock_porosity = ((1 + w_x + cls.window) *
                                 (1 + w_y + cls.window) *
                                 (1 + k + cls.window) + val_sum) * val_prod
                porosity_dset[w_x + cls.window, w_y + cls.window,
                              k] = (w_x + cls.window, w_y + cls.window, k,
                                    mock_porosity, common.RealValues.real, 0,
                                    well_id)

        return porosity_f