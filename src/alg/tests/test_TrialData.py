import unittest
import h5py
import os
import numpy as np
from parameterized import parameterized
from math import prod

from TrialDataNumpy import TrialDataNumpy
from data_filter import WellsSingleRingDataFilter
from FeatureDataH5 import FeatureDataH5
from config_parser import YAMLConfig
import common

concrete_classes = [TrialDataNumpy]


class Test_TrialDataAll(unittest.TestCase):

    # The area setup with well coords is:
    # All coords are with padding
    #             y
    #   . . . . . 1
    #   . . . . . 2
    #   0 . . . . 3
    #   . . 1 . . 4
    #   . . . . . 5
    #   . . . . . 6
    #   . . . . . 7
    # x 1 2 3 4 5

    # it 1
    #   . . . . . 1
    #   0 0 . . . 2
    #   x 0 1 1 . 3
    #   0 0 x 1 . 4
    #   . 1 1 1 . 5
    #   . . . . . 6
    #   . . . . . 7
    # x 1 2 3 4 5

    # it 2
    #   0 0 0 . . 1
    #   0 0 0 1 1 2
    #   x 0 1 1 1 3
    #   0 0 x 1 1 4
    #   0 1 1 1 1 5
    #   1 1 1 1 1 6
    #   . . . . . 7
    # x 1 2 3 4 5

    # All data info
    hypercube_test_shape = (5, 7, 5)

    # Window is hardcoded on the padding comprehension
    window = 1
    hypercube_test_shape_padded = [c + 2 * 1 for c in hypercube_test_shape]

    chunk_test_shape = (5, 4, 5)
    POROSITY_FILENAME = 'test_porosity.h5'
    FEATURE_FILENAME1 = 'test_feature1.h5'
    FEATURE_FILENAME2 = 'test_feature2.h5'
    config = None

    # Porosity data structures
    porosity_h5_f = None
    porosity_h5_dset = None

    # Feature data structures
    feature1_h5_f = None
    feature1_h5_dset = None
    feature2_h5_f = None
    feature2_h5_dset = None

    @classmethod
    def _fill_ring(cls, r, well_id, w_x, w_y):
        i_beg = max(w_x - r, cls.window)
        i_end = min(w_x + r + 1, cls.hypercube_test_shape_padded[0] - 1)
        j_beg = max(w_y - r, cls.window)
        j_end = min(w_y + r + 1, cls.hypercube_test_shape_padded[1] - 1)
        for i in range(i_beg, i_end):
            for j in range(j_beg, j_end):
                for k in range(
                        cls.window,
                        cls.hypercube_test_shape_padded[2] - cls.window):
                    if cls.porosity_h5_dset[
                            i, j, k]['real'] != common.RealValues.empty:
                        continue

                    # Only add points which are on the ring and were not
                    # previously propagated
                    if (i == i_beg or i == i_end - 1 or j == j_beg or
                            j == j_end - 1):
                        mock_porosity = w_x * w_y * k
                        cls.porosity_h5_dset[i, j, k] = (
                            i, j, k, mock_porosity,
                            common.RealValues.propagated, r, well_id)

    @classmethod
    def setUpClass(cls):
        # Create config object
        yaml_str = f"""
        wells:
          coords:
          - [0,2]
          - [2,3]
          window: {cls.window}
        alg:
          max_num_features: 2
          parallel:
            n_training_chunks: 1
        """
        cls.config = YAMLConfig(config_str=yaml_str)
        wells_list = cls.config.train_wells_coords

        # Create the porosity h5 file
        cls.porosity_h5_f = h5py.File(cls.POROSITY_FILENAME, 'w')

        # Create the dataset within the h5
        # TODO: Refactor: Move this to common and standardize this type
        porosity_data_type = np.dtype([
            ('x', np.int64),
            ('y', np.int64),
            ('z', np.int64),
            ('phi', np.float64),
            ('real', np.int64),
            ('ring', np.int64),
            ('well_id', np.int64),
        ])
        cls.porosity_h5_dset = cls.porosity_h5_f.create_dataset(
            common.POROSITY_DSET_NAME,
            cls.hypercube_test_shape_padded,
            dtype=porosity_data_type,
            chunks=cls.chunk_test_shape,
        )

        # Initialize all as empty data
        for i in range(cls.hypercube_test_shape_padded[0]):
            for j in range(cls.hypercube_test_shape_padded[1]):
                for k in range(cls.hypercube_test_shape_padded[2]):
                    cls.porosity_h5_dset[i, j,
                                         k] = (i, j, k, 0,
                                               common.RealValues.empty, -1, -1)

        # Fill wells
        for (well_id, (w_x, w_y)) in enumerate(wells_list):
            for k in range(cls.window,
                           cls.hypercube_test_shape[2] + cls.window):
                # # Well 1 would not have points throughout the whole depth
                # if well_id == 1 and (k == 1 or k == 4):
                #     continue

                mock_porosity = w_x * w_y * k
                cls.porosity_h5_dset[w_x, w_y,
                                     k] = (w_x, w_y, k, mock_porosity,
                                           common.RealValues.real, 0, well_id)

        # Fill more rings
        for r in range(1, 3):
            for (well_id, (w_x, w_y)) in enumerate(wells_list):
                cls._fill_ring(r, well_id, w_x, w_y)

        # Create the feature h5 file
        cls.feature1_h5_f = h5py.File(cls.FEATURE_FILENAME1, 'w')

        # Create the dataset within the h5
        cls.feature1_h5_dset = cls.feature1_h5_f.create_dataset(
            common.FEAT_DSET_NAME,
            cls.hypercube_test_shape_padded,
            dtype=np.float64,
        )

        # Initialize feature data
        for i in range(cls.hypercube_test_shape_padded[0]):
            for j in range(cls.hypercube_test_shape_padded[1]):
                for k in range(cls.hypercube_test_shape_padded[2]):
                    cls.feature1_h5_dset[i, j, k] = i * j * k + 1

        # Create the feature h5 file
        cls.feature2_h5_f = h5py.File(cls.FEATURE_FILENAME2, 'w')

        # Create the dataset within the h5
        cls.feature2_h5_dset = cls.feature2_h5_f.create_dataset(
            common.FEAT_DSET_NAME,
            cls.hypercube_test_shape_padded,
            dtype=np.float64,
        )

        # Initialize feature data
        for i in range(cls.hypercube_test_shape_padded[0]):
            for j in range(cls.hypercube_test_shape_padded[1]):
                for k in range(cls.hypercube_test_shape_padded[2]):
                    cls.feature2_h5_dset[i, j, k] = i * j * k + 10

    @classmethod
    def tearDownClass(cls):
        # Delete any old file
        cls.porosity_h5_f.close()
        os.remove(cls.POROSITY_FILENAME)
        cls.feature1_h5_f.close()
        os.remove(cls.FEATURE_FILENAME1)
        cls.feature2_h5_f.close()
        os.remove(cls.FEATURE_FILENAME2)

    @parameterized.expand(concrete_classes)
    def test_list_init(self, test_cls):
        '''
        Test the creation of a new type.
        '''
        # Load class data
        porosity_dset = self.__class__.porosity_h5_dset
        config = self.__class__.config
        wells_list = config.train_wells_ids

        td1 = test_cls(wells_list, porosity_dset, config)

        td2 = test_cls(wells_list, porosity_dset, config)

        self.assertTrue(True)

    @parameterized.expand(concrete_classes)
    def test_prepare_porosity_r1(self, test_cls):
        '''
        Test the use of prepare_porosity() the first ring.
        '''
        # Load class data
        porosity_dset = self.__class__.porosity_h5_dset
        config = self.__class__.config
        wells_list = config.train_wells_ids

        # Create TestData object
        td1 = test_cls(wells_list, porosity_dset, config)

        # Prepare first porosity
        td1.prepare_porosity(1)

        # Only one ring exists
        self.assertEqual(len(td1._data), 1)

        # Check if all points were added
        # Total of 2 full depths
        self.assertEqual(td1._ring_size(0),
                         self.__class__.hypercube_test_shape[2] * 2)

    @parameterized.expand(concrete_classes)
    def test_prepare_porosity_r2_3(self, test_cls):
        '''
        Test the use of prepare_porosity() by adding 3 rings,
        one at a time, from ring 1-3.
        '''
        # Load class data
        porosity_dset = self.__class__.porosity_h5_dset
        config = self.__class__.config
        wells_list = config.train_wells_ids

        # Create TestData object
        td1 = test_cls(wells_list, porosity_dset, config)

        # Prepare first porosity
        td1.prepare_porosity(1)
        td1.prepare_porosity(2)
        td1.prepare_porosity(3)

        # Only three ring should exist
        self.assertEqual(len(td1._data), 3)

        # Check if the points of all rings were added
        self.assertEqual(td1._ring_size(0),
                         self.__class__.hypercube_test_shape[2] * 2)
        self.assertEqual(td1._ring_size(1),
                         self.__class__.hypercube_test_shape[2] * 11)
        self.assertEqual(td1._ring_size(2),
                         self.__class__.hypercube_test_shape[2] * 15)

    @parameterized.expand(concrete_classes)
    def test_prepare_porosity_r3(self, test_cls):
        '''
        Test the use of prepare_porosity() by adding 3 rings all at
        once, simulating the resuming of previous iterations
        '''
        # Load class data
        porosity_dset = self.__class__.porosity_h5_dset
        config = self.__class__.config
        wells_list = config.train_wells_ids

        # Create TestData object
        td1 = test_cls(wells_list, porosity_dset, config)

        # Prepare all porosities up to iteration 3
        td1.prepare_porosity(3)

        # Assert if all 3 rings exist
        self.assertEqual(len(td1._data), 3)

        # Check if the points of all rings were added
        self.assertEqual(td1._ring_size(0),
                         self.__class__.hypercube_test_shape[2] * 2)
        self.assertEqual(td1._ring_size(1),
                         self.__class__.hypercube_test_shape[2] * 11)
        self.assertEqual(td1._ring_size(2),
                         self.__class__.hypercube_test_shape[2] * 15)

    @parameterized.expand(concrete_classes)
    def test_update_feature_r1(self, test_cls):
        # Load class data
        porosity_dset = self.__class__.porosity_h5_dset
        f1_filename = self.__class__.FEATURE_FILENAME1
        f2_filename = self.__class__.FEATURE_FILENAME2
        config = self.__class__.config
        wells_list = config.train_wells_ids

        disp = (0, 0, 0)

        # Create TestData object
        td1 = test_cls(wells_list, porosity_dset, config)

        # Prepare first porosity for iteration 1 (i.e., only ring0 is
        # used for training)
        td1.prepare_porosity(1)

        # =====================================================================
        # === Test first feature
        # =====================================================================
        f1 = FeatureDataH5(f1_filename, None)
        td1.update_feature(f1, disp)

        # Get points from ring 0
        cur_points = td1._data[0]

        for well_id in wells_list:
            # Get data through public interface
            # chunk_id=0 to return all data
            train_X, train_y = td1.get_train_values(well_id, chunk_id=0)
            val_X, val_y = td1.get_val_values(well_id)

            # Generate expected values
            expct_train_coordinates = []
            for w in wells_list:
                if w != well_id:
                    expct_train_coordinates.extend(
                        cur_points[w][['x', 'y', 'z']])
                else:
                    expct_val_coordinates = cur_points[w][['x', 'y', 'z']]

            expected_train_X = [[
                prod(c) + 1,
            ] for c in expct_train_coordinates]
            expected_train_y = [prod(c) for c in expct_train_coordinates]

            expected_val_X = [[
                prod(c) + 1,
            ] for c in expct_val_coordinates]
            expected_val_y = [prod(c) for c in expct_val_coordinates]

            self.assertTrue((expected_train_X == train_X).all())
            self.assertTrue((expected_train_y == train_y).all())
            self.assertTrue((expected_val_X == val_X).all())
            self.assertTrue((expected_val_y == val_y).all())

        # =====================================================================
        # === Test updating first feature to feature2
        # =====================================================================
        f2 = FeatureDataH5(f2_filename, None)
        td1.update_feature(f2, disp)

        for well_id in wells_list:
            # Get data through public interface
            # chunk_id=0 to return all data
            train_X, train_y = td1.get_train_values(well_id, chunk_id=0)
            val_X, val_y = td1.get_val_values(well_id)

            # Generate expected values
            expct_train_coordinates = []
            for w in wells_list:
                if w != well_id:
                    expct_train_coordinates.extend(
                        cur_points[w][['x', 'y', 'z']])
                else:
                    expct_val_coordinates = cur_points[w][['x', 'y', 'z']]

            expected_train_X = [[
                prod(c) + 10,
            ] for c in expct_train_coordinates]
            expected_train_y = [prod(c) for c in expct_train_coordinates]

            expected_val_X = [[
                prod(c) + 10,
            ] for c in expct_val_coordinates]
            expected_val_y = [prod(c) for c in expct_val_coordinates]

            self.assertTrue(np.array_equal(expected_train_X, train_X))
            # self.assertTrue((expected_train_X == train_X).all())
            self.assertTrue((expected_train_y == train_y).all())
            self.assertTrue((expected_val_X == val_X).all())
            self.assertTrue((expected_val_y == val_y).all())

        # =====================================================================
        # === Test committing feature2 and adding feature1 to col2
        # =====================================================================
        # # First, commit the feature2
        td1.commit_feature()
        # Now add the feature1
        td1.update_feature(f1, disp)
        td1.commit_feature()

        for well_id in wells_list:
            # Get data through public interface
            # chunk_id=0 to return all data
            train_X, train_y = td1.get_train_values(well_id, chunk_id=0)
            val_X, val_y = td1.get_val_values(well_id)

            # Generate expected values
            expct_train_coordinates = []
            for w in wells_list:
                if w != well_id:
                    expct_train_coordinates.extend(
                        cur_points[w][['x', 'y', 'z']])
                else:
                    expct_val_coordinates = cur_points[w][['x', 'y', 'z']]

            expected_train_X = [(
                prod(c) + 10,
                prod(c) + 1,
            ) for c in expct_train_coordinates]
            expected_train_y = [prod(c) for c in expct_train_coordinates]

            expected_val_X = [(
                prod(c) + 10,
                prod(c) + 1,
            ) for c in expct_val_coordinates]
            expected_val_y = [prod(c) for c in expct_val_coordinates]

            self.assertTrue(np.array_equal(expected_train_X, train_X))
            # self.assertTrue((expected_train_X == train_X).all())
            self.assertTrue((expected_train_y == train_y).all())
            self.assertTrue((expected_val_X == val_X).all())
            self.assertTrue((expected_val_y == val_y).all())

    @parameterized.expand(concrete_classes)
    def test_chunking(self, test_cls):
        '''
        Test the use of chunking for TrialData.
        It is checked whether all points are returned and if the memory
        usage is reduced.
        '''

        # Load class data
        porosity_dset = self.__class__.porosity_h5_dset
        config = self.__class__.config
        wells_list = config.train_wells_ids

        # Number of chunks tested. This parameter can be changed
        n_chunks_test = 6

        # Create custom config object
        yaml_str = f"""
        wells:
          coords:
          - [0,2]
          - [2,3]
          window: {self.__class__.window}
        alg:
          parallel:
            n_training_chunks: {n_chunks_test}
        """
        chunking_config = YAMLConfig(config_str=yaml_str)

        # Create TestData object
        td1 = test_cls(wells_list, porosity_dset, chunking_config)

        # Prepare all porosities up to iteration 3
        td1.prepare_porosity(3)

        # Regardless of chunking, all validation values should be returned
        # Number of points that should be present after iteration 2 were
        # counted manually, as per the diagrams in the beginning of this class.
        depth = self.__class__.hypercube_test_shape[2]
        val_X, _ = td1.get_val_values(0)
        self.assertTrue(len(val_X) == depth * 11)

        # Same, for well 1
        val_X, _ = td1.get_val_values(1)
        self.assertTrue(len(val_X) == depth * 17)

        # The sum of len's of all chunks should match the
        # total number of points.
        total_points = 0
        for chunk in range(n_chunks_test):
            train_X, _ = td1.get_train_values(0, chunk_id=chunk)
            total_points += len(train_X)
        self.assertTrue(total_points == depth * 17)

        # Same, for well 1
        total_points = 0
        for chunk in range(n_chunks_test):
            train_X, _ = td1.get_train_values(1, chunk_id=chunk)
            total_points += len(train_X)
        self.assertTrue(total_points == depth * 11)


if __name__ == '__main__':
    unittest.main()
