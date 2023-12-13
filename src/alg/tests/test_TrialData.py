import unittest
import h5py
import os
import numpy as np
from ddt import ddt, data
from math import prod

from TrialDataNumpy import TrialDataNumpy
from data_filter import WellsSingleRingDataFilter
from FeatureDataH5 import FeatureDataH5
from config_parser import YAMLConfig
import common

concrete_classes = (TrialDataNumpy)


@ddt
class Test_TrialDataAll(unittest.TestCase):

    # All data info
    hypercube_test_shape = (3, 4, 5)
    chunk_test_shape = (3, 2, 5)
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
    def setUpClass(cls):
        # Create config object
        yaml_str = """
        wells:
          coords:
          - [0,2]
          - [2,3]
          window: 0
        """
        cls.config = YAMLConfig(config_str=yaml_str)
        wells_list = cls.config.train_wells_coords

        # Create the porosity h5 file
        cls.porosity_h5_f = h5py.File(cls.POROSITY_FILENAME, 'w')

        # Create the dataset within the h5
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
            cls.hypercube_test_shape,
            dtype=porosity_data_type,
            chunks=cls.chunk_test_shape,
        )

        # Initialize all as empty data
        for i in range(cls.hypercube_test_shape[0]):
            for j in range(cls.hypercube_test_shape[1]):
                for k in range(cls.hypercube_test_shape[2]):
                    cls.porosity_h5_dset[i, j,
                                         k] = (i, j, k, 0,
                                               common.RealValues.empty, -1, -1)

        # Fill wells
        for (well_id, (w_x, w_y)) in enumerate(wells_list):
            for k in range(cls.hypercube_test_shape[2]):
                # Well 1 would not have points throughout the whole depth
                if well_id == 1 and (k == 1 or k == 4):
                    continue

                mock_porosity = w_x * w_y * k
                cls.porosity_h5_dset[w_x, w_y,
                                     k] = (w_x, w_y, k, mock_porosity,
                                           common.RealValues.real, 0, well_id)

        # Create the feature h5 file
        cls.feature1_h5_f = h5py.File(cls.FEATURE_FILENAME1, 'w')

        # Create the dataset within the h5
        cls.feature1_h5_dset = cls.feature1_h5_f.create_dataset(
            common.FEAT_DSET_NAME,
            cls.hypercube_test_shape,
            dtype=np.float64,
        )

        # Initialize feature data
        for i in range(cls.hypercube_test_shape[0]):
            for j in range(cls.hypercube_test_shape[1]):
                for k in range(cls.hypercube_test_shape[2]):
                    cls.feature1_h5_dset[i, j, k] = i * j * k + 1

        # Create the feature h5 file
        cls.feature2_h5_f = h5py.File(cls.FEATURE_FILENAME2, 'w')

        # Create the dataset within the h5
        cls.feature2_h5_dset = cls.feature2_h5_f.create_dataset(
            common.FEAT_DSET_NAME,
            cls.hypercube_test_shape,
            dtype=np.float64,
        )

        # Initialize feature data
        for i in range(cls.hypercube_test_shape[0]):
            for j in range(cls.hypercube_test_shape[1]):
                for k in range(cls.hypercube_test_shape[2]):
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

    @data(concrete_classes)
    def test_list_init(self, test_cls):
        '''
        Test the creation of a new type.
        '''
        # Load class data
        porosity_dset = self.__class__.porosity_h5_dset
        config = self.__class__.config
        wells_list = config.train_wells_coords

        f_sel_filter = WellsSingleRingDataFilter(wells_list)

        td1 = test_cls(features_only=True,
                       porosity_data=porosity_dset,
                       f_sel_filter=f_sel_filter,
                       config=config)
        self.assertTrue(True)
        td2 = test_cls(features_only=False,
                       porosity_data=porosity_dset,
                       f_sel_filter=f_sel_filter,
                       config=config)
        self.assertTrue(True)

    @data(concrete_classes)
    def test_prepare_porosity_r0(self, test_cls):
        # Load class data
        porosity_dset = self.__class__.porosity_h5_dset
        config = self.__class__.config
        wells_list = config.train_wells_coords

        f_sel_filter = WellsSingleRingDataFilter(list(range(len(wells_list))))

        # Create TestData object
        td1 = test_cls(features_only=False,
                       porosity_data=porosity_dset,
                       f_sel_filter=f_sel_filter,
                       config=config)

        # Prepare first porosity
        td1.prepare_porosity(1)

        # Only one ring exists
        self.assertEqual(len(td1._trial_data_dict), 1)

        # Check if all points were added
        # Total of 2 full depths, minus 2 non-added points
        # = 2*5-2 = 8
        self.assertEqual(len(td1._trial_data_dict[0]), 8)

    @data(concrete_classes)
    def test_update_feature_r0(self, test_cls):
        # Load class data
        porosity_dset = self.__class__.porosity_h5_dset
        f1_filename = self.__class__.FEATURE_FILENAME1
        f2_filename = self.__class__.FEATURE_FILENAME2
        config = self.__class__.config
        wells_list = config.train_wells_coords

        disp = (0, 0, 0)
        # disp_cube_shape = (1, 1, 1)

        f_sel_filter = WellsSingleRingDataFilter(list(range(len(wells_list))))

        # Create TestData object
        td1 = test_cls(features_only=False,
                       porosity_data=porosity_dset,
                       f_sel_filter=f_sel_filter,
                       config=config)

        # Prepare first porosity
        td1.prepare_porosity(1)

        # =====================================================================
        # === Test first feature
        # =====================================================================
        f1 = FeatureDataH5(f1_filename, None)
        td1.update_feature(f1, disp)

        cur_points = td1._trial_data_dict[0]

        for (well_id, well) in enumerate(wells_list):
            # Get data through public interface
            # chunk_id=0 to return all data
            train_X, train_y = td1.get_train_values(well_id, chunk_id=0)
            val_X, val_y = td1.get_val_values(well_id)

            # Generate expected values
            expected_train_coordinates = cur_points[cur_points['well_id'] !=
                                                    well_id][['x', 'y', 'z']]
            expected_val_coordinates = cur_points[cur_points['well_id'] ==
                                                  well_id][['x', 'y', 'z']]
            expected_train_X = [[
                prod(c) + 1,
            ] for c in expected_train_coordinates]
            expected_train_y = [prod(c) for c in expected_train_coordinates]

            expected_val_X = [[
                prod(c) + 1,
            ] for c in expected_val_coordinates]
            expected_val_y = [prod(c) for c in expected_val_coordinates]

            self.assertTrue((expected_train_X == train_X).all())
            self.assertTrue((expected_train_y == train_y).all())
            self.assertTrue((expected_val_X == val_X).all())
            self.assertTrue((expected_val_y == val_y).all())

        # =====================================================================
        # === Test updating first feature to feature2
        # =====================================================================
        f2 = FeatureDataH5(f2_filename, None)
        td1.update_feature(f2, disp)

        for (well_id, well) in enumerate(wells_list):
            # Get data through public interface
            # chunk_id=0 to return all data
            train_X, train_y = td1.get_train_values(well_id, chunk_id=0)
            val_X, val_y = td1.get_val_values(well_id)

            # Generate expected values
            expected_train_coordinates = cur_points[cur_points['well_id'] !=
                                                    well_id][['x', 'y', 'z']]
            expected_val_coordinates = cur_points[cur_points['well_id'] ==
                                                  well_id][['x', 'y', 'z']]
            expected_train_X = [[
                prod(c) + 10,
            ] for c in expected_train_coordinates]
            expected_train_y = [prod(c) for c in expected_train_coordinates]

            expected_val_X = [[
                prod(c) + 10,
            ] for c in expected_val_coordinates]
            expected_val_y = [prod(c) for c in expected_val_coordinates]

            self.assertTrue((expected_train_X == train_X).all())
            self.assertTrue((expected_train_y == train_y).all())
            self.assertTrue((expected_val_X == val_X).all())
            self.assertTrue((expected_val_y == val_y).all())

        # =====================================================================
        # === Test committing feature2 and adding feature1 to col2
        # =====================================================================
        td1.commit_feature()
        td1.update_feature(f1, disp)

        for (well_id, well) in enumerate(wells_list):
            # Get data through public interface
            # chunk_id=0 to return all data
            train_X, train_y = td1.get_train_values(well_id, chunk_id=0)
            val_X, val_y = td1.get_val_values(well_id)

            # Generate expected values
            expected_train_coordinates = cur_points[cur_points['well_id'] !=
                                                    well_id][['x', 'y', 'z']]
            expected_val_coordinates = cur_points[cur_points['well_id'] ==
                                                  well_id][['x', 'y', 'z']]
            expected_train_X = [(
                prod(c) + 10,
                prod(c) + 1,
            ) for c in expected_train_coordinates]
            expected_train_y = [prod(c) for c in expected_train_coordinates]

            expected_val_X = [(
                prod(c) + 10,
                prod(c) + 1,
            ) for c in expected_val_coordinates]
            expected_val_y = [prod(c) for c in expected_val_coordinates]

            self.assertTrue((expected_train_X == train_X).all())
            self.assertTrue((expected_train_y == train_y).all())
            self.assertTrue((expected_val_X == val_X).all())
            self.assertTrue((expected_val_y == val_y).all())


if __name__ == '__main__':
    unittest.main()
