from unittest import TestCase, main
from common import RealValues
import h5py
import tempfile
import numpy as np
from h5_expand_alg import H5ExpandAlg
from config_parser import YAMLConfig


class TestH5Expand(TestCase):

    def setUp(self):
        # SetUp h5 dset
        self.tmp_file = tempfile.TemporaryFile()
        self.h5_file = h5py.File(self.tmp_file, 'a')

        self.cur_data_type = [('x', np.int64), ('y', np.int64),
                              ('z', np.int64), ('well_id', np.int64),
                              ('real', np.int64)]

        x_size = 10
        y_size = 10
        self.z_size = 10
        well_id_size = 1
        real_size = 1
        self.cur_data_type = np.dtype(self.cur_data_type)
        data = np.empty((x_size, y_size, self.z_size, well_id_size, real_size),
                        dtype=self.cur_data_type)

        for i in range(x_size):
            for j in range(y_size):
                data[i, j]['x'] = i
                data[i, j]['y'] = j
                data[i, j]['z'] = np.arange(10)[:, np.newaxis, np.newaxis]
                # There will be 3 wells
                well_id = -1
                if i == 4 and j == 2:
                    well_id = 0
                elif i == 7 and j == 6:
                    well_id = 1
                elif i == 2 and j == 7:
                    well_id = 2
                data[i, j]['well_id'] = np.array(well_id)
                data[i, j]['real'] = RealValues.real if well_id != -1 else RealValues.empty

        self.dset = self.h5_file.create_dataset("default",
                                                dtype=self.cur_data_type,
                                                data=data,
                                                chunks=(10, 10, 10, 1, 1))

        # setup configs
        yaml_str = """
        wells:
          coords:
          - [4,2]
          - [7,6]
          - [2,7]
        """
        self.config_all_train = YAMLConfig(config_str=yaml_str)

        yaml_str = """
        wells:
          coords:
          - [4,2]
          - [7,6]
          - [2,7]
        alg:
          test_only_wells: [2]
        """
        self.config_one_test_well = YAMLConfig(config_str=yaml_str)

    def test_can_expand_all_train(self):
        expander = H5ExpandAlg(self.config_all_train)
        expander._expand(porosity_data_h5=self.dset,
                         it=1,
                         wells_coords=self.config_all_train.train_wells_coords,
                         full_depth_chunks=True)
        expected_n_points_per_well = 8 * np.array(
            [self.z_size, self.z_size, self.z_size])

        expanded_points = self.dset[self.dset['real'] == RealValues.expanded]
        _, exp_points_per_well = np.unique(expanded_points['well_id'],
                                           return_counts=True)
        self.assertTrue(
            np.array_equal(exp_points_per_well, expected_n_points_per_well))
    
    def test_dont_expand_test_well(self):
        expander = H5ExpandAlg(self.config_one_test_well)
        expander._expand(porosity_data_h5=self.dset,
                         it=1,
                         wells_coords=self.config_one_test_well.train_wells_coords,
                         full_depth_chunks=True)

        expanded_points = self.dset[self.dset['real'] == RealValues.expanded]
        wells_expanded = np.unique(expanded_points['well_id'])

        test_well = self.config_one_test_well.alg["test_only_wells"][0]

        self.assertNotIn(test_well, wells_expanded)
    
    def test_can_expand_many_times_without_test_well(self):
        expander = H5ExpandAlg(self.config_all_train)
        
        num_expansions = 2
        for it in range(1, num_expansions+1):
            expander._expand(porosity_data_h5=self.dset,
                            it=it,
                            wells_coords=self.config_all_train.train_wells_coords,
                            full_depth_chunks=True)
        
        # This is expected for 2 expansions
        # and as we didnt propagated, we never marked
        # any expanded points as propagated
        expected_exp_points_per_well = np.array(
            [24*self.z_size, 22*self.z_size, 24*self.z_size])

        expanded_points = self.dset[self.dset['real'] == RealValues.expanded]
        _, exp_points_per_well = np.unique(expanded_points['well_id'],
                                           return_counts=True)
        print(expected_exp_points_per_well)
        print(exp_points_per_well)
        self.assertTrue(
            np.array_equal(exp_points_per_well, expected_exp_points_per_well))
    
    def test_can_expand_many_times_with_test_well(self):
        expander = H5ExpandAlg(self.config_one_test_well)
        
        num_expansions = 2
        for it in range(1, num_expansions+1):
            expander._expand(porosity_data_h5=self.dset,
                            it=it,
                            wells_coords=self.config_one_test_well.train_wells_coords,
                            full_depth_chunks=True)
        
        expanded_points = self.dset[self.dset['real'] == RealValues.expanded]
        wells_expanded = np.unique(expanded_points['well_id'])
        test_well = self.config_one_test_well.alg["test_only_wells"][0]

        self.assertNotIn(test_well, wells_expanded)

    def tearDown(self):
        #this order matters
        self.h5_file.close()
        self.tmp_file.close()


if __name__ == "__main__":
    main()