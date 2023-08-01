from unittest import TestCase, main
from data_filter import DataFilter, FeatSelectionTrainDataFilter
from data_filter import PredTrainDataFilter, WellsDataFilter
from common import RealValues
import h5py
import tempfile
import numpy as np


class TestDataFilter(TestCase):

    def setUp(self) -> None:
        self.tmp_file = tempfile.TemporaryFile()
        self.h5_file = h5py.File(self.tmp_file, 'a')

        self.cur_data_type = [('x', np.int64), ('y', np.int64),
                              ('z', np.int64), ('phi', np.float64),
                              ('well_id', np.int64), ('ring', np.int8)]
        self.cur_data_type = np.dtype(self.cur_data_type)
        x_size = 50
        y_size = 50
        z_size = 20
        phi_size = 20
        well_id_size = 10
        ring_size = 1
        data = np.empty(
            (x_size, y_size, z_size, phi_size, well_id_size, ring_size),
            dtype=self.cur_data_type)

        middle_y = y_size // 2
        for i in range(x_size):
            for j in range(y_size):
                data[i, j]['x'] = i
                data[i, j]['y'] = j
                data[i, j]['z'] = np.arange(10)[:, np.newaxis]
                data[i, j]['phi'] = np.random.rand(z_size, phi_size,
                                                   well_id_size, ring_size)
                #The cube has x ranges associated with a well
                #example: if x in [0, 10], well = 0,
                #example: if x in [40, 50], well = 4
                data[i, j]['well_id'] = np.array([i // 10] * 10)[:, np.newaxis]

                #The ring num is based from the middle y
                data[i, j]['ring'] = abs(middle_y - j)

        self.dset = self.h5_file.create_dataset("default",
                                                dtype=self.cur_data_type,
                                                data=data,
                                                chunks=(10, 10, 10, 20, 10, 1))

        self.data_filter = DataFilter()

    def test_dont_filter_anything(self):
        result_size = len(self.data_filter.filter(self.dset[:]))
        self.assertEqual(result_size, self.dset.size)

    def test_add_min_ring_filter(self):
        self.data_filter.add_min_ring_filter(15)
        result_size = len(self.data_filter.filter(self.dset[:]))
        self.assertTrue(result_size > 0 and result_size < self.dset.size)

    def test_add_in_well_list_filter(self):
        well_ids = [1, 2]
        self.data_filter.add_in_well_list_filter(well_ids)
        result_size = len(self.data_filter.filter(self.dset[:]))
        self.assertTrue(result_size > 0 and result_size < self.dset.size)

    def test_add_in_well_empty_list_filter(self):
        well_ids = list()
        self.data_filter.add_in_well_list_filter(well_ids)
        result_size = len(self.data_filter.filter(self.dset[:]))
        self.assertTrue(result_size == 0)

    def test_add_not_in_well_list_filter_unused_well(self):
        well_ids = [10000]
        self.data_filter.add_not_in_well_list_filter(well_ids)
        result_size = len(self.data_filter.filter(self.dset[:]))
        self.assertEqual(result_size, self.dset.size)

    def test_add_not_in_well_list(self):
        well_ids = [1, 2]
        self.data_filter.add_not_in_well_list_filter(well_ids)
        result_size = len(self.data_filter.filter(self.dset[:]))
        self.assertTrue(result_size > 0 and result_size < self.dset.size)

    def test_add_not_in_well_empty_list(self):
        well_ids = []
        self.data_filter.add_not_in_well_list_filter(well_ids)
        result_size = len(self.data_filter.filter(self.dset[:]))
        self.assertEqual(result_size, self.dset.size)

    def test_can_count_data_given_filters(self):
        well_ids = [1, 2]
        self.data_filter.add_not_in_well_list_filter(well_ids)
        self.data_filter.add_min_ring_filter(15)
        result_size = self.data_filter.filter_count_dset(self.dset)
        self.assertTrue(result_size > 0 and result_size < self.dset.size)

    def test_can_filter_chunks(self):
        well_ids = [1, 2]
        self.data_filter.add_not_in_well_list_filter(well_ids)
        for chunk_slice in self.dset.iter_chunks():
            chunk_data = self.dset[chunk_slice]
            try:
                _ = self.data_filter.filter(chunk_data)
                self.assertTrue(True)
            except Exception as e:
                self.assertTrue(False)

    def tearDown(self):
        #this order matters
        self.h5_file.close()
        self.tmp_file.close()


class TestDataFilterWithoutData(TestCase):
    """
    This is so these tests run faster
    """

    def setUp(self) -> None:
        self.data_filter = DataFilter()

    def test_invalid_min_ring(self):
        with self.assertRaises(ValueError):
            self.data_filter.add_min_ring_filter(-1)

        with self.assertRaises(ValueError):
            self.data_filter.add_min_ring_filter(3.2)

        with self.assertRaises(ValueError):
            self.data_filter.add_min_ring_filter(3.0)

    def test_invalid_in_well_list(self):
        invalid_well_ids = 1
        with self.assertRaises(TypeError):
            self.data_filter.add_in_well_list_filter(invalid_well_ids)

    def test_invalid_not_in_well_list(self):
        invalid_well_ids = 1
        with self.assertRaises(TypeError):
            self.data_filter.add_not_in_well_list_filter(invalid_well_ids)

    def test_raise_on_invalid_dset(self):
        invalid_dset = np.arange(10)
        with self.assertRaises(TypeError):
            self.data_filter.filter_count_dset(invalid_dset)

    def test_can_get_min_ring(self):
        min_ring = 4
        self.data_filter.add_min_ring_filter(min_ring)
        self.assertTrue(min_ring, self.data_filter.min_ring)

    def test_can_get_in_wells(self):
        in_wells_list = [1, 3, 6]
        self.data_filter.add_in_well_list_filter(in_wells_list)
        self.assertListEqual(in_wells_list, self.data_filter.in_wells)

    def test_can_get_not_in_wells(self):
        not_in_wells_list = [1, 3, 6]
        self.data_filter.add_not_in_well_list_filter(not_in_wells_list)
        self.assertListEqual(not_in_wells_list, self.data_filter.not_in_wells)

    def test_can_get_n_filters(self):
        not_in_wells_list = [1, 3, 6]
        self.data_filter.add_not_in_well_list_filter(not_in_wells_list)
        in_wells_list = [0, 2, 4, 5]
        self.data_filter.add_in_well_list_filter(in_wells_list)
        self.assertTrue(self.data_filter.n_filters == 2)

    def test_can_get_str(self):
        self.assertTrue("min_ring" in str(self.data_filter))
        self.assertTrue("n_filters" in str(self.data_filter))


class TestTrainDataFilters(TestCase):

    def setUp(self) -> None:
        self.tmp_file = tempfile.TemporaryFile()
        self.h5_file = h5py.File(self.tmp_file, 'a')

        cur_data_type = [
            ('x', np.int64),
            ('y', np.int64),
            ('z', np.int64),
            ('phi', np.float64),
            ('real', np.int8),
            ('well_id', np.int64),
            ('ring', np.int8),
        ]
        cur_data_type = np.dtype(cur_data_type)
        #Dont need a big dataset here
        x_size = 10
        y_size = 10
        z_size = 10
        phi_size = 10
        well_id_size = 5
        ring_size = 1
        real_size = 1
        data = np.empty((x_size, y_size, z_size, phi_size, well_id_size,
                         ring_size, real_size),
                        dtype=cur_data_type)

        middle_y = y_size // 2
        for i in range(x_size):
            for j in range(y_size):
                data[i, j]['x'] = i
                data[i, j]['y'] = j
                data[i, j]['z'] = np.arange(well_id_size)[:, np.newaxis,
                                                          np.newaxis]
                data[i,
                     j]['phi'] = np.random.rand(z_size, phi_size, well_id_size,
                                                ring_size, real_size)
                #The cube has x ranges associated with a well
                #example: if x in [0, 10], well = 0,
                #example: if x in [40, 50], well = 4
                data[i, j]['well_id'] = np.array(
                    [i // 10] * well_id_size)[:, np.newaxis, np.newaxis]

                #The ring num is based from the middle y
                data[i, j]['ring'] = abs(middle_y - j)

                #Some logic for real
                data[i, j]['real'] = RealValues.real if (
                    i + j) % 2 == 0 else RealValues.expanded

        self.dset = self.h5_file.create_dataset("default",
                                                dtype=cur_data_type,
                                                data=data,
                                                chunks=(10, 10, 10, 10, 5, 1,
                                                        1))

        self.pred_data_filter = PredTrainDataFilter()
        self.feat_sel_data_filter = FeatSelectionTrainDataFilter()

    def test_pred_df_can_filter_automatically(self):
        result_size = self.pred_data_filter.filter_count_dset(self.dset)
        self.assertTrue(result_size > 0 and result_size < self.dset.size)

    def test_feat_sel_df_can_filter_automatically(self):
        result_size = self.feat_sel_data_filter.filter_count_dset(self.dset)
        self.assertTrue(result_size > 0 and result_size < self.dset.size)

    def tearDown(self):
        #this order matters
        self.h5_file.close()
        self.tmp_file.close()


class TestFeatSelTrainDataFilterWithoutData(TestCase):

    def setUp(self) -> None:
        self.data_filter = FeatSelectionTrainDataFilter()

    def test_n_filters_base_is_one(self):
        self.assertEqual(self.data_filter.n_filters, 1)

    def test_n_filters_after_add(self):
        self.data_filter.add_not_in_well_list_filter([1, 2])
        self.assertEqual(self.data_filter.n_filters, 2)


class TestPredTrainDataFilterWithoutData(TestCase):

    def setUp(self) -> None:
        self.data_filter = PredTrainDataFilter()

    def test_n_filters_base_is_one(self):
        self.assertEqual(self.data_filter.n_filters, 1)

    def test_n_filters_after_add(self):
        self.data_filter.add_not_in_well_list_filter([1, 2])
        self.assertEqual(self.data_filter.n_filters, 2)


class TestWellsDataFilter(TestCase):

    def setUp(self) -> None:
        self.tmp_file = tempfile.TemporaryFile()
        self.h5_file = h5py.File(self.tmp_file, 'a')

        self.cur_data_type = [('x', np.int64), ('y', np.int64),
                              ('z', np.int64), ('phi', np.float64),
                              ('well_id', np.int64), ('ring', np.int8)]
        self.cur_data_type = np.dtype(self.cur_data_type)
        #Dont need a big dataset here
        x_size = 30
        y_size = 10
        z_size = 10
        phi_size = 10
        well_id_size = 10
        ring_size = 1
        data = np.empty(
            (x_size, y_size, z_size, phi_size, well_id_size, ring_size),
            dtype=self.cur_data_type)

        middle_y = y_size // 2
        for i in range(x_size):
            for j in range(y_size):
                data[i, j]['x'] = i
                data[i, j]['y'] = j
                data[i, j]['z'] = np.arange(10)[:, np.newaxis]
                data[i, j]['phi'] = np.random.rand(z_size, phi_size,
                                                   well_id_size, ring_size)
                #The cube has x ranges associated with a well
                #example: if x in [0, 10], well = 0,
                #example: if x in [40, 50], well = 4
                #There is going to be x_size//10 wells
                data[i, j]['well_id'] = np.array([i // 10] * 10)[:, np.newaxis]

                #The ring num is based from the middle y
                data[i, j]['ring'] = abs(middle_y - j)

        self.dset = self.h5_file.create_dataset("default",
                                                dtype=self.cur_data_type,
                                                data=data,
                                                chunks=(10, 10, 10, 10, 10, 1))

        #There are 3 wells in total, 2 of them are test wells
        target_wells = [0, 1]
        self.data_filter = WellsDataFilter(target_wells)

    def test_can_filter_automatically(self):
        result_size = self.data_filter.filter_count_dset(self.dset)
        self.assertTrue(result_size > 0 and result_size < self.dset.size)

    def tearDown(self):
        #this order matters
        self.h5_file.close()
        self.tmp_file.close()


class TestWellsDataFilterWithoutData(TestCase):

    def setUp(self) -> None:
        self.data_filter = WellsDataFilter([4, 5])

    def test_n_filters_base_is_one(self):
        self.assertEqual(self.data_filter.n_filters, 1)

    def test_n_filters_after_add(self):
        self.data_filter.add_not_in_well_list_filter([1, 2])
        self.assertEqual(self.data_filter.n_filters, 2)


class TestIt0DataFilters(TestCase):

    def _is_well_coord(self, x: int, y: int) -> bool:
        return x == y and (x + y) % 4 == 0

    def setUp(self) -> None:
        self.tmp_file = tempfile.TemporaryFile()
        self.h5_file = h5py.File(self.tmp_file, 'a')

        cur_data_type = [
            ('x', np.int64),
            ('y', np.int64),
            ('z', np.int64),
            ('phi', np.float64),
            ('real', np.int8),
            ('well_id', np.int64),
            ('ring', np.int8),
        ]
        cur_data_type = np.dtype(cur_data_type)
        #Dont need a big dataset here
        x_size = 10
        y_size = 10
        z_size = 10
        phi_size = 10
        well_id_size = 5
        ring_size = 1
        real_size = 1
        data = np.empty((x_size, y_size, z_size, phi_size, well_id_size,
                         ring_size, real_size),
                        dtype=cur_data_type)

        well_count = 0
        for i in range(x_size):
            for j in range(y_size):
                data[i, j]['x'] = i
                data[i, j]['y'] = j
                data[i, j]['z'] = np.arange(well_id_size)[:, np.newaxis,
                                                          np.newaxis]
                data[i,
                     j]['phi'] = np.random.rand(z_size, phi_size, well_id_size,
                                                ring_size, real_size)
                #Wells at positions:
                # (2,2), (4,4), (6,6), (8,8) ...
                well_id = well_count if self._is_well_coord(i, j) else -1
                if well_id != -1:
                    well_count += 1

                data[i, j]['well_id'] = np.array(
                    [well_id] * well_id_size)[:, np.newaxis, np.newaxis]

                data[i, j]['ring'] = 0 if self._is_well_coord(i, j) else -1

                data[i, j]['real'] = RealValues.real if self._is_well_coord(
                    i, j) else RealValues.canal

        self.dset = self.h5_file.create_dataset("default",
                                                dtype=cur_data_type,
                                                data=data,
                                                chunks=(10, 10, 10, 10, 5, 1,
                                                        1))

    def test_can_filter_test_data(self):
        data_filter = WellsDataFilter([0])
        data_filter.add_min_ring_filter(0)
        result_size = data_filter.filter_count_dset(self.dset)
        self.assertTrue(result_size > 0 and result_size < self.dset.size)

    def test_can_filter_feat_sel_train_data(self):
        data_filter = FeatSelectionTrainDataFilter()
        data_filter.add_min_ring_filter(0)
        data_filter.add_not_in_well_list_filter([0])
        result_size = data_filter.filter_count_dset(self.dset)
        self.assertTrue(result_size > 0 and result_size < self.dset.size)

    def tearDown(self):
        #this order matters
        self.h5_file.close()
        self.tmp_file.close()


if __name__ == "__main__":
    main()