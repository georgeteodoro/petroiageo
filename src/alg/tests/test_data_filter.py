from unittest import TestCase, main
from data_filter import *
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
        result = self.data_filter.filter(self.dset[:])
        self.assertEqual(len(result), self.dset.size)
    
    def test_add_min_ring_filter(self):
        self.data_filter.add_min_ring_filter(15)
        result_size = len(self.data_filter.filter(self.dset[:]))
        self.assertTrue(result_size > 0 and result_size < self.dset.size)

    def tearDown(self):
        #this order matters
        self.h5_file.close()
        self.tmp_file.close()
    
class TestDataFilterWithoutData(TestCase):
    def setUp(self) -> None:
        self.data_filter = DataFilter()
        
    def test_invalid_min_ring(self):
        with self.assertRaises(ValueError):
            self.data_filter.add_min_ring_filter(-1)
        
        with self.assertRaises(ValueError):
            self.data_filter.add_min_ring_filter(3.2)
        
        with self.assertRaises(ValueError):
            self.data_filter.add_min_ring_filter(3.0)



if __name__ == "__main__":
    main()