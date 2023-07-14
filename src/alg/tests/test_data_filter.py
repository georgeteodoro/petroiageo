from unittest import TestCase, main
from data_filter import *
import h5py
import tempfile
import numpy as np


class TestDataFilter(TestCase):

    def setUp(self) -> None:
        self.tmp_file = tempfile.TemporaryFile()
        self.h5_file = h5py.File(self.tmp_file, 'a')

        self.cur_data_type = [
            ('x', np.int64),
            ('y', np.int64),
            ('z', np.int64),
            ('phi', np.float64),
            ('well_id', np.int64),
        ]
        self.cur_data_type = np.dtype(self.cur_data_type)
        x_size = 50
        y_size = 50
        z_size = 20
        phi_size = 20
        well_id_size = 10
        data = np.empty((x_size, y_size, z_size, phi_size, well_id_size),
                        dtype=self.cur_data_type)

        for i in range(x_size):
            for j in range(y_size):
                data[i, j]['x'] = i
                data[i, j]['y'] = j
                data[i, j]['z'] = np.arange(10)
                data[i, j]['phi'] = np.random.rand(z_size, phi_size,
                                                   well_id_size)
                #The cube has x ranges associated with a well
                #example: if x in [0, 10], well = 0,
                #example: if x in [40, 50], well = 4
                data[i, j]['well_id'] = np.array([i // 10] * 10)

        self.dset = self.h5_file.create_dataset("default",
                                                dtype=self.cur_data_type,
                                                data=data,
                                                chunks=(10, 10, 10, 20, 10))
        
    def test_dont_filter_anything(self):
        data_filter = DataFilter()
        result = data_filter.filter(self.dset[:])
        self.assertEqual(len(result), self.dset.size)
        

    def tearDown(self):
        #this order matters
        self.h5_file.close()
        self.tmp_file.close()


if __name__ == "__main__":
    main()