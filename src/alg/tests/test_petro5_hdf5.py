from unittest import TestCase, main
from petro5_hdf5 import _get_num_chunks_of_h5data
import h5py
import tempfile 
import numpy as np

class TestSampling(TestCase):
    def setUp(self):
        self.tmp_file = tempfile.TemporaryFile()
        self.h5_file = h5py.File(self.tmp_file,'a')
        data = np.random.random((100, 100, 20))
        self.dset = self.h5_file.create_dataset("default", data=data, chunks=(10, 10, 10))

    def test_can_get_correct_n_chunks(self):
        expected_n_chunks = 200
        n_chunks = _get_num_chunks_of_h5data(self.dset)
        self.assertEqual(n_chunks, expected_n_chunks)
    
    def tearDown(self):
        #this order matters
        self.h5_file.close()
        self.tmp_file.close()

if __name__ == "__main__":
    main()