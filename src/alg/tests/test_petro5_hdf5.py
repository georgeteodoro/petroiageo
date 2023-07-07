from unittest import TestCase, main
from petro5_hdf5 import _get_num_chunks_of_h5data, _sample_points
from petro5_hdf5 import _get_n_sampling_points_per_chunk, _is_well_in_list
import h5py
import tempfile
import numpy as np


class TestSampling(TestCase):

    def setUp(self):
        self.tmp_file = tempfile.TemporaryFile()
        self.h5_file = h5py.File(self.tmp_file, 'a')

        cur_data_type = [
            ('x', np.int64),
            ('y', np.int64),
            ('z', np.int64),
            ('phi', np.float64),
            ('well_id', np.int64),
        ]
        cur_data_type = np.dtype(cur_data_type)
        data = np.empty((100, 100, 20, 20, 10), dtype=cur_data_type)

        for i in range(100):
            for j in range(100):
                data[i, j]['x'] = i
                data[i, j]['y'] = j
                data[i, j]['z'] = np.arange(10)
                data[i, j]['phi'] = np.random.rand(20, 20, 10)
                #The cube has x ranges associated with a well
                #example: if x in [0, 10], well = 0,
                #example: if x in [40, 50], well = 4
                data[i, j]['well_id'] = np.array([i // 10] * 10)

        self.dset = self.h5_file.create_dataset("default",
                                                dtype=cur_data_type,
                                                data=data,
                                                chunks=(10, 10, 10, 20, 10))

    def test_can_get_correct_n_chunks(self):
        expected_n_chunks = 200
        n_chunks = _get_num_chunks_of_h5data(self.dset)
        self.assertEqual(n_chunks, expected_n_chunks)

    def test_can_sample_points(self):
        chunk_slice = self.dset.iter_chunks().__next__()
        data = self.dset[chunk_slice]
        max_points_to_sample = float("inf")
        n_points_to_sample_chunk = 10
        rng = np.random.default_rng()
        sample = _sample_points(max_points_to_sample, n_points_to_sample_chunk,
                                rng, data)

        self.assertTrue(n_points_to_sample_chunk, len(sample))

    def test_dont_sample_more_than_max_allowed(self):
        chunk_slice = self.dset.iter_chunks().__next__()
        data = self.dset[chunk_slice]
        max_points_to_sample = 2
        n_points_to_sample_chunk = 10
        rng = np.random.default_rng()
        sample = _sample_points(max_points_to_sample, n_points_to_sample_chunk,
                                rng, data)

        self.assertTrue(max_points_to_sample, len(sample))

    def test_can_define_n_sampling_points_per_chunk(self):
        is_training_point_f = lambda c: _is_well_in_list(c, [1, 3, 7])
        sampling_max_points = 1000
        samps_per_chunk = _get_n_sampling_points_per_chunk(
            self.dset, sampling_max_points, is_training_point_f)
        self.assertTrue(np.sum(samps_per_chunk) >= sampling_max_points)

    def tearDown(self):
        #this order matters
        self.h5_file.close()
        self.tmp_file.close()


if __name__ == "__main__":
    main()