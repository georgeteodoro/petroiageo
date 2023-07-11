from unittest import TestCase, main
from petro5_hdf5 import _get_num_chunks_of_h5data, _sample_points
from petro5_hdf5 import _get_n_sampling_points_per_chunk, _is_well_in_list
from petro5_hdf5 import _count_train_test_points_per_chunk
from petro5_hdf5 import _limit_training_points, _is_well_in_list
import h5py
import tempfile
import numpy as np


class TestSamplingWithData(TestCase):

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

    def test_can_get_train_test_points_per_chunk(self):
        train_wells = [1, 3, 7]
        test_wells = [2, 4, 5]
        is_training_point_f = lambda c: _is_well_in_list(c, train_wells)
        is_test_point_f = lambda c: _is_well_in_list(c, test_wells)
        n_train_per_chunk, n_test_per_chunk = _count_train_test_points_per_chunk(
            self.dset, is_training_point_f, is_test_point_f)

        n_chunks = 200
        expected_train_chunks_with_points = np.full(n_chunks,
                                                    False,
                                                    dtype=np.bool8)
        expected_test_chunks_with_points = np.full(n_chunks,
                                                   False,
                                                   dtype=np.bool8)
        for i in range(n_chunks):
            #Every 20 chunks, changes the well associated with it
            curr_well_of_chunk = i // 20
            if curr_well_of_chunk in train_wells:
                expected_train_chunks_with_points[i] = True

            if curr_well_of_chunk in test_wells:
                expected_test_chunks_with_points[i] = True
        self.assertTrue(
            np.all(n_train_per_chunk[expected_train_chunks_with_points] > 0))
        self.assertTrue(
            np.all(n_test_per_chunk[expected_test_chunks_with_points] > 0))

    def tearDown(self):
        #this order matters
        self.h5_file.close()
        self.tmp_file.close()


class TestSamplingWithoutData(TestCase):

    def test_can_calc_n_sampling_points_per_chunk(self):
        n_train_points_per_chunk = np.array(
            [0, 10, 10, 0, 0, 10, 0, 10, 0, 10])
        max_sampling_points = 20
        expected_n_samp_points_p_chunks = np.array(
            [0, 4, 4, 0, 0, 4, 0, 4, 0, 4])

        samps_per_chunk = _get_n_sampling_points_per_chunk(
            n_train_points_per_chunk, max_sampling_points)
        self.assertTrue(
            np.sum(expected_n_samp_points_p_chunks - samps_per_chunk) == 0)

    def test_sample_every_point_if_n_points_less_than_max_samps(self):
        n_train_points_per_chunk = np.array(
            [0, 10, 10, 0, 0, 10, 0, 10, 0, 10])
        max_sampling_points = 200
        expected_n_samp_points_p_chunks = np.array(
            [0, 10, 10, 0, 0, 10, 0, 10, 0, 10])

        samps_per_chunk = _get_n_sampling_points_per_chunk(
            n_train_points_per_chunk, max_sampling_points)
        self.assertTrue(
            np.sum(expected_n_samp_points_p_chunks - samps_per_chunk) == 0)

    def test_can_limit_n_training_points(self):
        curr_training_points = 10
        max_points_to_sample = 2
        new_n_train_points = _limit_training_points(max_points_to_sample,
                                                    curr_training_points)
        self.assertEqual(new_n_train_points, max_points_to_sample)

    def test_dont_limit_n_train_points(self):
        curr_training_points = 10
        max_points_to_sample = 20
        new_n_train_points = _limit_training_points(max_points_to_sample,
                                                    curr_training_points)
        self.assertEqual(new_n_train_points, curr_training_points)

    def test_dont_limit_n_train_points_if_negative(self):
        curr_training_points = 10
        max_points_to_sample = -1
        new_n_train_points = _limit_training_points(max_points_to_sample,
                                                    curr_training_points)
        self.assertEqual(new_n_train_points, curr_training_points)

    def test_dont_limit_n_train_points_if_zero(self):
        curr_training_points = 10
        max_points_to_sample = 0
        new_n_train_points = _limit_training_points(max_points_to_sample,
                                                    curr_training_points)
        self.assertEqual(new_n_train_points, curr_training_points)

    def test_is_well_in_list(self):
        my_dtype = [('well_id', np.int64)]
        my_dtype = np.dtype(my_dtype)
        data = np.empty(10, dtype=my_dtype)
        data['well_id'] = np.arange(10)
        target_wells = [1, 7, 9]
        expected_list = [
            False, True, False, False, False, False, False, True, False, True
        ]
        expected = np.array(expected_list, dtype=bool)
        result = _is_well_in_list(data, target_wells)
        self.assertTrue((expected == result).all())


if __name__ == "__main__":
    main()