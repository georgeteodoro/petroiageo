from unittest import TestCase, main
from filter_and_merge_seismic_por import interpolate_data

import numpy as np


class TestInterpolateData(TestCase):

    def test_can_interpolate_small_data(self):
        data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        data = np.array(data).reshape(2, 2, 3)

        curr_depths = np.array([1, 2, 3])
        target_depths = np.array([1, 1.5, 2, 2.5, 3])

        expected_data = [
            1, 1.5, 2, 2.5, 3, 4, 4.5, 5, 5.5, 6, 7, 7.5, 8, 8.5, 9, 10, 10.5,
            11, 11.5, 12
        ]
        expected_data = np.array(expected_data).reshape(
            2, 2, target_depths.size)

        result_data = interpolate_data(target_depths, curr_depths, data)
        self.assertTrue(np.array_equal(expected_data, result_data))

    def test_can_interpolate_big_data(self):
        orig_data_shape = (5, 3, 6)
        data_size = orig_data_shape[0] * orig_data_shape[1] * orig_data_shape[2]
        data = list(range(1, data_size + 1))
        data = np.array(data).reshape(orig_data_shape)

        # [1,..., orig_data_shape[2]]
        curr_depths = np.array(list(range(1, orig_data_shape[2] + 1)))
        # This makes so that the target_depths are egual to
        # [1, 1.5, 2, 2.5, ...., orig_data_shape[2]]
        target_depths = np.array(
            np.linspace(1, orig_data_shape[2], (orig_data_shape[2] * 2) - 1))

        # This creates the expected data
        first_level = data[:, :, 0]
        first_level = first_level[:, :, np.newaxis]
        expected_data = None
        for depth_idx, _ in enumerate(target_depths):
            if expected_data is None:
                expected_data = first_level
            else:
                expected_data = np.concatenate(
                    [expected_data, first_level + (0.5 * depth_idx)], axis=2)

        result_data = interpolate_data(target_depths, curr_depths, data)
        self.assertTrue(np.array_equal(expected_data, result_data))

if __name__ == "__main__":
    main()
