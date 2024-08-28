from unittest import TestCase, main
import numpy as np

from common import PointDtypeIdx
from config_parser import YAMLConfig
from sampler import AbstractChunkSampler, ChunkSamplerV1, target_based_sampler


class TestAbstractChunkSampler(TestCase):

    def test_cant_instantiate_abc_sampler(self):
        with self.assertRaises(Exception):
            _ = AbstractChunkSampler()


class TestChunkSamplerV1(TestCase):

    def setUp(self):
        # The area setup with well coords is:
        #                     y
        #   . . . . . . . . . 0
        #   . . . . . . . . . 1
        #   . . . . 0 . . . . 2
        #   . . . . . . . . . 3
        #   . . . . . . . . . 4
        #   . . . . . . . . . 5
        #   . . . . . . . 1 . 6
        #   . . 2 . . . . . . 7
        # x 0 1 2 3 4 5 6 7 8
        #
        # Ring After 1 expansion:
        # -1 -1 -1 -1 -1 -1 -1 -1 -1
        # -1 -1 -1  1  1  1 -1 -1 -1
        # -1 -1 -1  1  0  1 -1 -1 -1
        # -1 -1 -1  1  1  1 -1 -1 -1
        # -1 -1 -1 -1 -1 -1 -1 -1 -1
        # -1 -1 -1 -1 -1 -1  1  1  1
        # -1  1  1  1 -1 -1  1  0  1
        # -1  1  0  1 -1 -1  1  1  1
        #
        # Ring After 2 expansion:
        # -1 -1  2  2  2  2  2 -1 -1
        # -1 -1  2  1  1  1  2 -1 -1
        # -1 -1  2  1  0  1  2 -1 -1
        # -1 -1  2  1  1  1  2 -1 -1
        # -1 -1  2  2  2  2  2  2  2
        #  2  2  2  2  2  2  1  1  1
        #  2  1  1  1  2  2  1  0  1
        #  2  1  0  1  2  2  1  1  1
        #
        # Ring After 3 expansion:
        # -1  3  2  2  2  2  2  3 -1
        # -1  3  2  1  1  1  2  3 -1
        # -1  3  2  1  0  1  2  3 -1
        # -1  3  2  1  1  1  2  3  3
        #  3  3  2  2  2  2  2  2  2
        #  2  2  2  2  2  2  1  1  1
        #  2  1  1  1  2  2  1  0  1
        #  2  1  0  1  2  2  1  1  1
        #
        # Ring After 4 expansion
        #  4  3  2  2  2  2  2  3  4
        #  4  3  2  1  1  1  2  3  4
        #  4  3  2  1  0  1  2  3  4
        #  4  3  2  1  1  1  2  3  3
        #  3  3  2  2  2  2  2  2  2
        #  2  2  2  2  2  2  1  1  1
        #  2  1  1  1  2  2  1  0  1
        #  2  1  0  1  2  2  1  1  1

        # All points in a well coord have ring == 0

        self.cur_data_type = [('x', np.int64), ('y', np.int64),
                              ('phi', np.float64), ('well_id', np.int64),
                              ('ring', np.int8)]
        self.cur_data_type = np.dtype(self.cur_data_type)

        self.x_size = 9
        self.y_size = 8
        phi_size = 1
        well_id_size = 1
        ring_size = 1

        self.data = np.empty(
            (self.x_size, self.y_size, phi_size, well_id_size, ring_size),
            dtype=self.cur_data_type)

        self.wells_coords = [(4, 2), (7, 6), (2, 7)]

        for i in range(self.x_size):
            for j in range(self.y_size):
                self.data[i, j]['x'] = i
                self.data[i, j]['y'] = j
                self.data[i, j]['phi'] = np.random.rand(phi_size, well_id_size,
                                                        ring_size)

                if (i, j) in self.wells_coords:
                    self.data[i, j]['well_id'] = self.wells_coords.index((i, j))
                    self.data[i, j]['ring'] = 0
                else:
                    self.data[i, j]['well_id'] = -1
                    self.data[i, j]['ring'] = -1

        self.yaml_str_fmt = """
        wells:
          coords: [[1,1]]
          window: 0
        alg:
          sampling:
            seed: 42
            max_points: {}
            layers_window_size: {}
            beta_dist:
                alpha: {}
                beta: {}
        """

    def _expand_data_n_times(self, data: np.ndarray, n_expansions: int):
        """
        Does n_expansions expansions on the data marking the ring on each point
        """
        for expansion in range(1, n_expansions + 1):
            for well_coord in self.wells_coords:
                filtered_data = data[(data['x'] >= well_coord[0] - expansion)
                                     & (data['x'] <= well_coord[0] + expansion)
                                     & (data['y'] >= well_coord[1] - expansion)
                                     & (data['y'] <= well_coord[1] + expansion)
                                     & (data['ring'] == -1)]

                filtered_data['ring'] = np.full(filtered_data.shape, expansion)

                data[(data['x'] >= well_coord[0] - expansion)
                     & (data['x'] <= well_coord[0] + expansion) &
                     (data['y'] >= well_coord[1] - expansion) &
                     (data['y'] <= well_coord[1] + expansion) &
                     (data['ring'] == -1)] = filtered_data

        return data

    def _get_yaml_str_formated(self,
                               max_points: int = -1,
                               layers_window_size: int = -1,
                               alpha: int = 1,
                               beta: int = 1):
        return self.yaml_str_fmt.format(max_points, layers_window_size, alpha,
                                        beta)

    def _agg_samples_by_ring(self, sampler: AbstractChunkSampler,
                             data: np.ndarray, curr_final_layer: int):
        """Do sampling for each ring at a time and aggregate the sampled data.
        \n The sampling may sample just a little more than the max_points"""
        rings = np.unique(data['ring'])
        n_trial_points = data.size
        sampled_data = None
        for ring in rings:
            ring_data = data[data['ring'] == ring]
            curr_sampled_data = sampler.sample(ring_data, n_trial_points,
                                               curr_final_layer, ring)
            if sampled_data is None:
                sampled_data = curr_sampled_data
            else:
                sampled_data = np.concatenate([sampled_data, curr_sampled_data])
        return sampled_data

    def test_sample_basic_config_no_expansions(self):
        my_config = YAMLConfig(config_str=self._get_yaml_str_formated())

        sampler = ChunkSamplerV1(my_config)
        target_ring = 0
        filtered_data = self.data[self.data['ring'] == target_ring]
        n_trial_points = filtered_data.size
        curr_final_layer = 1
        sampled_data = sampler.sample(filtered_data, n_trial_points,
                                      curr_final_layer, target_ring)
        self.assertEqual(sampled_data.size, n_trial_points)

    def test_sample_all_data_config_3_expansions(self):
        my_config = YAMLConfig(config_str=self._get_yaml_str_formated())
        sampler = ChunkSamplerV1(my_config)

        n_expansions = 3
        expanded_data = self._expand_data_n_times(self.data, n_expansions)

        filtered_data = expanded_data[(expanded_data['ring'] <= n_expansions)
                                      & (expanded_data['ring'] >= 0)]
        curr_final_layer = n_expansions + 1
        n_trial_points = filtered_data.size
        sampled_data = self._agg_samples_by_ring(sampler, filtered_data,
                                                 curr_final_layer)

        self.assertEqual(sampled_data.size, n_trial_points)

    def test_sample_max_points_no_expansions(self):
        max_points = 10
        my_config = YAMLConfig(config_str=self._get_yaml_str_formated(
            max_points=max_points))
        sampler = ChunkSamplerV1(my_config)
        filtered_data = self.data[self.data['ring'] == 0]
        n_trial_points = filtered_data.size
        curr_final_layer = 1
        sampled_data = self._agg_samples_by_ring(sampler, filtered_data,
                                                 curr_final_layer)
        self.assertEqual(sampled_data.size, n_trial_points)

    def test_sample_max_points_3_expansions(self):
        max_points = 10
        my_config = YAMLConfig(config_str=self._get_yaml_str_formated(
            max_points=max_points))
        sampler = ChunkSamplerV1(my_config)
        n_expansions = 3
        expanded_data = self._expand_data_n_times(self.data, n_expansions)

        filtered_data = expanded_data[(expanded_data['ring'] <= n_expansions)
                                      & (expanded_data['ring'] >= 0)]
        curr_final_layer = n_expansions + 1
        sampled_data = self._agg_samples_by_ring(sampler, filtered_data,
                                                 curr_final_layer)
        self.assertGreaterEqual(sampled_data.size, max_points)

    def test_sample_max_points_with_layers_no_expansions(self):
        max_points = 10
        layers_window_size = 2
        my_config = YAMLConfig(config_str=self._get_yaml_str_formated(
            max_points=max_points, layers_window_size=layers_window_size))
        sampler = ChunkSamplerV1(my_config)

        n_expansions = 3
        expanded_data = self._expand_data_n_times(self.data, n_expansions)
        curr_final_layer = n_expansions + 1

        filtered_data = expanded_data[
            (expanded_data['ring'] <= n_expansions)
            & (expanded_data['ring'] >= curr_final_layer - layers_window_size)]

        sampled_data = self._agg_samples_by_ring(sampler, filtered_data,
                                                 curr_final_layer)
        self.assertGreaterEqual(sampled_data.size, max_points)

    def test_sample_max_points_with_layers_3_expansions(self):
        max_points = 10
        layers_window_size = 2
        my_config = YAMLConfig(config_str=self._get_yaml_str_formated(
            max_points=max_points, layers_window_size=layers_window_size))
        sampler = ChunkSamplerV1(my_config)

        n_expansions = 3
        curr_final_layer = n_expansions + 1
        expanded_data = self._expand_data_n_times(self.data, n_expansions)
        filtered_data = expanded_data[
            (expanded_data['ring'] <= n_expansions)
            & (expanded_data['ring'] >= curr_final_layer - layers_window_size)]

        sampled_data = self._agg_samples_by_ring(sampler, filtered_data,
                                                 curr_final_layer)
        self.assertGreaterEqual(sampled_data.size, max_points)

    def test_sample_beta_dist_3_expansions(self):
        alpha = 1
        beta = 5
        my_config = YAMLConfig(
            config_str=self._get_yaml_str_formated(alpha=alpha, beta=beta))
        sampler = ChunkSamplerV1(my_config)

        n_expansions = 3
        expanded_data = self._expand_data_n_times(self.data, n_expansions)

        filtered_data = expanded_data[(expanded_data['ring'] <= n_expansions)
                                      & (expanded_data['ring'] >= 0)]
        curr_final_layer = n_expansions + 1
        n_trial_points = filtered_data.size
        sampled_data = self._agg_samples_by_ring(sampler, filtered_data,
                                                 curr_final_layer)
        self.assertEqual(sampled_data.size, n_trial_points)

    def test_sample_max_points_beta_dist_with_layers_3_expansions(self):
        max_points = 10
        layers_window_size = 2
        alpha = 1
        beta = 5
        my_config = YAMLConfig(config_str=self._get_yaml_str_formated(
            max_points=max_points,
            layers_window_size=layers_window_size,
            alpha=alpha,
            beta=beta))
        sampler = ChunkSamplerV1(my_config)

        n_expansions = 3
        curr_final_layer = n_expansions + 1
        expanded_data = self._expand_data_n_times(self.data, n_expansions)
        filtered_data = expanded_data[
            (expanded_data['ring'] <= n_expansions)
            & (expanded_data['ring'] >= curr_final_layer - layers_window_size)]
        sampled_data = self._agg_samples_by_ring(sampler, filtered_data,
                                                 curr_final_layer)
        self.assertGreaterEqual(sampled_data.size, max_points)

    def test_sample_max_points_beta_dist_3_expansions(self):
        max_points = 10
        alpha = 1
        beta = 5
        my_config = YAMLConfig(config_str=self._get_yaml_str_formated(
            max_points=max_points, alpha=alpha, beta=beta))
        sampler = ChunkSamplerV1(my_config)
        n_expansions = 3
        expanded_data = self._expand_data_n_times(self.data, n_expansions)

        filtered_data = expanded_data[(expanded_data['ring'] <= n_expansions)
                                      & (expanded_data['ring'] >= 0)]
        curr_final_layer = n_expansions + 1
        sampled_data = self._agg_samples_by_ring(sampler, filtered_data,
                                                 curr_final_layer)
        self.assertGreaterEqual(sampled_data.size, max_points)

    def test_sample_max_points_greater_than_n_points_3_expansions(self):
        max_points = 1000
        my_config = YAMLConfig(config_str=self._get_yaml_str_formated(
            max_points=max_points))
        sampler = ChunkSamplerV1(my_config)
        n_expansions = 3
        expanded_data = self._expand_data_n_times(self.data, n_expansions)

        filtered_data = expanded_data[(expanded_data['ring'] <= n_expansions)
                                      & (expanded_data['ring'] >= 0)]
        curr_final_layer = n_expansions + 1
        n_trial_points = filtered_data.size
        sampled_data = self._agg_samples_by_ring(sampler, filtered_data,
                                                 curr_final_layer)
        self.assertEqual(sampled_data.size, n_trial_points)

    def test_sample_none_data(self):
        my_config = YAMLConfig(config_str=self._get_yaml_str_formated())
        sampler = ChunkSamplerV1(my_config)

        sampled_data = sampler.sample(None, 100, 1, 0)
        self.assertIsNone(sampled_data)

    def test_sample_empty_data(self):
        my_config = YAMLConfig(config_str=self._get_yaml_str_formated())
        sampler = ChunkSamplerV1(my_config)
        data = np.array([])
        sampled_data = sampler.sample(data, 100, 1, 0)
        self.assertIsNone(sampled_data)


class TestTargetBasedSampler(TestCase):

    def _get_n_points_with_pors(
            self, pors_and_qts: list[tuple[float, int]]) -> list[list]:
        points = list()
        count = 0
        for por, n_points in pors_and_qts:
            for _ in range(n_points):
                new_p = [count] * 7
                new_p[PointDtypeIdx.phi] = por
                points.append(new_p)
                count += 1
        return points

    def test_dont_sample_empty_propagated_points(self):
        propagated_points = list()
        buckets_len = {i: i for i in range(3)}
        expected_buckets_len = {i: i for i in range(3)}
        buckets_max_size = 4
        alpha = 0.1
        sampled_must_add, sampled_for_update = target_based_sampler(
            propagated_points, buckets_len, buckets_max_size, alpha)

        self.assertDictEqual(buckets_len, expected_buckets_len)
        self.assertTrue(len(sampled_must_add) == 0)
        self.assertTrue(len(sampled_for_update) == 0)

    def test_add_points_to_empty_buckets(self):
        propagated_points = list()
        propagated_points = self._get_n_points_with_pors([(1.1, 1), (2.2, 2),
                                                          (3.3, 3)])

        buckets_len = dict()
        buckets_max_size = 4
        alpha = 0.1
        sampled_must_add, sampled_for_update = target_based_sampler(
            propagated_points, buckets_len, buckets_max_size, alpha)

        expected_buckets_len = {1: 1, 2: 2, 3: 3}
        self.assertDictEqual(buckets_len, expected_buckets_len)
        self.assertTrue(len(sampled_for_update) == 0)
        expected_sampled_must_add = {
            1: [propagated_points[0]],
            2: propagated_points[1:3],
            3: propagated_points[3:]
        }
        self.assertDictEqual(sampled_must_add, expected_sampled_must_add)

    def test_add_points_to_not_empty_buckets(self):
        propagated_points = list()
        propagated_points = self._get_n_points_with_pors([(1.1, 1), (2.2, 2),
                                                          (3.3, 3)])

        buckets_len = {1: 1, 2: 1, 3: 1}
        buckets_max_size = 4
        alpha = 0.1
        sampled_must_add, sampled_for_update = target_based_sampler(
            propagated_points, buckets_len, buckets_max_size, alpha)

        expected_buckets_len = {1: 2, 2: 3, 3: 4}
        self.assertDictEqual(buckets_len, expected_buckets_len)
        self.assertTrue(len(sampled_for_update) == 0)
        expected_sampled_must_add = {
            1: [propagated_points[0]],
            2: propagated_points[1:3],
            3: propagated_points[3:]
        }
        self.assertDictEqual(sampled_must_add, expected_sampled_must_add)

    def test_add_points_to_full_buckets_alpha_1(self):
        propagated_points = list()
        propagated_points = self._get_n_points_with_pors([(1.1, 1), (2.2, 2),
                                                          (3.3, 3)])

        buckets_max_size = 4
        buckets_len = {
            1: buckets_max_size,
            2: buckets_max_size,
            3: buckets_max_size
        }
        alpha = 1
        sampled_must_add, sampled_for_update = target_based_sampler(
            propagated_points, buckets_len, buckets_max_size, alpha)

        expected_buckets_len = {
            1: buckets_max_size,
            2: buckets_max_size,
            3: buckets_max_size
        }
        self.assertDictEqual(buckets_len, expected_buckets_len)
        expected_sampled_for_update = {
            1: [propagated_points[0]],
            2: propagated_points[1:3],
            3: propagated_points[3:]
        }
        self.assertDictEqual(sampled_for_update, expected_sampled_for_update)
        self.assertTrue(len(sampled_must_add) == 0)

    def test_fills_almost_full_bucket(self):
        propagated_points = list()
        propagated_points = self._get_n_points_with_pors([(1.1, 1), (2.2, 2),
                                                          (3.3, 3)])

        buckets_max_size = 4
        buckets_len = {1: buckets_max_size, 2: 3, 3: 3}
        alpha = 0.1
        sampled_must_add, sampled_for_update = target_based_sampler(
            propagated_points, buckets_len, buckets_max_size, alpha)

        expected_buckets_len = {
            1: buckets_max_size,
            2: buckets_max_size,
            3: buckets_max_size
        }
        self.assertDictEqual(buckets_len, expected_buckets_len)
        expected_sampled_must_add = {
            2: [propagated_points[1]],
            3: [propagated_points[3]]
        }
        self.assertDictEqual(sampled_must_add, expected_sampled_must_add)

    def test_dont_sample_alpha_0(self):
        propagated_points = list()
        propagated_points = self._get_n_points_with_pors([(1.1, 1), (2.2, 2),
                                                          (3.3, 3)])

        buckets_max_size = 4
        buckets_len = {
            1: buckets_max_size,
            2: buckets_max_size,
            3: buckets_max_size
        }
        alpha = 0
        sampled_must_add, sampled_for_update = target_based_sampler(
            propagated_points, buckets_len, buckets_max_size, alpha)
        expected_buckets_len = {
            1: buckets_max_size,
            2: buckets_max_size,
            3: buckets_max_size
        }
        self.assertDictEqual(buckets_len, expected_buckets_len)
        self.assertTrue(len(sampled_must_add) == 0)
        self.assertTrue(len(sampled_for_update) == 0)

    def test_generate_sample_for_update(self):
        propagated_points = list()
        propagated_points = self._get_n_points_with_pors([(1.1, 1), (2.2, 2),
                                                          (3.3, 3)])

        buckets_max_size = 4
        buckets_len = {1: buckets_max_size, 2: 3, 3: 3}
        alpha = 0.7
        seed = 42
        sampled_must_add, sampled_for_update = target_based_sampler(
            propagated_points, buckets_len, buckets_max_size, alpha, seed=seed)

        expected_buckets_len = {
            1: buckets_max_size,
            2: buckets_max_size,
            3: buckets_max_size
        }
        self.assertDictEqual(buckets_len, expected_buckets_len)
        expected_sampled_must_add = {
            2: [propagated_points[1]],
            3: [propagated_points[3]]
        }
        self.assertDictEqual(sampled_must_add, expected_sampled_must_add)

        # If alpha or seed changes this might not pass anymore
        expected_sampled_for_update = {
            2: [propagated_points[2]],
            3: [propagated_points[5]]
        }
        self.assertDictEqual(sampled_for_update, expected_sampled_for_update)


if __name__ == "__main__":
    main()
