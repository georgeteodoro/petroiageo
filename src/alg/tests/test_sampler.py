from unittest import TestCase, main
import numpy as np

from sampler import AbstractChunkSampler, ChunkSamplerV1
from config_parser import YAMLConfig


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
