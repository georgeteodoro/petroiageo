from unittest import TestCase, main

import config_parser


class TestYAMLConfig(TestCase):
    def test_can_update_base_conf_from_str(self):
        starting_it = 10
        num_its = 20
        generate_porosity_cube = "true"
        feature_selection_type = "NONE"
        coords = [[1, 2], [3, 4], {"x": 4, "y": 5}]
        expected_coords = [{
            "x": 1,
            "y": 2
        }, {
            "x": 3,
            "y": 4
        }, {
            "x": 4,
            "y": 5
        }]
        base_save_cube_every_n_its_expected = -1
        yaml_str = f"""
        alg:
          starting_it: {starting_it}
          num_its: {num_its}
          generate_porosity_cube: {generate_porosity_cube}
          feature_selection_type: {feature_selection_type}

        wells:
          coords: {str(coords)}
          window: 0
        """

        my_config = config_parser.YAMLConfig(config_str=yaml_str)
        self.assertEqual(my_config.alg["starting_it"], starting_it)
        self.assertEqual(my_config.alg["num_its"], num_its)
        self.assertEqual(my_config.alg["generate_porosity_cube"], True)
        self.assertEqual(
            my_config.alg["feature_selection_type"],
            config_parser.FeatureSelection[feature_selection_type],
        )
        self.assertEqual(
            my_config.alg["save_cube_every_n_its"],
            base_save_cube_every_n_its_expected,
        )
        self.assertEqual(my_config.wells["coords"], expected_coords)

    def test_raise_invalid_feature_selection_mode(self):
        feature_selection = "DIAGONAL"
        yaml_str = f"""
        wells:
          coords: [[1, 1]]
          window: 0
        alg:
          feature_selection_type: {feature_selection}
        """
        with self.assertRaises(KeyError):
            config_parser.YAMLConfig(config_str=yaml_str)

    def test_accepts_valid_feature_selection_mode(self):
        feature_selection_1 = "FORWARD"
        yaml_str_fmt = """
        wells:
          coords: [[1, 1]]
          window: 0
        alg:
          feature_selection_type: {}
        """
        my_config = config_parser.YAMLConfig(
            config_str=yaml_str_fmt.format(feature_selection_1))
        self.assertEqual(
            my_config.alg["feature_selection_type"],
            config_parser.FeatureSelection[feature_selection_1],
        )

        feature_selection_2 = "NONE"
        my_config = config_parser.YAMLConfig(
            config_str=yaml_str_fmt.format(feature_selection_2))
        self.assertEqual(
            my_config.alg["feature_selection_type"],
            config_parser.FeatureSelection[feature_selection_2],
        )

    def test_raise_invalid_save_model_types(self):
        yaml_str_fmt = """
        wells:
          coords: [[1, 1]]
          window: 0
        alg:
          save_models_on: {}
        """

        save_model_type = "every_2.2"
        with self.assertRaises(ValueError):
            my_config = config_parser.YAMLConfig(
                config_str=yaml_str_fmt.format(save_model_type))

        save_model_type = "every_0"
        with self.assertRaises(ValueError):
            my_config = config_parser.YAMLConfig(
                config_str=yaml_str_fmt.format(save_model_type))

    def test_accepts_valid_save_model_type(self):
        save_model_types = ["last", "all", "every_4", "every_1"]
        yaml_str_fmt = """
        wells:
          coords: [[1, 1]]
          window: 0
        alg:
          save_models_on: {}
        """
        for model in save_model_types:
            my_config = config_parser.YAMLConfig(
                config_str=yaml_str_fmt.format(model))
            self.assertEqual(my_config.alg["save_models_on"], model)

    def test_accepts_list_of_dict_for_wells_coords(self):
        yaml_str = """
        wells:
          coords:
          - x: 1
            y: 1
          - x: 2
            y: 2
          - x: 3
            y: 3
          window: 0
        """
        expected_wells_coords_dicts = [
            {
                "x": 1,
                "y": 1
            },
            {
                "x": 2,
                "y": 2
            },
            {
                "x": 3,
                "y": 3
            },
        ]
        my_config = config_parser.YAMLConfig(config_str=yaml_str)
        self.assertListEqual(my_config.wells["coords"],
                             expected_wells_coords_dicts)

    def test_accepts_list_of_lists_for_wells_coords(self):
        yaml_str = """
        wells:
          coords: [[1,1],[2,2],[3,3]]
          window: 0
        """
        expected_wells_coords_dicts = [
            {
                "x": 1,
                "y": 1
            },
            {
                "x": 2,
                "y": 2
            },
            {
                "x": 3,
                "y": 3
            },
        ]
        my_config = config_parser.YAMLConfig(config_str=yaml_str)
        self.assertListEqual(my_config.wells["coords"],
                             expected_wells_coords_dicts)

    def test_accepts_lists_as_items_for_wells_coords(self):
        yaml_str = """
        wells:
          coords:
          - [1,1]
          - [2,2]
          - [3,3]
          window: 0
        """
        expected_wells_coords_dicts = [
            {
                "x": 1,
                "y": 1
            },
            {
                "x": 2,
                "y": 2
            },
            {
                "x": 3,
                "y": 3
            },
        ]
        my_config = config_parser.YAMLConfig(config_str=yaml_str)
        self.assertListEqual(my_config.wells["coords"],
                             expected_wells_coords_dicts)

    def test_accepts_multiple_ways_to_wells_coords(self):
        yaml_str = """
        wells:
          coords:
          - [1,1]
          - x: 2
            y: 2
          - [3, 3]
          window: 0
        """
        expected_wells_coords_dicts = [
            {
                "x": 1,
                "y": 1
            },
            {
                "x": 2,
                "y": 2
            },
            {
                "x": 3,
                "y": 3
            },
        ]
        my_config = config_parser.YAMLConfig(config_str=yaml_str)
        self.assertListEqual(my_config.wells["coords"],
                             expected_wells_coords_dicts)

    def test_raise_on_negative_wells_coords(self):
        yaml_str = """
        wells:
          coords: [[-1, 1]]
          window: 0
        """
        with self.assertRaises(ValueError):
            my_config = config_parser.YAMLConfig(config_str=yaml_str)

    def test_raise_on_empty_wells_coords(self):
        yaml_str = """
       wells:
         coords:
         window: 0

       """
        with self.assertRaises(ValueError):
            my_config = config_parser.YAMLConfig(config_str=yaml_str)

    def test_get_wells_as_simple_list(self):
        yaml_str = """
        wells:
          coords:
          - [1,1]
          - [2,2]
          - [3,3]
          window: 0
        """
        expected_list = [(1, 1), (2, 2), (3, 3)]
        my_config = config_parser.YAMLConfig(config_str=yaml_str)
        self.assertListEqual(my_config.wells_as_simple_list, expected_list)

    def test_raise_invalid_starting_it(self):
        invalid_values = [-1, 'a', 3.4]
        yaml_str_fmt = """
        wells:
          coords: [[1, 1]]
          window: 0
        alg:
          starting_it: {}
        """
        for invalid_value in invalid_values:
            with self.assertRaises(ValueError):
                _ = config_parser.YAMLConfig(
                    config_str=yaml_str_fmt.format(invalid_value))

    def test_raise_invalid_num_its(self):
        invalid_values = [-1, 'a', 3.4]
        yaml_str_fmt = """
        wells:
          coords: [[1, 1]]
          window: 0
        alg:
          num_its: {}
        """
        for invalid_value in invalid_values:
            with self.assertRaises(ValueError):
                _ = config_parser.YAMLConfig(
                    config_str=yaml_str_fmt.format(invalid_value))

    def test_raises_validation_only_wells_index_out_of_bounds(self):
        yaml_str = """
        wells:
          coords: [[1,1], [2,2]]
          window: 0
        alg:
          validation_only_wells: [1, 3]
        """
        with self.assertRaises(ValueError):
            my_config = config_parser.YAMLConfig(config_str=yaml_str)

    def test_raises_validation_only_wells_negative_index(self):
        yaml_str = """
        wells:
          coords: [[1,1], [2,2]]
          window: 0
        alg:
          validation_only_wells: [1, -1]
        """
        with self.assertRaises(ValueError):
            my_config = config_parser.YAMLConfig(config_str=yaml_str)

    def test_raises_test_only_wells_index_out_of_bounds(self):
        yaml_str = """
        wells:
          coords: [[1,1], [2,2]]
          window: 0
        alg:
          test_only_wells: [1, 3]
        """
        with self.assertRaises(ValueError):
            my_config = config_parser.YAMLConfig(config_str=yaml_str)

    def test_raises_test_only_wells_negative_index(self):
        yaml_str = """
        wells:
          coords: [[1,1], [2,2]]
          window: 0
        alg:
          test_only_wells: [1, -1]
        """
        with self.assertRaises(ValueError):
            my_config = config_parser.YAMLConfig(config_str=yaml_str)

    def test_can_get_train_only_wells_coords(self):
        yaml_str = """
        wells:
          coords: [[1,1], [2,2]]
          window: 0
        alg:
          test_only_wells: [1]
        """
        my_config = config_parser.YAMLConfig(config_str=yaml_str)
        expected_train_only_wells_coords = [(1, 1)]
        self.assertListEqual(expected_train_only_wells_coords,
                             my_config.train_wells_coords)

    def test_can_get_train_only_wells_ids(self):
        yaml_str = """
        wells:
          coords: [[1,1], [2,2], [3,3]]
          window: 0
        alg:
          test_only_wells: [1]
        """
        my_config = config_parser.YAMLConfig(config_str=yaml_str)
        expected_train_only_wells_ids = [0, 2]
        self.assertListEqual(expected_train_only_wells_ids,
                             my_config.train_wells_ids)

    def test_can_get_base_max_exec_time(self):
        yaml_str = """
        wells:
          coords: [[1,1], [2,2], [3,3]]
          window: 0
        """
        my_config = config_parser.YAMLConfig(config_str=yaml_str)
        expected_base_max_exec_time = -1
        self.assertEqual(my_config.alg['max_exec_time'],
                         expected_base_max_exec_time)

    def test_raise_on_invalid_max_exec_time(self):
        invalid_values = [100.3, 'ab']
        yaml_str_fmt = """
        wells:
          coords: [[1,1]]
          window: 0
        alg:
          max_exec_time: {}
        """
        for invalid_value in invalid_values:
            with self.assertRaises(ValueError):
                _ = config_parser.YAMLConfig(
                    config_str=yaml_str_fmt.format(invalid_value))

    def test_raise_on_invalid_max_num_features(self):
        invalid_values = [100.3, 'ab', -2]
        yaml_str_fmt = """
        wells:
          coords: [[1,1]]
          window: 0
        alg:
          max_num_features: {}
        """
        for invalid_value in invalid_values:
            with self.assertRaises(ValueError):
                _ = config_parser.YAMLConfig(
                    config_str=yaml_str_fmt.format(invalid_value))

    def test_raise_on_invalid_window(self):
        invalid_values = [100.3, 'ab', -2]
        yaml_str_fmt = """
        wells:
          coords: [[1,1]]
          window: 0
        alg:
          window: {}
        """
        for invalid_value in invalid_values:
            with self.assertRaises(ValueError):
                _ = config_parser.YAMLConfig(
                    config_str=yaml_str_fmt.format(invalid_value))

    def test_can_set_window(self):
        window = 2
        yaml_str_fmt = """
        wells:
          coords: [[1,1]]
          window: 0
        alg:
          window: {}
        """
        config = config_parser.YAMLConfig(
            config_str=yaml_str_fmt.format(window))
        self.assertEqual(config.alg['window'], window)

    def test_can_get_base_window(self):
        expected_base_window = 3
        yaml_str_fmt = """
        wells:
          coords: [[1,1]]
          window: 0
        """
        config = config_parser.YAMLConfig(config_str=yaml_str_fmt)
        self.assertEqual(config.alg['window'], expected_base_window)

    def test_can_get_base_layer_w_size(self):
        expected_layer_w_size = -1
        yaml_str_fmt = """
        wells:
          coords: [[1,1]]
          window: 0
        """
        config = config_parser.YAMLConfig(config_str=yaml_str_fmt)
        self.assertEqual(config.alg['sampling']['layers_window_size'],
                         expected_layer_w_size)

    def test_raise_on_invalid_sampling_window_size(self):
        invalid_values = [3.2, "a"]
        yaml_str_fmt = """
        wells:
          coords: [[1,1]]
          window: 0

        alg:
          sampling:
            layers_window_size: {}
        """
        for invalid_value in invalid_values:
            with self.assertRaises(ValueError):
                my_config = config_parser.YAMLConfig(
                    config_str=yaml_str_fmt.format(invalid_value))

    def test_raise_on_invalid_sampling_max_points(self):
        invalid_values = [3.2, "a"]
        yaml_str_fmt = """
        wells:
          coords: [[1,1]]
          window: 0

        alg:
          sampling:
            max_points: {}
        """
        for invalid_value in invalid_values:
            with self.assertRaises(ValueError):
                my_config = config_parser.YAMLConfig(
                    config_str=yaml_str_fmt.format(invalid_value))

    def test_raise_on_invalid_seed(self):
        invalid_values = [3.2, "a", -2, True]
        yaml_str_fmt = """
        wells:
          coords: [[1,1]]
          window: 0

        alg:
          sampling:
            seed: {}
        """
        for invalid_value in invalid_values:
            with self.assertRaises(ValueError):
                _ = config_parser.YAMLConfig(
                    config_str=yaml_str_fmt.format(invalid_value))

    def test_can_update_seed_value(self):
        seed_value = 1
        yaml_str_fmt = """
        wells:
          coords: [[1,1]]
          window: 0

        alg:
          sampling:
            seed: {}
        """.format(seed_value)
        my_config = config_parser.YAMLConfig(config_str=yaml_str_fmt)
        self.assertEqual(seed_value, my_config.alg['sampling']['seed'])

    def test_can_update_beta_dist(self):
        alpha = 2
        beta = 3
        yaml_str_fmt = """
        wells:
          coords: [[1,1]]
          window: 0

        alg:
          sampling:
            beta_dist:
              alpha: {}
              beta: {}
        """.format(alpha, beta)
        my_config = config_parser.YAMLConfig(config_str=yaml_str_fmt)
        self.assertEqual(alpha,
                         my_config.alg['sampling']['beta_dist']['alpha'])
        self.assertEqual(beta, my_config.alg['sampling']['beta_dist']['beta'])

    def test_raise_invalid_sampling_beta_dist_params(self):
        invalid_params = [
            (-1, 2),
            (1, -2),
            (-2, -2),
            (1, "a"),
            ("a", 1),
            ("a", -1),
        ]
        yaml_str_fmt = """
        wells:
          coords: [[1,1]]
          window: 0

        alg:
          sampling:
            beta_dist:
              alpha: {}
              beta: {}
        """
        for alpha, beta in invalid_params:
            with self.assertRaises(ValueError):
                my_config = config_parser.YAMLConfig(
                    config_str=yaml_str_fmt.format(alpha, beta))

    def test_can_add_new_attribute(self):
        yaml_str = """
        wells:
          coords: [[1,1],[2,2]]
          window: 0
        """
        my_config = config_parser.YAMLConfig(config_str=yaml_str)
        param_name = "new_param"
        param_value = 123
        my_config.add_param(param_name, param_value)
        self.assertEqual(my_config.get_param(param_name), param_value)

    def test_raises_if_add_new_param_with_name_equals_to_base_config(self):
        yaml_str = """
        wells:
          coords: [[1,1],[2,2]]
          window: 0
        """
        my_config = config_parser.YAMLConfig(config_str=yaml_str)
        for base_param in config_parser.Config.TOP_LEVEL_BASE_CONFIGS:
            new_value = "new_value"
            with self.assertRaises(config_parser.InvalidNewParamError):
                my_config.add_param(base_param, new_value)

    def test_can_remove_new_param(self):
        yaml_str = """
        wells:
          coords: [[1,1],[2,2]]
          window: 0
        """
        my_config = config_parser.YAMLConfig(config_str=yaml_str)
        param_name = "new_param"
        param_value = 123
        my_config.add_param(param_name, param_value)
        my_config.remove_param(param_name)
        self.assertIsNone(my_config.get_param(param_name))

    def test_raises_if_trying_to_remove_base_param(self):
        yaml_str = """
        wells:
          coords: [[1,1],[2,2]]
          window: 0
        """
        my_config = config_parser.YAMLConfig(config_str=yaml_str)
        for base_param in config_parser.Config.TOP_LEVEL_BASE_CONFIGS:
            with self.assertRaises(config_parser.InvalidNewParamError):
                my_config.remove_param(base_param)

    def test_can_get_max_chunk_size(self):
        yaml_str = """
        wells:
          coords: [[1,1],[2,2]]
          window: 0
        """
        my_config = config_parser.YAMLConfig(config_str=yaml_str)
        try:
            _ = my_config.alg['parallel']['max_points_per_chunk']
            self.assertTrue(True)
        except:
            self.assertTrue(False)

    def test_raise_invalid_parallel_chunksize(self):
        invalid_params = [0, 'a']
        yaml_str = """
        wells:
          coords: [[1,1],[2,2]]
          window: 0

        alg:
          parallel:
            max_points_per_chunk: {}
        """
        for invalid_param in invalid_params:
            with self.assertRaises(ValueError):
                _ = config_parser.YAMLConfig(
                    config_str=yaml_str.format(invalid_param))

    def test_can_get_base_layers_to_predict(self):
        yaml_str = """
        wells:
          coords: [[1,1],[2,2]]
          window: 0
        """
        my_config = config_parser.YAMLConfig(config_str=yaml_str)
        expected_layers_to_predict = 1
        self.assertEqual(my_config.alg['layers_to_predict'],
                         expected_layers_to_predict)

    def test_can_set_layers_to_predict(self):
        yaml_str = """
        wells:
          coords: [[1,1],[2,2]]
          window: 0
        alg:
          layers_to_predict: {}
        """
        expected_layers_to_predict = 2
        my_config = config_parser.YAMLConfig(
            config_str=yaml_str.format(expected_layers_to_predict))
        self.assertEqual(my_config.alg['layers_to_predict'],
                         expected_layers_to_predict)

    def test_raise_invalid_layers_to_predict(self):
        invalid_params = [0, 'a', 1.3, 0.5]
        yaml_str = """
        wells:
          coords: [[1,1],[2,2]]
          window: 0

        alg:
          layers_to_predict: {}
        """
        for invalid_param in invalid_params:
            with self.assertRaises(ValueError):
                _ = config_parser.YAMLConfig(
                    config_str=yaml_str.format(invalid_param))

    def test_can_get_rings_range(self):
        yaml_str = """
        wells:
          coords:
          - [4,2]
          - [7,6]
          - [2,7]
          window: 0

        alg:
          layers_to_predict: {}
        """
        layers_to_predict = [1, 2, 5]
        curr_it = [1, 3, 7]

        expected_start_ring = [1, 5, 31]
        expected_end_ring = [1, 6, 35]

        for test_idx, (layer, it) in enumerate(zip(layers_to_predict,
                                                   curr_it)):
            config = config_parser.YAMLConfig(
                config_str=yaml_str.format(layer))
            ring_range = config.ring_range_to_expand(it)
            result_expected = (expected_start_ring[test_idx],
                               expected_end_ring[test_idx])
            self.assertTupleEqual(ring_range, result_expected)


if __name__ == "__main__":
    main()
