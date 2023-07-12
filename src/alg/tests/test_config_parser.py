import config_parser

from unittest import TestCase, main


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
        """
        with self.assertRaises(ValueError):
            my_config = config_parser.YAMLConfig(config_str=yaml_str)

    def test_raise_on_empty_wells_coords(self):
        yaml_str = """
       wells:
         coords:

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
        """
        expected_list = [(1, 1), (2, 2), (3, 3)]
        my_config = config_parser.YAMLConfig(config_str=yaml_str)
        self.assertListEqual(my_config.wells_as_simple_list, expected_list)

    def test_raise_invalid_starting_it(self):
        yaml_str_1 = """
        wells:
          coords: [[1, 1]]
        alg:
          starting_it: -1
        """
        yaml_str_2 = """
        wells:
          coords: [[1, 1]]
        alg:
          starting_it: a
        """
        yaml_str_3 = """
        wells:
          coords: [[1, 1]]
        alg:
          starting_it: 3.4
        """
        for yaml_str in [yaml_str_1, yaml_str_2, yaml_str_3]:
            with self.assertRaises(ValueError):
                my_config = config_parser.YAMLConfig(config_str=yaml_str)

    def test_raise_invalid_num_its(self):
        yaml_str_1 = """
        wells:
          coords: [[1, 1]]
        alg:
          num_its: -1
        """
        yaml_str_2 = """
        wells:
          coords: [[1, 1]]
        alg:
          num_its: a
        """
        yaml_str_3 = """
        wells:
          coords: [[1, 1]]
        alg:
          num_its: 3.4
        """
        for yaml_str in [yaml_str_1, yaml_str_2, yaml_str_3]:
            with self.assertRaises(ValueError):
                my_config = config_parser.YAMLConfig(config_str=yaml_str)

    def test_raises_validation_only_wells_index_out_of_bounds(self):
        yaml_str = """
        wells:
          coords: [[1,1], [2,2]]
        alg:
          validation_only_wells: [1, 3]
        """
        with self.assertRaises(ValueError):
            my_config = config_parser.YAMLConfig(config_str=yaml_str)

    def test_raises_validation_only_wells_negative_index(self):
        yaml_str = """
        wells:
          coords: [[1,1], [2,2]]
        alg:
          validation_only_wells: [1, -1]
        """
        with self.assertRaises(ValueError):
            my_config = config_parser.YAMLConfig(config_str=yaml_str)

    def test_raises_test_only_wells_index_out_of_bounds(self):
        yaml_str = """
        wells:
          coords: [[1,1], [2,2]]
        alg:
          test_only_wells: [1, 3]
        """
        with self.assertRaises(ValueError):
            my_config = config_parser.YAMLConfig(config_str=yaml_str)

    def test_raises_test_only_wells_negative_index(self):
        yaml_str = """
        wells:
          coords: [[1,1], [2,2]]
        alg:
          test_only_wells: [1, -1]
        """
        with self.assertRaises(ValueError):
            my_config = config_parser.YAMLConfig(config_str=yaml_str)

    def test_raise_on_invalid_max_exec_time(self):
        yaml_str_1 = """
       wells:
         coords: [[1,1]]
       alg:
         max_exec_time: 100.3
       """
        yaml_str_2 = """
       wells:
         coords: [[1,1]]
       alg:
         max_exec_time: ab
       """
        for yaml_str in [yaml_str_1, yaml_str_2]:
            with self.assertRaises(ValueError):
                my_config = config_parser.YAMLConfig(config_str=yaml_str)

    def test_raise_on_invalid_max_num_features(self):
        yaml_str_1 = """
       wells:
         coords: [[1,1]]
       alg:
         max_num_features: 100.3
       """
        yaml_str_2 = """
       wells:
         coords: [[1,1]]
       alg:
         max_num_features: ab
       """
        yaml_str_2 = """
       wells:
         coords: [[1,1]]
       alg:
         max_num_features: -2
       """
        for yaml_str in [yaml_str_1, yaml_str_2]:
            with self.assertRaises(ValueError):
                my_config = config_parser.YAMLConfig(config_str=yaml_str)

    def test_raise_on_invalid_sampling_window_size(self):
        invalid_values = [3.2, "a"]
        yaml_str_fmt = """
       wells:
         coords: [[1,1]]
       
       alg:
         sampling:
           its_window_size: {}
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
       
       alg:
         sampling:
           max_points: {}
       """
        for invalid_value in invalid_values:
            with self.assertRaises(ValueError):
                my_config = config_parser.YAMLConfig(
                    config_str=yaml_str_fmt.format(invalid_value))

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
      """
        my_config = config_parser.YAMLConfig(config_str=yaml_str)
        for base_param in config_parser.Config.TOP_LEVEL_BASE_CONFIGS:
            with self.assertRaises(config_parser.InvalidNewParamError):
                my_config.remove_param(base_param)

    def test_can_get_max_chunk_size(self):
        yaml_str = """
        wells:
          coords: [[1,1],[2,2]]
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
        
        alg:
          parallel:
            max_points_per_chunk: {}
        """
        for invalid_param in invalid_params:
            with self.assertRaises(ValueError):
                _ = config_parser.YAMLConfig(
                    config_str=yaml_str.format(invalid_param))


if __name__ == "__main__":
    main()
