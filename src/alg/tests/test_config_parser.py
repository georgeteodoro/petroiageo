import config_parser

from unittest import TestCase, main

class TestYAMLConfig(TestCase):
    
    def test_can_update_base_conf_from_str(self):
        starting_it = 10
        num_its = 20
        generate_porosity_cube = 'true'
        feature_selection_type = 'None'
        coords = [[1,2],[3,4],{'x': 4, 'y':5}]
        expected_coords = [{'x':1, 'y':2},
                           {'x':3, 'y':4},
                           {'x': 4, 'y':5}]
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
        self.assertEqual(my_config.alg['starting_it'], starting_it)
        self.assertEqual(my_config.alg['num_its'], num_its)
        self.assertEqual(my_config.alg['generate_porosity_cube'], True)
        self.assertEqual(my_config.alg['feature_selection_type'], feature_selection_type)
        self.assertEqual(my_config.alg['save_cube_every_n_its'], base_save_cube_every_n_its_expected)
        self.assertEqual(my_config.wells['coords'], expected_coords)


if __name__ == "__main__":
    main()