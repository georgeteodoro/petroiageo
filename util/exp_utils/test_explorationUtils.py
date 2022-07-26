from unittest import TestCase, main
from explorationUtils import ExplorationCube, NonNegativeInteger2DPoint, Well
    
class TestWell(TestCase):

    def test_cant_init_with_invalid_coords(self):
        invalid_x = -1
        valid_y = 2
        self.assertRaises(ValueError, Well, invalid_x, valid_y)

        invalid_x = True
        self.assertRaises(TypeError, Well, invalid_x, valid_y)

        invalid_y = 'a'
        valid_x = 1
        self.assertRaises(TypeError, Well, valid_x, invalid_y)

        invalid_y = 3.7
        valid_x = 1
        self.assertRaises(TypeError, Well, valid_x, invalid_y)
    
    def test_can_init_with_valid_coords(self):
        valid_x = 1
        valid_y = 2
        point = NonNegativeInteger2DPoint(valid_x, valid_y)
        self.assertEqual(point.x, valid_x)
        self.assertEqual(point.y, valid_y)
    
    def test_cant_set_invalid_key(self):
        point = Well(0, 1)

        with self.assertRaises(IndexError):
            point[2] = 3
    
    def test_cant_set_key_invalid_value(self):
        valid_x = 0
        valid_y = 1
        point = Well(valid_x, valid_y)

        with self.assertRaises(ValueError):
            point[1] = -1
        
        with self.assertRaises(TypeError):
            point[0] = 'a'

        self.assertEqual(point[0], valid_x)
        self.assertEqual(point[1], valid_y)
    
    def test_can_get_coords(self):
        valid_x = 0
        valid_y = 1
        point = Well(valid_x, valid_y)
        self.assertTupleEqual(point.coords, (valid_x, valid_y))
    
    def test_well_equality(self):
        x = 0
        y = 1

        well_1 = Well(x, y)
        well_2 = Well(x, y)
        self.assertEqual(well_1, well_2)
    
    def test_well_inequality(self):
        x = 0
        y_1 = 1
        y_2 = 2

        well_1 = Well(x, y_1)
        well_2 = Well(x, y_2)
        self.assertNotEqual(well_1, well_2)
    
    def test_num_its_until_point_with_less_x_less_y_than_well(self):
        well_x = 1
        well_y = 2

        well = Well(well_x, well_y)
        target_point = (0, 0)

        self.assertEqual(well.its_to_point(target_point), 2)
    
    def test_num_its_until_point_with_bigger_x_less_y_than_well(self):
        well_x = 1
        well_y = 2

        well = Well(well_x, well_y)
        target_point = (2, 0)

        self.assertEqual(well.its_to_point(target_point), 2)
    
    def test_num_its_until_point_with_less_x_bigger_y_than_well(self):
        well_x = 1
        well_y = 2

        well = Well(well_x, well_y)
        target_point = (0, 4)

        self.assertEqual(well.its_to_point(target_point), 2)
    
    def test_num_its_until_point_with_bigger_x_bigger_y_than_well(self):
        well_x = 1
        well_y = 2

        well = Well(well_x, well_y)
        target_point = (2, 3)

        self.assertEqual(well.its_to_point(target_point), 1)
    
    def test_num_its_until_point_with_same_x_bigger_y_than_well(self):
        well_x = 1
        well_y = 2

        well = Well(well_x, well_y)
        target_point = (1, 4)

        self.assertEqual(well.its_to_point(target_point), 2)
    
    def test_num_its_until_point_with_same_x_less_y_than_well(self):
        well_x = 1
        well_y = 2

        well = Well(well_x, well_y)
        target_point = (1, 0)

        self.assertEqual(well.its_to_point(target_point), 2)
    
    def test_num_its_until_point_with_same_x_same_y_than_well(self):
        well_x = 1
        well_y = 2

        well = Well(well_x, well_y)
        target_point = (1, 2)

        self.assertEqual(well.its_to_point(target_point), 0)
    
    def test_num_its_until_point_with_bigger_x_same_y_than_well(self):
        well_x = 1
        well_y = 2

        well = Well(well_x, well_y)
        target_point = (2, 2)

        self.assertEqual(well.its_to_point(target_point), 1)
    
    def test_num_its_until_point_with_less_x_same_y_than_well(self):
        well_x = 1
        well_y = 2

        well = Well(well_x, well_y)
        target_point = (0, 2)

        self.assertEqual(well.its_to_point(target_point), 1)
    
    def test_num_its_until_point_with_both_coords_negative(self):
        well_x = 1
        well_y = 2

        well = Well(well_x, well_y)
        target_point = (-1, -1)

        self.assertEqual(well.its_to_point(target_point), 3)
    
    def test_num_its_until_point_with_x_coord_negative(self):
        well_x = 1
        well_y = 2

        well = Well(well_x, well_y)
        target_point = (-1, 2)

        self.assertEqual(well.its_to_point(target_point), 2)
    
    def test_num_its_until_point_with_y_coord_negative(self):
        well_x = 1
        well_y = 2

        well = Well(well_x, well_y)
        target_point = (0, -3)

        self.assertEqual(well.its_to_point(target_point), 5)

class TestExplorationCube(TestCase):

    def test_can_init_with_valid_points(self):
        top_left = (0, 0)
        bottom_right = (1, 1)
        depth = 1

        cube = ExplorationCube(top_left, bottom_right, depth)

        self.assertTupleEqual(cube.top_left, top_left)
        self.assertTupleEqual(cube.bottom_right, bottom_right)
        self.assertEqual(cube.depth, depth)
    
    def test_raise_when_init_with_wrong_type_points(self):
        wrong_top_left = (0, True)
        bottom_right = (1, 1)
        depth = 1

        self.assertRaises(TypeError, ExplorationCube, wrong_top_left, bottom_right, depth)

        wrong_bottom_right = (1, ['a','b'])
        top_left = (0, 0)
        self.assertRaises(TypeError, ExplorationCube, top_left, wrong_bottom_right)
    
    def test_raise_when_init_with_negative_points(self):
        wrong_top_left = (0, -1)
        bottom_right = (1, 1)
        depth = 1

        self.assertRaises(ValueError, ExplorationCube, wrong_top_left, bottom_right, depth)

        wrong_bottom_right = (-3, -4)
        top_left = (0, 0)
        self.assertRaises(ValueError, ExplorationCube, top_left, wrong_bottom_right, depth)
    
    def test_raise_when_init_with_top_left_after_bottom_right(self):
        top_left = (0, 1)
        bottom_right = (1, 0)
        depth = 1

        self.assertRaises(ValueError, ExplorationCube, top_left, bottom_right, depth)
    
    def test_raise_when_init_top_left_same_x_bottom_right(self):
        top_left = (1, 1)
        bottom_right = (1, 2)
        depth = 1

        self.assertRaises(ValueError, ExplorationCube, top_left, bottom_right, depth)
    
    def test_raise_when_init_top_left_same_y_bottom_right(self):
        top_left = (0, 1)
        bottom_right = (1, 1)
        depth = 1

        self.assertRaises(ValueError, ExplorationCube, top_left, bottom_right, depth)
    
    def test_raise_when_init_top_left_same_as_bottom_right(self):
        top_left = (1, 1)
        bottom_right = (1, 1)
        depth = 1

        self.assertRaises(ValueError, ExplorationCube, top_left, bottom_right, depth)
    
    def test_raise_when_init_non_positive_depth(self):
        top_left = (0, 0)
        bottom_right = (1, 1)
        depth = -1

        self.assertRaises(ValueError, ExplorationCube, top_left, bottom_right, depth)

        depth = 0

        self.assertRaises(ValueError, ExplorationCube, top_left, bottom_right, depth)
    
    def test_raise_when_init_non_int_depth(self):
        top_left = (0, 0)
        bottom_right = (1, 1)
        depth = True

        self.assertRaises(TypeError, ExplorationCube, top_left, bottom_right, depth)

        depth = 'a'

        self.assertRaises(TypeError, ExplorationCube, top_left, bottom_right, depth)
    
    def test_raise_when_set_top_left_same_x_bottom_right(self):
        valid_top_left = (0, 0)
        valid_bot_right = (1, 1)
        depth = 1

        cube = ExplorationCube(valid_top_left, valid_bot_right, depth)

        invalid_top_left = (1, 0)
        with self.assertRaises(ValueError):
            cube.top_left = invalid_top_left
    
        self.assertEqual(cube.top_left, valid_top_left)
    
    def test_raise_when_set_top_left_same_y_bottom_right(self):
        valid_top_left = (0, 0)
        valid_bot_right = (1, 1)
        depth = 1

        cube = ExplorationCube(valid_top_left, valid_bot_right, depth)

        invalid_top_left = (1, 1)
        with self.assertRaises(ValueError):
            cube.top_left = invalid_top_left
    
        self.assertEqual(cube.top_left, valid_top_left)

    def test_raise_when_set_non_positive_depth(self):
        top_left = (0, 0)
        bottom_right = (1, 1)
        valid_depth = 1

        cube = ExplorationCube(top_left, bottom_right, valid_depth)

        invalid_depth = -1
        with self.assertRaises(ValueError):
            cube.depth = invalid_depth

        self.assertEqual(cube.depth, valid_depth)
    
    def test_exp_cube_equality(self):
        top_left = (0, 0)
        bottom_right = (1, 1)
        depth = (1)

        cube1 = ExplorationCube(top_left, bottom_right, depth)
        cube2 = ExplorationCube(top_left, bottom_right, depth)

        self.assertEqual(cube1, cube2)
    
    def test_exp_cube_inequality(self):
        top_left = (0, 0)
        bottom_right_1 = (1, 1)
        bottom_right_2 = (2, 2)
        depth = (1)

        cube1 = ExplorationCube(top_left, bottom_right_1, depth)
        cube2 = ExplorationCube(top_left, bottom_right_2, depth)

        self.assertNotEqual(cube1, cube2)

if __name__ == '__main__':
    main()