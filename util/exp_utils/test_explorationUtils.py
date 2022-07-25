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

class TestExplorationCube(TestCase):

    def test_can_init_with_valid_points(self):
        top_left = (0, 0)
        bottom_right = (1, 1)
        depth = 1

        cube = ExplorationCube(top_left, bottom_right, depth)

        self.assertTupleEqual(cube.top_left, top_left)
        self.assertTupleEqual(cube.bottom_right, bottom_right)
    
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


if __name__ == '__main__':
    main()