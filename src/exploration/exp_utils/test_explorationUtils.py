from unittest import TestCase, main
from explorationUtils import (
    ExplorationCube,
    NonNegativeInteger2DPoint,
    Well,
    WellSet,
)


class TestWell(TestCase):
    def test_cant_init_with_invalid_coords(self):
        invalid_x = -1
        valid_y = 2
        self.assertRaises(ValueError, Well, invalid_x, valid_y)

        invalid_x = True
        self.assertRaises(TypeError, Well, invalid_x, valid_y)

        invalid_y = "a"
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
        well = Well(0, 1)

        with self.assertRaises(IndexError):
            well[2] = 3

    def test_cant_set_key_invalid_value(self):
        valid_x = 0
        valid_y = 1
        well = Well(valid_x, valid_y)

        with self.assertRaises(ValueError):
            well[1] = -1

        with self.assertRaises(TypeError):
            well[0] = "a"

        self.assertEqual(well[0], valid_x)
        self.assertEqual(well[1], valid_y)

    def test_can_get_coords(self):
        valid_x = 0
        valid_y = 1
        well = Well(valid_x, valid_y)
        self.assertTupleEqual(well.coords, (valid_x, valid_y))

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

    def test_get_extremity_points_at_some_it(self):
        max_x = 10
        max_y = 10

        well = Well(1, 1)

        it = 1
        expected_extremity_points = ((0, 0), (2, 2))
        self.assertTupleEqual(
            well.extremity_points_at_it(it, max_x, max_y),
            expected_extremity_points,
        )

        well = Well(4, 6)
        it = 3
        expected_extremity_points = ((1, 3), (7, 9))
        self.assertTupleEqual(
            well.extremity_points_at_it(it, max_x, max_y),
            expected_extremity_points,
        )

    def test_has_overlap_at_some_it(self):
        well1 = Well(1, 1)
        well2 = Well(4, 4)
        max_x = 10
        max_y = 10

        it = 1
        self.assertFalse(well1.overlap_with_well_at_it(well2, it, max_x, max_y))

        it = 2
        self.assertTrue(well1.overlap_with_well_at_it(well2, it, max_x, max_y))

        well1 = Well(1, 1)
        well2 = Well(1, 2)
        it = 1
        self.assertTrue(well1.overlap_with_well_at_it(well2, it, max_x, max_y))

        it = 0
        self.assertFalse(well1.overlap_with_well_at_it(well2, it, max_x, max_y))

    def test_num_predicted_points(self):
        plane_max_x = 3
        plane_max_y = 3

        depth = 30

        well = Well(1, 1)

        it = 1
        expected_num_predicted_points = (plane_max_x * plane_max_y - 1) * depth
        self.assertEqual(
            well.num_predicted(plane_max_x, plane_max_y, depth, it),
            expected_num_predicted_points,
        )

    def test_num_predicted_points_passing_plane_size(self):
        plane_max_x = 9
        plane_max_y = 9
        depth = 10

        well = Well(2, 2)

        it = 10
        expected_num_predicted_points = (
            (plane_max_x + 1) * (plane_max_y + 1) - 1
        ) * depth
        self.assertEqual(
            well.num_predicted(plane_max_x, plane_max_y, depth, it),
            expected_num_predicted_points,
        )

    def test_no_volume_overlap_between_wells(self):
        well1 = Well(0, 0)
        well2 = Well(2, 2)

        plane_max_x = 9
        plane_max_y = 9
        depth = 10

        it = 0
        expected_result = 0
        self.assertEqual(
            well1.overlap_volume_at_it(
                well2, it, plane_max_x, plane_max_y, depth
            ),
            expected_result,
        )

    def test_has_one_volume_overlap_between_wells(self):
        well1 = Well(0, 0)
        well2 = Well(2, 2)

        plane_max_x = 9
        plane_max_y = 9
        depth = 10

        it = 1
        expected_result = depth * 1
        self.assertEqual(
            well1.overlap_volume_at_it(
                well2, it, plane_max_x, plane_max_y, depth
            ),
            expected_result,
        )

    def test_has_x_range_volume_overlap_between_wells(self):
        well1 = Well(1, 1)
        well2 = Well(3, 1)

        plane_max_x = 9
        plane_max_y = 9
        depth = 10

        it = 1
        expected_result = depth * 3
        self.assertEqual(
            well1.overlap_volume_at_it(
                well2, it, plane_max_x, plane_max_y, depth
            ),
            expected_result,
        )

    def test_has_y_range_volume_overlap_between_wells(self):
        well1 = Well(1, 1)
        well2 = Well(1, 3)

        plane_max_x = 9
        plane_max_y = 9
        depth = 10

        it = 1
        expected_result = depth * 3
        self.assertEqual(
            well1.overlap_volume_at_it(
                well2, it, plane_max_x, plane_max_y, depth
            ),
            expected_result,
        )

    def test_has_volume_overlap_between_wells(self):
        well1 = Well(1, 1)
        well2 = Well(4, 4)

        plane_max_x = 9
        plane_max_y = 9
        depth = 10

        it = 2
        expected_result = depth * 4
        self.assertEqual(
            well1.overlap_volume_at_it(
                well2, it, plane_max_x, plane_max_y, depth
            ),
            expected_result,
        )


class TestWellSet(TestCase):
    def setUp(self):
        self.well_set = WellSet()

    def test_can_add_well_at(self):
        self.well_set.add_well_at(1, 1)
        self.assertTrue(self.well_set.has_well_at(1, 1))

    def test_cant_add_well_with_negative_coords_at(self):
        with self.assertRaises(ValueError):
            self.well_set.add_well_at(-1, 1)

    def test_cant_add_well_with_non_int_coords_at(self):
        with self.assertRaises(TypeError):
            self.well_set.add_well_at(1.7, 1)

        with self.assertRaises(TypeError):
            self.well_set.add_well_at(True, 1)

    def test_can_add_well(self):
        well = Well(1, 1)
        self.well_set.add_well(well)
        self.assertTrue(self.well_set.has_well_at(1, 1))
        self.assertTrue(self.well_set.has_well(well))

    def test_can_add_all_wells(self):
        wells = [Well(1, 1), Well(1, 2)]
        self.well_set.add_all_wells(wells)
        self.assertTrue(self.well_set.has_well_at(1, 1))
        self.assertTrue(self.well_set.has_well(wells[1]))

    def test_can_add_all_wells_at(self):
        wells = [(1, 1), (1, 2)]
        self.well_set.add_all_wells_at(wells)
        self.assertTrue(self.well_set.has_well_at(1, 1))
        self.assertTrue(self.well_set.has_well_at(*wells[1]))

    def test_can_remove_well(self):
        well = Well(1, 1)
        self.well_set.add_well(well)
        self.well_set.remove_well(well)
        self.assertFalse(self.well_set.has_well(well))

    def test_can_remove_well_at(self):
        well = Well(1, 1)
        self.well_set.add_well(well)
        self.well_set.remove_well_at(well.coords[0], well.coords[1])
        self.assertFalse(self.well_set.has_well(well))

    def test_can_remove_all_wells(self):
        wells = [Well(1, 1), Well(0, 0)]
        self.well_set.add_all_wells(wells)
        self.well_set.remove_all()
        self.assertFalse(self.well_set.has_well(wells[0]))
        self.assertFalse(self.well_set.has_well(wells[1]))

    def test_can_get_well_at(self):
        well = Well(1, 1)
        self.well_set.add_well(well)
        retrieved_well = self.well_set.get_well_at(
            well.coords[0], well.coords[1]
        )
        self.assertEqual(well, retrieved_well)

    def test_dont_get_well_that_doesnt_exists(self):
        retrieved_well = self.well_set.get_well_at(1, 1)
        self.assertIsNone(retrieved_well)

    def test_dont_raise_if_remove_when_dont_have_well(self):
        self.assertIsNone(self.well_set.remove_well_at(1, 1))

    def test_its_until_point(self):
        wells = [(1, 1), (1, 2)]
        self.well_set.add_all_wells_at(wells)

        target_point = (1, 3)

        self.assertEqual(self.well_set.its_until_point(target_point), 1)

        target_point = (-1, 0)
        self.assertEqual(self.well_set.its_until_point(target_point), 2)

    def test_its_until_point_with_no_wells(self):
        target_point = (1, 3)

        self.assertEqual(
            self.well_set.its_until_point(target_point), float("inf")
        )

    def test_first_well_to_point(self):
        wells = [Well(1, 1), Well(1, 2)]
        self.well_set.add_all_wells(wells)

        target_point = (1, 3)
        self.assertEqual(self.well_set.first_to_point(target_point), wells[1])

        target_point = (0, -1)
        self.assertEqual(self.well_set.first_to_point(target_point), wells[0])

        target_point = (0, 1)
        self.assertEqual(self.well_set.first_to_point(target_point), wells[0])

        target_point = (1, 2)
        self.assertEqual(self.well_set.first_to_point(target_point), wells[1])

    def test_has_zero_overlap_points(self):
        wells = [Well(1, 1), Well(2, 2), Well(3, 3), Well(4, 4)]
        self.well_set.add_all_wells(wells)

        depth = 10
        self.assertEqual(
            self.well_set.num_overlap_points_at_it(0, 5, 5, depth), 0
        )

        self.well_set.remove_all()
        wells = [Well(1, 0), Well(4, 0)]
        self.well_set.add_all_wells(wells)

        self.assertEqual(
            self.well_set.num_overlap_points_at_it(0, 5, 5, depth), 0
        )

    def test_has_extremity_overlap_points(self):
        wells = [Well(0, 0), Well(2, 2), Well(4, 4)]
        self.well_set.add_all_wells(wells)

        depth = 10
        expected_overlap = 2 * depth
        self.assertEqual(
            self.well_set.num_overlap_points_at_it(1, 10, 10, depth),
            expected_overlap,
        )

    def test_num_overlap_points(self):
        wells = [Well(1, 1), Well(4, 4)]
        self.well_set.add_all_wells(wells)

        depth = 10
        expected_overlap = 4 * depth
        it = 2
        self.assertEqual(
            self.well_set.num_overlap_points_at_it(it, 10, 10, depth),
            expected_overlap,
        )

    def test_num_overlap_points_considering_wells_coords(self):
        wells = [Well(1, 1), Well(3, 3)]
        self.well_set.add_all_wells(wells)

        depth = 10
        expected_overlap = 7 * depth
        it = 2
        self.assertEqual(
            self.well_set.num_overlap_points_at_it(it, 10, 10, depth),
            expected_overlap,
        )

    def test_num_predicted_points_one_well(self):
        self.well_set.add_well_at(1, 1)

        depth = 10
        expected_predicted = depth * 8
        it = 1
        max_x = 2
        max_y = 2
        self.assertEqual(
            self.well_set.num_predicted_points_at_it(it, max_x, max_y, depth),
            expected_predicted,
        )

    def test_num_predicted_points_two_non_overlapping_wells(self):
        self.well_set.add_well_at(1, 1)
        self.well_set.add_well_at(4, 4)

        depth = 10
        expected_predicted = depth * 8 * 2
        it = 1
        max_x = 5
        max_y = 5
        self.assertEqual(
            self.well_set.num_predicted_points_at_it(it, max_x, max_y, depth),
            expected_predicted,
        )

    def test_num_predicted_points_two_overlapping_wells(self):
        self.well_set.add_well_at(1, 1)
        self.well_set.add_well_at(3, 3)

        depth = 10
        expected_predicted = depth * ((8 * 2) - 1)
        it = 1
        max_x = 4
        max_y = 4
        self.assertEqual(
            self.well_set.num_predicted_points_at_it(it, max_x, max_y, depth),
            expected_predicted,
        )

    def test_num_predicted_points_three_non_overlapping_wells(self):
        self.well_set.add_well_at(1, 1)
        self.well_set.add_well_at(4, 4)
        self.well_set.add_well_at(7, 7)

        depth = 10
        expected_predicted = depth * (8 * 3)
        it = 1
        max_x = 8
        max_y = 8
        self.assertEqual(
            self.well_set.num_predicted_points_at_it(it, max_x, max_y, depth),
            expected_predicted,
        )

    def test_num_predicted_points_three_wells_with_two_overlapping(self):
        self.well_set.add_well_at(1, 1)
        self.well_set.add_well_at(4, 4)
        self.well_set.add_well_at(6, 6)

        depth = 10
        expected_predicted = depth * ((8 * 3) - 1)
        it = 1
        max_x = 8
        max_y = 8
        self.assertEqual(
            self.well_set.num_predicted_points_at_it(it, max_x, max_y, depth),
            expected_predicted,
        )

    def test_num_predicted_points_three_overlapping_wells(self):
        self.well_set.add_well_at(1, 1)
        self.well_set.add_well_at(1, 3)
        self.well_set.add_well_at(3, 2)

        depth = 10
        expected_predicted = depth * 18
        it = 1
        max_x = 8
        max_y = 8
        self.assertEqual(
            self.well_set.num_predicted_points_at_it(it, max_x, max_y, depth),
            expected_predicted,
        )


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

        self.assertRaises(
            TypeError, ExplorationCube, wrong_top_left, bottom_right, depth
        )

        wrong_bottom_right = (1, ["a", "b"])
        top_left = (0, 0)
        self.assertRaises(
            TypeError, ExplorationCube, top_left, wrong_bottom_right
        )

    def test_raise_when_init_with_negative_points(self):
        wrong_top_left = (0, -1)
        bottom_right = (1, 1)
        depth = 1

        self.assertRaises(
            ValueError, ExplorationCube, wrong_top_left, bottom_right, depth
        )

        wrong_bottom_right = (-3, -4)
        top_left = (0, 0)
        self.assertRaises(
            ValueError, ExplorationCube, top_left, wrong_bottom_right, depth
        )

    def test_raise_when_init_with_top_left_after_bottom_right(self):
        top_left = (0, 1)
        bottom_right = (1, 0)
        depth = 1

        self.assertRaises(
            ValueError, ExplorationCube, top_left, bottom_right, depth
        )

    def test_raise_when_init_top_left_same_x_bottom_right(self):
        top_left = (1, 1)
        bottom_right = (1, 2)
        depth = 1

        self.assertRaises(
            ValueError, ExplorationCube, top_left, bottom_right, depth
        )

    def test_raise_when_init_top_left_same_y_bottom_right(self):
        top_left = (0, 1)
        bottom_right = (1, 1)
        depth = 1

        self.assertRaises(
            ValueError, ExplorationCube, top_left, bottom_right, depth
        )

    def test_raise_when_init_top_left_same_as_bottom_right(self):
        top_left = (1, 1)
        bottom_right = (1, 1)
        depth = 1

        self.assertRaises(
            ValueError, ExplorationCube, top_left, bottom_right, depth
        )

    def test_raise_when_init_non_positive_depth(self):
        top_left = (0, 0)
        bottom_right = (1, 1)
        depth = -1

        self.assertRaises(
            ValueError, ExplorationCube, top_left, bottom_right, depth
        )

        depth = 0

        self.assertRaises(
            ValueError, ExplorationCube, top_left, bottom_right, depth
        )

    def test_raise_when_init_non_int_depth(self):
        top_left = (0, 0)
        bottom_right = (1, 1)
        depth = True

        self.assertRaises(
            TypeError, ExplorationCube, top_left, bottom_right, depth
        )

        depth = "a"

        self.assertRaises(
            TypeError, ExplorationCube, top_left, bottom_right, depth
        )

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
        depth = 1

        cube1 = ExplorationCube(top_left, bottom_right, depth)
        cube2 = ExplorationCube(top_left, bottom_right, depth)

        self.assertEqual(cube1, cube2)

    def test_exp_cube_inequality(self):
        top_left = (0, 0)
        bottom_right_1 = (1, 1)
        bottom_right_2 = (2, 2)
        depth = 1

        cube1 = ExplorationCube(top_left, bottom_right_1, depth)
        cube2 = ExplorationCube(top_left, bottom_right_2, depth)

        self.assertNotEqual(cube1, cube2)

    def test_can_add_wells(self):
        top_left = (0, 0)
        bottom_right = (5, 5)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)
        wells = [Well(1, 1), Well(3, 3)]

        [cube.add_well(well) for well in wells]
        self.assertListEqual(
            [True] * len(wells), [cube.has_well(well) for well in wells]
        )

    def test_can_remove_all_wells(self):
        top_left = (0, 0)
        bottom_right = (5, 5)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)
        wells = [Well(1, 1), Well(3, 3)]

        [cube.add_well(well) for well in wells]
        cube.remove_all_wells()

        self.assertListEqual(
            [False] * len(wells), [cube.has_well(well) for well in wells]
        )

    def test_can_remove_well(self):
        top_left = (0, 0)
        bottom_right = (5, 5)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)
        wells = [Well(1, 1), Well(3, 3)]

        [cube.add_well(well) for well in wells]
        cube.remove_well(wells[0])

        self.assertListEqual(
            [False, True], [cube.has_well(well) for well in wells]
        )

        cube.remove_well_at(wells[1].coords[0], wells[1].coords[1])

        self.assertListEqual(
            [False] * len(wells), [cube.has_well(well) for well in wells]
        )

    def test_raise_if_well_isnt_inside_cube(self):
        top_left = (0, 0)
        bottom_right = (5, 5)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)
        with self.assertRaises(ValueError):
            cube.add_well_at(6, 6)

    def test_raise_cant_set_wellset(self):
        top_left = (0, 0)
        bottom_right = (5, 5)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)
        with self.assertRaises(AttributeError):
            well_set = WellSet()
            cube.wells = well_set

    def test_raise_add_non_well(self):
        top_left = (0, 0)
        bottom_right = (5, 5)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)
        with self.assertRaises(TypeError):
            well = 1
            cube.add_well(well)

    def test_raise_add_non_int_coords_well(self):
        top_left = (0, 0)
        bottom_right = (5, 5)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)
        with self.assertRaises(TypeError):
            x = 1.9
            y = 2.0
            cube.add_well_at(x, y)

        with self.assertRaises(TypeError):
            x = 1
            y = 2.5
            cube.add_well_at(x, y)

    def test_base_num_its_until_n_points_predicted(self):
        top_left = (0, 0)
        bottom_right = (5, 5)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)

        wells = [Well(1, 1), Well(3, 3)]

        [cube.add_well(well) for well in wells]
        num_points_to_predict = 0
        expected_num_its = 0
        result = cube.its_to_predict_n(num_points_to_predict)
        self.assertEqual(expected_num_its, result)

    def test_num_its_until_n_points_predicted(self):
        top_left = (0, 0)
        bottom_right = (5, 5)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)

        wells = [Well(1, 1), Well(3, 3)]

        [cube.add_well(well) for well in wells]

        num_points_to_predict = 8 * 15
        expected_num_its = 1

        result = cube.its_to_predict_n(num_points_to_predict)
        self.assertEqual(expected_num_its, result)

    def test_raise_if_pass_max_points_num_its_to_n_points_predicted(self):
        top_left = (0, 0)
        bottom_right = (5, 5)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)

        wells = [Well(1, 1), Well(3, 3)]

        [cube.add_well(well) for well in wells]

        num_points_to_predict = 1000000000
        with self.assertRaises(ValueError):
            self.assertEqual(0, cube.its_to_predict_n(num_points_to_predict))

    def test_raise_if_negative_num_points_num_its_to_n_points_predicted(self):
        top_left = (0, 0)
        bottom_right = (5, 5)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)

        wells = [Well(1, 1), Well(3, 3)]

        [cube.add_well(well) for well in wells]

        num_points_to_predict = -1
        with self.assertRaises(ValueError):
            self.assertEqual(0, cube.its_to_predict_n(num_points_to_predict))

    def test_raise_if_cube_doesnt_have_wells_num_its_to_n_points_predicted(
        self,
    ):
        top_left = (0, 0)
        bottom_right = (5, 5)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)

        num_points_to_predict = -1
        with self.assertRaises(ValueError):
            self.assertEqual(0, cube.its_to_predict_n(num_points_to_predict))

    def test_num_its_to_complete_cube_with_2_wells(self):
        top_left = (0, 0)
        bottom_right = (5, 5)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)

        wells = [Well(1, 1), Well(3, 3)]

        [cube.add_well(well) for well in wells]

        expected_num_its = 3
        result = cube.its_to_predict_complete()
        self.assertEqual(expected_num_its, result)

    def test_num_its_to_complete_cube_with_1_well(self):
        top_left = (0, 0)
        bottom_right = (4, 4)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)

        well = Well(2, 2)
        cube.add_well(well)

        expected_num_its = 2
        result = cube.its_to_predict_complete()
        self.assertEqual(expected_num_its, result)

    def test_raise_if_cube_doesnt_have_wells_num_its_to_complete_cube(self):
        top_left = (0, 0)
        bottom_right = (5, 5)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)
        expected_num_its = 2

        with self.assertRaises(ValueError):
            self.assertEqual(expected_num_its, cube.its_to_predict_complete())

    def test_num_pred_at_it_one_well(self):
        top_left = (0, 0)
        bottom_right = (4, 4)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)

        well = Well(2, 2)
        cube.add_well(well)

        it = 1
        expected_result = 8 * depth
        self.assertEqual(cube.num_predicted_at_it(it), expected_result)

        it = 2
        expected_result = 24 * depth
        self.assertEqual(cube.num_predicted_at_it(it), expected_result)

    def test_num_pred_at_it_two_well(self):
        top_left = (0, 0)
        bottom_right = (4, 4)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)

        wells = [Well(2, 2), Well(4, 4)]
        [cube.add_well(well) for well in wells]

        it = 1
        expected_result = 10 * depth
        self.assertEqual(cube.num_predicted_at_it(it), expected_result)

        it = 2
        expected_result = 23 * depth
        self.assertEqual(cube.num_predicted_at_it(it), expected_result)

    def test_raise_when_it_not_positive_num_pred_at_it(self):
        top_left = (0, 0)
        bottom_right = (4, 4)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)

        well = Well(2, 2)
        cube.add_well(well)

        it = 0
        expected_result = 0
        with self.assertRaises(ValueError):
            self.assertEqual(cube.num_predicted_at_it(it), expected_result)

    def test_raise_when_it_not_positive_num_overlap_at_it(self):
        top_left = (0, 0)
        bottom_right = (4, 4)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)

        well = Well(2, 2)
        cube.add_well(well)

        it = 0
        expected_result = 0
        with self.assertRaises(ValueError):
            self.assertEqual(cube.num_overlaps_at_it(it), expected_result)

    def test_num_overlap_at_it_one_well(self):
        top_left = (0, 0)
        bottom_right = (4, 4)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)

        well = Well(2, 2)
        cube.add_well(well)

        it = 1
        expected_result = 0
        self.assertEqual(cube.num_overlaps_at_it(it), expected_result)

        it = 2
        expected_result = 0
        self.assertEqual(cube.num_overlaps_at_it(it), expected_result)

    def test_num_overlap_at_it_two_well(self):
        top_left = (0, 0)
        bottom_right = (4, 4)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)

        wells = [Well(2, 2), Well(4, 4)]
        [cube.add_well(well) for well in wells]

        it = 1
        expected_result = 1 * depth
        self.assertEqual(cube.num_overlaps_at_it(it), expected_result)

        it = 2
        expected_result = 7 * depth
        self.assertEqual(cube.num_overlaps_at_it(it), expected_result)

    def test_can_get_well_influence_at_point_at_it(self):
        top_left = (0, 0)
        bottom_right = (5, 5)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)

        well = Well(1, 1)
        cube.add_well(well)

        it = 3
        target_point = (1, 4)

        self.assertListEqual(
            cube.wells_influencing_point_at_it(target_point, it), [(1, 1)]
        )

    def test_get_no_wells_influencing_point_at_it(self):
        top_left = (0, 0)
        bottom_right = (5, 5)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)

        well = Well(1, 1)
        cube.add_well(well)

        it = 1
        target_point = (1, 4)

        self.assertListEqual(
            cube.wells_influencing_point_at_it(target_point, it), []
        )

    def test_get_many_wells_influencing_point_at_minimun_it_to_reach(self):
        top_left = (0, 0)
        bottom_right = (5, 5)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)

        wells = [Well(1, 1), Well(5, 3)]
        for well in wells:
            cube.add_well(well)

        it = 4
        target_point = (1, 3)
        expected_result = [(1, 1), (5, 3)]
        self.assertListEqual(
            cube.wells_influencing_point_at_it(target_point, it),
            expected_result,
        )

    def test_get_many_wells_influencing_point_pass_minimun_it_to_reach(self):
        top_left = (0, 0)
        bottom_right = (5, 5)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)

        wells = [Well(1, 1), Well(5, 3)]
        for well in wells:
            cube.add_well(well)

        it = 10
        target_point = (1, 3)
        expected_result = [(1, 1), (5, 3)]
        self.assertListEqual(
            cube.wells_influencing_point_at_it(target_point, it),
            expected_result,
        )

    def test_get_only_wells_influencing_point_at_it(self):
        top_left = (0, 0)
        bottom_right = (5, 5)
        depth = 10

        cube = ExplorationCube(top_left, bottom_right, depth)

        wells = [Well(1, 1), Well(5, 3)]
        for well in wells:
            cube.add_well(well)

        it = 2
        target_point = (4, 3)
        expected_result = [(5, 3)]
        self.assertListEqual(
            cube.wells_influencing_point_at_it(target_point, it),
            expected_result,
        )


if __name__ == "__main__":
    main()
