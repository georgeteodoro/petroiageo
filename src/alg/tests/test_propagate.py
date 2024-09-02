import numpy as np

from common import RealValues
from propagate import _get_coords_to_propagate
from unittest import TestCase, main


class TestPropagate(TestCase):

    def setUp(self):
        # Create full cube with only 4 dimensions and 2 wells
        # - - - - - 0
        # - 0 - - - 1
        # - - - - - 2
        # - - - - - 3
        # - - - 1 - 4
        # 0 1 2 3 4
        cur_data_type = [('x', np.int64), ('y', np.int64), ('z', np.int64),
                         ('real', np.int64)]
        cur_data_type = np.dtype(cur_data_type)
        x_size = 5
        y_size = 5
        z_size = 5
        #Each point in the cube has 4 dimensions
        self.data = np.empty((x_size, y_size, z_size), dtype=cur_data_type)
        # Fill cube
        for x in range(x_size):
            all_yz = []
            for y in range(y_size):
                all_z = []
                for z in range(z_size):
                    all_z = all_z + [(x, y, z, RealValues.empty)]
                all_yz = all_yz + [all_z]

            self.data[x, ...] = all_yz

        # Mark wells points
        self.wells_coords = [(1, 1), (4, 3)]
        for (x, y) in self.wells_coords:
            all_z = [() for _ in range(z_size)]
            for z in range(z_size):
                all_z[z] = (x, y, z, RealValues.real)
            self.data[x, y, ...] = all_z

    def test_can_get_coords_to_propagate_ring_0_full_z(self):
        ring_to_expand = 0
        for well_x, well_y in self.wells_coords:
            target_coords = _get_coords_to_propagate(ring_to_expand, self.data,
                                                     well_x, well_y)
            self.assertTrue(len(target_coords[0]) == 0)

    def test_can_get_coords_to_propagate_ring_0_not_full_z(self):
        ring_to_expand = 0
        for well_x, well_y in self.wells_coords:
            #Mark first depth of well as not real
            self.data[well_x, well_y, 0]['real'] = RealValues.empty
            target_coords = _get_coords_to_propagate(ring_to_expand, self.data,
                                                     well_x, well_y)
            #Should return the first depth of well as a target
            expected_target_x = np.array([well_x])
            expected_target_y = np.array([well_y])
            expected_target_z = np.array([0])
            self.assertTrue(np.equal(target_coords[0], expected_target_x))
            self.assertTrue(np.equal(target_coords[1], expected_target_y))
            self.assertTrue(np.equal(target_coords[2], expected_target_z))

    def test_can_get_coords_to_propagate_ring_1(self):
        ring_to_expand = 1
        well_to_expected_target_coords = {
            0: [
                [0] * 15 + [1] * 10 + [2] * 15,  # First dim
                [0] * 5 + [1] * 5 + [2] * 5 + [0] * 5 + [2] * 5 + [0] * 5 +
                [1] * 5 + [2] * 5,  # Second dim
                [0, 1, 2, 3, 4] * 8
            ],  # Third dim
            1: [[3] * 15 + [4] * 10, # First dim
                [2] * 5 + [3] * 5 + [4] * 5 + [2] * 5 + [4] * 5, # Secong dim
                [0, 1, 2, 3, 4] * 5] # Third dim
        }
        for well_id, (well_x, well_y) in enumerate(self.wells_coords):
            #Mark first depth of well as not real
            self.data[well_x, well_y, 0]['real'] = RealValues.empty
            target_coords = _get_coords_to_propagate(ring_to_expand, self.data,
                                                     well_x, well_y)
            #Should return the first depth of well as a target
            self.assertTrue(
                np.array_equal(
                    target_coords[0],
                    np.array(well_to_expected_target_coords[well_id][0])))
            self.assertTrue(
                np.array_equal(
                    target_coords[1],
                    np.array(well_to_expected_target_coords[well_id][1])))
            self.assertTrue(
                np.array_equal(
                    target_coords[2],
                    np.array(well_to_expected_target_coords[well_id][2])))


if __name__ == "__main__":
    main()
