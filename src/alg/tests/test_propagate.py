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
        self.data = np.empty((x_size, y_size, z_size),
                             dtype=cur_data_type)
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
        self.wells_coords = [(1, 1), (3, 4)]
        for (x, y) in self.wells_coords:
            all_z = [() for _ in range(z_size)] 
            for z in range(z_size):
                all_z[z] = (x, y, z, RealValues.real)
            self.data[x, y, ...] = all_z

    def test_can_get_coords_to_propagate_ring_0_full_z(self):
        for well_x, well_y in self.wells_coords:
            target_coords = _get_coords_to_propagate(0, (5, 5, 5), 0, self.data,
                                                     well_x,
                                                     well_y)
            self.assertTrue(len(target_coords[0]) == 0)
    
    def test_can_get_coords_to_propagate_ring_0_not_full_z(self):
        for well_x, well_y in self.wells_coords:
            #Mark first depth of well as not real
            self.data[well_x, well_y, 0]['real'] = RealValues.empty
            target_coords = _get_coords_to_propagate(0, (5, 5, 5), 0, self.data,
                                                     well_x,
                                                     well_y)
            expected_target_x = np.array([well_x])
            expected_target_y = np.array([well_y])
            expected_target_z = np.array([0])
            self.assertTrue(np.equal(target_coords[0], expected_target_x))
            self.assertTrue(np.equal(target_coords[1], expected_target_y))
            self.assertTrue(np.equal(target_coords[2], expected_target_z))


if __name__ == "__main__":
    main()
