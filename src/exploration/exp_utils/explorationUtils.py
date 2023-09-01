from contextlib import suppress
from typing import Iterable, List
import numpy as np


class NonNegativeIntegerSingleCoordinate:
    """
    This is a single coordinate that can't be a negative number.
    For exemple, a 2-dimensional space would have two NonNegativeIntegerSingleCoordinates as a point, one per
    dimension
    """

    def __init__(self, coord: int, dim_name: str = None):
        self._dim_name = dim_name
        self.value = coord

    @property
    def value(self) -> int:
        """
        The value of this coordinate
        """
        return self._value

    @value.setter
    def value(self, new_value: int):
        self._raise_if_invalid(new_value)
        self._value = new_value

    @property
    def name(self) -> str:
        """
        The dimension name for this coordinate
        """
        return self._dim_name

    @name.setter
    def name(self, new_name: str):
        if isinstance(new_name, str):
            self._dim_name = new_name
        else:
            raise TypeError("name should be str!")

    def _raise_if_invalid(self, coord):
        """
        Raise appropriate exception if coord is a negative number
        """
        if not isinstance(coord, int) or type(coord) == bool:
            raise_msg = None
            if self._dim_name:
                raise_msg = f"{self._dim_name} value should be an int!"
            else:
                raise_msg = f"Value should be an int!"

            raise TypeError(raise_msg)

        if coord < 0:
            raise_msg = None
            if self._dim_name:
                raise_msg = (
                    f"{self._dim_name}  value should be a non negative integer!"
                )
            else:
                raise_msg = f"Value should be a non negative integer!"

            raise ValueError(raise_msg)

    def __eq__(self, other):
        if isinstance(other, NonNegativeIntegerSingleCoordinate):
            return self.value == other.value and self.name == other.name

        return False

    def __hash__(self):
        return hash(self.value) + hash(self.name)


class NonNegativeInteger2DPoint:
    """
    This is a 2D point that can't have negative coordinates
    """

    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __eq__(self, other):
        if isinstance(other, NonNegativeInteger2DPoint):
            return self.x == other.x and self.y == other.y

        return False

    def __hash__(self):
        return hash(self.x) + hash(self.y)

    @property
    def x(self):
        """
        The x value of this point
        """
        return self._x.value

    @x.setter
    def x(self, new_x: int):
        self._x = NonNegativeIntegerSingleCoordinate(new_x)

    @property
    def y(self):
        """
        The y value of this point
        """
        return self._y.value

    @y.setter
    def y(self, new_y: int):
        self._y = NonNegativeIntegerSingleCoordinate(new_y)

    def as_tuple(self):
        """
        Returns (x, y)
        """
        return (self._x.value, self._y.value)

    def __getitem__(self, key: int) -> int:
        if key == 0:
            return self._x.value
        elif key == 1:
            return self._y.value
        else:
            raise IndexError("key should be 0 or 1")

    def __str__(self):
        return f"(x={self._x.value}, y={self._y.value})"


class Well:
    """
    This represents a Well
    """

    def __init__(self, x: int, y: int):
        self.coords = (x, y)

    @property
    def coords(self) -> tuple:
        """
        The well (x, y) coordinates
        """
        return self._coords.as_tuple()

    @coords.setter
    def coords(self, new_coords: tuple):
        self._coords = NonNegativeInteger2DPoint(
            x=new_coords[0], y=new_coords[1]
        )

    def __getitem__(self, key: int) -> int:
        if key == 0:
            return self._coords.x
        elif key == 1:
            return self._coords.y
        else:
            raise IndexError("Key must be 0 or 1")

    def __setitem__(self, key: int, value: int):
        if key == 0:
            self._coords.x = value
        elif key == 1:
            self._coords.y = value
        else:
            raise IndexError("Key must be 0 or 1")

    def __eq__(self, other):
        if isinstance(other, Well):
            return self.coords == other.coords

        return False

    def __hash__(self):
        return hash(self.coords)

    def __str__(self):
        return f"Well(x={self._coords.x}, y={self._coords.y})"

    def __repr__(self):
        return f"Well(x={self._coords.x}, y={self._coords.y})"

    def its_to_point(self, target_point: tuple) -> int:
        """
        Returns the number of iterations to reach the target_point

        target_point: Tuple of (x, y)
        """
        if not type(target_point) == tuple:
            raise TypeError("target_point should be a tuple!")

        if not len(target_point) >= 2:
            raise ValueError("target_point should have lenght of at least 2")

        return max(
            abs(target_point[0] - self._coords.x),
            abs(target_point[1] - self._coords.y),
        )

    def overlap_with_well_at_it(
        self, other: "Well", it: int, max_x: int, max_y: int
    ) -> bool:
        """
        Returns if at iteration 'it' this well overlaps with 'other' well

        other: Another Well instance to compare
        it: Iteration number
        max_x: The max x value starting at 0
        max_y: The max y value starting at 0
        """
        if not isinstance(other, Well):
            raise TypeError("other_well should be a Well!")

        (
            other_well_top_left,
            other_well_bot_right,
        ) = other.extremity_points_at_it(it, max_x, max_y)
        my_well_top_left, my_well_bot_right = self.extremity_points_at_it(
            it, max_x, max_y
        )

        if self._has_area_zero(
            my_well_top_left, my_well_bot_right
        ) or self._has_area_zero(other_well_top_left, other_well_bot_right):
            return False

        if self._some_is_on_left_side_of_other(
            my_well_top_left,
            my_well_bot_right,
            other_well_top_left,
            other_well_bot_right,
        ):
            return False

        if self._some_is_above_other(
            my_well_top_left,
            my_well_bot_right,
            other_well_top_left,
            other_well_bot_right,
        ):
            return False

        return True

    def _has_area_zero(self, top_left: tuple, bottom_right: tuple) -> bool:
        return top_left[0] == bottom_right[0] or top_left[1] == bottom_right[1]

    def _some_is_on_left_side_of_other(
        self, first_top_left, first_bot_right, second_top_left, second_bot_right
    ) -> bool:
        return (
            first_top_left[0] > second_bot_right[0]
            or second_top_left[0] > first_bot_right[0]
        )

    def _some_is_above_other(
        self, first_top_left, first_bot_right, second_top_left, second_bot_right
    ) -> bool:
        # As the y axis begins at the top with 0 and increases the y downwards, we should use the '<' sign
        return (
            first_bot_right[1] < second_top_left[1]
            or second_bot_right[1] < first_top_left[1]
        )

    def overlap_volume_at_it(
        self, other: "Well", it: int, max_x: int, max_y: int, depth: int
    ) -> int:
        """
        Returns the num of points overlapping for this well with other well at iteration it.
        Based on: https://www.geeksforgeeks.org/total-area-two-overlapping-rectangles/

        it: The iteration
        max_x: The max x value starting at 0
        max_y: The max y value starting at 0
        depth: The cube depth
        """
        area_well1 = self.num_predicted(max_x, max_y, depth, it)
        area_well2 = other.num_predicted(max_x, max_y, depth, it)

        my_top_left, my_bot_right = self.extremity_points_at_it(
            it, max_x, max_y
        )

        other_top_left, other_bot_right = other.extremity_points_at_it(
            it, max_x, max_y
        )

        x_dist = (
            min(my_bot_right[0], other_bot_right[0])
            - max(my_top_left[0], other_top_left[0])
            + 1
        )

        y_dist = (
            min(my_bot_right[1], other_bot_right[1])
            - max(my_top_left[1], other_top_left[1])
            + 1
        )

        intersection_volume = 0
        if x_dist > 0 and y_dist > 0:
            intersection_area = x_dist * y_dist

            intersection_volume = intersection_area * depth

        return intersection_volume

    def extremity_points_at_it(self, it: int, max_x: int, max_y: int) -> tuple:
        """
        Returns the top_left and bottom_right x, y coordinates of the predicted layer at iteration it
        for this well
        """
        if not type(max_x) == int or max_x < 1:
            raise TypeError("'max_x' should be a positive int!")

        if not type(max_y) == int or max_y < 1:
            raise TypeError("'max_y' should be a positive int!")

        if not type(it) == int:
            raise TypeError("it should be an int!")

        if it < 0:
            raise ValueError("it should be a non negative integer!")

        top_left = (self._coords.x - it, self._coords.y - it)
        bottom_right = (self._coords.x + it, self._coords.y + it)

        top_left, bottom_right = self._resolve_extremities_with_range(
            top_left, bottom_right, max_x, max_y
        )

        return (top_left, bottom_right)

    def _resolve_extremities_with_range(
        self, top_left: tuple, bot_right: tuple, max_x: int, max_y: int
    ):
        top_left_copy = list(top_left)
        bot_right_copy = list(bot_right)

        if top_left_copy[0] < 0:
            top_left_copy[0] = 0

        if top_left_copy[1] < 0:
            top_left_copy[1] = 0

        if bot_right_copy[0] > max_x:
            bot_right_copy[0] = max_x

        if bot_right_copy[1] > max_y:
            bot_right_copy[1] = max_y

        return tuple(top_left_copy), tuple(bot_right_copy)

    def num_predicted(self, max_x: int, max_y: int, depth: int, it: int) -> int:
        """
        Returns the num of points this well predicts at some it. Dont consider the points the well is already in.

        max_x: The max x value starting at 0
        max_y: The max y value starting at 0
        depth: The cube depth
        it: Iteration number

        """
        if not type(depth) == int or depth < 1:
            raise TypeError("'depth' should be a positive int!")

        top_left, bottom_right = self.extremity_points_at_it(it, max_x, max_y)

        x_range = bottom_right[0] - top_left[0] + 1
        y_range = bottom_right[1] - top_left[1] + 1

        # Dont consider the x, y coords that the well already is
        return ((x_range * y_range) - 1) * depth


class WellSet:
    """
    This represents a wells set
    """

    def __init__(self):
        self._wells = set()

    def add_well(self, well: Well):
        """
        Adds a Well
        """
        self._wells.add(well)

    def add_well_at(self, x: int, y: int):
        """
        Adds a Well with the specified coords
        """
        self.add_well(Well(x, y))

    def add_all_wells(self, wells: Iterable):
        """
        Adds all Well objects inside the iterable

        wells: Should be an iterable of Well instances
        """
        for well in wells:
            self.add_well(well)

    def add_all_wells_at(self, coords: Iterable):
        """
        Adds wells for each tuple inside the coords

        coords: Should be an iterable of tuples. Each tuple should be of (x, y) coordinates
        """
        for coord in coords:
            self.add_well_at(coord[0], coord[1])

    def has_well_at(self, x: int, y: int) -> bool:
        """
        Checks if there is a Well with the specified coords
        """
        return self.has_well(Well(x, y))

    def has_well(self, well: Well) -> bool:
        """
        Checks if the well is in this set
        """
        return well in self._wells

    def remove_well_at(self, x: int, y: int):
        """
        Remove well with the specified coords from this set
        """
        self.remove_well(Well(x, y))

    def remove_well(self, well: Well):
        """
        Remove the well from this set
        """
        with suppress(KeyError):
            self._wells.remove(well)

    def remove_all(self):
        """
        Removes all wells from this set
        """
        self._wells.clear()

    def get_well_at(self, x: int, y: int) -> Well:
        """
        Returns the Well with the specified position if it is present. Otherwise, it returns None

        x: The x coordinate
        y: The y coordinate
        """
        target_well = None
        wells_it = iter(self._wells)
        while True:
            well = next(wells_it, None)
            if not well:
                break

            if well.coords[0] == x and well.coords[1] == y:
                target_well = well
                break

        return target_well

    def its_until_point(self, target_point: tuple) -> int:
        """
        Returns the minimun number of its until target_point is reached.
        If there are no wells present, it returns float('inf')

        target_point: Should be a (x, y) tuple
        """
        min_its = float("inf")
        for well in self._wells:
            min_its = min(min_its, well.its_to_point(target_point))

        return min_its

    def wells_in_point_at_it(self, target_point: tuple, it: int) -> List[tuple]:
        """
        Returns a list of wells that get to the target_point at iteration it. The wells are
        ordered by arrival it.
        target_point: Should be a (x, y) tuple
        it: The iteration (int) to check arrival

        returns:
        A list of tuples with the pattern: [(x, y)] where (x, y) are the well's coordinates.
        This list is in ascending order by its arrival it.
        """
        if not type(it) == int:
            raise TypeError("it should be an integer!")

        if it < 1:
            raise ValueError("it should be a positive integer!")

        wells_and_num_its_till_point = list()
        for well in self._wells:
            well_its_to_point = well.its_to_point(target_point)
            if well_its_to_point <= it:
                wells_and_num_its_till_point.append(
                    (well.coords, well_its_to_point)
                )

        sorted_list = sorted(wells_and_num_its_till_point, key=lambda t: t[1])

        only_wells_list = [t[0] for t in sorted_list]

        return only_wells_list

    def first_to_point(self, target_point: tuple) -> Well:
        """
        Returns the first well to get to the target_point

        target_point: Should be a (x, y) tuple
        """
        min_its = float("inf")
        first_well = None
        for well in self._wells:
            its_to_point = well.its_to_point(target_point)
            if its_to_point < min_its:
                first_well = well
                min_its = its_to_point

        return first_well

    def num_overlap_points_at_it(
        self, it: int, max_x: int, max_y: int, depth: int
    ) -> int:
        """
        Returns the number of overlap points predicted at some iteration considering all wells.
        Based on: https://stackoverflow.com/a/25355331/16264901

        it: The iteration
        max_x, max_y: The sizes of the area considered. If The area has a x range of 10, the max_x should be 9. Same for max_y
        depth: The cube depth. It assumes that every Well can predict along all this depth
        """
        total_area = self.get_prediction_area(it, max_x, max_y)

        area_of_intersect = np.sum(total_area > 1, dtype="int64")

        return area_of_intersect * depth

    def get_prediction_area(
        self, it: int, max_x: int, max_y: int
    ) -> np.ndarray:
        wells_extremities = list()

        max_well_x_coord = -1
        max_well_y_coord = -1

        for well in self._wells:
            top_left, bot_right = well.extremity_points_at_it(it, max_x, max_y)
            if bot_right[0] > max_well_x_coord:
                max_well_x_coord = bot_right[0]

            if bot_right[1] > max_well_y_coord:
                max_well_y_coord = bot_right[1]

            wells_extremities.append((top_left, bot_right))

        total_area = np.zeros((max_well_x_coord + 1, max_well_y_coord + 1))

        for well_extremities in wells_extremities:
            top_left, bot_right = well_extremities
            total_area[
                top_left[0] : bot_right[0] + 1, top_left[1] : bot_right[1] + 1
            ] += 1

        # Zeroes the wells coordinates as one well can't predict there
        for well in self._wells:
            x, y = well.coords
            total_area[x, y] = 0

        return total_area

    def num_predicted_points_at_it(
        self, it: int, max_x: int, max_y: int, depth: int
    ) -> int:
        """
        Returns the number of predicted points at some iteration considering all wells and overlaps.
        Based on: https://stackoverflow.com/a/25355331/16264901

        it: The iteration
        max_x, max_y: The sizes of the area considered. If The area has a x range of 10, the max_x should be 9. Same for max_y
        depth: The cube depth. It assumes that every Well can predict along all this depth
        """
        total_area = self.get_prediction_area(it, max_x, max_y)

        area_of_intersect = np.sum(total_area > 0, dtype="int64")

        return area_of_intersect * depth

    def __len__(self):
        return len(self._wells)


class ExplorationCube:
    """
    This represents an Exploration Cube.
    """

    def __init__(
        self, top_left_point: tuple, bottom_right_point: tuple, depth: int
    ):
        self.depth = depth
        self._wells = WellSet()
        self._set_extremity_points(top_left_point, bottom_right_point)

    @property
    def wells(self) -> WellSet:
        """
        The wells inside this cube
        """
        return self._wells

    @wells.setter
    def wells(self, new_well_set: WellSet):
        raise AttributeError("Can't set well_set!")

    @property
    def top_left(self) -> tuple:
        """
        The top left point of the Cube
        """
        return self._top_left_point.as_tuple()

    @top_left.setter
    def top_left(self, new_top_left: tuple):
        self._set_extremity_points(new_top_left, self.bottom_right)

    @property
    def bottom_right(self) -> tuple:
        """
        The bottom right point of the Cube
        """
        return self._bottom_right_point.as_tuple()

    @bottom_right.setter
    def bottom_right(self, new_bottom_right: tuple):
        self._set_extremity_points(self.top_left, new_bottom_right)

    @property
    def depth(self):
        """
        The depth of this cube
        """
        return self._depth

    @depth.setter
    def depth(self, new_depth):
        if not type(new_depth) == int:
            raise TypeError("depth should be an int!")

        if new_depth < 1:
            raise ValueError("depth should be a positive int!")

        self._depth = new_depth

    @property
    def x_range(self):
        """
        The x range of this cube
        """
        return self.bottom_right[0] - self.top_left[0] + 1

    @property
    def y_range(self):
        """
        The y range of this cube
        """
        return self.bottom_right[1] - self.top_left[1] + 1

    @property
    def max_x(self):
        """
        The max x coordinate this cube occupies
        """
        return self.bottom_right[0]

    @property
    def max_y(self):
        """
        The max y coordinate this cube occupies
        """
        return self.bottom_right[1]

    @property
    def min_x(self):
        """
        The min x coordinate this cube occupies
        """
        return self.top_left[0]

    @property
    def min_y(self):
        """
        The min y coordinate this cube occupies
        """
        return self.top_left[1]

    def _set_extremity_points(
        self, new_top_left: tuple, new_bottom_right: tuple
    ):
        if not any([new_top_left, new_bottom_right]):
            raise TypeError("A point should not be NoneType")

        new_top_left = self._return_point_or_raise(new_top_left, "new_top_left")
        new_bottom_right = self._return_point_or_raise(
            new_bottom_right, "new_bottom_right"
        )

        if self._is_top_left_before_bottom_right(
            new_top_left, new_bottom_right
        ):
            self._top_left_point = new_top_left
            self._bottom_right_point = new_bottom_right
        else:
            raise ValueError("This configuration of points is invalid!")

    def _return_point_or_raise(
        self, my_tuple: tuple, name: str
    ) -> NonNegativeInteger2DPoint:
        self._raise_if_invalid_tuple(my_tuple, name)
        return NonNegativeInteger2DPoint(my_tuple[0], my_tuple[1])

    def _raise_if_invalid_tuple(self, my_tuple: tuple, name: str):
        if not isinstance(my_tuple, tuple):
            raise TypeError(f"{name} should be a tuple!")

        if len(my_tuple) < 2:
            raise ValueError(f"{name} should have size equals 2 at least!")

    def _is_top_left_before_bottom_right(
        self,
        top_left: NonNegativeInteger2DPoint,
        bottom_right: NonNegativeInteger2DPoint,
    ) -> bool:
        if top_left.x >= bottom_right.x:
            return False

        if top_left.y >= bottom_right.y:
            return False

        if top_left == bottom_right:
            return False

        return True

    def add_well_at(self, x: int, y: int):
        """
        Adds a Well at the x and y positions. Raises ValueError if Well isn't inside the cube.
        """
        if not type(x) == int:
            raise TypeError("x should be an int!")

        if not type(y) == int:
            raise TypeError("y should be an int!")

        self.add_well(Well(x, y))

    def add_well(self, well: Well):
        """
        Adds a Well to this cube. Raises ValueError if Well isn't inside the cube.
        """
        if not isinstance(well, Well):
            raise TypeError("well should be a Well!")

        if self._well_is_inside_cube(well):
            self._wells.add_well(well)
        else:
            raiseMsg = "Well is not inside ExplorationCube!"
            raiseMsg += f"Well coords: {well}."
            raiseMsg += f"ExplorationCube coords: {self}."
            raise ValueError(raiseMsg)

    def _well_is_inside_cube(self, well: Well) -> bool:
        x_valid = well.coords[0] >= 0 and well.coords[0] <= self.bottom_right[0]
        y_valid = well.coords[1] >= 0 and well.coords[1] <= self.bottom_right[1]
        return x_valid and y_valid

    def remove_all_wells(self):
        """
        Removes all wells inside this cube
        """
        self._wells.remove_all()

    def remove_well(self, well: Well):
        """
        Remove a well from this cube
        """
        if not isinstance(well, Well):
            raise TypeError("well should be a Well!")

        self._wells.remove_well(well)

    def remove_well_at(self, x: int, y: int):
        """
        Remove a Well with x and y positions from this cube
        """
        if not type(x) == int:
            raise TypeError("x should be an int!")

        if not type(y) == int:
            raise TypeError("y should be an int!")

        self._wells.remove_well_at(x, y)

    def has_well(self, well: Well) -> bool:
        """
        Returns if this cube has a specific well
        """
        if not isinstance(well, Well):
            raise TypeError("well should be a Well!")

        return self._wells.has_well(well)

    def has_well_at(self, x: int, y: int) -> bool:
        """
        Returns if this cube has a specific well at x and y coordinates
        """
        return self._wells.has_well_at(x, y)

    def its_to_predict_n(self, n_points: int) -> int:
        if not type(n_points) == int:
            raise TypeError("n_points should be an int!")

        if n_points < 0 or n_points > self.max_predictable_points_possible():
            raiseMsg = "n_points should be a non negative integer and less-equal than the maximum number of "
            raiseMsg += (
                f"predictable points ({self.max_predictable_points_possible()})"
            )
            raise ValueError(raiseMsg)

        if len(self._wells) < 1:
            raise ValueError(
                f"{self.__class__.__name__} doesnt have any wells!"
            )

        if n_points == 0:
            return 0

        it_count = 0
        while True:
            n_predicted = self._wells.num_predicted_points_at_it(
                it_count, self.max_x, self.max_y, self.depth
            )
            if n_predicted < n_points:
                it_count += 10
            else:
                break

        # passou dos pontos
        while True:
            n_predicted = self._wells.num_predicted_points_at_it(
                it_count, self.max_x, self.max_y, self.depth
            )
            if n_predicted >= n_points:
                it_count -= 1
            else:
                if it_count < 0:
                    return 0

                if (
                    self._wells.num_predicted_points_at_it(
                        it_count + 1, self.max_x, self.max_y, self.depth
                    )
                    >= n_points
                ):
                    return it_count + 1
                else:
                    return it_count

    def max_predictable_points_possible(self) -> int:
        """
        Returns the total number of points inside this cube minus the amount of points that wells occupy
        """
        area_points = self.x_range * self.y_range
        return (area_points - len(self._wells)) * self.depth

    def its_to_predict_complete(self) -> int:
        return self.its_to_predict_n(self.max_predictable_points_possible())

    def num_predicted_at_it(self, it: int) -> int:
        """
        Returns the total number of points predicted until iteration it
        """
        if not type(it) == int:
            raise TypeError("it should be an integer!")

        if it < 1:
            raise ValueError("it should be a positive integer!")

        return self._wells.num_predicted_points_at_it(
            it, self.max_x, self.max_y, self.depth
        )

    def num_overlaps_at_it(self, it: int) -> int:
        if not type(it) == int:
            raise TypeError("it should be an integer!")

        if it < 1:
            raise ValueError("it should be a positive integer!")

        return self._wells.num_overlap_points_at_it(
            it, self.max_x, self.max_y, self.depth
        )

    def pred_config_at_it(self, it: int) -> np.ndarray:
        """
        Returns the num of wells that predicted some point at some it as a numpy.ndarray
        """
        if not type(it) == int:
            raise TypeError("it should be an integer!")

        if it < 1:
            raise ValueError("it should be a positive integer!")

        return self._wells.get_prediction_area(it, self.max_x, self.max_y)

    def wells_influencing_point_at_it(
        self, point: tuple, it: int
    ) -> List[tuple]:
        """
        Returns a list of wells that get to the target_point at iteration it. The wells are
        ordered by arrival it.
        target_point: Should be a (x, y) tuple
        it: The iteration (int) to check arrival

        returns:
        A list of tuples with the pattern: [(x, y)] where (x, y) are the well's coordinates.
        This list is in ascending order by its arrival it.
        """
        if not type(it) == int:
            raise TypeError("it should be an integer!")

        if it < 1:
            raise ValueError("it should be a positive integer!")

        # This checks if it is a valid point
        cube_point = NonNegativeInteger2DPoint(point[0], point[1])

        return self.wells.wells_in_point_at_it(cube_point.as_tuple(), it)

    def __eq__(self, other):
        if isinstance(other, ExplorationCube):
            my_values = [self.top_left, self.bottom_right, self.depth]
            other_values = [other.top_left, other.bottom_right, other.depth]
            return my_values == other_values

        return False

    def __hash__(self):
        return sum(
            hash(value)
            for value in [self.top_left, self.bottom_right, self.depth]
        )

    def __repr__(self):
        return f"ExplorationCube(top_left={self.top_left}, bottom_right={self.bottom_right}, depth={self.depth})"

    def __str__(self):
        return f"ExplorationCube(top_left={self.top_left}, bottom_right={self.bottom_right}, depth={self.depth})"
