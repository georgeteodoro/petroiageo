from contextlib import suppress
from typing import Iterable


class NonNegativeIntegerSingleCoordinate():
    """
    This is a single coordinate that can't be a negative number.
    For exemple, a 2-dimensional space would have two NonNegativeIntegerSingleCoordinates as a point, one per
    dimension
    """
    def __init__(self, coord:int, dim_name:str = None):
        self._dim_name = dim_name
        self.value = coord

    @property
    def value(self) -> int:
        """
        The value of this coordinate
        """
        return self._value
    
    @value.setter
    def value(self, new_value:int):
        self._raise_if_invalid(new_value)
        self._value = new_value
    
    @property
    def name(self) -> str:
        """
        The dimension name for this coordinate
        """
        return self._dim_name
    
    @name.setter
    def name(self, new_name:str):
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
                raise_msg = f"{self._dim_name}  value should be a non negative integer!"
            else:
                raise_msg = f"Value should be a non negative integer!"

            raise ValueError(raise_msg)
    
    def __eq__(self, other):
        if isinstance(other, NonNegativeIntegerSingleCoordinate):
            return self.value == other.value and self.name == other.name
        
        return False
    
    def __hash__(self):
        return hash(self.value) + hash(self.name)

class NonNegativeInteger2DPoint():
    """
    This is a 2D point that can't have negative coordinates
    """
    def __init__(self, x:int, y:int):
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
    def x(self, new_x:int):
        self._x = NonNegativeIntegerSingleCoordinate(new_x)
    
    @property
    def y(self):
        """
        The y value of this point
        """
        return self._y.value
    
    @y.setter
    def y(self, new_y:int):
        self._y = NonNegativeIntegerSingleCoordinate(new_y)
    
    def as_tuple(self):
        return (self._x.value, self._y.value)
    
    def __getitem__(self, key:int) -> int:
        if key == 0:
            return self._x.value
        elif key == 1:
            return self._y.value
        else:
            raise IndexError("key should be 0 or 1")
    
    def __str__(self):
        return f"(x={self._x.value}, y={self._y.value})"

class Well():
    """
    This represents a Well
    """
    def __init__(self, x:int, y:int):
        self.coords = (x, y)
    
    @property
    def coords(self) -> tuple:
        """
        The well (x, y) coordinates
        """
        return self._coords.as_tuple()
    
    @coords.setter
    def coords(self, new_coords:tuple):
        self._coords = NonNegativeInteger2DPoint(x=new_coords[0], y=new_coords[1])
    
    def __getitem__(self, key:int) -> int:
        if key == 0:
            return self._coords.x
        elif key == 1:
            return self._coords.y
        else:
            raise IndexError("Key must be 0 or 1")
    
    def __setitem__(self, key:int, value:int):
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

    def its_to_point(self, target_point:tuple) -> int:
        if not type(target_point) == tuple:
            raise TypeError("target_point should be a tuple!")
        
        if not len(target_point) >= 2:
            raise ValueError("target_point should have lenght of at least 2")
        
        return max(abs(target_point[0] - self._coords.x), abs(target_point[1] - self._coords.y))
    
    def overlap_with_well_at_it(self, other:'Well', it:int) -> bool:
        """
        Returns if at iteration 'it' this well overlaps with 'other' well
        """
        if not isinstance(other, Well):
            raise TypeError("other_well should be a Well!")
        
        other_well_top_left, other_well_bot_right = other.extremity_points_at_it(it)
        my_well_top_left, my_well_bot_right = self.extremity_points_at_it(it)

        if (self._has_area_zero(my_well_top_left, my_well_bot_right) or 
                self._has_area_zero(other_well_top_left, other_well_bot_right)):
            return False
     
        if(self._some_is_on_left_side_of_other(my_well_top_left, my_well_bot_right, other_well_top_left, other_well_bot_right)):
            return False


        if(self._some_is_above_other(my_well_top_left, my_well_bot_right, other_well_top_left, other_well_bot_right)):
            return False
    
        return True
    
    def _has_area_zero(self, top_left:tuple, bottom_right:tuple) -> bool:
        return top_left[0] == bottom_right[0] or top_left[1] == bottom_right[1]
    
    def _some_is_on_left_side_of_other(self, first_top_left, first_bot_right, second_top_left, second_bot_right) -> bool:
        return first_top_left[0] > second_bot_right[0] or second_top_left[0] > first_bot_right[0]
    
    def _some_is_above_other(self, first_top_left, first_bot_right, second_top_left, second_bot_right) -> bool:
        
        #As the y axis begins at the top with 0 and increases the y downwards, we should use the '<' sign
        return first_bot_right[1] < second_top_left[1] or second_bot_right[1] < first_top_left[1]
    
    def overlap_volume_at_it(self, other:'Well', it:int, max_x:int, max_y:int, depth:int) -> int:
        """
        Returns the num of points overlapping for this well with other well at iteration it.
        Based on: https://www.geeksforgeeks.org/total-area-two-overlapping-rectangles/
        """
        area_well1 = self.num_predicted(max_x, max_y, depth, it)
        area_well2 = other.num_predicted(max_x, max_y, depth, it)
        
        my_top_left, my_bot_right = self.extremity_points_at_it(it)
        other_top_left, other_bot_right = other.extremity_points_at_it(it)

        x_dist = (min(my_bot_right[0], other_bot_right[0]) -
              max(my_top_left[0], other_top_left[0]) + 1)
 
        y_dist = (min(my_bot_right[1], other_bot_right[1]) -
                max(my_top_left[1], other_top_left[1]) + 1)

        intersection_volume = 0
        if x_dist > 0 and y_dist > 0:
            intersection_area = x_dist * y_dist
            
            intersection_volume = intersection_area * depth
    
        return intersection_volume
    
    def extremity_points_at_it(self, it:int) -> tuple:
        """
        Returns the top_left and bottom_right x, y coordinates of the predicted layer at iteration it
        for this well   
        """
        if not type(it) == int:
            raise TypeError("it should be an int!")
        
        if it < 0:
            raise ValueError("it should be a non negative integer!")

        top_left = (self._coords.x - it, self._coords.y - it)
        bottom_right = (self._coords.x + it, self._coords.y + it)

        return (top_left, bottom_right)
    
    def num_predicted(self, max_x:int, max_y:int, depth:int, it:int) -> int:
        if not type(max_x) == int or max_x < 1:
            raise TypeError("'max_x' should be a positive int!")
        
        if not type(max_y) == int or max_y < 1:
            raise TypeError("'max_y' should be a positive int!")
        
        if not type(it) == int or it < 0:
            raise TypeError("'it' should be a non negative int!")
        
        if not type(depth) == int or depth < 1:
            raise TypeError("'depth' should be a positive int!")
        
        top_left, bottom_right = self.extremity_points_at_it(it)
        top_left = list(top_left)
        bottom_right = list(bottom_right)

        if top_left[0] < 0:
            top_left[0] = 0
        
        if top_left[1] < 0:
            top_left[1] = 0
        
        if bottom_right[0] > max_x:
            bottom_right[0] = max_x

        if bottom_right[1] > max_y:
            bottom_right[1] = max_y
        
        x_range = bottom_right[0] - top_left[0] + 1
        y_range = bottom_right[1] - top_left[1] + 1
        
        return (x_range * y_range) * depth

class WellSet():
    """
    This represents a wells set
    """

    def __init__(self):
        self._wells = set()
    
    def add_well(self, well:Well):
        """
        Adds a Well
        """
        self._wells.add(well)
    
    def add_well_at(self, x:int, y:int):
        """
        Adds a Well with the specified coords
        """
        self.add_well(Well(x, y))
    
    def add_all_wells(self, wells:Iterable):
        """
        Adds all Well objects inside the iterable
        """
        for well in wells:
            self.add_well(well)
    
    def add_all_wells_at(self, coords:Iterable):
        """
        Adds wells for each tuple inside the coords
        """
        for coord in coords:
            self.add_well_at(coord[0], coord[1])
    
    def has_well_at(self, x:int, y:int) -> bool:
        """
        Checks if there is a Well with the specified coords
        """
        return self.has_well(Well(x, y))
    
    def has_well(self, well:Well) -> bool:
        """
        Checks if the well is in this set
        """
        return well in self._wells
    
    def remove_well_at(self, x:int, y:int):
        """
        Remove well with the specified coords from this set
        """
        self.remove_well(Well(x, y))
    
    def remove_well(self, well:Well):
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
    
    def get_well_at(self, x:int, y:int) -> Well:
        """
        Returns the Well with the specified position if it is present. Otherwise, it returns None
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
    
    def its_until_point(self, target_point:tuple) -> int:
        """
        Returns the minimun number of its until target_point is reached.
        If there is no wells present, it returns float('inf')
        """
        min_its = float("inf")
        for well in self._wells:
            min_its = min(min_its, well.its_to_point(target_point))
        
        return min_its
    
    def first_to_point(self, target_point:tuple) -> Well:
        """
        Returns the first well to get to the target_point
        """
        min_its = float("inf")
        first_well = None
        for well in self._wells:
            its_to_point = well.its_to_point(target_point)
            if its_to_point < min_its:
                first_well = well
                min_its = its_to_point
        
        return first_well

class ExplorationCube():
    """
    This represents an Exploration Cube.
    """
    def __init__(self, top_left_point:tuple, bottom_right_point:tuple, depth:int):
        self.depth = depth
        self._set_extremity_points(top_left_point, bottom_right_point)
    
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
    
    def _set_extremity_points(self, new_top_left:tuple, new_bottom_right:tuple):
        if not any([new_top_left, new_bottom_right]):
            raise TypeError("A point should not be NoneType")
        
        new_top_left = self._return_point_or_raise(new_top_left, "new_top_left")
        new_bottom_right = self._return_point_or_raise(new_bottom_right, "new_bottom_right")
        
        if self._is_top_left_before_bottom_right(new_top_left, new_bottom_right):
            self._top_left_point = new_top_left
            self._bottom_right_point = new_bottom_right
        else:
            raise ValueError("This configuration of points is invalid!")
    
    def _return_point_or_raise(self, my_tuple:tuple, name:str) -> NonNegativeInteger2DPoint:
        self._raise_if_invalid_tuple(my_tuple, name)
        return NonNegativeInteger2DPoint(my_tuple[0], my_tuple[1])
    
    def _raise_if_invalid_tuple(self, my_tuple:tuple, name:str):
        if not isinstance(my_tuple, tuple):
            raise TypeError(f"{name} should be a tuple!")
        
        if len(my_tuple) < 2:
            raise ValueError(f"{name} should have size equals 2 at least!")

    def _is_top_left_before_bottom_right(self, top_left: NonNegativeInteger2DPoint, bottom_right: NonNegativeInteger2DPoint) -> bool:
        
        if top_left.x >= bottom_right.x:
            return False

        if top_left.y >= bottom_right.y:
            return False
        
        if top_left == bottom_right:
            return False
        
        return True
    
    def __eq__(self, other):
        if isinstance(other, ExplorationCube):
            my_values = [self.top_left, self.bottom_right, self.depth]
            other_values = [other.top_left, other.bottom_right, other.depth]
            return my_values == other_values
        
        return False
    
    def __hash__(self):
        return sum(hash(value) for value in [self.top_left, self.bottom_right, self.depth])