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

    def its_to_point(self, target_point:tuple) -> int:
        if not type(target_point) == tuple:
            raise TypeError("target_point should be a tuple!")
        
        if not len(target_point) >= 2:
            raise ValueError("target_point should have lenght of at least 2")
        
        return max(abs(target_point[0] - self._coords.x), abs(target_point[1] - self._coords.y))

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