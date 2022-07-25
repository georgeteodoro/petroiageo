class NonNegativeIntegerPoint():
    """
    This is a point that can't have negative coordinates
    """
    def __init__(self, x:int, y:int):
        self.x = x
        self.y = y
    
    def _raise_if_invalid_x_coord(self, x:int) -> None:
        self._raise_if_invalid(x, 'x')
    
    def _raise_if_invalid_y_coord(self, y:int) -> None:
        self._raise_if_invalid(y, 'y')
    
    def _raise_if_invalid(self, coord, coord_name:str):
        if not isinstance(coord, int) or type(coord) == bool:
            raise TypeError(f"{coord_name} value should be an int!")
        
        if coord < 0:
            raise ValueError(f"{coord_name} should be a non negative integer!")
    
    def __eq__(self, other):
        if isinstance(other, NonNegativeIntegerPoint):
            return self.x == other.x and self.y == other.y
        
        return False
    
    def __hash__(self):
        return hash(self.x) + hash(self.y)

    @property
    def x(self):
        """
        The x value of this point
        """
        return self._x
    
    @x.setter
    def x(self, new_x:int):
        self._raise_if_invalid_x_coord(new_x)
        self._x = new_x
    
    @property
    def y(self):
        """
        The y value of this point
        """
        return self._y
    
    @y.setter
    def y(self, new_y:int):
        self._raise_if_invalid_y_coord(new_y)
        self._y = new_y
    
    def as_tuple(self):
        return (self._x, self._y)
    
    def __getitem__(self, key:int) -> int:
        if key == 0:
            return self._x
        elif key == 1:
            return self._y
        else:
            raise IndexError("key should be 0 or 1")
    
    def __str__(self):
        return f"(x={self._x}, y={self._y})"

class Well():
    def __init__(self, x:int, y:int):
        self._coords = NonNegativeIntegerPoint(x, y)
    
    @property
    def coords(self) -> tuple:
        """
        The well (x, y) coordinates
        """
        return self._coords.as_tuple()
    
    @coords.setter
    def coords(self, new_coords:tuple):
        self._coords = NonNegativeIntegerPoint(x=new_coords[0], y=new_coords[1])
    
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


class ExplorationCube():
    def __init__(self, top_left_point:tuple, bottom_right_point:tuple):
        self._top_left_point = None
        self._bottom_right_point = None


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
       self._set_extremity_points(self.top_left, NonNegativeIntegerPoint(new_bottom_right[0], new_bottom_right[1]))
    
    def _set_extremity_points(self, new_top_left: NonNegativeIntegerPoint, new_bottom_right:NonNegativeIntegerPoint):
        if not any([new_top_left, new_bottom_right]):
            raise TypeError("A point should not be NoneType")
        
        if isinstance(new_top_left, tuple):
            new_top_left = NonNegativeIntegerPoint(new_top_left[0], new_top_left[1])
        
        if isinstance(new_bottom_right, tuple):
            new_bottom_right = NonNegativeIntegerPoint(new_bottom_right[0], new_bottom_right[1])
        
        if self._is_top_left_before_bottom_right(new_top_left, new_bottom_right):
            self._top_left_point = new_top_left
            self._bottom_right_point = new_bottom_right
        else:
            raise ValueError("This configuration of points is invalid!")

    def _is_top_left_before_bottom_right(self, top_left: NonNegativeIntegerPoint, bottom_right: NonNegativeIntegerPoint) -> bool:
        
        
        if top_left.x >= bottom_right.x:
            return False

        if top_left.y >= bottom_right.y:
            return False
        
        if top_left == bottom_right:
            return False
        
        return True