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

class NonNegativeInteger3DPoint():
    """
    This is a 3D point that can't have negative coordinates
    """
    def __init__(self, x:int, y:int, z:int):
        self.x = x
        self.y = y
        self.z = z
    
    def __eq__(self, other):
        if isinstance(other, NonNegativeInteger3DPoint):
            my_values = [self.x, self.y, self.z]
            other_values = [other.x, other.y, other.z]
            return my_values == other_values
        
        return False
    
    def __hash__(self):
        my_values = [self.x, self.y, self.z]
        return sum(hash(value) for value in my_values)

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
    
    @property
    def z(self):
        """
        The z value of this point
        """
        return self._z.value
    
    @z.setter
    def z(self, new_z:int):
        self._z = NonNegativeIntegerSingleCoordinate(new_z)
    
    def as_tuple(self):
        return (self._x.value, self._y.value, self._z.value)
    
    def __getitem__(self, key:int) -> int:
        if key == 0:
            return self._x.value
        elif key == 1:
            return self._y.value
        elif key == 2:
            return self._z.value
        else:
            raise IndexError("key should be 0, 1 or 2")
    
    def __str__(self):
        my_values = (('x', self._x.value), ('y', self._y.value), ('z', self._z.value))
        my_str = "("

        for values_pair in my_values[:-1]:
            my_str += values_pair[0] +"="+ values_pair[1]+", "
        else:
            my_str += my_values[-1][0] +"="+ my_values[-1][1]
        
        my_str += ")"
        return my_str

class Well():
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


class ExplorationCube():
    def __init__(self, top_left_point:tuple, bottom_right_point:tuple):
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
       self._set_extremity_points(self.top_left, NonNegativeInteger2DPoint(new_bottom_right[0], new_bottom_right[1]))
    
    def _set_extremity_points(self, new_top_left: NonNegativeInteger2DPoint, new_bottom_right:NonNegativeInteger2DPoint):
        if not any([new_top_left, new_bottom_right]):
            raise TypeError("A point should not be NoneType")
        
        if isinstance(new_top_left, tuple):
            new_top_left = NonNegativeInteger2DPoint(new_top_left[0], new_top_left[1])
        
        if isinstance(new_bottom_right, tuple):
            new_bottom_right = NonNegativeInteger2DPoint(new_bottom_right[0], new_bottom_right[1])
        
        if self._is_top_left_before_bottom_right(new_top_left, new_bottom_right):
            self._top_left_point = new_top_left
            self._bottom_right_point = new_bottom_right
        else:
            raise ValueError("This configuration of points is invalid!")

    def _is_top_left_before_bottom_right(self, top_left: NonNegativeInteger2DPoint, bottom_right: NonNegativeInteger2DPoint) -> bool:
        
        if top_left.x >= bottom_right.x:
            return False

        if top_left.y >= bottom_right.y:
            return False
        
        if top_left == bottom_right:
            return False
        
        return True