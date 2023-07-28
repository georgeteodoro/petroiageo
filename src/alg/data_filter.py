from __future__ import annotations
import numpy as np
from typing import Callable
import numpy as np
from common import RealValues
import h5py
from hdf5_util import fold_h5_all_clusters


class DataFilter():
    """
    Custom data filter to be used on np.ndarray or h5py:Dataset data
    """

    def __init__(self):
        self._filter_list = list()
        self._min_ring = None
        self._in_wells = list()
        self._not_in_wells = list()

    def add_min_ring_filter(self, min_ring: int) -> DataFilter:
        """
        Adds the min ring filter to be used when sampling
        window is defined.
        """
        if min_ring < 0 or not isinstance(min_ring, int):
            raise ValueError("min_ring should be a non negative integer!")

        self._filter_list.append(lambda d: (d['ring'] >= min_ring))
        self._min_ring = min_ring
        return self

    def add_not_in_well_list_filter(self, well_ids_list: list) -> DataFilter:
        """
        Adds the not in well filter based on the well ids list.
        It will check that the data is not associated with the wells
        in well_id_list
        """
        if not isinstance(well_ids_list, list) and not isinstance(
                well_ids_list, tuple):
            raise TypeError("well_ids_list should be a list or a tuple!")

        self._filter_list.append(
            self._is_well_not_in_list_decorator(well_ids_list))

        self._not_in_wells.extend(well_ids_list)
        return self

    def _is_well_not_in_list_decorator(self, well_ids_list: list) -> Callable:
        """
        This is a decorator. It returns the _is_well_not_in_list
        function that uses the well_list internally.
        The returned function expects a ndarray as parameter.
        """

        def _is_well_not_in_list(d: np.ndarray) -> np.ndarray:
            ret = np.full((d.shape), True, dtype=bool)
            for x in well_ids_list:
                ret *= d['well_id'] != x
            return ret

        return _is_well_not_in_list

    def add_in_well_list_filter(self, well_ids_list: list) -> DataFilter:
        """
        Adds the in well filter based on the well ids list.
        It will check that the data is associated with the wells
        in well_id_list
        """
        if not isinstance(well_ids_list, list) and not isinstance(
                well_ids_list, tuple):
            raise TypeError("well_ids_list should be a list or a tuple!")

        self._filter_list.append(
            self._is_well_in_list_decorator(well_ids_list))

        self._in_wells.extend(well_ids_list)
        return self

    def _is_well_in_list_decorator(self, well_ids_list: list) -> Callable:
        """
        This is a decorator. It returns the _is_well_in_list
        function that uses the well_list internally.
        The returned function expects a ndarray as parameter.
        """

        def _is_well_in_list(d: np.ndarray) -> np.ndarray:
            ret = np.full((d.shape), False, dtype=bool)
            for x in well_ids_list:
                ret += d['well_id'] == x
            return ret

        return _is_well_in_list

    def satisfies(self, data: np.ndarray) -> np.ndarray:
        """
        Returns a boolean array indicating the data that
        satisfies all current filters.
        If no filters were added, return a ndarray full of True.
        """
        ret = np.full((data.shape), True, dtype=bool)
        for filter_f in self._filter_list:
            ret &= filter_f(data)

        return ret

    def filter(self, data: np.ndarray) -> np.ndarray:
        """
        Filters from data all elements that satisfies the current filters.
        Returns a ndarray with shape: (num_points_filtered,)
        """
        satisfies = self.satisfies(data)
        return data[satisfies]

    def filter_count_dset(self, dset: h5py.Dataset) -> int:
        """
        Counts how many data satisfy the current filters
        in the dset.
        """
        if not isinstance(dset, h5py.Dataset):
            raise TypeError(
                f"dset should be a h5py.Dataset but was {type(dset)}")

        # return fold_h5_all_clusters(
        #     dset,
        #     lambda a: self.satisfies(a).sum(),
        #     0,
        # )

        count = 0
        for chunk_slice in dset.iter_chunks():
            data = dset[chunk_slice]
            count += self.satisfies(data).sum()

        return count

    @property
    def in_wells(self) -> list:
        return self._in_wells.copy()

    @in_wells.setter
    def in_wells(self, new_in_wells: list) -> None:
        raise AttributeError("cant set in_wells!")

    @property
    def not_in_wells(self) -> list:
        return self._not_in_wells.copy()

    @not_in_wells.setter
    def not_in_wells(self, new_not_in_wells: list) -> None:
        raise AttributeError("cant set not_in_wells!")

    @property
    def min_ring(self) -> int:
        return self._min_ring

    @min_ring.setter
    def min_ring(self, new_min_ring) -> None:
        raise AttributeError("cant set min_ring!")

    def __str__(self) -> str:
        return f"min_ring: {self._min_ring}, in_wells: {self._in_wells}, \
            not_in_wells: {self._not_in_wells}"


class PredTrainDataFilter(DataFilter):
    """
    The final prediction train data filter. It automatically
    uses data that are RealValue.real or RealValues.propagated.
    If used along with a WellsDataFilter, should use the
    add_not_in_well_list_filter method with the well_ids_list
    of the Test Data.
    """

    def __init__(self):
        super().__init__()
        self._filter_list.append(lambda d:
                                 ((d["real"] == RealValues.real)
                                  | (d["real"] == RealValues.propagated)))


class FeatSelectionTrainDataFilter(PredTrainDataFilter):
    """
    The Feature Selection train data filter. It automatically
    uses data that are RealValue.real, RealValues.propagated
    or RealValues.canal_expanded.
    If used along with a WellsDataFilter, should use the
    add_not_in_well_list_filter method with the well_ids_list
    of the Test Data.
    """

    def __init__(self):
        super().__init__()
        self._filter_list.append(lambda d:
                                 ((d["real"] == RealValues.real)
                                  | (d["real"] == RealValues.canal_expanded)
                                  | (d["real"] == RealValues.propagated)))


class WellsDataFilter(DataFilter):
    """
    The Test data filter. As test data is defined based on the well id,
    it requires a well_ids_list and automatically adds the
    necessary filter to itself. 
    
    Had to name it WellsDataFilter instead of TestDataFilter because of 
    clashes with unittest's naming convention.
    """

    def __init__(self, well_ids_list: list):
        super().__init__()
        self.add_in_well_list_filter(well_ids_list)