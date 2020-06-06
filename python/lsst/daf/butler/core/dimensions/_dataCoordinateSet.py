# This file is part of daf_butler.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (http://www.lsst.org).
# See the COPYRIGHT file at the top-level directory of this distribution
# for details of code ownership.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import annotations

__all__ = (
    "DataCoordinateSet",
)

from typing import (
    AbstractSet,
    Any,
    Iterator,
    Type,
    TypeVar,
)

from .coordinate import DataCoordinate
from .graph import DimensionGraph
from ._dataCoordinateIterable import DataCoordinateIterable


D = TypeVar("D", bound=DataCoordinate)


class DataCoordinateSet(DataCoordinateIterable[D]):
    """A `DataCoordinateIterable` implementation that adds some set-like
    functionality, and is backed by a true set-like object.

    Parameters
    ----------
    dataIds : `collections.abc.Set` [ `DataCoordinate` ]
         A set of instances of ``dtype``, with dimensions equal to
        ``graph``.  No runtime checking is performed, to allow
        `DataCoordinateSet` to be used as a lightweight view of a true
        `set`, dictionary keys, or other object implementing the set
        interface.
    graph : `DimensionGraph`
        Dimensions identified by all data IDs in the set.
    dtype : `type` (`DataCoordinate` or subclass thereof), optional
        A `DataCoordinate` subclass that all elements' classes must inherit
        from.  Calling code that uses static typing are responsible for
        ensuring this is equal to the ``D`` generic type parameter, as
        there is no good way for runtime code to inspect that.

    Notes
    -----
    `DataCoordinateSet` does not formally implement the `collections.abc.Set`
    interface, because that requires many binary operations to accept any
    set-like object as the other argument (regardless of what its elements
    might be), and it's much easier to ensure those operations never behave
    surprisingly if we restrict them to `DataCoordinateSet` or (sometimes)
    `DataCoordinateIterable`, and in most cases restrict that they identify
    the same dimensions.  In particular:

    - a `DataCoordinateSet` will compare as not equal to any object that is
    not a `DataCoordinateSet`, even native Python sets containing the exact
    same elements;

    - subset/superset comparison _operators_ (``<``, ``>``, ``<=``, ``>=``)
    require both operands to be `DataCoordinateSet` instances that have the
    same dimensions (i.e. ``graph`` attribute);

    - `issubset`, `issuperset`, and `isdisjoint` require the other argument to
    be a `DataCoordinateIterable` with the same dimensions;

    - operators that create new sets (``&``, ``|``, ``^``, ``-``) require both
    operands to be `DataCoordinateSet` instances that have the same dimensions
    _and_ the same ``dtype``;

    - named methods that create new sets (`intersection`, `union`,
    `symmetric_difference`, `difference`) require the other operand to be a
    `DataCoordinateIterable` with the same dimensions _and_ the same ``dtype``.

    """
    def __init__(self, dataIds: AbstractSet[D], graph: DimensionGraph, *,
                 dtype: Type[DataCoordinate] = DataCoordinate, check: bool = False):
        self._graph = graph
        self._dtype = dtype
        self._native = dataIds
        if check:
            for dataId in self._native:
                if not isinstance(dataId, self._dtype):
                    raise TypeError(f"Bad DataCoordinate subclass instance '{type(dataId).__name__}'; "
                                    f"{self._dtype.__name__} required in this context.")
                if dataId.graph != self._graph:
                    raise ValueError(f"Bad dimensions {dataId.graph}; expected {self._graph}.")

    __slots__ = ("_graph", "_dtype", "_native")

    def __iter__(self) -> Iterator[D]:
        return iter(self._native)

    def __len__(self) -> int:
        return len(self._native)

    def __contains__(self, key: Any) -> bool:
        key = DataCoordinate.standardize(key, graph=self.graph)
        return key in self._native

    def __str__(self) -> str:
        return str(self._native)

    def __repr__(self) -> str:
        return f"DataCoordinateSet({set(self._native)}, {self._graph!r}, dtype={self._dtype})"

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, DataCoordinateSet):
            return (
                self._graph == other._graph
                and self._native == other._native
            )
        return False

    def __le__(self, other: DataCoordinateSet[DataCoordinate]) -> bool:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set comparision: {self.graph} != {other.graph}.")
        return self._native <= other._native

    def __ge__(self, other: DataCoordinateSet[DataCoordinate]) -> bool:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set comparision: {self.graph} != {other.graph}.")
        return self._native >= other._native

    def __lt__(self, other: DataCoordinateSet[DataCoordinate]) -> bool:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set comparision: {self.graph} != {other.graph}.")
        return self._native < other._native

    def __gt__(self, other: DataCoordinateSet[DataCoordinate]) -> bool:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set comparision: {self.graph} != {other.graph}.")
        return self._native > other._native

    def issubset(self, other: DataCoordinateIterable[DataCoordinate]) -> bool:
        """Test whether ``self`` contains all data IDs in ``other``.

        Parameters
        ----------
        other : `DataCoordinateIterable`
            An iterable of data IDs with ``other.graph == self.graph``.

        Returns
        -------
        issubset : `bool`
            `True` if all data IDs in ``self`` are also in ``other``, and
            `False` otherwise.
        """
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set comparision: {self.graph} != {other.graph}.")
        return self._native <= other.toSet()._native

    def issuperset(self, other: DataCoordinateIterable[DataCoordinate]) -> bool:
        """Test whether ``other`` contains all data IDs in ``self``.

        Parameters
        ----------
        other : `DataCoordinateIterable`
            An iterable of data IDs with ``other.graph == self.graph``.

        Returns
        -------
        issuperset : `bool`
            `True` if all data IDs in ``other`` are also in ``self``, and
            `False` otherwise.
        """
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set comparision: {self.graph} != {other.graph}.")
        return self._native >= other.toSet()._native

    def isdisjoint(self, other: DataCoordinateIterable[DataCoordinate]) -> bool:
        """Test whether there are no data IDs in both ``self`` and ``other``.

        Parameters
        ----------
        other : `DataCoordinateIterable`
            An iterable of data IDs with ``other.graph == self.graph``.

        Returns
        -------
        isdisjoint : `bool`
            `True` if there are no data IDs in both ``self`` and ``other``, and
            `False` otherwise.
        """
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set comparision: {self.graph} != {other.graph}.")
        return self._native.isdisjoint(other.toSet()._native)

    def __and__(self, other: DataCoordinateSet[D]) -> DataCoordinateSet[D]:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set operation: {self.graph} != {other.graph}.")
        if self.dtype != other.dtype:
            raise ValueError(f"Inconsistent item types in set operation: {self.dtype} != {other.dtype}.")
        return DataCoordinateSet(self._native & other._native, self.graph, dtype=self.dtype)

    def __or__(self, other: DataCoordinateSet[D]) -> DataCoordinateSet[D]:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set operation: {self.graph} != {other.graph}.")
        if self.dtype != other.dtype:
            raise ValueError(f"Inconsistent item types in set operation: {self.dtype} != {other.dtype}.")
        return DataCoordinateSet(self._native | other._native, self.graph, dtype=self.dtype)

    def __xor__(self, other: DataCoordinateSet[D]) -> DataCoordinateSet[D]:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set operation: {self.graph} != {other.graph}.")
        if self.dtype != other.dtype:
            raise ValueError(f"Inconsistent item types in set operation: {self.dtype} != {other.dtype}.")
        return DataCoordinateSet(self._native ^ other._native, self.graph, dtype=self.dtype)

    def __sub__(self, other: DataCoordinateSet[D]) -> DataCoordinateSet[D]:
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set operation: {self.graph} != {other.graph}.")
        if self.dtype != other.dtype:
            raise ValueError(f"Inconsistent item types in set operation: {self.dtype} != {other.dtype}.")
        return DataCoordinateSet(self._native - other._native, self.graph, dtype=self.dtype)

    def intersection(self, other: DataCoordinateIterable[D]) -> DataCoordinateSet[D]:
        """Return a new set that contains all data IDs in both ``self`` and
        ``other``.

        Parameters
        ----------
        other : `DataCoordinateIterable`
            An iterable of data IDs with ``other.graph == self.graph`` and
            ``other.dtype == self.dtype``.

        Returns
        -------
        intersection : `DataCoordinateSet`
            A new `DataCoordinateSet` instance.
        """
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set operation: {self.graph} != {other.graph}.")
        if self.dtype != other.dtype:
            raise ValueError(f"Inconsistent item types in set operation: {self.dtype} != {other.dtype}.")
        return DataCoordinateSet(self._native & other.toSet()._native, self.graph, dtype=self.dtype)

    def union(self, other: DataCoordinateIterable[D]) -> DataCoordinateSet[D]:
        """Return a new set that contains all data IDs in either ``self`` or
        ``other``.

        Parameters
        ----------
        other : `DataCoordinateIterable`
            An iterable of data IDs with ``other.graph == self.graph`` and
            ``other.dtype == self.dtype``.

        Returns
        -------
        intersection : `DataCoordinateSet`
            A new `DataCoordinateSet` instance.
        """
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set operation: {self.graph} != {other.graph}.")
        if self.dtype != other.dtype:
            raise ValueError(f"Inconsistent item types in set operation: {self.dtype} != {other.dtype}.")
        return DataCoordinateSet(self._native | other.toSet()._native, self.graph, dtype=self.dtype)

    def symmetric_difference(self, other: DataCoordinateIterable[D]) -> DataCoordinateSet[D]:
        """Return a new set that contains all data IDs in either ``self`` or
        ``other``, but not both.

        Parameters
        ----------
        other : `DataCoordinateIterable`
            An iterable of data IDs with ``other.graph == self.graph`` and
            ``other.dtype == self.dtype``.

        Returns
        -------
        intersection : `DataCoordinateSet`
            A new `DataCoordinateSet` instance.
        """
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set operation: {self.graph} != {other.graph}.")
        if self.dtype != other.dtype:
            raise ValueError(f"Inconsistent item types in set operation: {self.dtype} != {other.dtype}.")
        return DataCoordinateSet(self._native ^ other.toSet()._native, self.graph, dtype=self.dtype)

    def difference(self, other: DataCoordinateIterable[D]) -> DataCoordinateSet[D]:
        """Return a new set that contains all data IDs in ``self`` that are not
        in ``other``.

        Parameters
        ----------
        other : `DataCoordinateIterable`
            An iterable of data IDs with ``other.graph == self.graph`` and
            ``other.dtype == self.dtype``.

        Returns
        -------
        intersection : `DataCoordinateSet`
            A new `DataCoordinateSet` instance.
        """
        if self.graph != other.graph:
            raise ValueError(f"Inconsistent dimensions in set operation: {self.graph} != {other.graph}.")
        if self.dtype != other.dtype:
            raise ValueError(f"Inconsistent item types in set operation: {self.dtype} != {other.dtype}.")
        return DataCoordinateSet(self._native - other.toSet()._native, self.graph, dtype=self.dtype)

    def toSet(self) -> DataCoordinateSet[D]:
        # Docstring inherited from DataCoordinateIterable.
        return self

    @property
    def graph(self) -> DimensionGraph:
        # Docstring inherited from DataCoordinateIterable.
        return self._graph

    @property
    def dtype(self) -> Type[DataCoordinate]:
        # Docstring inherited from DataCoordinateIterable.
        return self._dtype

    def subset(self, graph: DimensionGraph) -> DataCoordinateSet[D]:
        # Docstring inherited from DataCoordinateIterable.
        if not graph.issubset(self.graph):
            raise ValueError(f"{graph} is not a subset of {self.graph}")
        if graph == self.graph:
            return self
        # See comment in _ScalarDataCoordinateIterable on type: ignore there.
        return DataCoordinateSet(
            {dataId.subset(graph) for dataId in self._native},  # type: ignore
            graph,
            dtype=self.dtype
        )
