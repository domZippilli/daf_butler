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
    "DataCoordinateIterable",
)

from abc import abstractmethod
from typing import (
    AbstractSet,
    Any,
    Callable,
    Generic,
    Iterable,
    Iterator,
    List,
    TYPE_CHECKING,
    Type,
    TypeVar,
)

import sqlalchemy

from ..simpleQuery import SimpleQuery
from .coordinate import DataCoordinate, MinimalDataCoordinate
from .graph import DimensionGraph

if TYPE_CHECKING:
    from ._dataCoordinateSet import DataCoordinateSet


D = TypeVar("D", bound=DataCoordinate)


class DataCoordinateIterable(Iterable[D]):
    """An abstract base class for homogeneous iterables of data IDs.

    All elements of a `DataCoordinateIterable` identify the same set of
    dimensions (given by the `graph` property) and are generally instances
    of the same `DataCoordinate` subclass (given by the `dtype` property, as
    well as the generic parameter in its type annotation).
    """

    __slots__ = ()

    @staticmethod
    def fromScalar(dataId: D) -> _ScalarDataCoordinateIterable[D]:
        """Return a `DataCoordinateIterable` containing the single data ID
        given.

        Parameters
        ----------
        dataId : `DataCoordinate`
            Data ID to adapt.  Must be a true `DataCoordinate` instance, not
            an arbitrary mapping.  No runtime checking is performed.

        Returns
        -------
        iterable : `DataCoordinateIterable`
            A `DataCoordinateIterable` instance of unspecified (i.e.
            implementation-detail) subclass.  Guaranteed to implement
            the `collections.abc.Sized` (i.e. `__len__`) and
            `collections.abc.Container` (i.e. `__contains__`) interfaces as
            well as that of `DataCoordinateIterable`.
        """
        return _ScalarDataCoordinateIterable(dataId)

    @staticmethod
    def fromSet(dataIds: AbstractSet[D], graph: DimensionGraph, *,
                dtype: Type[DataCoordinate] = DataCoordinate) -> DataCoordinateSet[D]:
        """Return a `DataCoordinateSet` (a subclass of
        `DataCoordinateIterable`) backed by the given set-like object.

        Parameters
        ----------
        dataIds : `collections.abc.Set` [ `DataCoordinate` ]
            A set of instances of ``dtype``, with dimensions equal to
            ``graph``.  No runtime checking is performed, to allow
            `DataCoordinateSet` to be used as a lightweight view of a true
            `set`, dictionary keys, or other object implementing the set
            interface.
        graph : `DimensionGraph`
            The dimensions all data IDs in the set identify.
        dtype : `type` (`DataCoordinate` or subclass thereof), optional
            A `DataCoordinate` subclass that all elements' classes must inherit
            from.  Calling code that uses static typing are responsible for
            ensuring this is equal to the ``D`` generic type parameter, as
            there is no good way for runtime code to inspect that.
        """
        from ._dataCoordinateSet import DataCoordinateSet
        return DataCoordinateSet(dataIds, graph=graph, dtype=dtype)

    @property
    @abstractmethod
    def graph(self) -> DimensionGraph:
        """The dimensions identified by thes data IDs (`DimensionGraph`).
        """
        raise NotImplementedError()

    @property
    @abstractmethod
    def dtype(self) -> Type[DataCoordinate]:
        """A `DataCoordinate` subclass that all elements' classes inherit
        from (`type`).
        """
        raise NotImplementedError()

    def minimal(self) -> DataCoordinateIterable[MinimalDataCoordinate]:
        """Return an equivalent `DataCoordinateIterable` containing
        `MinimalDataCoordinate` elements.

        Returns
        -------
        minimal : `DataCoordinateIterable` [ `MinimalDataCoordinate` ]
            An equivalent iterable.  This should always either a lightweight
            view or ``self``.
        """
        if issubclass(self.dtype, MinimalDataCoordinate):
            return self  # type: ignore
        else:
            return _MinimalViewDataCoordinateIterable(self)

    def toSet(self) -> DataCoordinateSet[D]:
        """Transform this iterable into a `DataCoordinateSet`.

        Returns
        -------
        set : `DataCoordinateSet`
            A new `DatasetCoordinateSet` backed by a `frozenset` object,
            constructed by iterating over ``self``.
        """
        from ._dataCoordinateSet import DataCoordinateSet
        return DataCoordinateSet(frozenset(self), graph=self.graph, dtype=self.dtype)

    def constrain(self, query: SimpleQuery, columns: Callable[[str], sqlalchemy.sql.ColumnElement]) -> None:
        """Constrain a SQL query to include or relate to only data IDs in
        this iterable.

        Parameters
        ----------
        query : `SimpleQuery`
            Struct that represents the SQL query to constrain, either by
            appending to its WHERE clause, joining a new table or subquery,
            or both.
        columns : `Callable`
            A callable that accepts `str` dimension names and returns
            SQLAlchemy objects representing a column for that dimension's
            primary key value in the query.
        """
        toOrTogether: List[sqlalchemy.sql.ColumnElement] = []
        for dataId in self:
            toOrTogether.append(
                sqlalchemy.sql.and_(*[
                    columns(dimension.name) == dataId[dimension.name]
                    for dimension in self.graph.required
                ])
            )
        query.where.append(sqlalchemy.sql.or_(*toOrTogether))

    @abstractmethod
    def subset(self, graph: DimensionGraph) -> DataCoordinateIterable[D]:
        """Return an iterable whose data IDs identify a subset of the
        dimensions that this one's do.

        Parameters
        ----------
        graph : `DimensionGraph`
            Dimensions to be identified by the data IDs in the returned
            iterable.  Must be a subset of ``self.graph``.

        Returns
        -------
        iterable : `DataCoordinateIterable`
            A `DataCoordinateIterable` with ``iterable.graph == graph``.
            May be ``self`` if ``graph == self.graph``.  Elements are
            equivalent to those that would be created by calling
            `DataCoordinate.subset` on all elements in ``self``, possibly
            with deduplication (depending on the subclass).
        """
        raise NotImplementedError()


class _ScalarDataCoordinateIterable(DataCoordinateIterable[D]):
    """A `DataCoordinateIterable` implementation that adapts a single
    `DataCoordinate` instance.

    This class should only be used directly by other code in the module in
    which it is defined; all other code should interact with it only through
    the `DataCoordinateIterable` interface.

    Parameters
    ----------
    dataId : `DataCoordinate`
        The data ID to adapt.
    """
    def __init__(self, dataId: D):
        self._dataId = dataId

    __slots__ = ("_dataId",)

    def __iter__(self) -> Iterator[D]:
        yield self._dataId

    def __len__(self) -> int:
        return 1

    def __contains__(self, key: Any) -> bool:
        if isinstance(key, DataCoordinate):
            return key == self._dataId
        else:
            return False

    @property
    def graph(self) -> DimensionGraph:
        # Docstring inherited from DataCoordinateIterable.
        return self._dataId.graph

    @property
    def dtype(self) -> Type[DataCoordinate]:
        # Docstring inherited from DataCoordinateIterable.
        return type(self._dataId)

    def subset(self, graph: DimensionGraph) -> _ScalarDataCoordinateIterable[D]:
        # Docstring inherited from DataCoordinateIterable.
        # No good way to tell MyPy the return type is covariant here (i.e. that
        # we know D.subset(graph) -> D when D is one of Minimal-, Complete- or
        # ExtendedDataCoordinate); we can't use generics in
        # DataCoordinate.subset because that would demand that all
        # implementations return their own type, not just their most
        # appropriate ABC.
        return _ScalarDataCoordinateIterable(self._dataId.subset(graph))  # type: ignore


class _MinimalViewDataCoordinateIterable(DataCoordinateIterable[MinimalDataCoordinate], Generic[D]):
    """A `DataCoordinateIterable` implementation provides a view into another
    iterable with ``dtype=MinimalDataCoordinate``.

    This class should only be used directly by other code in the module in
    which it is defined; all other code should interact with it only through
    the `DataCoordinateIterable` interface.

    Parameters
    ----------
    target : `DataCoordinateIterable`
        The iterable this object is a view into.
    """
    def __init__(self, target: DataCoordinateIterable[D]):
        self._target = target

    __slots__ = ("_target",)

    def __iter__(self) -> Iterator[MinimalDataCoordinate]:
        for dataId in self._target:
            yield dataId.minimal()

    @property
    def graph(self) -> DimensionGraph:
        # Docstring inherited from DataCoordinateIterable.
        return self._target.graph

    @property
    def dtype(self) -> Type[DataCoordinate]:
        # Docstring inherited from DataCoordinateIterable.
        return MinimalDataCoordinate

    def minimal(self) -> _MinimalViewDataCoordinateIterable:
        # Docstring inherited from DataCoordinateIterable.
        return self

    def subset(self, graph: DimensionGraph) -> DataCoordinateIterable[MinimalDataCoordinate]:
        # Docstring inherited from DataCoordinateIterable.
        return self._target.subset(graph).minimal()
