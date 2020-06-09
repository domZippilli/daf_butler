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
    "DataCoordinate",
    "CompleteDataCoordinate",
    "EmptyDataCoordinate",
    "ExpandedDataCoordinate",
    "MinimalDataCoordinate",
    "DataId",
    "DataIdKey",
    "DataIdValue",
)

from abc import ABC, abstractmethod
import numbers
from typing import (
    AbstractSet,
    Any,
    Callable,
    Dict,
    Iterator,
    Mapping,
    Optional,
    Tuple,
    TYPE_CHECKING,
    Union,
)

import astropy.time

from lsst.sphgeom import Region
from ..named import NamedKeyMapping, NamedValueSet
from ..timespan import Timespan
from .elements import Dimension, DimensionElement
from .graph import DimensionGraph

if TYPE_CHECKING:  # Imports needed only for type annotations; may be circular.
    from .universe import DimensionUniverse
    from .records import DimensionRecord


DataIdKey = Union[str, Dimension]
DataIdValue = Union[str, int, None]


class DataCoordinate(ABC):
    """An abstract base class for data IDs that have been validated against
    the dimensions they are intended to identify.

    Notes
    -----

    `DataCoordinate` supports many `collections.abc.Mapping` operations (`get`
    and `__getitem__`), but different subclass interfaces define different keys
    for the same set of dimensions in order to fully implement the mapping
    interface:

    - `MinimalDataCoordinate` has key-value pairs for only the _required_
    dimensions, the minimal set needed to identify the rest (given access to
    a `Registry` database).

    - `CompleteDataCoordinate` has key-values pairs for all dimensions (as does
    its subclass, `ExpandedDataCoordinate`).

    Note that `CompleteDataCoordinate` does not (and cannot!) inherit from
    `MinimalDataCoordinate`, because it changes the meaning of ``keys()`` and
    other iteration methods in a non-substitutable way.

    Only the required dimensions are used in equality comparisons and hashing,
    though, so instances of different subclasses with the same values for
    the required dimensions do compare as equal.

    All `DataCoordinate` subclasses should be immutable.
    """

    __slots__ = ()

    @staticmethod
    def standardize(mapping: Optional[Union[Mapping[str, DataIdValue], DataCoordinate]] = None, *,
                    graph: Optional[DimensionGraph] = None,
                    universe: Optional[DimensionUniverse] = None,
                    **kwargs: Any) -> DataCoordinate:
        """Adapt an arbitrary mapping and/or additional arguments into a true
        `DataCoordinate`, or augment an existing one.

        Parameters
        ----------
        mapping : `~collections.abc.Mapping`, optional
            An informal data ID that maps dimension names to their primary key
            values (may also be a true `DataCoordinate`).
        graph : `DimensionGraph`
            The dimensions to be identified by the new `DataCoordinate`.
            If not provided, will be inferred from the keys of ``mapping``,
            and ``universe`` must be provided unless ``mapping`` is already a
            `DataCoordinate`.
        universe : `DimensionUniverse`
            All known dimensions and their relationships; used to expand
            and validate dependencies when ``graph`` is not provided.
        **kwargs
            Additional keyword arguments are treated like additional key-value
            pairs in ``mapping``.

        Returns
        -------
        coordinate : `DataCoordinate`
            A validated `DataCoordinate` instance.  Will be a
            `CompleteDataCoordinate` if all implied dimensions are identified,
            an `ExpandedDataCoordinate` if ``mapping`` is already an
            `ExpandedDataCoordinate` and ``kwargs`` is empty (and hence no new
            dimensions are being added), and a `MinimalDataCoordinate`
            otherwise.

        Raises
        ------
        TypeError
            Raised if the set of optional arguments provided is not supported.
        KeyError
            Raised if a key-value pair for a required dimension is missing.

        Notes
        -----
        Because `MinimalDataCoordinate` stores only values for required
        dimensions, key-value pairs for other implied dimensions will be
        ignored and excluded from the result unless _all_ implied dimensions
        are identified (and hence a `CompleteDataCoordinate` can be returned).
        This means that a `DataCoordinate` may contain *fewer* key-value pairs
        than the informal data ID dictionary it was constructed from.
        """
        d: Dict[str, DataIdValue] = {}
        if isinstance(mapping, DataCoordinate):
            if graph is None:
                if not kwargs:
                    # Already standardized to exactly what we want.
                    return mapping
            elif (isinstance(mapping, CompleteDataCoordinate)
                    and kwargs.keys().isdisjoint(graph.dimensions.names)):
                # User provided kwargs, but told us not to use them by
                # passing in a disjoint graph.
                return mapping.subset(graph)
            assert universe is None or universe == mapping.universe
            universe = mapping.universe
            d.update((name, mapping[name]) for name in mapping.graph.required.names)
            if isinstance(mapping, CompleteDataCoordinate):
                d.update((name, mapping[name]) for name in mapping.graph.implied.names)
        elif mapping is not None:
            d.update(mapping)
        d.update(kwargs)
        if graph is None:
            if universe is None:
                raise TypeError("universe must be provided if graph is not.")
            graph = DimensionGraph(universe, names=d.keys())
        if not graph.dimensions:
            return DataCoordinate.makeEmpty(graph.universe)
        cls: Callable[[DimensionGraph, Tuple[DataIdValue, ...]], DataCoordinate]
        if d.keys() >= graph.dimensions.names:
            cls = CompleteDataCoordinate.fromValues
            values = tuple(d[name] for name in graph.dimensions.names)
        else:
            cls = MinimalDataCoordinate.fromValues
            try:
                values = tuple(d[name] for name in graph.required.names)
            except KeyError as err:
                raise KeyError(f"No value in data ID ({mapping}) for required dimension {err}.") from err
        # Some backends cannot handle numpy.int64 type which is a subclass of
        # numbers.Integral; convert that to int.
        values = tuple(int(val) if isinstance(val, numbers.Integral)  # type: ignore
                       else val for val in values)
        return cls(graph, values)

    @staticmethod
    def makeEmpty(universe: DimensionUniverse) -> EmptyDataCoordinate:
        """Return an empty `DataCoordinate` that identifies the null set of
        dimensions.

        Parameters
        ----------
        universe : `DimensionUniverse`
            Universe to which this null dimension set belongs.

        Returns
        -------
        dataId : `EmptyDataCoordinate`
            A special object that implements both the `MinimalDataCoordinate`
            and `ExpandedDataCoordinate` interfaces, which is only possible in
            this special case where there are no dimensions and hence no
            mapping keys.
        """
        return EmptyDataCoordinate(universe)

    @abstractmethod
    def __getitem__(self, key: DataIdKey) -> DataIdValue:
        """Return the primary key value associated with a dimension.

        Parameters
        ----------
        key : `str` or `Dimension`
            The dimension to be identified.  All `DataCoordinate` subclasses
            support dimensions in ``self.graph.required`` (or their names);
            some subclasses may extend this to those in ``self.graph.implied``.

        Returns
        -------
        value : `int`, `str`, or `None`
            Primary key value associated with just this dimension (the full
            primary key for a dimension may be compound).

        Raises
        ------
        KeyError
            Raised if ``key`` is not a dimension or dimension name known to
            this data ID.
        """
        raise NotImplementedError()

    def get(self, key: DataIdKey, default: Any = None) -> Any:
        """Return the primary key value associated with a dimension or a
        default.

        Parameters
        ----------
        key : `str` or `Dimension`
            The dimension to be identified.  All `DataCoordinate` subclasses
            support dimensions in ``self.graph.required`` (or their names);
            some subclasses may extend this to those in ``self.graph.implied``.
        default
            Default value to return if ``key`` is not known to this object.

        Returns
        -------
        value
            Primary key value associated with just this dimension (the full
            primary key for a dimension may be compound), or ``default`` if it
            is not known to this object.
        """
        try:
            return self[key]
        except KeyError:
            return default

    @property
    @abstractmethod
    def graph(self) -> DimensionGraph:
        """The dimensions identified by this data ID (`DimensionGraph`).

        Note that values are only required to be present for dimensions in
        ``self.graph.required``; all others may be retrieved (from a
        `Registry`) given these.
        """
        raise NotImplementedError()

    def minimal(self) -> MinimalDataCoordinate:
        """Return a `MinimalDataCoordinate` that compares equal to ``self``.

        Returns
        -------
        minimal : `MinimalDataCoordinate`
            A data ID mapping whose keys only include the dimensions in
            ``self.graph.required``.  Should generally be ``self`` if it is
            already a `MinimalDataCoordinate`, or a lightweight view otherwise.
        """
        return MinimalDataCoordinate.fromValues(
            self.graph,
            tuple(self[name] for name in self.graph.required.names)
        )

    @abstractmethod
    def subset(self, graph: DimensionGraph) -> DataCoordinate:
        """Return a new `DataCoordinate` whose graph is a subset of
        ``self.graph``.

        Subclasses may override this method to return a subclass instance or
        operate more efficiently.

        Parameters
        ----------
        graph : `DimensionGraph`
            The dimensions identified by the returned `DataCoordinate`.

        Returns
        -------
        coordinate : `DataCoordinate`
            A `DataCoordinate` instance that identifies only the given
            dimensions.

        Raises
        ------
        KeyError
            Raised if ``graph`` is not a subset of ``self.graph``, and hence
            one or more dimensions has no associated primary key value.
        NotImplementedError
            Raised if ``graph`` is a subset of ``self.graph``, but its
            required dimensions are only implied dimensions in ``self.graph``,
            and ``self`` is a `MinimalDataCoordinate` (and hence those
            dimension values are unknown).
        """
        raise NotImplementedError()

    def __repr__(self) -> str:
        # We can't make repr yield something that could be exec'd here without
        # printing out the whole DimensionUniverse the graph is derived from.
        # So we print something that mostly looks like a dict, but doesn't
        # quote its keys: that's both more compact and something that can't
        # be mistaken for an actual dict or something that could be exec'd.
        return "{{{}}}".format(
            ', '.join(f"{d}: {self.get(d, '?')!r}" for d in self.graph.dimensions.names)
        )

    def __hash__(self) -> int:
        return hash((self.graph,) + tuple(self[d.name] for d in self.graph.required))

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, DataCoordinate):
            other = DataCoordinate.standardize(other)
        return self.graph == other.graph and all(self[d.name] == other[d.name] for d in self.graph.required)

    @property
    def universe(self) -> DimensionUniverse:
        """The universe that defines all known dimensions compatible with
        this coordinate (`DimensionUniverse`).
        """
        return self.graph.universe


class MinimalDataCoordinate(DataCoordinate, NamedKeyMapping[Dimension, DataIdValue]):
    """An abstract base class that combines the `DataCoordinate` interface with
    the `NamedKeyMapping` interface, with key-value pairs only for required
    dimensions.

    Notes
    -----
    The iteration order of a `MinimalDataCoordinate` is the same as the order
    of `DimensionGraph.required`.

    `MinimalDataCoordinate` is abstract, but it includes ``staticmethod``
    constructors (`fromValues` and `fromMapping`) that return instances whose
    exact concrete type should be considered an implementation detail.
    """

    __slots__ = ()

    @staticmethod
    def fromValues(graph: DimensionGraph, values: Tuple[DataIdValue, ...]) -> MinimalDataCoordinate:
        """Construct a `MinimalDataCoordinate` from a tuple of values.

        Parameters
        ----------
        graph : `DimensionGraph`
            Dimensions this data ID will identify.
        values : `tuple` of any of (`int`, `str`, `None`)
            Values for ``graph.required``, in that order.

        Returns
        -------
        dataId : `MinimalDataCoordinate`
            A minimal data ID whose exact concrete type is an implementation
            detail.
        """
        return _MinimalTupleDataCoordinate(graph, values)

    @staticmethod
    def fromMapping(graph: DimensionGraph, mapping: Mapping[str, DataIdValue]) -> MinimalDataCoordinate:
        """Construct a `MinimalDataCoordinate` from a mapping with `str` keys.

        Parameters
        ----------
        graph : `DimensionGraph`
            Dimensions this data ID will identify.
        values : `Mapping` from `str` to of any of (`int`, `str`, `None`)
            Mapping from dimension name to values for at least the dimensions
            in ``graph.required`` (all other key-value pairs are ignored).

        Returns
        -------
        dataId : `MinimalDataCoordinate`
            A minimal data ID whose exact concrete type is an implementation
            detail.
        """
        return MinimalDataCoordinate.fromValues(
            graph,
            tuple(mapping[name] for name in graph.required.names)
        )

    def keys(self) -> NamedValueSet[Dimension]:
        """The keys of the mapping as `Dimension` objects.

        Returns
        -------
        keys : `NamedValueSet` [ `Dimension` ]
            A set of `Dimension` objects.  This is guaranteed to be equivalent
            to ``self.graph.required``.
        """
        return self.graph.required

    def minimal(self) -> MinimalDataCoordinate:
        # Docstring inherited from DataCoordinate.
        return self

    def subset(self, graph: DimensionGraph) -> MinimalDataCoordinate:
        # Docstring inherited from DataCoordinate.
        if graph == self.graph:
            return self
        if not (graph <= self.graph):
            raise KeyError(f"{graph} is not a subset of {self.graph}.")
        if not (graph.required <= self.graph.required):
            raise NotImplementedError(
                f"No value for implied dimension(s) {graph.required - self.graph.required} in "
                f"MinimalDataCoordinate {self}."
            )
        return _MinimalTupleDataCoordinate(
            graph,
            tuple(self[name] for name in graph.dimensions.names)
        )


class CompleteDataCoordinate(DataCoordinate, NamedKeyMapping[Dimension, DataIdValue]):
    """An abstract base class that combines the `DataCoordinate` interface with
    the `NamedKeyMapping` interface, with key-value pairs for all identified
    dimensions.

    Notes
    -----
    The iteration order of a `CompleteDataCoordinate` is the same as the order
    of `DimensionGraph.dimensions`.

    `CompleteDataCoordinate` is abstract, but it includes ``staticmethod``
    constructors (`fromValues` and `fromMapping`) that return instances whose
    exact concrete type should be considered an implementation detail.

    Note that `CompleteDataCoordinate` does not (and cannot!) inherit from
    `MinimalDataCoordinate`, because it changes the meaning of ``keys()`` and
    other iteration methods in a non-substitutable way.  An equivalent
    `MinimalDataCoordinate` view can still be obtained by via the `minimal`
    method.
    """

    __slots__ = ()

    @staticmethod
    def fromValues(graph: DimensionGraph, values: Tuple[DataIdValue, ...]) -> CompleteDataCoordinate:
        """Construct a `CompleteDataCoordinate` from a tuple of values.

        Parameters
        ----------
        graph : `DimensionGraph`
            Dimensions this data ID will identify.
        values : `tuple` of any of (`int`, `str`, `None`)
            Values for ``graph.dimensions``, in that order.

        Returns
        -------
        dataId : `CompleteDataCoordinate`
            A complete data ID whose exact concrete type is an implementation
            detail.
        """
        return _CompleteTupleDataCoordinate(graph, values)

    @staticmethod
    def fromMapping(graph: DimensionGraph, mapping: Mapping[str, DataIdValue]) -> CompleteDataCoordinate:
        """Construct a `CompleteDataCoordinate` from a mapping with `str` keys.

        Parameters
        ----------
        graph : `DimensionGraph`
            Dimensions this data ID will identify.
        values : `Mapping` from `str` to of any of (`int`, `str`, `None`)
            Mapping from dimension name to values for at least the dimensions
            in ``graph.dimensions`` (all other key-value pairs are ignored).

        Returns
        -------
        dataId : `CompleteDataCoordinate`
            A complete data ID whose exact concrete type is an implementation
            detail.
        """
        return CompleteDataCoordinate.fromValues(
            graph,
            tuple(mapping[name] for name in graph.dimensions.names)
        )

    def expanded(self, records: Mapping[str, Optional[DimensionRecord]]) -> ExpandedDataCoordinate:
        """Return an equivalent data ID object that also includes
        `DimensionRecord` structs for all identified `DimensionElement`
        instances.

        Parameters
        ----------
        records : `Mapping` [ `str`, `DimensionRecord` or `None` ]
            Records to associate with identified dimension elements.  Keys must
            be exactly the names in ``self.graph.elements.names``, and values
            should only be `None` if ``self[key]`` is `None` for that key.

        Returns
        -------
        expanded : `ExpandedDataCoordinate`
            A data ID that includes the given record objects.  May be ``self``
            if it is already an `ExpandedDataCoordinate` instance, in which
            case ``records`` is ignored.

        Notes
        -----
        This method is intended primarily for use by code internal to Registry,
        as it requires the caller to be able to construct the ``records`` dict
        correctly and performs no checking.  Most code should use Registry
        interfaces for expanding data IDs via database lookups instead.
        """
        return _ExpandedTupleDataCoordinate(self.graph, tuple(self.values()), records=records)

    def keys(self) -> NamedValueSet[Dimension]:
        """The keys of the mapping as `Dimension` objects.

        Returns
        -------
        keys : `NamedValueSet` [ `Dimension` ]
            A set of `Dimension` objects.  This is guaranteed to be equivalent
            to ``self.graph.dimensions``.
        """
        return self.graph.dimensions

    def subset(self, graph: DimensionGraph) -> CompleteDataCoordinate:
        # Docstring inherited from DataCoordinate.
        if graph == self.graph:
            return self
        return _CompleteTupleDataCoordinate(
            graph,
            tuple(self[name] for name in graph.dimensions.names)
        )


def _intersectRegions(*args: Region) -> Optional[Region]:
    """Return the intersection of several regions.

    For internal use by `ExpandedDataCoordinate` only.

    If no regions are provided, returns `None`.

    This is currently a placeholder; it actually raises `NotImplementedError`
    (it does *not* raise an exception) when multiple regions are given, which
    propagates to `ExpandedDataCoordinate`.
    """
    if len(args) == 0:
        return None
    elif len(args) == 1:
        return args[0]
    else:
        raise NotImplementedError(
            "The region for dimension graphs with multiple unrelated "
            "spatial elements is logically the intersection of all of "
            "per-element, but `Region` intersection is not implemented in "
            "lsst.sphgeom."
        )


class ExpandedDataCoordinate(CompleteDataCoordinate):
    """An abstract base class that adds additional `DimensionRecord` and
    spatiotemporal information to the `CompleteDataCoordinate` interface.
    """

    __slots__ = ()

    @abstractmethod
    def record(self, key: Union[DimensionElement, str]) -> Optional[DimensionRecord]:
        """Return the `DimensionRecord` associated with the given element.

        Parameters
        ----------
        key : `str` or `DimensionElement`
            A `Dimension` or other element in the dimension system, or the name
            of one.

        Returns
        -------
        record : `DimensionRecord` or `None`
            The record for the given element, or `None` if and only if
            ``self[key]`` is `None`.
        """
        raise NotImplementedError()

    @property
    def region(self) -> Optional[Region]:
        """The spatial region associated with this data ID
        (`lsst.sphgeom.Region` or `None`).

        This is `None` if and only if ``self.graph.spatial`` is empty.
        """
        regions = []
        for element in self.graph.spatial:
            record = self.record(element.name)
            # DimensionRecord subclasses for spatial elements always have a
            # .region, but they're dynamic so this can't be type-checked.
            if record is None or record.region is None:  # type: ignore
                return None
            else:
                regions.append(record.region)  # type:ignore
        return _intersectRegions(*regions)

    @property
    def timespan(self) -> Optional[Timespan[astropy.time.Time]]:
        """The temporal interval associated with this data ID
        (`Timespan` or `None`).

        This is `None` if and only if ``self.graph.timespan`` is empty.
        """
        timespans = []
        for element in self.graph.temporal:
            record = self.record(element.name)
            # DimensionRecord subclasses for temporal elements always have
            # .timespan, but they're dynamic so this can't be type-checked.
            if record is None or record.timespan is None:  # type:ignore
                return None
            else:
                timespans.append(record.timespan)  # type:ignore
        return Timespan.intersection(*timespans)

    def pack(self, name: str, *, returnMaxBits: bool = False) -> Union[Tuple[int, int], int]:
        """Pack this data ID into an integer.

        Parameters
        ----------
        name : `str`
            Name of the `DimensionPacker` algorithm (as defined in the
            dimension configuration).
        returnMaxBits : `bool`, optional
            If `True` (`False` is default), return the maximum number of
            nonzero bits in the returned integer across all data IDs.

        Returns
        -------
        packed : `int`
            Integer ID.  This ID is unique only across data IDs that have
            the same values for the packer's "fixed" dimensions.
        maxBits : `int`, optional
            Maximum number of nonzero bits in ``packed``.  Not returned unless
            ``returnMaxBits`` is `True`.
        """
        return self.universe.makePacker(name, self).pack(self, returnMaxBits=returnMaxBits)

    def subset(self, graph: DimensionGraph) -> ExpandedDataCoordinate:
        # Docstring inherited from DataCoordinate.
        if graph == self.graph:
            return self
        return _ExpandedTupleDataCoordinate(
            graph,
            tuple(self[d] for d in graph.dimensions),
            records={e.name: self.record(e.name) for e in graph.elements},
        )


DataId = Union[DataCoordinate, Mapping[str, DataIdValue]]
"""A type-annotation alias for signatures that accept both informal data ID
dictionaries and validated `DataCoordinate` instances.
"""


class _DataCoordinateTupleMixin(NamedKeyMapping[Dimension, DataIdValue]):
    """A mixin class that uses a tuple of values to help implement one of the
    `DataCoordinate` subclass interfaces.

    Parameters
    ----------
    graph : `DimensionGraph`
        The dimensions to be identified.
    indices : `dict` [ `str`, `int` ]
        Dictionary mapping dimension name to sequentially increasing integers
        (i.e. indices into ``values``, which must already be ordered to be
        consistent with ``graph``).
    values : `tuple` [ `int` or `str` or `None` ]
        Data ID values, ordered to match ``keys()``.

    Notes
    -----
    This mixin is probably only useful within the module where it is defined,
    where it is used to provide tuple-based implementations of the three main
    `DataCoordinate` subclass ABCs.
    """

    def __init__(self, graph: DimensionGraph,
                 indices: Dict[str, int],
                 values: Tuple[DataIdValue, ...]):
        assert len(indices) == len(values)
        self._graph = graph
        self._indices = indices
        self._values = values

    __slots__ = ("_graph", "_indices", "_values")

    @abstractmethod
    def keys(self) -> NamedValueSet[Dimension]:
        """The keys of the mapping as `Dimension` objects.

        Returns
        -------
        keys : `NamedValueSet` [ `Dimension` ]
            A set of `Dimension` objects.  Subclasses must implement this
            method to define which dimensions in ``self.graph`` are included
            in the mapping, and must do so consistently with the ``values``
            and ``indices`` arguments to `__init__`.
        """
        raise NotImplementedError()

    def values(self) -> Tuple[DataIdValue, ...]:  # type: ignore
        return self._values

    @property
    def names(self) -> AbstractSet[str]:
        # Docstring inherited from NamedKeyMapping.
        return self.keys().names

    @property
    def graph(self) -> DimensionGraph:
        """The dimensions identified by this data ID (`DimensionGraph`).

        Note that values are only required to be present for dimensions in
        ``self.graph.required``; all others may be retrieved (from a
        `Registry`) given these.
        """
        # Docstring can't be inherited from DataCoordinate because this is a
        # mixin that will appear first in the inheritance list, so we just
        # copy it here.
        return self._graph

    def __getitem__(self, key: DataIdKey) -> DataIdValue:
        """Return the primary key value associated with a dimension.

        Parameters
        ----------
        key : `str` or `Dimension`
            The dimension to be identified.  All `DataCoordinate` subclasses
            support dimensions in ``self.graph.required`` (or their names);
            some subclasses may extend this to those in ``self.graph.implied``.

        Returns
        -------
        value : `int`, `str`, or `None`
            Primary key value associated with just this dimension (the full
            primary key for a dimension may be compound).

        Raises
        ------
        KeyError
            Raised if ``key`` is not a dimension or dimension name known to
            this data ID.
        """
        # Docstring can't be inherited from DataCoordinate because this is a
        # mixin that will appear first in the inheritance list, so we just
        # copy it here.
        if isinstance(key, Dimension):
            key = key.name
        return self._values[self._indices[key]]

    def __len__(self) -> int:
        return len(self._indices)

    def __iter__(self) -> Iterator[Dimension]:
        return iter(self.keys())


class _MinimalTupleDataCoordinate(_DataCoordinateTupleMixin, MinimalDataCoordinate):
    """A tuple-based implementation of `MinimalDataCoordinate`.

    This class should only be used directly by other code in the module in
    which it is defined; all other code should interact with it only through
    the `MinimalDataCoordinate` interface.
    """

    __slots__ = ()

    def __init__(self, graph: DimensionGraph, values: Tuple[DataIdValue, ...]):
        super().__init__(graph, graph._requiredIndices, values)

    def keys(self) -> NamedValueSet[Dimension]:
        return self.graph.required


class _CompleteTupleDataCoordinate(_DataCoordinateTupleMixin, CompleteDataCoordinate):
    """A tuple-based implementation of `CompleteDataCoordinate`.

    This class should only be used directly by other code in the module in
    which it is defined; all other code should interact with it only through
    the `CompleteDataCoordinate` interface.
    """

    __slots__ = ()

    def __init__(self, graph: DimensionGraph, values: Tuple[DataIdValue, ...]):
        super().__init__(graph, graph._dimensionIndices, values)

    def keys(self) -> NamedValueSet[Dimension]:
        return self.graph.dimensions


class _ExpandedTupleDataCoordinate(_CompleteTupleDataCoordinate, ExpandedDataCoordinate):
    """A tuple-based implementation of `ExpandedDataCoordinate`.

    This class should only be used directly by other code in the module in
    which it is defined; all other code should interact with it only through
    the `ExpandedDataCoordinate` interface.
    """

    __slots__ = ("_records",)

    def __init__(self, graph: DimensionGraph, values: Tuple[DataIdValue, ...], *,
                 records: Mapping[str, Optional[DimensionRecord]]):
        super().__init__(graph, values)
        self._records = records

    def record(self, key: Union[DimensionElement, str]) -> Optional[DimensionRecord]:
        # Docstring inherited from ExpandedDataCoordinate.
        if isinstance(key, DimensionElement):
            return self._records[key.name]
        else:
            return self._records[key]


class EmptyDataCoordinate(MinimalDataCoordinate, ExpandedDataCoordinate):
    """A special `DataCoordinate` implementation for null dimension graphs.
    """

    __slots__ = ("_universe",)

    def __init__(self, universe: DimensionUniverse):
        self._universe = universe

    @staticmethod
    def fromValues(graph: DimensionGraph, values: Tuple[DataIdValue, ...]) -> EmptyDataCoordinate:
        # Docstring inherited from ExpandedDataCoordinate.
        assert not graph and not values
        return EmptyDataCoordinate(graph.universe)

    @staticmethod
    def fromMapping(graph: DimensionGraph, mapping: Mapping[str, DataIdValue]) -> EmptyDataCoordinate:
        # Docstring inherited from ExpandedDataCoordinate.
        assert not graph
        return EmptyDataCoordinate(graph.universe)

    @property
    def graph(self) -> DimensionGraph:
        # Docstring inherited from ExpandedDataCoordinate.
        return self._universe.empty

    @property
    def universe(self) -> DimensionUniverse:
        # Docstring inherited from ExpandedDataCoordinate.
        return self._universe

    @property
    def names(self) -> AbstractSet[str]:
        # Docstring inherited from ExpandedDataCoordinate.
        return frozenset()

    def __getitem__(self, key: DataIdKey) -> DataIdValue:
        # Docstring inherited from ExpandedDataCoordinate.
        raise KeyError(f"Empty data ID indexed with {key}.")

    def keys(self) -> NamedValueSet[Dimension]:
        # Docstring inherited from ExpandedDataCoordinate.
        return NamedValueSet()

    def __len__(self) -> int:
        return 0

    def __iter__(self) -> Iterator[Dimension]:
        return iter(())

    def record(self, key: Union[DimensionElement, str]) -> Optional[DimensionRecord]:
        # Docstring inherited from ExpandedDataCoordinate.
        return None

    def subset(self, graph: DimensionGraph) -> EmptyDataCoordinate:
        # Docstring inherited from ExpandedDataCoordinate.
        if graph:
            raise KeyError(f"Cannot subset EmptyDataCoordinate with non-empty graph {graph}.")
        return self
