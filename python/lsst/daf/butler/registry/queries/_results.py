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
    "ChainedDatasetQueryResults",
    "DataCoordinateQueryResults",
    "DatasetQueryResults",
    "ParentDatasetQueryResults",
)

from abc import abstractmethod
from contextlib import contextmanager, ExitStack
import itertools
from typing import (
    Any,
    Callable,
    ContextManager,
    Generic,
    Iterable,
    Iterator,
    Sequence,
    Mapping,
    Optional,
    Type,
    TypeVar,
)

import sqlalchemy

from ...core import (
    CompleteDataCoordinate,
    DataCoordinateIterable,
    DatasetRef,
    DatasetType,
    DimensionGraph,
    DimensionRecord,
    ExpandedDataCoordinate,
    SimpleQuery,
)
from ..interfaces import Database
from ._query import Query


D = TypeVar("D", bound=CompleteDataCoordinate)


class DataCoordinateQueryResults(DataCoordinateIterable[D]):
    """An enhanced implementation of `DataCoordinateIterable` that represents
    data IDs retrieved from a database query.

    Notes
    -----
    Query results always return at least `CompleteDataCoordinate` objects, and
    can return `ExpandedDataCoordinate` objects.  While this class is declared
    as generic, it can only return `ExpandedDataCoordinate` objects if the
    ``records`` parameter is not `None` at construction.  That sort of runtime
    relationship to typing is not something MyPy handles well, and the result
    is a lot of ``type: ignore`` in the implementation and an expectation that
    code that constructs this class (which should be rare and localized to the
    ``daf.butler.registry`` subpackage) will take care to guarantee this.  The
    alternative would be replacing this single generic class with a pair of
    classes that are almost identical.  Code that only interacts with this
    class after construction (and obtains instances through properly typed
    interfaces) can rely on generic typing working as expected.  Code that does
    not use type annotations can ignore this entire paragraph.
    """

    def __init__(self, db: Database, query: Query, *,
                 records: Optional[Mapping[str, Mapping[tuple, DimensionRecord]]] = None):
        self._db = db
        self._query = query
        self._records = records
        assert query.datasetType is None, \
            "Query used to initialize data coordinate results should not have any datasets."

    __slots__ = ("_db", "_query", "_records")

    def __iter__(self) -> Iterator[D]:
        return (self._query.extractDataId(row, records=self._records)  # type: ignore
                for row in self._query.rows(self._db))

    @property
    def graph(self) -> DimensionGraph:
        return self._query.graph

    @property
    def dtype(self) -> Type[CompleteDataCoordinate]:
        return CompleteDataCoordinate if self._records is None else ExpandedDataCoordinate

    @contextmanager
    def materialize(self) -> Iterator[DataCoordinateQueryResults[D]]:
        with self._query.materialize(self._db) as materialized:
            yield DataCoordinateQueryResults(self._db, materialized, records=self._records)

    def expanded(self) -> DataCoordinateQueryResults[ExpandedDataCoordinate]:
        if self._records is None:
            records = {}
            for element in self.graph.elements:
                subset = self.subset(graph=element.graph, unique=True)
                records[element.name] = {
                    tuple(record.dataId.values()): record
                    for record in self._query.managers.dimensions[element].fetch(subset)
                }
            return DataCoordinateQueryResults(self._db, self._query, records=records)
        else:
            return self  # type: ignore

    def subset(self, graph: Optional[DimensionGraph] = None, *,
               unique: bool = False) -> DataCoordinateQueryResults[D]:
        if graph is None:
            graph = self.graph
        if not graph.issubset(self.graph):
            raise ValueError(f"{graph} is not a subset of {self.graph}")
        if graph == self.graph and (not unique or self._query.isUnique()):
            return self
        records: Optional[Mapping[str, Mapping[tuple, DimensionRecord]]]
        if self._records is not None:
            records = {element.name: self._records[element.name] for element in graph.elements}
        else:
            records = None
        return DataCoordinateQueryResults(
            self._db,
            self._query.subset(graph=graph, datasets=False, unique=unique),
            records=records,
        )

    def constrain(self, query: SimpleQuery, columns: Callable[[str], sqlalchemy.sql.ColumnElement]) -> None:
        fromClause = self._query.sql.alias("c")
        query.join(
            fromClause,
            onclause=sqlalchemy.sql.and_(*[
                columns(dimension.name) == fromClause.columns[dimension.name]
                for dimension in self.graph.required
            ])
        )

    def findDatasets(self, datasetType: DatasetType, collections: Any, *,
                     deduplicate: bool = True) -> ParentDatasetQueryResults:
        # TODO: allow dataset type name to be passed here; probably involves
        # moving component handling down into managers.
        if datasetType.dimensions != self.graph:
            raise ValueError(f"findDatasets requires that the dataset type have the same dimensions as "
                             f"the DataCoordinateQueryResult used as input to the search, but "
                             f"{datasetType.name} has dimensions {datasetType.dimensions}, while the input "
                             f"dimensions are {self.graph}.")
        builder = self._query.makeBuilder()
        if datasetType.isComponent():
            # We were given a true DatasetType instance, but it's a component.
            parentName, componentName = datasetType.nameAndComponent()
            storage = self._query.managers.datasets.find(parentName)
            if storage is None:
                raise KeyError(f"Parent DatasetType '{parentName}' could not be found.")
            datasetType = storage.datasetType
            components = [componentName]
        else:
            components = [None]
        builder.joinDataset(datasetType, collections=collections, deduplicate=deduplicate)
        query = builder.finish()
        return ParentDatasetQueryResults(db=self._db, query=query, components=components,
                                         records=self._records)


class DatasetQueryResults(Iterable[DatasetRef]):

    @abstractmethod
    def byParentDatasetType(self) -> Iterator[ParentDatasetQueryResults]:
        raise NotImplementedError()

    @abstractmethod
    def materialize(self) -> ContextManager[DatasetQueryResults]:
        raise NotImplementedError()


class ParentDatasetQueryResults(DatasetQueryResults, Generic[D]):

    def __init__(self, db: Database, query: Query, *,
                 components: Sequence[Optional[str]],
                 records: Optional[Mapping[str, Mapping[tuple, DimensionRecord]]] = None):
        self._db = db
        self._query = query
        self._components = components
        self._records = records
        assert query.datasetType is not None, \
            "Query used to initialize dataset results must have a dataset."
        assert query.datasetType.dimensions == query.graph

    __slots__ = ("_db", "_query", "_dimensions", "_components", "_records")

    def __iter__(self) -> Iterator[DatasetRef]:
        for row in self._query.rows(self._db):
            parentRef = self._query.extractDatasetRef(row, records=self._records)
            for component in self._components:
                if component is None:
                    yield parentRef
                else:
                    yield parentRef.makeComponentRef(component)

    def byParentDatasetType(self) -> Iterator[ParentDatasetQueryResults]:
        yield self

    @contextmanager
    def materialize(self) -> Iterator[ParentDatasetQueryResults[D]]:
        with self._query.materialize(self._db) as materialized:
            yield ParentDatasetQueryResults(self._db, materialized,
                                            components=self._components,
                                            records=self._records)

    @property
    def parentDatasetType(self) -> DatasetType:
        assert self._query.datasetType is not None
        return self._query.datasetType

    @property
    def dataIds(self) -> DataCoordinateQueryResults[D]:
        return DataCoordinateQueryResults(
            self._db,
            self._query.subset(graph=self.parentDatasetType.dimensions, datasets=False, unique=False),
            records=self._records,
        )

    def withComponents(self, components: Sequence[Optional[str]]) -> ParentDatasetQueryResults:
        return ParentDatasetQueryResults(self._db, self._query, records=self._records,
                                         components=components)

    def expanded(self) -> ParentDatasetQueryResults[ExpandedDataCoordinate]:
        if self._records is None:
            records = self.dataIds.expanded()._records
            return ParentDatasetQueryResults(self._db, self._query, records=records,
                                             components=self._components)
        else:
            return self  # type: ignore


class ChainedDatasetQueryResults(DatasetQueryResults):

    def __init__(self, chain: Sequence[ParentDatasetQueryResults]):
        self._chain = chain

    __slots__ = ("_chain",)

    def __iter__(self) -> Iterator[DatasetRef]:
        return itertools.chain.from_iterable(self._chain)

    def byParentDatasetType(self) -> Iterator[ParentDatasetQueryResults]:
        return iter(self._chain)

    @contextmanager
    def materialize(self) -> Iterator[ChainedDatasetQueryResults]:
        with ExitStack() as stack:
            yield ChainedDatasetQueryResults(
                [stack.enter_context(r.materialize()) for r in self._chain]
            )
