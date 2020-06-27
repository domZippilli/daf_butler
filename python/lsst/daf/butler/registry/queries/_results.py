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
    "DataCoordinateQueryResults",
)

from contextlib import contextmanager
from typing import (
    Callable,
    Iterator,
    Mapping,
    Optional,
    Type,
    TypeVar,
)

import sqlalchemy

from ...core import (
    CompleteDataCoordinate,
    DataCoordinate,
    DataCoordinateIterable,
    DimensionGraph,
    DimensionRecord,
    ExpandedDataCoordinate,
    SimpleQuery,
)
from ..interfaces import Database, DimensionRecordStorageManager
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

    def __init__(self, db: Database, query: Query, dimensions: DimensionRecordStorageManager, *,
                 records: Optional[Mapping[str, Mapping[tuple, DimensionRecord]]] = None):
        self._db = db
        self._query = query
        self._dimensions = dimensions
        self._records = records
        assert query.datasetType is None, \
            "Query used to initialize data coordinate results should not have any datasets."

    __slots__ = ("_db", "_query", "_dimensions", "_records")

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
            yield DataCoordinateQueryResults(self._db, materialized, self._dimensions,
                                             records=self._records)

    def expanded(self) -> DataCoordinateQueryResults[ExpandedDataCoordinate]:
        if self._records is None:
            records = {}
            for element in self.graph.elements:
                subset = self.subset(graph=element.graph, unique=True)
                records[element.name] = {
                    tuple(record.dataId.values()): record
                    for record in self._dimensions[element].fetch(subset)
                }
            return DataCoordinateQueryResults(self._db, self._query, self._dimensions, records=records)
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
            self._dimensions,
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
