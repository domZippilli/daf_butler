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
    "CompleteDataCoordinateQueryResults",
    "ExpandedDataCoordinateQueryResults",
)

from contextlib import contextmanager
from typing import Callable, Dict, Iterator, Optional, Type

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


class CompleteDataCoordinateQueryResults(DataCoordinateIterable[CompleteDataCoordinate]):

    __slots__ = ("_db", "_query", "_dimensions")

    def __init__(self, db: Database, query: Query, dimensions: DimensionRecordStorageManager):
        self._db = db
        self._query = query
        self._dimensions = dimensions
        assert next(query.datasetTypes, None) is None, \
            "Query used to initialize data coordinate results should not have any datasets."

    def __iter__(self) -> Iterator[CompleteDataCoordinate]:
        predicate = self._query.predicate()
        for row in self._db.query(self._query.sql):
            if predicate(row):
                yield self._query.extractDataId(row)

    @property
    def graph(self) -> DimensionGraph:
        return self._query.graph

    @property
    def dtype(self) -> Type[DataCoordinate]:
        return CompleteDataCoordinate

    @contextmanager
    def materialize(self) -> Iterator[CompleteDataCoordinateQueryResults]:
        with self._query.materialize(self._db) as materialized:
            yield CompleteDataCoordinateQueryResults(self._db, materialized, self._dimensions)

    def expanded(self) -> ExpandedDataCoordinateQueryResults:
        records = {}
        for element in self.graph.elements:
            subset = self.subset(graph=element.graph, unique=True)
            records[element.name] = {
                tuple(record.dataId.values()): record
                for record in self._dimensions[element].fetch(subset)
            }
        return ExpandedDataCoordinateQueryResults(self, records)

    def subset(self, graph: Optional[DimensionGraph] = None, *,
               unique: bool = False) -> CompleteDataCoordinateQueryResults:
        if graph is None:
            graph = self.graph
        if not graph.issubset(self.graph):
            raise ValueError(f"{graph} is not a subset of {self.graph}")
        if graph == self.graph and (not unique or self._query.isUnique()):
            return self
        return CompleteDataCoordinateQueryResults(
            self._db,
            self._query.subset(graph=graph, datasetTypes=(), unique=unique),
            self._dimensions,
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


class ExpandedDataCoordinateQueryResults(DataCoordinateIterable[ExpandedDataCoordinate]):

    __slots__ = ("_complete", "_records")

    def __init__(self, complete: CompleteDataCoordinateQueryResults,
                 records: Dict[str, Dict[tuple, DimensionRecord]]):
        self._complete = complete
        self._records = records

    def __iter__(self) -> Iterator[ExpandedDataCoordinate]:
        for dataId in self._complete:
            records = {}
            for element in self.graph.elements:
                key = tuple(dataId.subset(element.graph).minimal().values())
                records[element.name] = self._records[element.name].get(key)
            yield dataId.expanded(records)

    @property
    def graph(self) -> DimensionGraph:
        return self._complete.graph

    @property
    def dtype(self) -> Type[DataCoordinate]:
        return ExpandedDataCoordinate

    @contextmanager
    def materialize(self) -> Iterator[ExpandedDataCoordinateQueryResults]:
        with self._complete.materialize() as complete:
            yield ExpandedDataCoordinateQueryResults(complete, self._records)

    def expanded(self) -> ExpandedDataCoordinateQueryResults:
        return self

    def subset(self, graph: Optional[DimensionGraph] = None, *,
               unique: bool = False) -> ExpandedDataCoordinateQueryResults:
        if graph is None:
            graph = self.graph
        if not graph.issubset(self.graph):
            raise ValueError(f"{graph} is not a subset of {self.graph}")
        if graph == self.graph:
            return self
        return ExpandedDataCoordinateQueryResults(
            self._complete.subset(graph, unique=unique),
            records={element.name: self._records[element.name] for element in graph.elements}
        )

    def constrain(self, query: SimpleQuery, columns: Callable[[str], sqlalchemy.sql.ColumnElement]) -> None:
        self._complete.constrain(query, columns)
