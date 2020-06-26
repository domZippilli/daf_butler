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

__all__ = ("Query",)

from abc import ABC, abstractmethod
from contextlib import contextmanager
import copy
import enum
import itertools
from typing import Callable, Iterable, Iterator, Optional, Tuple

import sqlalchemy

from lsst.sphgeom import Region

from ...core import (
    addDimensionForeignKey,
    CompleteDataCoordinate,
    DataCoordinate,
    DatasetRef,
    DatasetType,
    ddl,
    Dimension,
    DimensionElement,
    DimensionGraph,
    REGION_FIELD_SPEC,
    SimpleQuery,
)
from ..interfaces import (
    CollectionManager,
    Database,
    DatasetRecordStorageManager,
)
from ._structs import DatasetQueryColumns, QueryColumns


class Query(ABC):
    """A wrapper for a SQLAlchemy query that knows how to transform result rows
    into data IDs and dataset references.

    A `Query` should almost always be constructed directly by a call to
    `QueryBuilder.finish`; direct construction will make it difficult to be
    able to maintain invariants between arguments (see the documentation for
    `QueryColumns` for more information).

    Parameters
    ----------
    graph : `DimensionGraph`
        Object describing the dimensions included in the query.
    whereRegion : `lsst.sphgeom.Region`, optional
        Region that all region columns in all returned rows must overlap.
    columns : `QueryColumns`
        Columns that are referenced in the query in any clause.
    collections : `CollectionsManager`,
        Manager object for collection tables.
    TODO

    Notes
    -----
    SQLAlchemy is used in the public interface of `Query` rather than just its
    implementation simply because avoiding this would entail writing wrappers
    for the `sqlalchemy.engine.RowProxy` and `sqlalchemy.engine.ResultProxy`
    classes that are themselves generic wrappers for lower-level Python DBAPI
    classes.  Another layer would entail another set of computational
    overheads, but the only reason we would seriously consider not using
    SQLAlchemy here in the future would be to reduce computational overheads.
    """
    def __init__(self, *,
                 graph: DimensionGraph,
                 whereRegion: Optional[Region],
                 collections: CollectionManager,
                 datasets: DatasetRecordStorageManager):
        self.graph = graph
        self.whereRegion = whereRegion
        self._collections = collections
        self._datasets = datasets

    @abstractmethod
    def isUnique(self) -> bool:
        """Return `True` if this queries rows are guaranteed to be unique, and
        `False` otherwise.
        """
        raise NotImplementedError()

    @abstractmethod
    def getDimensionColumn(self, name: str) -> sqlalchemy.sql.ColumnElement:
        # TODO: docs
        raise NotImplementedError()

    @property
    @abstractmethod
    def spatial(self) -> Iterator[DimensionElement]:
        # TODO: docs
        raise NotImplementedError()

    @abstractmethod
    def getRegionColumn(self, name: str) -> sqlalchemy.sql.ColumnElement:
        # TODO: docs
        raise NotImplementedError()

    @property
    def datasetType(self) -> Optional[DatasetType]:
        # TODO: docs
        cols = self.getDatasetColumns()
        if cols is None:
            return None
        return cols.datasetType

    @abstractmethod
    def getDatasetColumns(self) -> Optional[DatasetQueryColumns]:
        # TODO: docs
        raise NotImplementedError()

    @property
    @abstractmethod
    def sql(self) -> sqlalchemy.sql.FromClause:
        """Return a SQLAlchemy object representing the full query.

        Returns
        -------
        sql : `sqlalchemy.sql.FromClause`
            A SQLAlchemy object representing the query.
        """
        raise NotImplementedError()

    def predicate(self, region: Optional[Region] = None) -> Callable[[sqlalchemy.engine.RowProxy], bool]:
        """Return a callable that can perform extra Python-side filtering of
        query results.

        To get the expected results from a query, the returned predicate *must*
        be used to ignore rows for which it returns `False`; this permits the
        `QueryBuilder` implementation to move logic from the database to Python
        without changing the public interface.

        Parameters
        ----------
        region : `sphgeom.Region`, optional
            A region that any result-row regions must overlap in order for the
            predicate to return `True`.  If not provided, this will be
            ``self.region``, if that exists.

        Returns
        -------
        func : `Callable`
            A callable that takes a single `sqlalchemy.engine.RowProxy`
            argmument and returns `bool`.
        """
        whereRegion = region if region is not None else self.whereRegion

        def closure(row: sqlalchemy.engine.RowProxy) -> bool:
            rowRegions = [row[self.getRegionColumn(element.name)] for element in self.spatial]
            if whereRegion and any(r.isDisjointFrom(whereRegion) for r in rowRegions):
                return False
            return not any(a.isDisjointFrom(b) for a, b in itertools.combinations(rowRegions, 2))

        return closure

    def extractDimensionsTuple(self, row: sqlalchemy.engine.RowProxy,
                               dimensions: Iterable[Dimension]) -> tuple:
        """Extract a tuple of data ID values from a result row.

        Parameters
        ----------
        row : `sqlalchemy.engine.RowProxy`
            A result row from a SQLAlchemy SELECT query.
        dimensions : `Iterable` [ `Dimension` ]
            The dimensions to include in the returned tuple, in order.

        Returns
        -------
        values : `tuple`
            A tuple of dimension primary key values.
        """
        return tuple(row[self.getDimensionColumn(dimension.name)] for dimension in dimensions)

    def extractDataId(self, row: sqlalchemy.engine.RowProxy, *, graph: Optional[DimensionGraph] = None
                      ) -> CompleteDataCoordinate:
        """Extract a data ID from a result row.

        Parameters
        ----------
        row : `sqlalchemy.engine.RowProxy`
            A result row from a SQLAlchemy SELECT query.
        graph : `DimensionGraph`, optional
            The dimensions the returned data ID should identify.  If not
            provided, this will be all dimensions in `QuerySummary.requested`.

        Returns
        -------
        dataId : `CompleteDataCoordinate`
            A data ID that identifies all required and implied dimensions.
        """
        if graph is None:
            graph = self.graph
        return CompleteDataCoordinate.fromValues(graph, self.extractDimensionsTuple(row, graph.dimensions))

    def extractDatasetRef(self, row: sqlalchemy.engine.RowProxy,
                          dataId: Optional[DataCoordinate] = None) -> DatasetRef:
        """Extract a `DatasetRef` from a result row.

        Parameters
        ----------
        row : `sqlalchemy.engine.RowProxy`
            A result row from a SQLAlchemy SELECT query.
        dataId : `DataCoordinate`
            Data ID to attach to the `DatasetRef`.  A minimal (i.e. base class)
            `DataCoordinate` is constructed from ``row`` if `None`.

        Returns
        -------
        ref : `DatasetRef`
            Reference to the dataset; guaranteed to have `DatasetRef.id` not
            `None`.
        """
        datasetColumns = self.getDatasetColumns()
        assert datasetColumns is not None
        if dataId is None:
            dataId = self.extractDataId(row, graph=datasetColumns.datasetType.dimensions)
        runRecord = self._collections[row[datasetColumns.runKey]]
        return DatasetRef(datasetColumns.datasetType, dataId, id=row[datasetColumns.id], run=runRecord.name)

    def _makeTableSpec(self, constraints: bool = False) -> ddl.TableSpec:
        # TODO: docs
        unique = self.isUnique()
        spec = ddl.TableSpec(fields=())
        for dimension in self.graph:
            addDimensionForeignKey(spec, dimension, primaryKey=unique, constraint=constraints)
        for element in self.spatial:
            field = copy.copy(REGION_FIELD_SPEC)
            field.name = f"{element.name}_region"
            spec.fields.add(field)
        datasetColumns = self.getDatasetColumns()
        if datasetColumns is not None:
            self._datasets.addDatasetForeignKey(spec, primaryKey=unique, constraint=constraints)
            self._collections.addRunForeignKey(spec, nullable=False, constraint=constraints)
        return spec

    def _makeSubsetQueryColumns(self, *, graph: Optional[DimensionGraph] = None,
                                datasets: bool = True,
                                unique: bool = False) -> Tuple[DimensionGraph, Optional[QueryColumns]]:
        # TODO: docs
        if graph is None:
            graph = self.graph
        if (graph == self.graph and (self.getDatasetColumns() is None or datasets)
                and (self.isUnique() or not unique)):
            return graph, None
        columns = QueryColumns()
        for dimension in graph.dimensions:
            col = self.getDimensionColumn(dimension.name)
            columns.keys[dimension] = [col]
        for element in self.spatial:
            col = self.getRegionColumn(element.name)
            columns.regions[element] = col
        if not datasets and self.getDatasetColumns() is not None:
            columns.datasets = self.getDatasetColumns()
        return graph, columns

    @contextmanager
    def materialize(self, db: Database) -> Iterator[MaterializedQuery]:
        # TODO: docs
        table = db.makeTemporaryTable(self._makeTableSpec())
        db.insert(table, select=self.sql(), names=table.fields.names)
        yield MaterializedQuery(table=table,
                                spatial=self.spatial,
                                datasetType=self.datasetType,
                                isUnique=self.isUnique(),
                                graph=self.graph,
                                whereRegion=self.whereRegion,
                                collections=self._collections,
                                datasets=self._datasets)
        db.dropTemporaryTable(db)

    @abstractmethod
    def subset(self, *, graph: Optional[DimensionGraph] = None,
               datasets: bool = True,
               unique: bool = False) -> Query:
        # TODO: docs
        raise NotImplementedError()


class DirectQueryUniqueness(enum.Enum):
    NOT_UNIQUE = enum.auto()
    NATURALLY_UNIQUE = enum.auto()
    NEEDS_DISTINCT = enum.auto()


class DirectQuery(Query):
    """A `Query` implementation that represents a direct SELECT query that
    usually joins many tables.

    Parameters
    ----------
    simpleQuery : `SimpleQuery`
        Struct representing the actual SELECT, FROM, and WHERE clauses.
    graph : `DimensionGraph`
        Object describing the dimensions included in the query.
    whereRegion : `lsst.sphgeom.Region`, optional
        Region that all region columns in all returned rows must overlap.
    columns : `QueryColumns`
        Columns that are referenced in the query in any clause.
    collections : `CollectionsManager`,
        Manager object for collection tables.
    TODO
    """
    def __init__(self, *,
                 simpleQuery: SimpleQuery,
                 columns: QueryColumns,
                 uniqueness: DirectQueryUniqueness,
                 graph: DimensionGraph,
                 whereRegion: Optional[Region],
                 collections: CollectionManager,
                 datasets: DatasetRecordStorageManager):
        super().__init__(graph=graph, whereRegion=whereRegion,
                         collections=collections, datasets=datasets)
        assert not simpleQuery.columns, "Columns should always be set on a copy in .sql"
        self._simpleQuery = simpleQuery
        self._columns = columns
        self._uniqueness = uniqueness

    def isUnique(self) -> bool:
        # Docstring inherited from Query.
        return self._uniqueness is not DirectQueryUniqueness.NOT_UNIQUE

    def getDimensionColumn(self, name: str) -> sqlalchemy.sql.ColumnElement:
        # Docstring inherited from Query.
        return self._columns.getKeyColumn(name).label(name)

    @property
    def spatial(self) -> Iterator[DimensionElement]:
        # Docstring inherited from Query.
        return iter(self._columns.regions)

    def getRegionColumn(self, name: str) -> sqlalchemy.sql.ColumnElement:
        # Docstring inherited from Query.
        return self._columns.regions[name].label(f"{name}_region")

    def getDatasetColumns(self) -> Optional[DatasetQueryColumns]:
        # Docstring inherited from Query.
        base = self._columns.datasets
        if base is None:
            return None
        return DatasetQueryColumns(
            datasetType=base.datasetType,
            id=base.id.label("dataset_id"),
            runKey=base.runKey.label(self._collections.getRunForeignKeyName()),
        )

    @property
    def sql(self) -> sqlalchemy.sql.FromClause:
        # Docstring inherited from Query.
        simpleQuery = self._simpleQuery.copy()
        for dimension in self.graph:
            simpleQuery.columns.append(self.getDimensionColumn(dimension.name))
        for element in self.spatial:
            simpleQuery.columns.append(self.getRegionColumn(element.name))
        datasetColumns = self.getDatasetColumns()
        if datasetColumns is not None:
            simpleQuery.columns.extend(datasetColumns)
        sql = simpleQuery.combine()
        if self._uniqueness is DirectQueryUniqueness.NEEDS_DISTINCT:
            return sql.distinct()
        else:
            return sql

    def subset(self, *, graph: Optional[DimensionGraph] = None,
               datasets: bool = True,
               unique: bool = False) -> DirectQuery:
        # Docstring inherited from Query.
        graph, columns = self._makeSubsetQueryColumns(graph=graph, datasets=datasets, unique=unique)
        if columns is None:
            return self
        return DirectQuery(
            simpleQuery=self._simpleQuery.copy(),
            columns=columns,
            uniqueness=DirectQueryUniqueness.NEEDS_DISTINCT if unique else DirectQueryUniqueness.NOT_UNIQUE,
            graph=graph,
            whereRegion=self.whereRegion,
            collections=self._collections,
            datasets=self._datasets,
        )


class MaterializedQuery(Query):
    """A `Query` implementation that represents query results saved in a
    temporary table.

    `MaterializedQuery` instances should not be constructed directly; use
    `Query.materialize()` instead.

    Parameters
    ----------
    table : `sqlalchemy.schema.Table`
        SQLAlchemy table object.
    graph : `DimensionGraph`
        Object describing the dimensions included in the query.
    region : `lsst.sphgeom.Region`, optional
        Region that all region columns in all returned rows must overlap.
    collections : `CollectionsManager`,
        Manager object for collection tables.
    """
    def __init__(self, *,
                 table: sqlalchemy.schema.Table,
                 spatial: Iterable[DimensionElement],
                 datasetType: Optional[DatasetType],
                 isUnique: bool,
                 graph: DimensionGraph,
                 whereRegion: Optional[Region],
                 collections: CollectionManager,
                 datasets: DatasetRecordStorageManager):
        super().__init__(graph=graph, whereRegion=whereRegion, collections=collections, datasets=datasets)
        self._table = table
        self._spatial = tuple(spatial)
        self._datasetType = datasetType
        self._isUnique = isUnique

    def isUnique(self) -> bool:
        # Docstring inherited from Query.
        return self._isUnique

    def getDimensionColumn(self, name: str) -> sqlalchemy.sql.ColumnElement:
        # Docstring inherited from Query.
        return self._table.columns[name]

    @property
    def spatial(self) -> Iterator[DimensionElement]:
        # Docstring inherited from Query.
        return iter(self._spatial)

    def getRegionColumn(self, name: str) -> sqlalchemy.sql.ColumnElement:
        # Docstring inherited from Query.
        return self._table.columns[f"{name}_region"]

    @property
    def datasetTypes(self) -> Optional[DatasetType]:
        # Docstring inherited from Query.
        return self._datasetType

    def getDatasetColumns(self) -> Optional[DatasetQueryColumns]:
        # Docstring inherited from Query.
        if self._datasetType is not None:
            return DatasetQueryColumns(
                datasetType=self._datasetType,
                id=self._table.columns["dataset_id"],
                runKey=self._table.columns[self._collections.getRunForeignKeyName()],
            )
        else:
            return None

    @property
    def sql(self) -> sqlalchemy.sql.FromClause:
        # Docstring inherited from Query.
        return self._table.select()

    @contextmanager
    def materialize(self, db: Database) -> Iterator[MaterializedQuery]:
        # Docstring inherited from Query.
        yield self

    def subset(self, *, graph: Optional[DimensionGraph] = None,
               datasets: bool = True,
               unique: bool = False) -> Query:
        # Docstring inherited from Query.
        graph, columns = self._makeSubsetQueryColumns(graph=graph, datasets=datasets, unique=unique)
        if columns is None:
            return self
        simpleQuery = SimpleQuery()
        simpleQuery.join(self._table)
        return DirectQuery(
            simpleQuery=simpleQuery,
            columns=columns,
            uniqueness=DirectQueryUniqueness.NEEDS_DISTINCT if unique else DirectQueryUniqueness.NOT_UNIQUE,
            graph=graph,
            whereRegion=self.whereRegion,
            collections=self._collections,
            datasets=self._datasets,
        )
