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

__all__ = ("QueryBuilder",)

from typing import Any, Iterable, List, Optional, TYPE_CHECKING

import sqlalchemy.sql

from ...core import (
    DimensionElement,
    SkyPixDimension,
    Dimension,
    DatasetType,
    NamedKeyDict,
    NamedValueSet,
    SimpleQuery,
)

from ._structs import QuerySummary, QueryColumns, DatasetQueryColumns
from .expressions import ClauseVisitor
from ._query import DirectQuery, DirectQueryUniqueness
from ..wildcards import CollectionSearch, CollectionQuery

if TYPE_CHECKING:
    from ..interfaces import CollectionManager, DimensionRecordStorageManager, DatasetRecordStorageManager


class QueryBuilder:
    """A builder for potentially complex queries that join tables based
    on dimension relationships.

    Parameters
    ----------
    summary : `QuerySummary`
        Struct organizing the dimensions involved in the query.
    collections : `CollectionManager`
        Manager object for collection tables.
    dimensions : `DimensionRecordStorageManager`
        Manager for storage backend objects that abstract access to dimension
        tables.
    datasets : `DatasetRegistryStorage`
        Storage backend object that abstracts access to dataset tables.
    """

    def __init__(self, summary: QuerySummary, *,
                 collections: CollectionManager,
                 dimensions: DimensionRecordStorageManager,
                 datasets: DatasetRecordStorageManager):
        self.summary = summary
        self._collections = collections
        self._dimensions = dimensions
        self._datasets = datasets
        self._simpleQuery = SimpleQuery()
        self._elements: NamedKeyDict[DimensionElement, sqlalchemy.sql.FromClause] = NamedKeyDict()
        self._columns = QueryColumns()
        self._isNaturallyUnique = True

    def hasDimensionKey(self, dimension: Dimension) -> bool:
        """Return `True` if the given dimension's primary key column has
        been included in the query (possibly via a foreign key column on some
        other table).
        """
        return dimension in self._columns.keys

    def joinDimensionElement(self, element: DimensionElement) -> None:
        """Add the table for a `DimensionElement` to the query.

        This automatically joins the element table to all other tables in the
        query with which it is related, via both dimension keys and spatial
        and temporal relationships.

        External calls to this method should rarely be necessary; `finish` will
        automatically call it if the `DimensionElement` has been identified as
        one that must be included.

        Parameters
        ----------
        element : `DimensionElement`
            Element for which a table should be added.  The element must be
            associated with a database table (see `DimensionElement.hasTable`).
        """
        assert element not in self._elements, "Element already included in query."
        storage = self._dimensions[element]
        fromClause = storage.join(
            self,
            regions=self._columns.regions if element in self.summary.spatial else None,
            timespans=self._columns.timespans if element in self.summary.temporal else None,
        )
        self._elements[element] = fromClause

    def joinDataset(self, datasetType: DatasetType, collections: Any, *,
                    isResult: bool = True, addRank: bool = False) -> bool:
        """Add a dataset search or constraint to the query.

        Unlike other `QueryBuilder` join methods, this *must* be called
        directly to search for datasets of a particular type or constrain the
        query results based on the exists of datasets.  However, all dimensions
        used to identify the dataset type must have already been included in
        `QuerySummary.requested` when initializing the `QueryBuilder`.

        Parameters
        ----------
        datasetType : `DatasetType`
            The type of datasets to search for.
        collections : `Any`
            An expression that fully or partially identifies the collections
            to search for datasets, such as a `str`, `re.Pattern`, or iterable
            thereof.  `...` can be used to return all collections. See
            :ref:`daf_butler_collection_expressions` for more information.
        isResult : `bool`, optional
            If `True` (default), include the dataset ID column in the
            result columns of the query, allowing complete `DatasetRef`
            instances to be produced from the query results for this dataset
            type.  If `False`, the existence of datasets of this type is used
            only to constrain the data IDs returned by the query.
        addRank : `bool`, optional
            If `True` (`False` is default), also include a calculated column
            that ranks the collection in which the dataset was found (lower
            is better).  Requires that all entries in ``collections`` be
            regular strings, so there is a clear search order.  Ignored if
            ``isResult`` is `False`.

        Returns
        -------
        anyRecords : `bool`
            If `True`, joining the dataset table was successful and the query
            should proceed.  If `False`, we were able to determine (from the
            combination of ``datasetType`` and ``collections``) that there
            would be no results joined in from this dataset, and hence (due to
            the inner join that would normally be present), the full query will
            return no results.
        """
        assert datasetType.dimensions.issubset(self.summary.requested)
        if isResult and addRank:
            collections = CollectionSearch.fromExpression(collections)
        else:
            collections = CollectionQuery.fromExpression(collections)
        datasetRecordStorage = self._datasets.find(datasetType.name)
        if datasetRecordStorage is None:
            # Unrecognized dataset type means no results.  It might be better
            # to raise here, but this is consistent with previous behavior,
            # which is expected by QuantumGraph generation code in pipe_base.
            return False
        subsubqueries = []
        for rank, collectionRecord in enumerate(collections.iter(self._collections, datasetType=datasetType)):
            ssq = datasetRecordStorage.select(collection=collectionRecord,
                                              dataId=SimpleQuery.Select,
                                              id=SimpleQuery.Select if isResult else None,
                                              run=SimpleQuery.Select if isResult else None)
            if ssq is None:
                continue
            if addRank:
                ssq.columns.append(sqlalchemy.sql.literal(rank).label("rank"))
            subsubqueries.append(ssq.combine())
        if not subsubqueries:
            return False
        subquery = sqlalchemy.sql.union_all(*subsubqueries).alias(datasetType.name)
        self.joinTable(subquery, datasetType.dimensions.required)
        if isResult:
            self._columns.datasets[datasetType] = DatasetQueryColumns(
                id=subquery.columns["id"],
                runKey=subquery.columns[self._collections.getRunForeignKeyName()],
                rank=subquery.columns["rank"] if addRank else None
            )
        else:
            # Joining in dataset tables just to constrain the data IDs may
            # result in non-unique result rows, because we could get matches
            # from multiple collections.
            self._isNaturallyUnique = False
        return True

    def joinTable(self, table: sqlalchemy.sql.FromClause, dimensions: NamedValueSet[Dimension]) -> None:
        """Join an arbitrary table to the query via dimension relationships.

        External calls to this method should only be necessary for tables whose
        records represent neither dataset nor dimension elements (i.e.
        extensions to the standard `Registry` schema).

        Parameters
        ----------
        table : `sqlalchemy.sql.FromClause`
            SQLAlchemy object representing the logical table (which may be a
            join or subquery expression) to be joined.
        dimensions : iterable of `Dimension`
            The dimensions that relate this table to others that may be in the
            query.  The table must have columns with the names of the
            dimensions.
        """
        joinOn = self.startJoin(table, dimensions, dimensions.names)
        self.finishJoin(table, joinOn)

    def startJoin(self, table: sqlalchemy.sql.FromClause, dimensions: Iterable[Dimension],
                  columnNames: Iterable[str]
                  ) -> List[sqlalchemy.sql.ColumnElement]:
        """Begin a join on dimensions.

        Must be followed by call to `finishJoin`.

        Parameters
        ----------
        table : `sqlalchemy.sql.FromClause`
            SQLAlchemy object representing the logical table (which may be a
            join or subquery expression) to be joined.
        dimensions : iterable of `Dimension`
            The dimensions that relate this table to others that may be in the
            query.  The table must have columns with the names of the
            dimensions.
        columnNames : iterable of `str`
            Names of the columns that correspond to dimension key values; must
            be `zip` iterable with ``dimensions``.

        Returns
        -------
        joinOn : `list` of `sqlalchemy.sql.ColumnElement`
            Sequence of boolean expressions that should be combined with AND
            to form (part of) the ON expression for this JOIN.
        """
        joinOn = []
        for dimension, columnName in zip(dimensions, columnNames):
            columnInTable = table.columns[columnName]
            columnsInQuery = self._columns.keys.setdefault(dimension, [])
            for columnInQuery in columnsInQuery:
                joinOn.append(columnInQuery == columnInTable)
            columnsInQuery.append(columnInTable)
        return joinOn

    def finishJoin(self, table: sqlalchemy.sql.FromClause, joinOn: List[sqlalchemy.sql.ColumnElement]
                   ) -> None:
        """Complete a join on dimensions.

        Must be preceded by call to `startJoin`.

        Parameters
        ----------
        table : `sqlalchemy.sql.FromClause`
            SQLAlchemy object representing the logical table (which may be a
            join or subquery expression) to be joined.  Must be the same object
            passed to `startJoin`.
        joinOn : `list` of `sqlalchemy.sql.ColumnElement`
            Sequence of boolean expressions that should be combined with AND
            to form (part of) the ON expression for this JOIN.  Should include
            at least the elements of the list returned by `startJoin`.
        """
        onclause: Optional[sqlalchemy.sql.ColumnElement]
        if len(joinOn) == 0:
            onclause = None
        elif len(joinOn) == 1:
            onclause = joinOn[0]
        else:
            onclause = sqlalchemy.sql.and_(*joinOn)
        self._simpleQuery.join(table, onclause=onclause)

    def _joinMissingDimensionElements(self) -> None:
        """Join all dimension element tables that were identified as necessary
        by `QuerySummary` and have not yet been joined.

        For internal use by `QueryBuilder` only; will be called (and should
        only by called) by `finish`.
        """
        # Join all DimensionElement tables that we need for spatial/temporal
        # joins/filters or a nontrivial WHERE expression.
        # We iterate over these in *reverse* topological order to minimize the
        # number of tables joined.  For example, the "visit" table provides
        # the primary key value for the "instrument" table it depends on, so we
        # don't need to join "instrument" as well unless we had a nontrivial
        # expression on it (and hence included it already above).
        for element in self.summary.universe.sorted(self.summary.mustHaveTableJoined, reverse=True):
            self.joinDimensionElement(element)
        # Join in any requested Dimension tables that don't already have their
        # primary keys identified by the query.
        for dimension in self.summary.universe.sorted(self.summary.mustHaveKeysJoined, reverse=True):
            if dimension not in self._columns.keys:
                self.joinDimensionElement(dimension)

    def _addWhereClause(self) -> None:
        """Add a WHERE clause to the query under construction, connecting all
        joined dimensions to the expression and data ID dimensions from
        `QuerySummary`.

        For internal use by `QueryBuilder` only; will be called (and should
        only by called) by `finish`.
        """
        if self.summary.expression.tree is not None:
            visitor = ClauseVisitor(self.summary.universe, self._columns, self._elements)
            self._simpleQuery.where.append(self.summary.expression.tree.visit(visitor))
        for dimension, columnsInQuery in self._columns.keys.items():
            if dimension in self.summary.dataId.graph:
                givenKey = self.summary.dataId[dimension]
                # Add a WHERE term for each column that corresponds to each
                # key.  This is redundant with the JOIN ON clauses that make
                # them equal to each other, but more constraints have a chance
                # of making things easier on the DB's query optimizer.
                for columnInQuery in columnsInQuery:
                    self._simpleQuery.where.append(columnInQuery == givenKey)
            else:
                # Dimension is not fully identified, but it might be a skypix
                # dimension that's constrained by a given region.
                if self.summary.dataId.graph.spatial and isinstance(dimension, SkyPixDimension):
                    # We know the region now.
                    givenSkyPixIds: List[int] = []
                    for begin, end in dimension.pixelization.envelope(self.summary.dataId.region):
                        givenSkyPixIds.extend(range(begin, end))
                    for columnInQuery in columnsInQuery:
                        self._simpleQuery.where.append(columnInQuery.in_(givenSkyPixIds))
        # If we are given an dataId with a timespan, and there are one or more
        # timespans in the query that aren't given, add a WHERE expression for
        # each of them.
        if self.summary.dataId.graph.temporal and self.summary.temporal:
            # Timespan is known now.
            givenInterval = self.summary.dataId.timespan
            assert givenInterval is not None
            for element, intervalInQuery in self._columns.timespans.items():
                assert element not in self.summary.dataId.graph.elements
                self._simpleQuery.where.append(intervalInQuery.overlaps(givenInterval, ops=sqlalchemy.sql))

    def finish(self) -> DirectQuery:
        """Finish query constructing, returning a new `Query` instance.

        This automatically joins any missing dimension element tables
        (according to the categorization of the `QuerySummary` the builder was
        constructed with).

        This consumes the `QueryBuilder`; no other methods should be called
        after this one.

        Returns
        -------
        query : `Query`
            A `Query` object that can be executed (possibly multiple times
            with different bind parameter values) and used to interpret result
            rows.
        """
        self._joinMissingDimensionElements()
        self._addWhereClause()
        uniqueness = (DirectQueryUniqueness.NATURALLY_UNIQUE if self._isNaturallyUnique
                      else DirectQueryUniqueness.NOT_UNIQUE)
        return DirectQuery(graph=self.summary.requested,
                           uniqueness=uniqueness,
                           whereRegion=self.summary.dataId.region,
                           simpleQuery=self._simpleQuery,
                           columns=self._columns,
                           collections=self._collections,
                           datasets=self._datasets)
