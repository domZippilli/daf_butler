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
"""The default concrete implementations of the classes that manage
opaque tables for `Registry`.
"""

__all__ = ["ByNameOpaqueTableStorage", "ByNameOpaqueTableStorageManager"]

from typing import (
    Any,
    ClassVar,
    Dict,
    Iterator,
    Optional,
)

import sqlalchemy

from ..core.ddl import TableSpec, FieldSpec
from .interfaces import Database, OpaqueTableStorageManager, OpaqueTableStorage, StaticTablesContext


class ByNameOpaqueTableStorage(OpaqueTableStorage):
    """An implementation of `OpaqueTableStorage` that simply creates a true
    table for each different named opaque logical table.

    A `ByNameOpaqueTableStorageManager` instance should always be used to
    construct and manage instances of this class.

    Parameters
    ----------
    db : `Database`
        Database engine interface for the namespace in which this table lives.
    name : `str`
        Name of the logical table (also used as the name of the actual table).
    table : `sqlalchemy.schema.Table`
        SQLAlchemy representation of the table, which must have already been
        created in the namespace managed by ``db`` (this is the responsibility
        of `ByNameOpaqueTableStorageManager`).
    """
    def __init__(self, *, db: Database, name: str, table: sqlalchemy.schema.Table):
        super().__init__(name=name)
        self._db = db
        self._table = table

    def insert(self, *data: dict) -> None:
        # Docstring inherited from OpaqueTableStorage.
        self._db.insert(self._table, *data)

    def fetch(self, **where: Any) -> Iterator[dict]:
        # Docstring inherited from OpaqueTableStorage.
        sql = self._table.select().where(
            sqlalchemy.sql.and_(*[self._table.columns[k] == v for k, v in where.items()])
        )
        for row in self._db.query(sql):
            yield dict(row)

    def delete(self, **where: Any) -> None:
        # Docstring inherited from OpaqueTableStorage.
        self._db.delete(self._table, where.keys(), where)


class ByNameOpaqueTableStorageManager(OpaqueTableStorageManager):
    """An implementation of `OpaqueTableStorageManager` that simply creates a
    true table for each different named opaque logical table.

    Instances of this class should generally be constructed via the
    `initialize` class method instead of invoking ``__init__`` directly.

    Parameters
    ----------
    db : `Database`
        Database engine interface for the namespace in which this table lives.
    metaTable : `sqlalchemy.schema.Table`
        SQLAlchemy representation of the table that records which opaque
        logical tables exist.
    """
    def __init__(self, db: Database, metaTable: sqlalchemy.schema.Table):
        self._db = db
        self._metaTable = metaTable
        self._storage: Dict[str, OpaqueTableStorage] = {}

    _META_TABLE_NAME: ClassVar[str] = "opaque_meta"

    _META_TABLE_SPEC: ClassVar[TableSpec] = TableSpec(
        fields=[
            FieldSpec("table_name", dtype=sqlalchemy.String, length=128, primaryKey=True),
        ],
    )

    @classmethod
    def initialize(cls, db: Database, context: StaticTablesContext) -> OpaqueTableStorageManager:
        # Docstring inherited from OpaqueTableStorageManager.
        metaTable = context.addTable(cls._META_TABLE_NAME, cls._META_TABLE_SPEC)
        return cls(db=db, metaTable=metaTable)

    def get(self, name: str) -> Optional[OpaqueTableStorage]:
        # Docstring inherited from OpaqueTableStorageManager.
        return self._storage.get(name)

    def register(self, name: str, spec: TableSpec) -> OpaqueTableStorage:
        # Docstring inherited from OpaqueTableStorageManager.
        result = self._storage.get(name)
        if result is None:
            # Create the table itself.  If it already exists but wasn't in
            # the dict because it was added by another client since this one
            # was initialized, that's fine.
            table = self._db.ensureTableExists(name, spec)
            # Add a row to the meta table so we can find this table in the
            # future.  Also okay if that already exists, so we use sync.
            self._db.sync(self._metaTable, keys={"table_name": name})
            result = ByNameOpaqueTableStorage(name=name, table=table, db=self._db)
            self._storage[name] = result
        return result
