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

__all__ = ("EphemeralDatastoreRegistryBridge",)

from contextlib import contextmanager
from typing import Iterable, Iterator, Set

from lsst.daf.butler import DatasetRef, FakeDatasetRef
from lsst.daf.butler.registry.interfaces import DatastoreRegistryBridge


class EphemeralDatastoreRegistryBridge(DatastoreRegistryBridge):
    """An implementation of `DatastoreRegistryBridge` for ephemeral datastores
    - those whose artifacts never outlive the current process.

    Parameters
    ----------
    datastoreName : `str`
        Name of the `Datastore` as it should appear in `Registry` tables
        referencing it.

    Notes
    -----
    The current implementation just uses a Python set to remember the dataset
    IDs associated with the datastore.  This will probably need to be converted
    to use in-database temporary tables instead in the future to support
    "in-datastore" constraints in `Registry.queryDatasets`.
    """
    def __init__(self, datastoreName: str):
        super().__init__(datastoreName)
        self._datasetIds: Set[int] = set()
        self._trashedIds: Set[int] = set()

    def insert(self, refs: Iterable[DatasetRef]) -> None:
        # Docstring inherited from DatastoreRegistryBridge
        self._datasetIds.update(ref.getCheckedId() for ref in refs)

    def moveToTrash(self, refs: Iterable[DatasetRef]) -> None:
        # Docstring inherited from DatastoreRegistryBridge
        self._trashedIds.update(ref.getCheckedId() for ref in refs)

    def check(self, refs: Iterable[DatasetRef]) -> Iterable[DatasetRef]:
        # Docstring inherited from DatastoreRegistryBridge
        yield from (ref for ref in refs if ref in self)

    def __contains__(self, ref: DatasetRef) -> bool:
        return ref.getCheckedId() in self._datasetIds and ref.getCheckedId() not in self._trashedIds

    @contextmanager
    def emptyTrash(self) -> Iterator[Iterable[FakeDatasetRef]]:
        # Docstring inherited from DatastoreRegistryBridge
        yield (FakeDatasetRef(id) for id in self._trashedIds)
        self._datasetIds.difference_update(self._trashedIds)
        self._trashedIds = set()
