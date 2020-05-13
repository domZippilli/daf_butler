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
    "Registry",
)

from collections import defaultdict
import contextlib
import sys
from typing import (
    Any,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Type,
    TYPE_CHECKING,
    Union,
)

import sqlalchemy

from ..core import (
    Config,
    DataCoordinate,
    DataId,
    DatasetRef,
    DatasetType,
    Dimension,
    DimensionElement,
    DimensionGraph,
    DimensionRecord,
    DimensionUniverse,
    ExpandedDataCoordinate,
    StorageClassFactory,
)
from ..core import ddl
from ..core.utils import doImport, iterable, transactional
from ._config import RegistryConfig
from .queries import (
    QueryBuilder,
    QuerySummary,
)
from .tables import makeRegistryTableSpecs
from ._collectionType import CollectionType
from ._dimensionRecordCache import ConsistentDataIds, DimensionRecordCache
from ._exceptions import ConflictingDefinitionError, OrphanedRecordError
from .wildcards import CategorizedWildcard, CollectionQuery, CollectionSearch

if TYPE_CHECKING:
    from ..butlerConfig import ButlerConfig
    from ..core import (
        Quantum
    )
    from .interfaces import (
        CollectionManager,
        Database,
        OpaqueTableStorageManager,
        DimensionRecordStorageManager,
        DatasetRecordStorageManager,
        DatastoreRegistryBridgeManager,
    )


class Registry:
    """Registry interface.

    Parameters
    ----------
    config : `ButlerConfig`, `RegistryConfig`, `Config` or `str`
        Registry configuration
    """

    defaultConfigFile = None
    """Path to configuration defaults. Relative to $DAF_BUTLER_DIR/config or
    absolute path. Can be None if no defaults specified.
    """

    @classmethod
    def fromConfig(cls, config: Union[ButlerConfig, RegistryConfig, Config, str], create: bool = False,
                   butlerRoot: Optional[str] = None, writeable: bool = True) -> Registry:
        """Create `Registry` subclass instance from `config`.

        Uses ``registry.cls`` from `config` to determine which subclass to
        instantiate.

        Parameters
        ----------
        config : `ButlerConfig`, `RegistryConfig`, `Config` or `str`
            Registry configuration
        create : `bool`, optional
            Assume empty Registry and create a new one.
        butlerRoot : `str`, optional
            Path to the repository root this `Registry` will manage.
        writeable : `bool`, optional
            If `True` (default) create a read-write connection to the database.

        Returns
        -------
        registry : `Registry` (subclass)
            A new `Registry` subclass instance.
        """
        if not isinstance(config, RegistryConfig):
            if isinstance(config, str) or isinstance(config, Config):
                config = RegistryConfig(config)
            else:
                raise ValueError("Incompatible Registry configuration: {}".format(config))
        config.replaceRoot(butlerRoot)
        DatabaseClass = config.getDatabaseClass()
        database = DatabaseClass.fromUri(str(config.connectionString), origin=config.get("origin", 0),
                                         namespace=config.get("namespace"), writeable=writeable)
        universe = DimensionUniverse(config)
        opaque = doImport(config["managers", "opaque"])
        dimensions = doImport(config["managers", "dimensions"])
        collections = doImport(config["managers", "collections"])
        datasets = doImport(config["managers", "datasets"])
        datastoreBridges = doImport(config["managers", "datastores"])
        return cls(database, universe, dimensions=dimensions, opaque=opaque, collections=collections,
                   datasets=datasets, datastoreBridges=datastoreBridges, create=create)

    def __init__(self, database: Database, universe: DimensionUniverse, *,
                 opaque: Type[OpaqueTableStorageManager],
                 dimensions: Type[DimensionRecordStorageManager],
                 collections: Type[CollectionManager],
                 datasets: Type[DatasetRecordStorageManager],
                 datastoreBridges: Type[DatastoreRegistryBridgeManager],
                 create: bool = False):
        self._db = database
        self.storageClasses = StorageClassFactory()
        with self._db.declareStaticTables(create=create) as context:
            self._dimensions = dimensions.initialize(self._db, context, universe=universe)
            self._collections = collections.initialize(self._db, context)
            self._datasets = datasets.initialize(self._db, context,
                                                 collections=self._collections,
                                                 universe=self.dimensions)
            self._opaque = opaque.initialize(self._db, context)
            self._datastoreBridges = datastoreBridges.initialize(self._db, context,
                                                                 opaque=self._opaque,
                                                                 datasets=datasets,
                                                                 universe=self.dimensions)
            self._tables = context.addTableTuple(makeRegistryTableSpecs(self.dimensions,
                                                                        self._collections,
                                                                        self._datasets))
        self._collections.refresh()
        self._datasets.refresh(universe=self._dimensions.universe)

    def __str__(self) -> str:
        return str(self._db)

    def __repr__(self) -> str:
        return f"Registry({self._db!r}, {self.dimensions!r})"

    def isWriteable(self) -> bool:
        """Return `True` if this registry allows write operations, and `False`
        otherwise.
        """
        return self._db.isWriteable()

    @property
    def dimensions(self) -> DimensionUniverse:
        """All dimensions recognized by this `Registry` (`DimensionUniverse`).
        """
        return self._dimensions.universe

    @contextlib.contextmanager
    def transaction(self):
        """Return a context manager that represents a transaction.
        """
        # TODO make savepoint=False the default.
        try:
            with self._db.transaction():
                yield
        except BaseException:
            # TODO: this clears the caches sometimes when we wouldn't actually
            # need to.  Can we avoid that?
            self._dimensions.clearCaches()
            raise

    def registerOpaqueTable(self, tableName: str, spec: ddl.TableSpec):
        """Add an opaque (to the `Registry`) table for use by a `Datastore` or
        other data repository client.

        Opaque table records can be added via `insertOpaqueData`, retrieved via
        `fetchOpaqueData`, and removed via `deleteOpaqueData`.

        Parameters
        ----------
        tableName : `str`
            Logical name of the opaque table.  This may differ from the
            actual name used in the database by a prefix and/or suffix.
        spec : `ddl.TableSpec`
            Specification for the table to be added.
        """
        self._opaque.register(tableName, spec)

    @transactional
    def insertOpaqueData(self, tableName: str, *data: dict):
        """Insert records into an opaque table.

        Parameters
        ----------
        tableName : `str`
            Logical name of the opaque table.  Must match the name used in a
            previous call to `registerOpaqueTable`.
        data
            Each additional positional argument is a dictionary that represents
            a single row to be added.
        """
        self._opaque[tableName].insert(*data)

    def fetchOpaqueData(self, tableName: str, **where: Any) -> Iterator[dict]:
        """Retrieve records from an opaque table.

        Parameters
        ----------
        tableName : `str`
            Logical name of the opaque table.  Must match the name used in a
            previous call to `registerOpaqueTable`.
        where
            Additional keyword arguments are interpreted as equality
            constraints that restrict the returned rows (combined with AND);
            keyword arguments are column names and values are the values they
            must have.

        Yields
        ------
        row : `dict`
            A dictionary representing a single result row.
        """
        yield from self._opaque[tableName].fetch(**where)

    @transactional
    def deleteOpaqueData(self, tableName: str, **where: Any):
        """Remove records from an opaque table.

        Parameters
        ----------
        tableName : `str`
            Logical name of the opaque table.  Must match the name used in a
            previous call to `registerOpaqueTable`.
        where
            Additional keyword arguments are interpreted as equality
            constraints that restrict the deleted rows (combined with AND);
            keyword arguments are column names and values are the values they
            must have.
        """
        self._opaque[tableName].delete(**where)

    def registerCollection(self, name: str, type: CollectionType = CollectionType.TAGGED):
        """Add a new collection if one with the given name does not exist.

        Parameters
        ----------
        name : `str`
            The name of the collection to create.
        type : `CollectionType`
            Enum value indicating the type of collection to create.

        Notes
        -----
        This method cannot be called within transactions, as it needs to be
        able to perform its own transaction to be concurrent.
        """
        self._collections.register(name, type)

    def getCollectionType(self, name: str) -> CollectionType:
        """Return an enumeration value indicating the type of the given
        collection.

        Parameters
        ----------
        name : `str`
            The name of the collection.

        Returns
        -------
        type : `CollectionType`
            Enum value indicating the type of this collection.

        Raises
        ------
        MissingCollectionError
            Raised if no collection with the given name exists.
        """
        return self._collections.find(name).type

    def registerRun(self, name: str):
        """Add a new run if one with the given name does not exist.

        Parameters
        ----------
        name : `str`
            The name of the run to create.

        Notes
        -----
        This method cannot be called within transactions, as it needs to be
        able to perform its own transaction to be concurrent.
        """
        self._collections.register(name, CollectionType.RUN)

    @transactional
    def removeCollection(self, name: str):
        """Completely remove the given collection.

        Parameters
        ----------
        name : `str`
            The name of the collection to remove.

        Raises
        ------
        MissingCollectionError
            Raised if no collection with the given name exists.

        Notes
        -----
        If this is a `~CollectionType.RUN` collection, all datasets and quanta
        in it are also fully removed.  This requires that those datasets be
        removed (or at least trashed) from any datastores that hold them first.

        A collection may not be deleted as long as it is referenced by a
        `~CollectionType.CHAINED` collection; the ``CHAINED`` collection must
        be deleted or redefined first.
        """
        self._collections.remove(name)

    def getCollectionChain(self, parent: str) -> CollectionSearch:
        """Return the child collections in a `~CollectionType.CHAINED`
        collection.

        Parameters
        ----------
        parent : `str`
            Name of the chained collection.  Must have already been added via
            a call to `Registry.registerCollection`.

        Returns
        -------
        children : `CollectionSearch`
            An object that defines the search path of the collection.
            See :ref:`daf_butler_collection_expressions` for more information.

        Raises
        ------
        MissingCollectionError
            Raised if ``parent`` does not exist in the `Registry`.
        TypeError
            Raised if ``parent`` does not correspond to a
            `~CollectionType.CHAINED` collection.
        """
        record = self._collections.find(parent)
        if record.type is not CollectionType.CHAINED:
            raise TypeError(f"Collection '{parent}' has type {record.type.name}, not CHAINED.")
        return record.children

    @transactional
    def setCollectionChain(self, parent: str, children: Any):
        """Define or redefine a `~CollectionType.CHAINED` collection.

        Parameters
        ----------
        parent : `str`
            Name of the chained collection.  Must have already been added via
            a call to `Registry.registerCollection`.
        children : `Any`
            An expression defining an ordered search of child collections,
            generally an iterable of `str`.  Restrictions on the dataset types
            to be searched can also be included, by passing mapping or an
            iterable containing tuples; see
            :ref:`daf_butler_collection_expressions` for more information.

        Raises
        ------
        MissingCollectionError
            Raised when any of the given collections do not exist in the
            `Registry`.
        TypeError
            Raised if ``parent`` does not correspond to a
            `~CollectionType.CHAINED` collection.
        ValueError
            Raised if the given collections contains a cycle.
        """
        record = self._collections.find(parent)
        if record.type is not CollectionType.CHAINED:
            raise TypeError(f"Collection '{parent}' has type {record.type.name}, not CHAINED.")
        children = CollectionSearch.fromExpression(children)
        if children != record.children:
            record.update(self._collections, children)

    def registerDatasetType(self, datasetType: DatasetType) -> bool:
        """
        Add a new `DatasetType` to the Registry.

        It is not an error to register the same `DatasetType` twice.

        Parameters
        ----------
        datasetType : `DatasetType`
            The `DatasetType` to be added.

        Returns
        -------
        inserted : `bool`
            `True` if ``datasetType`` was inserted, `False` if an identical
            existing `DatsetType` was found.  Note that in either case the
            DatasetType is guaranteed to be defined in the Registry
            consistently with the given definition.

        Raises
        ------
        ValueError
            Raised if the dimensions or storage class are invalid.
        ConflictingDefinitionError
            Raised if this DatasetType is already registered with a different
            definition.

        Notes
        -----
        This method cannot be called within transactions, as it needs to be
        able to perform its own transaction to be concurrent.
        """
        _, inserted = self._datasets.register(datasetType)
        return inserted

    def getDatasetType(self, name: str) -> DatasetType:
        """Get the `DatasetType`.

        Parameters
        ----------
        name : `str`
            Name of the type.

        Returns
        -------
        type : `DatasetType`
            The `DatasetType` associated with the given name.

        Raises
        ------
        KeyError
            Requested named DatasetType could not be found in registry.
        """
        storage = self._datasets.find(name)
        if storage is None:
            raise KeyError(f"DatasetType '{name}' could not be found.")
        return storage.datasetType

    def findDataset(self, datasetType: Union[DatasetType, str], dataId: Optional[DataId] = None, *,
                    collections: Any, **kwargs: Any) -> Optional[DatasetRef]:
        """Find a dataset given its `DatasetType` and data ID.

        This can be used to obtain a `DatasetRef` that permits the dataset to
        be read from a `Datastore`. If the dataset is a component and can not
        be found using the provided dataset type, a dataset ref for the parent
        will be returned instead but with the correct dataset type.

        Parameters
        ----------
        datasetType : `DatasetType` or `str`
            A `DatasetType` or the name of one.
        dataId : `dict` or `DataCoordinate`, optional
            A `dict`-like object containing the `Dimension` links that identify
            the dataset within a collection.
        collections
            An expression that fully or partially identifies the collections
            to search for the dataset, such as a `str`, `re.Pattern`, or
            iterable  thereof.  `...` can be used to return all collections.
            See :ref:`daf_butler_collection_expressions` for more information.
        **kwargs
            Additional keyword arguments passed to
            `DataCoordinate.standardize` to convert ``dataId`` to a true
            `DataCoordinate` or augment an existing one.

        Returns
        -------
        ref : `DatasetRef`
            A reference to the dataset, or `None` if no matching Dataset
            was found.

        Raises
        ------
        LookupError
            Raised if one or more data ID keys are missing or the dataset type
            does not exist.
        MissingCollectionError
            Raised if any of ``collections`` does not exist in the registry.
        """
        if isinstance(datasetType, DatasetType):
            storage = self._datasets.find(datasetType.name)
            if storage is None:
                raise LookupError(f"DatasetType '{datasetType}' has not been registered.")
        else:
            storage = self._datasets.find(datasetType)
            if storage is None:
                raise LookupError(f"DatasetType with name '{datasetType}' has not been registered.")
        dataId = DataCoordinate.standardize(dataId, graph=storage.datasetType.dimensions,
                                            universe=self.dimensions, **kwargs)
        collections = CollectionSearch.fromExpression(collections)
        for collectionRecord in collections.iter(self._collections, datasetType=storage.datasetType):
            result = storage.find(collectionRecord, dataId)
            if result is not None:
                if result.datasetType.isComposite():
                    result = self._datasets.fetchComponents(result)
                return result

        # fallback to the parent if we got nothing and this was a component
        if storage.datasetType.isComponent():
            parentType, _ = storage.datasetType.nameAndComponent()
            parentRef = self.findDataset(parentType, dataId, collections=collections, **kwargs)
            if parentRef is not None:
                # Should already conform and we know no components
                return DatasetRef(storage.datasetType, parentRef.dataId, id=parentRef.id,
                                  run=parentRef.run, conform=False, hasParentId=True)

        return None

    @transactional
    def insertDatasets(self, datasetType: Union[DatasetType, str], dataIds: Iterable[DataId],
                       run: str, *, producer: Optional[Quantum] = None, recursive: bool = False
                       ) -> List[DatasetRef]:
        """Insert one or more datasets into the `Registry`

        This always adds new datasets; to associate existing datasets with
        a new collection, use ``associate``.

        Parameters
        ----------
        datasetType : `DatasetType` or `str`
            A `DatasetType` or the name of one.
        dataIds :  `~collections.abc.Iterable` of `dict` or `DataCoordinate`
            Dimension-based identifiers for the new datasets.
        run : `str`
            The name of the run that produced the datasets.
        producer : `Quantum`
            Unit of work that produced the datasets.  May be `None` to store
            no provenance information, but if present the `Quantum` must
            already have been added to the Registry.
        recursive : `bool`
            If True, recursively add datasets and attach entries for component
            datasets as well.

        Returns
        -------
        refs : `list` of `DatasetRef`
            Resolved `DatasetRef` instances for all given data IDs (in the same
            order).

        Raises
        ------
        ConflictingDefinitionError
            If a dataset with the same dataset type and data ID as one of those
            given already exists in ``run``.
        MissingCollectionError
            Raised if ``run`` does not exist in the registry.
        """
        if isinstance(datasetType, DatasetType):
            storage = self._datasets.find(datasetType.name)
            if storage is None:
                raise LookupError(f"DatasetType '{datasetType}' has not been registered.")
        else:
            storage = self._datasets.find(datasetType)
            if storage is None:
                raise LookupError(f"DatasetType with name '{datasetType}' has not been registered.")
        runRecord = self._collections.find(run)
        dataIds = [self.expandDataId(dataId, graph=storage.datasetType.dimensions) for dataId in dataIds]
        try:
            refs = list(storage.insert(runRecord, dataIds, quantum=producer))
        except sqlalchemy.exc.IntegrityError as err:
            raise ConflictingDefinitionError(f"A database constraint failure was triggered by inserting "
                                             f"one or more datasets of type {storage.datasetType} into "
                                             f"collection '{run}'. "
                                             f"This probably means a dataset with the same data ID "
                                             f"and dataset type already exists, but it may also mean a "
                                             f"dimension row is missing.") from err
        if recursive and storage.datasetType.isComposite():
            # Insert component rows by recursing.
            composites = defaultdict(dict)
            # TODO: we really shouldn't be inserting all components defined by
            # the storage class, because there's no guarantee all of them are
            # actually present in these datasets.
            for componentName in storage.datasetType.storageClass.components:
                componentDatasetType = storage.datasetType.makeComponentDatasetType(componentName)
                componentRefs = self.insertDatasets(componentDatasetType,
                                                    dataIds=dataIds,
                                                    run=run,
                                                    producer=producer,
                                                    recursive=True)
                for parentRef, componentRef in zip(refs, componentRefs):
                    composites[parentRef][componentName] = componentRef
            if composites:
                refs = list(self._datasets.attachComponents(composites.items()))
        return refs

    def getDataset(self, id: int) -> Optional[DatasetRef]:
        """Retrieve a Dataset entry.

        Parameters
        ----------
        id : `int`
            The unique identifier for the dataset.

        Returns
        -------
        ref : `DatasetRef` or `None`
            A ref to the Dataset, or `None` if no matching Dataset
            was found.
        """
        ref = self._datasets.getDatasetRef(id)
        if ref is None:
            return None
        if ref.datasetType.isComposite():
            return self._datasets.fetchComponents(ref)
        return ref

    @transactional
    def removeDatasets(self, refs: Iterable[DatasetRef], *, recursive: bool = True):
        """Remove datasets from the Registry.

        The datasets will be removed unconditionally from all collections, and
        any `Quantum` that consumed this dataset will instead be marked with
        having a NULL input.  `Datastore` records will *not* be deleted; the
        caller is responsible for ensuring that the dataset has already been
        removed from all Datastores.

        Parameters
        ----------
        refs : `Iterable` of `DatasetRef`
            References to the datasets to be removed.  Must include a valid
            ``id`` attribute, and should be considered invalidated upon return.
        recursive : `bool`, optional
            If `True`, remove all component datasets as well.  Note that
            this only removes components that are actually included in the
            given `DatasetRef` instances, which may not be the same as those in
            the database (especially if they were obtained from
            `queryDatasets`, which does not populate `DatasetRef.components`).

        Raises
        ------
        AmbiguousDatasetError
            Raised if any ``ref.id`` is `None`.
        OrphanedRecordError
            Raised if any dataset is still present in any `Datastore`.
        """
        for datasetType, refsForType in DatasetRef.groupByType(refs, recursive=recursive).items():
            storage = self._datasets.find(datasetType.name)
            try:
                storage.delete(refsForType)
            except sqlalchemy.exc.IntegrityError as err:
                raise OrphanedRecordError("One or more datasets is still "
                                          "present in one or more Datastores.") from err

    @transactional
    def attachComponents(self, parent: DatasetRef, components: Mapping[str, DatasetRef]):
        """Attach components to a dataset.

        Parameters
        ----------
        parent : `DatasetRef`
            A reference to the parent dataset.
        components : `Mapping` [ `str`, `DatasetRef` ]
            Mapping from component name to the `DatasetRef` for that component.

        Returns
        -------
        ref : `DatasetRef`
            An updated version of ``parent`` with components included.

        Raises
        ------
        AmbiguousDatasetError
            Raised if ``parent.id`` or any `DatasetRef.id` in ``components``
            is `None`.
        """
        for name, ref in components.items():
            if ref.datasetType.storageClass != parent.datasetType.storageClass.components[name]:
                raise TypeError(f"Expected storage class "
                                f"'{parent.datasetType.storageClass.components[name].name}' "
                                f"for component '{name}' of dataset {parent}; got "
                                f"dataset {ref} with storage class "
                                f"'{ref.datasetType.storageClass.name}'.")
        ref, = self._datasets.attachComponents([(parent, components)])
        return ref

    @transactional
    def associate(self, collection: str, refs: Iterable[DatasetRef], *, recursive: bool = True):
        """Add existing datasets to a `~CollectionType.TAGGED` collection.

        If a DatasetRef with the same exact integer ID is already in a
        collection nothing is changed. If a `DatasetRef` with the same
        `DatasetType` and data ID but with different integer ID
        exists in the collection, `ConflictingDefinitionError` is raised.

        Parameters
        ----------
        collection : `str`
            Indicates the collection the datasets should be associated with.
        refs : `Iterable` [ `DatasetRef` ]
            An iterable of resolved `DatasetRef` instances that already exist
            in this `Registry`.
        recursive : `bool`, optional
            If `True`, associate all component datasets as well.  Note that
            this only associates components that are actually included in the
            given `DatasetRef` instances, which may not be the same as those in
            the database (especially if they were obtained from
            `queryDatasets`, which does not populate `DatasetRef.components`).

        Raises
        ------
        ConflictingDefinitionError
            If a Dataset with the given `DatasetRef` already exists in the
            given collection.
        AmbiguousDatasetError
            Raised if ``any(ref.id is None for ref in refs)``.
        MissingCollectionError
            Raised if ``collection`` does not exist in the registry.
        TypeError
            Raise adding new datasets to the given ``collection`` is not
            allowed.
        """
        collectionRecord = self._collections.find(collection)
        if collectionRecord.type is not CollectionType.TAGGED:
            raise TypeError(f"Collection '{collection}' has type {collectionRecord.type.name}, not TAGGED.")
        for datasetType, refsForType in DatasetRef.groupByType(refs, recursive=recursive).items():
            storage = self._datasets.find(datasetType.name)
            try:
                storage.associate(collectionRecord, refsForType)
            except sqlalchemy.exc.IntegrityError as err:
                raise ConflictingDefinitionError(
                    f"Constraint violation while associating dataset of type {datasetType.name} with "
                    f"collection {collection}.  This probably means that one or more datasets with the same "
                    f"dataset type and data ID already exist in the collection, but it may also indicate "
                    f"that the datasets do not exist."
                ) from err

    @transactional
    def disassociate(self, collection: str, refs: Iterable[DatasetRef], *, recursive: bool = True):
        """Remove existing datasets from a `~CollectionType.TAGGED` collection.

        ``collection`` and ``ref`` combinations that are not currently
        associated are silently ignored.

        Parameters
        ----------
        collection : `str`
            The collection the datasets should no longer be associated with.
        refs : `Iterable` [ `DatasetRef` ]
            An iterable of resolved `DatasetRef` instances that already exist
            in this `Registry`.
        recursive : `bool`, optional
            If `True`, disassociate all component datasets as well.  Note that
            this only disassociates components that are actually included in
            the given `DatasetRef` instances, which may not be the same as
            those in the database (especially if they were obtained from
            `queryDatasets`, which does not populate `DatasetRef.components`).

        Raises
        ------
        AmbiguousDatasetError
            Raised if any of the given dataset references is unresolved.
        MissingCollectionError
            Raised if ``collection`` does not exist in the registry.
        TypeError
            Raise adding new datasets to the given ``collection`` is not
            allowed.
        """
        collectionRecord = self._collections.find(collection)
        if collectionRecord.type is not CollectionType.TAGGED:
            raise TypeError(f"Collection '{collection}' has type {collectionRecord.type.name}; "
                            "expected TAGGED.")
        for datasetType, refsForType in DatasetRef.groupByType(refs, recursive=recursive).items():
            storage = self._datasets.find(datasetType.name)
            storage.disassociate(collectionRecord, refsForType)

    def getDatastoreBridgeManager(self) -> DatastoreRegistryBridgeManager:
        # TODO docs
        return self._datastoreBridges

    def getDatasetLocations(self, ref: DatasetRef) -> Iterator[str]:
        """Retrieve datastore locations for a given dataset.

        Typically used by `Datastore`.

        Parameters
        ----------
        ref : `DatasetRef`
            A reference to the dataset for which to retrieve storage
            information.

        Returns
        -------
        datastores : `Iterable` [ `str` ]
            All the matching datastores holding this dataset.

        Raises
        ------
        AmbiguousDatasetError
            Raised if ``ref.id`` is `None`.
        """
        return self._datastoreBridges.findDatastores(ref)

    @contextlib.contextmanager
    def cachedDimensions(self) -> Iterator[DimensionRecordCache]:
        """Return a context manager that supports `expandDataId` and
        `relateDataIds` operations while caching all `DimensionRecord`
        instances it sees.

        Returns
        -------
        cache : `DimensionRecordCache`
            Object that provides caching versions of the `relateDataIds` and
            `expandDataId` methods.
        """
        yield DimensionRecordCache(self._dimensions)

    def relateDataIds(self, a: DataId, b: DataId) -> Optional[ConsistentDataIds]:
        """Compare the keys and values of a pair of data IDs for consistency.

        See `ConsistentDataIds` for more information.

        Parameters
        ----------
        a : `dict` or `DataCoordinate`
            First data ID to be compared.
        b : `dict` or `DataCoordinate`
            Second data ID to be compared.

        Returns
        -------
        relationship : `ConsistentDataIds` or `None`
            Relationship information.  This is not `None` and coerces to
            `True` in boolean contexts if and only if the data IDs are
            consistent in terms of all common key-value pairs, all many-to-many
            join tables, and all spatial andtemporal relationships.
        """
        with self.cachedDimensions() as cache:
            return cache.relateDataIds(a, b)

    def expandDataId(self, dataId: Optional[DataId] = None, *, graph: Optional[DimensionGraph] = None,
                     records: Optional[Mapping[DimensionElement, DimensionRecord]] = None,
                     **kwargs: Any) -> ExpandedDataCoordinate:
        """Expand a dimension-based data ID to include additional information.

        Parameters
        ----------
        dataId : `DataCoordinate` or `dict`, optional
            Data ID to be expanded; augmented and overridden by ``kwds``.
        graph : `DimensionGraph`, optional
            Set of dimensions for the expanded ID.  If `None`, the dimensions
            will be inferred from the keys of ``dataId`` and ``kwds``.
            Dimensions that are in ``dataId`` or ``kwds`` but not in ``graph``
            are silently ignored, providing a way to extract and expand a
            subset of a data ID.
        records : mapping [`DimensionElement`, `DimensionRecord`], optional
            Dimension record data to use before querying the database for that
            data.
        **kwds
            Additional keywords are treated like additional key-value pairs for
            ``dataId``, extending and overriding

        Returns
        -------
        expanded : `ExpandedDataCoordinate`
            A data ID that includes full metadata for all of the dimensions it
            identifieds.
        """
        with self.cachedDimensions() as cache:
            return cache.expandDataId(dataId, graph=graph, records=records, **kwargs)

    def insertDimensionData(self, element: Union[DimensionElement, str],
                            *data: Union[dict, DimensionRecord],
                            conform: bool = True):
        """Insert one or more dimension records into the database.

        Parameters
        ----------
        element : `DimensionElement` or `str`
            The `DimensionElement` or name thereof that identifies the table
            records will be inserted into.
        data : `dict` or `DimensionRecord` (variadic)
            One or more records to insert.
        conform : `bool`, optional
            If `False` (`True` is default) perform no checking or conversions,
            and assume that ``element`` is a `DimensionElement` instance and
            ``data`` is a one or more `DimensionRecord` instances of the
            appropriate subclass.
        """
        if conform:
            element = self.dimensions[element]  # if this is a name, convert it to a true DimensionElement.
            records = [element.RecordClass.fromDict(row) if not type(row) is element.RecordClass else row
                       for row in data]
        else:
            records = data
        storage = self._dimensions[element]
        storage.insert(*records)

    def syncDimensionData(self, element: Union[DimensionElement, str],
                          row: Union[dict, DimensionRecord],
                          conform: bool = True) -> bool:
        """Synchronize the given dimension record with the database, inserting
        if it does not already exist and comparing values if it does.

        Parameters
        ----------
        element : `DimensionElement` or `str`
            The `DimensionElement` or name thereof that identifies the table
            records will be inserted into.
        row : `dict` or `DimensionRecord`
           The record to insert.
        conform : `bool`, optional
            If `False` (`True` is default) perform no checking or conversions,
            and assume that ``element`` is a `DimensionElement` instance and
            ``data`` is a one or more `DimensionRecord` instances of the
            appropriate subclass.

        Returns
        -------
        inserted : `bool`
            `True` if a new row was inserted, `False` otherwise.

        Raises
        ------
        ConflictingDefinitionError
            Raised if the record exists in the database (according to primary
            key lookup) but is inconsistent with the given one.

        Notes
        -----
        This method cannot be called within transactions, as it needs to be
        able to perform its own transaction to be concurrent.
        """
        if conform:
            element = self.dimensions[element]  # if this is a name, convert it to a true DimensionElement.
            record = element.RecordClass.fromDict(row) if not type(row) is element.RecordClass else row
        else:
            record = row
        storage = self._dimensions[element]
        return storage.sync(record)

    def queryDatasetTypes(self, expression: Any = ...) -> Iterator[DatasetType]:
        """Iterate over the dataset types whose names match an expression.

        Parameters
        ----------
        expression : `Any`, optional
            An expression that fully or partially identifies the dataset types
            to return, such as a `str`, `re.Pattern`, or iterable thereof.
            `...` can be used to return all dataset types, and is the default.
            See :ref:`daf_butler_dataset_type_expressions` for more
            information.

        Yields
        ------
        datasetType : `DatasetType`
            A `DatasetType` instance whose name matches ``expression``.
        """
        wildcard = CategorizedWildcard.fromExpression(expression, coerceUnrecognized=lambda d: d.name)
        if wildcard is ...:
            yield from self._datasets
            return
        done = set()
        for name in wildcard.strings:
            storage = self._datasets.find(name)
            if storage is not None:
                done.add(storage.datasetType)
                yield storage.datasetType
        if wildcard.patterns:
            for datasetType in self._datasets:
                if datasetType.name in done:
                    continue
                if any(p.fullmatch(datasetType.name) for p in wildcard.patterns):
                    yield datasetType

    def queryCollections(self, expression: Any = ...,
                         datasetType: Optional[DatasetType] = None,
                         collectionType: Optional[CollectionType] = None,
                         flattenChains: bool = False,
                         includeChains: Optional[bool] = None) -> Iterator[str]:
        """Iterate over the collections whose names match an expression.

        Parameters
        ----------
        expression : `Any`, optional
            An expression that fully or partially identifies the collections
            to return, such as a `str`, `re.Pattern`, or iterable thereof.
            `...` can be used to return all collections, and is the default.
            See :ref:`daf_butler_collection_expressions` for more
            information.
        datasetType : `DatasetType`, optional
            If provided, only yield collections that should be searched for
            this dataset type according to ``expression``.  If this is
            not provided, any dataset type restrictions in ``expression`` are
            ignored.
        collectionType : `CollectionType`, optional
            If provided, only yield collections of this type.
        flattenChains : `bool`, optional
            If `True` (`False` is default), recursively yield the child
            collections of matching `~CollectionType.CHAINED` collections.
        includeChains : `bool`, optional
            If `True`, yield records for matching `~CollectionType.CHAINED`
            collections.  Default is the opposite of ``flattenChains``: include
            either CHAINED collections or their children, but not both.

        Yields
        ------
        collection : `str`
            The name of a collection that matches ``expression``.
        """
        query = CollectionQuery.fromExpression(expression)
        for record in query.iter(self._collections, datasetType=datasetType, collectionType=collectionType,
                                 flattenChains=flattenChains, includeChains=includeChains):
            yield record.name

    def makeQueryBuilder(self, summary: QuerySummary) -> QueryBuilder:
        """Return a `QueryBuilder` instance capable of constructing and
        managing more complex queries than those obtainable via `Registry`
        interfaces.

        This is an advanced interface; downstream code should prefer
        `Registry.queryDimensions` and `Registry.queryDatasets` whenever those
        are sufficient.

        Parameters
        ----------
        summary : `QuerySummary`
            Object describing and categorizing the full set of dimensions that
            will be included in the query.

        Returns
        -------
        builder : `QueryBuilder`
            Object that can be used to construct and perform advanced queries.
        """
        return QueryBuilder(summary=summary,
                            collections=self._collections,
                            dimensions=self._dimensions,
                            datasets=self._datasets)

    def queryDimensions(self, dimensions: Union[Iterable[Union[Dimension, str]], Dimension, str], *,
                        dataId: Optional[DataId] = None,
                        datasets: Any = None,
                        collections: Any = None,
                        where: Optional[str] = None,
                        expand: bool = True,
                        **kwds) -> Iterator[DataCoordinate]:
        """Query for and iterate over data IDs matching user-provided criteria.

        Parameters
        ----------
        dimensions : `Dimension` or `str`, or iterable thereof
            The dimensions of the data IDs to yield, as either `Dimension`
            instances or `str`.  Will be automatically expanded to a complete
            `DimensionGraph`.
        dataId : `dict` or `DataCoordinate`, optional
            A data ID whose key-value pairs are used as equality constraints
            in the query.
        datasets : `Any`, optional
            An expression that fully or partially identifies dataset types
            that should constrain the yielded data IDs.  For example, including
            "raw" here would constrain the yielded ``instrument``,
            ``exposure``, ``detector``, and ``physical_filter`` values to only
            those for which at least one "raw" dataset exists in
            ``collections``.  Allowed types include `DatasetType`, `str`,
            `re.Pattern`, and iterables thereof.  Unlike other dataset type
            expressions, `...` is not permitted - it doesn't make sense to
            constrain data IDs on the existence of *all* datasets.
            See :ref:`daf_butler_dataset_type_expressions` for more
            information.
        collections: `Any`, optional
            An expression that fully or partially identifies the collections
            to search for datasets, such as a `str`, `re.Pattern`, or iterable
            thereof.  `...` can be used to return all collections.  Must be
            provided if ``datasets`` is, and is ignored if it is not.  See
            :ref:`daf_butler_collection_expressions` for more information.
        where : `str`, optional
            A string expression similar to a SQL WHERE clause.  May involve
            any column of a dimension table or (as a shortcut for the primary
            key column of a dimension table) dimension name.  See
            :ref:`daf_butler_dimension_expressions` for more information.
        expand : `bool`, optional
            If `True` (default) yield `ExpandedDataCoordinate` instead of
            minimal `DataCoordinate` base-class instances.
        kwds
            Additional keyword arguments are forwarded to
            `DataCoordinate.standardize` when processing the ``dataId``
            argument (and may be used to provide a constraining data ID even
            when the ``dataId`` argument is `None`).

        Yields
        ------
        dataId : `DataCoordinate`
            Data IDs matching the given query parameters.  Order is
            unspecified.
        """
        dimensions = iterable(dimensions)
        standardizedDataId = self.expandDataId(dataId, **kwds)
        standardizedDatasetTypes = []
        requestedDimensionNames = set(self.dimensions.extract(dimensions).names)
        if datasets is not None:
            if collections is None:
                raise TypeError("Cannot pass 'datasets' without 'collections'.")
            for datasetType in self.queryDatasetTypes(datasets):
                requestedDimensionNames.update(datasetType.dimensions.names)
                standardizedDatasetTypes.append(datasetType)
            # Preprocess collections expression in case the original included
            # single-pass iterators (we'll want to use it multiple times
            # below).
            collections = CollectionQuery.fromExpression(collections)

        summary = QuerySummary(
            requested=DimensionGraph(self.dimensions, names=requestedDimensionNames),
            dataId=standardizedDataId,
            expression=where,
        )
        builder = self.makeQueryBuilder(summary)
        for datasetType in standardizedDatasetTypes:
            builder.joinDataset(datasetType, collections, isResult=False)
        query = builder.finish()
        predicate = query.predicate()
        # We start a dimension cache even though we only need it if
        # expand=True, just to keep this code simple; it should be quite cheap
        # if we don't use it.
        with self.cachedDimensions() as cache:
            for row in self._db.query(query.sql):
                if predicate(row):
                    result = query.extractDataId(row)
                    if expand:
                        yield cache.expandDataId(result, records=standardizedDataId.records)
                    else:
                        yield result

    def queryDatasets(self, datasetType: Any, *,
                      collections: Any,
                      dimensions: Optional[Iterable[Union[Dimension, str]]] = None,
                      dataId: Optional[DataId] = None,
                      where: Optional[str] = None,
                      deduplicate: bool = False,
                      expand: bool = True,
                      **kwds) -> Iterator[DatasetRef]:
        """Query for and iterate over dataset references matching user-provided
        criteria.

        Parameters
        ----------
        datasetType
            An expression that fully or partially identifies the dataset types
            to be queried.  Allowed types include `DatasetType`, `str`,
            `re.Pattern`, and iterables thereof.  The special value `...` can
            be used to query all dataset types.  See
            :ref:`daf_butler_dataset_type_expressions` for more information.
        collections
            An expression that fully or partially identifies the collections
            to search for datasets, such as a `str`, `re.Pattern`, or iterable
            thereof.  `...` can be used to return all collections.  See
            :ref:`daf_butler_collection_expressions` for more information.
        dimensions : `~collections.abc.Iterable` of `Dimension` or `str`
            Dimensions to include in the query (in addition to those used
            to identify the queried dataset type(s)), either to constrain
            the resulting datasets to those for which a matching dimension
            exists, or to relate the dataset type's dimensions to dimensions
            referenced by the ``dataId`` or ``where`` arguments.
        dataId : `dict` or `DataCoordinate`, optional
            A data ID whose key-value pairs are used as equality constraints
            in the query.
        where : `str`, optional
            A string expression similar to a SQL WHERE clause.  May involve
            any column of a dimension table or (as a shortcut for the primary
            key column of a dimension table) dimension name.  See
            :ref:`daf_butler_dimension_expressions` for more information.
        deduplicate : `bool`, optional
            If `True` (`False` is default), for each result data ID, only
            yield one `DatasetRef` of each `DatasetType`, from the first
            collection in which a dataset of that dataset type appears
            (according to the order of ``collections`` passed in).  If `True`,
            ``collections`` must not contain regular expressions and may not
            be `...`.
        expand : `bool`, optional
            If `True` (default) attach `ExpandedDataCoordinate` instead of
            minimal `DataCoordinate` base-class instances.
        kwds
            Additional keyword arguments are forwarded to
            `DataCoordinate.standardize` when processing the ``dataId``
            argument (and may be used to provide a constraining data ID even
            when the ``dataId`` argument is `None`).

        Yields
        ------
        ref : `DatasetRef`
            Dataset references matching the given query criteria.  These
            are grouped by `DatasetType` if the query evaluates to multiple
            dataset types, but order is otherwise unspecified.

        Raises
        ------
        TypeError
            Raised when the arguments are incompatible, such as when a
            collection wildcard is passed when ``deduplicate`` is `True`.

        Notes
        -----
        When multiple dataset types are queried in a single call, the
        results of this operation are equivalent to querying for each dataset
        type separately in turn, and no information about the relationships
        between datasets of different types is included.  In contexts where
        that kind of information is important, the recommended pattern is to
        use `queryDimensions` to first obtain data IDs (possibly with the
        desired dataset types and collections passed as constraints to the
        query), and then use multiple (generally much simpler) calls to
        `queryDatasets` with the returned data IDs passed as constraints.
        """
        # Standardize the collections expression.
        if deduplicate:
            collections = CollectionSearch.fromExpression(collections)
        else:
            collections = CollectionQuery.fromExpression(collections)
        # Standardize and expand the data ID provided as a constraint.
        standardizedDataId = self.expandDataId(dataId, **kwds)
        # If the datasetType passed isn't actually a DatasetType, expand it
        # (it could be an expression that yields multiple DatasetTypes) and
        # recurse.
        if not isinstance(datasetType, DatasetType):
            for trueDatasetType in self.queryDatasetTypes(datasetType):
                yield from self.queryDatasets(trueDatasetType, collections=collections,
                                              dimensions=dimensions, dataId=standardizedDataId,
                                              where=where, deduplicate=deduplicate)
            return
        # The full set of dimensions in the query is the combination of those
        # needed for the DatasetType and those explicitly requested, if any.
        requestedDimensionNames = set(datasetType.dimensions.names)
        if dimensions is not None:
            requestedDimensionNames.update(self.dimensions.extract(dimensions).names)
        # Construct the summary structure needed to construct a QueryBuilder.
        summary = QuerySummary(
            requested=DimensionGraph(self.dimensions, names=requestedDimensionNames),
            dataId=standardizedDataId,
            expression=where,
        )
        builder = self.makeQueryBuilder(summary)
        # Add the dataset subquery to the query, telling the QueryBuilder to
        # include the rank of the selected collection in the results only if we
        # need to deduplicate.  Note that if any of the collections are
        # actually wildcard expressions, and we've asked for deduplication,
        # this will raise TypeError for us.
        if not builder.joinDataset(datasetType, collections, isResult=True, addRank=deduplicate):
            return
        query = builder.finish()
        predicate = query.predicate()
        # We start a dimension cache even though we only need it if
        # expand=True, just to keep this code simple; it should be quite cheap
        # if we don't use it.
        with self.cachedDimensions() as cache:
            if not deduplicate:
                # No need to de-duplicate across collections.
                for row in self._db.query(query.sql):
                    if predicate(row):
                        dataId = query.extractDataId(row, graph=datasetType.dimensions)
                        if expand:
                            dataId = cache.expandDataId(dataId, records=standardizedDataId.records)
                        yield query.extractDatasetRef(row, datasetType, dataId)[0]
            else:
                # For each data ID, yield only the DatasetRef with the lowest
                # collection rank.
                bestRefs = {}
                bestRanks = {}
                for row in self._db.query(query.sql):
                    if predicate(row):
                        ref, rank = query.extractDatasetRef(row, datasetType)
                        bestRank = bestRanks.get(ref.dataId, sys.maxsize)
                        if rank < bestRank:
                            bestRefs[ref.dataId] = ref
                            bestRanks[ref.dataId] = rank
                # If caller requested expanded data IDs, we defer that until
                # here so we do as little expansion as possible.
                if expand:
                    for ref in bestRefs.values():
                        dataId = cache.expandDataId(ref.dataId, records=standardizedDataId.records)
                        yield ref.expanded(dataId)
                else:
                    yield from bestRefs.values()

    dimensions: DimensionUniverse
    """The universe of all dimensions known to the registry
    (`DimensionUniverse`).
    """

    storageClasses: StorageClassFactory
    """All storage classes known to the registry (`StorageClassFactory`).
    """
