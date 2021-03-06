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

import click
import yaml

from ..opt import (collection_type_option, dataset_type_option, directory_argument, glob_parameter,
                   repo_argument, run_option, transfer_option, verbose_option)
from ..utils import split_commas, cli_handle_exception, ParameterType, typeStrAcceptsMultiple
from ...script import (butlerImport, createRepo, configDump, configValidate, queryCollections,
                       queryDatasetTypes)


# The conversion from the import command name to the butler_import function
# name for subcommand lookup is implemented in the cli/butler.py, in
# funcNameToCmdName and cmdNameToFuncName. If name changes are made here they
# must be reflected in that location. If this becomes a common pattern a better
# mechanism should be implemented.
@click.command("import")
@repo_argument(required=True, help=repo_argument.will_create_repo)
@directory_argument(required=True)
@transfer_option()
@run_option(required=True)
@click.option("--export-file",
              help="Name for the file that contains database information associated with the exported "
                   "datasets.  If this is not an absolute path, does not exist in the current working "
                   "directory, and --dir is provided, it is assumed to be in that directory.  Defaults "
                   "to \"export.yaml\".",
              type=click.File("r"))
def butler_import(*args, **kwargs):
    """Import data into a butler repository."""
    cli_handle_exception(butlerImport, *args, **kwargs)


@click.command()
@repo_argument(required=True, help=repo_argument.will_create_repo)
@click.option("--seed-config", help="Path to an existing YAML config file to apply (on top of defaults).")
@click.option("--standalone", is_flag=True, help="Include all defaults in the config file in the repo, "
              "insulating the repo from changes in package defaults.")
@click.option("--override", is_flag=True, help="Allow values in the supplied config to override all "
              "repo settings.")
@click.option("--outfile", "-f", default=None, type=str, help="Name of output file to receive repository "
              "configuration. Default is to write butler.yaml into the specified repo.")
def create(*args, **kwargs):
    """Create an empty Gen3 Butler repository."""
    cli_handle_exception(createRepo, *args, **kwargs)


@click.command(short_help="Dump butler config to stdout.")
@repo_argument(required=True)
@click.option("--subset", "-s", type=str,
              help="Subset of a configuration to report. This can be any key in the hierarchy such as "
              "'.datastore.root' where the leading '.' specified the delimiter for the hierarchy.")
@click.option("--searchpath", "-p", type=str, multiple=True, callback=split_commas,
              metavar=typeStrAcceptsMultiple,
              help="Additional search paths to use for configuration overrides")
@click.option("--file", "outfile", type=click.File("w"), default="-",
              help="Print the (possibly-expanded) configuration for a repository to a file, or to stdout "
              "by default.")
def config_dump(*args, **kwargs):
    """Dump either a subset or full Butler configuration to standard output."""
    cli_handle_exception(configDump, *args, **kwargs)


@click.command(short_help="Validate the configuration files.")
@repo_argument(required=True)
@click.option("--quiet", "-q", is_flag=True, help="Do not report individual failures.")
@dataset_type_option(help="Specific DatasetType(s) to validate.", multiple=True)
@click.option("--ignore", "-i", type=str, multiple=True, callback=split_commas,
              metavar=typeStrAcceptsMultiple,
              help="DatasetType(s) to ignore for validation.")
def config_validate(*args, **kwargs):
    """Validate the configuration files for a Gen3 Butler repository."""
    cli_handle_exception(configValidate, *args, **kwargs)


@click.command(short_help="Search for collections.")
@repo_argument(required=True)
@collection_type_option()
@click.option("--flatten-chains/--no-flatten-chains",
              help="Recursively get the child collections of matching CHAINED collections. Default is "
                   "--no-flatten-chains.")
@click.option("--include-chains/--no-include-chains",
              default=None,
              help="For --include-chains, return records for matching CHAINED collections. For "
                   "--no-include-chains do not return records for CHAINED collections. Default is the "
                   "opposite of --flatten-chains: include either CHAINED collections or their children, but "
                   "not both.")
def query_collections(*args, **kwargs):
    """Get the collections whose names match an expression."""
    print(yaml.dump(cli_handle_exception(queryCollections, *args, **kwargs)))


@click.command()
@repo_argument(required=True)
@glob_parameter(parameterType=ParameterType.ARGUMENT, multiple=True)
@verbose_option(help="Include dataset type name, dimensions, and storage class in output.")
@click.option("--components/--no-components",
              default=None,
              help="For --components, apply all expression patterns to component dataset type names as well. "
                   "For --no-components, never apply patterns to components. Default (where neither is "
                   "specified) is to apply patterns to components only if their parent datasets were not "
                   "matched by the expression. Fully-specified component datasets (`str` or `DatasetType` "
                   "instances) are always included.")
def query_dataset_types(*args, **kwargs):
    """Get the dataset types in a repository."""
    print(yaml.dump(cli_handle_exception(queryDatasetTypes, *args, **kwargs), sort_keys=False))
