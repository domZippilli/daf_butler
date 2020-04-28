
_daf_butler_cli:

The Butler Command
==================

``daf_butler`` provides a command line interface command called ``butler``. It supports subcommands, some of
which are implemented in ``daf_butler``. Help about subcommands is available by setting up daf_butler and
running ``butler --help``.

Other packages can add subcommands to the butler command by way of a plugin system, described below.

The ``butler`` command and subcommands are implemented in `Click_`, which is well documented and has a good
quickstart guide.

.. _Click: https://click.palletsprojects.com/

This guide includes a very quick overview of how Click commands, options, and arguments. There is more to know
about these and the Click documentation is a good resource.

.. _subcommands:

Subcommands
-----------

Subcommands are like the ``pull`` in ``git pull``. The subcommand is implemented as a function and decorated
with ``@click.command`` to make it a Click Command. The subcommand name will be the same as the name of the
function (underscores in the funciton name will be changed to dashes in the command name).

.. code-block:: py
   :name: command-example

    @click.command
    def pull():
        ...

Naming
^^^^^^

For two-word commands, the words in the function name should be separated by an underscore. Click will convert
the underscore to a dash so the separater in the command name is a dash. e.g. ``def register_instrument`
becomes ``butler register-instrument``.

.. _options:

Options
-------

Options are like the ``-a`` and the ``-m <msg>`` in ``git commit -a -m <msg>``. They are declared by adding
decorators to Command functions. The long name of the option (``--message``) becomes the argument name
(``message``) when Click calls the function.

.. code-block:: py
   :name: option-example

    @click.command
    @click.option("-a", "--all", is_flag=True)
    @click.option("-m", "--message")
    def commit(all, message):
        ...

.. _click-shared-options:

Shared Options
~~~~~~~~~~~~~~

Option definitions can be shared and should be used to improve consistency and reduce code duplication. Shared
Options should be placed in a package that is as high in the dependcy tree as is reasonable for that option.
By convention the option should go in its own file in ``.../cli/opt/option_name.py`` in the package's python
directory tree.

Two common ways to create a shared option are:

1. Use a function to return an option decorator with preset values:

.. code-block:: py
   :name: simple-shared-option

    def all_option(f):
        return click.option("-a", "--all",
                            is_flag=True,
                            help="Tell the command to automatically stage "
                            "files that have been modified and deleted, but "
                            "new files you have not told Git about are not "
                            "affected.")(f)

2. Use a class to accept values to pass to the click option:

.. code-block:: py
   :name: shared-option-with-parameters

    class message_option:  # noqa: N801 (see :doc:`Why noqa: N801? <Why noqa: N801?>`)
        def __init__(self, help=None):
            self.help = help if help is not None else "Use the given <msg> "
            "as the commit message. If multiple -m options are given, their "
            "values are concatenated as separate paragraphs."

        def __call__(self, f):
            return click.option("-m", "--message",
                                help=self.help)(f)

These optons can be used like so:

.. code-block:: py

    from ..opt import all_option, message_option

    @click.command
    @all_option
    @message_option(help="My help message.")
    def commit(message):
        ...

.. _why-noqa-n801

Why noqa: N801?
"""""""""""""""

The `PEP8 section on class names`_ says class names should use the CapWords convention, but for a decorator
this is unexpected. Consider the above example using CapsWords:

.. _PEP8 section on class names: https://www.python.org/dev/peps/pep-0008/#class-names

.. code-block:: py

    @click.command
    @all_option
    @MessageOption(help="My help message.")
    def commit(message):
        ...

Using lowercase and underscores instead of CapWords results in more consistent decorator naming.


Click Arguments
---------------

Arguments are unnamed parameters like ``my_branch`` in ``git checkout my_branch``. They are declared much
like options

.. code-block:: py

    @click.command
    @click.argument("branch")
    def checkout(branch):
        ...


Shared Arguments
~~~~~~~~~~~~~~~~

Arguments can be shared, similar to options.

Butler Command Plugins
======================

Package Layout
--------------

By convention, all command line interface code should go in a folder called ``cli`` under the package's python
hierarchy e.g. ``python/lsst/daf/butler/cli``. Commands, shared arguments, and shared options should each go
in their own file, and in a category folder ``cmd``, ``arg``, or ``opt``. There must be a manifest file,
usually named ``resources.yaml`` in the ``cli`` folder. ``cli`` may also contain a ``utils.py`` file by
convention.

.. code-block:: text

   cli
   ├── arg
   ├── cmd
   ├── opt
   ├── resources.yaml
   └── utils.yaml

Manifest
--------

The ``butler`` command finds plugin commands by way of a resource manifest published in an environment
variable.

Create a file ``resources.yaml`` in the ``cli`` folder, as below. ``cmd`` names commands, ``import`` names the
package that the commands can be imported from and commands is a list of importable commands. Use the
dash-separated command name, not the underscore-separated function name.

.. code-block:: yaml

    cmd:
      import: lsst.obs.base.cli.cmd
      commands:
        - register-instrument
        - write-curated-calibrations

Publish the resource manifest in an environment variable: in the package's ``ups/<pkg>.table`` file, add a
command to prepend ``DAF_BUTLER_PLUGINS`` with the location of the resource manifest. Make sure to use the
environment variable for the location of the package.

.. code-block:: py

    envPrepend(DAF_BUTLER_PLUGINS, $OBS_BASE_DIR/python/lsst/obs/base/cli/resources.yaml)
