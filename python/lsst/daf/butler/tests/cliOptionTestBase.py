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

import abc
import click
import copy
import inspect
import os
from unittest.mock import MagicMock

from ..cli.utils import clickResultMsg, LogCliRunner, ParameterType, split_kv_separator
from ..core.utils import iterable


class MockCliTestHelper:
    """Contains objects associated with a CLI test that calls a mock.
    """
    def __init__(self, cli=None, mock=None, expectedArgs=None, expectedKwargs=None):
        self.cli = cli
        self.mock = mock
        self.expectedArgs = [] if expectedArgs is None else expectedArgs
        self.expectedKwargs = {} if expectedKwargs is None else expectedKwargs


class CliFactory:

    @staticmethod
    def noOp(optTestBase, parameterKwargs=None, cmdInitKwArgs=None):
        """Produces a no-op cli function that supports the option class being
        tested, and initializes the option class with the expected and
        passed-in keyword arguments.

        Uses `optTestBase.isArgument` and `optTestBase.isParameter` to
        determine if the option should be initialzied with a `parameterType`
        keyword argument, and sets it accordingly (`ParameterType.OPTION` is
        the standard default for parameters so this only sets it for
        `ARGUMENT`).

        Parameters
        ----------
        optTestBase : `OptTestBase` subclass
            The test being executed.
        parameterKwargs : `dict` [`str`, `Any`], optional
            A list of keyword arguments to pass to the parameter (Argument or
            Option) constructor.
        cmdInitKwArgs : `dict` [`str`, `Any`], optional
            A list of keyword arguments to pass to the command constructor.

        Returns
        -------
        cli : a click.Command.
            A click command that can be invoked by the click test invoker.
        """
        cliArgs = copy.copy(parameterKwargs) if parameterKwargs is not None else {}
        if "parameterType" not in cliArgs and optTestBase.isArgument and optTestBase.isParameter:
            cliArgs["parameterType"] = ParameterType.ARGUMENT

        if cmdInitKwArgs is None:
            cmdInitKwArgs = {}

        @click.command(**cmdInitKwArgs)
        @optTestBase.optionClass(**cliArgs)
        def cli(*args, **kwargs):
            pass
        return cli

    @staticmethod
    def mocked(optTestBase, expectedArgs=None, expectedKwargs=None, parameterKwargs=None):
        """Produces a helper object with a cli function that supports the
        option class being tested, the mock it will call, and args & kwargs
        that the mock is expected to be called with. Initializes the option
        class with the expected and passed-in keyword arguments.

        Uses `optTestBase.isArgument` and `optTestBase.isParameter` to
        determine if the option should be initialzied with a `parameterType`
        keyword argument, and sets it accordingly (`ParameterType.OPTION` is
        the standard default for parameters so this only sets it for
        `ARGUMENT`).

        Parameters
        ----------
        optTestBase : `OptTestBase` subclass
            The test being executed.
        expectedArgs : `list [`Any`], optional
            A list of arguments the mock is expected to be called with, by
            default None.
        expectedKwargs : `dict` [`str`, `Any`], optional
            A list of keyword arguments the mock is expected to be called with,
            by default None.
        parameterKwargs : `dict` [`str`, `Any`], optional
            A list of keyword arguments to pass to the parameter (Argument or
            Option) constructor.

        Returns
        -------
        helper : `MockCliTestHelper`
            The helper object.
        """
        cliArgs = copy.copy(parameterKwargs) if parameterKwargs is not None else {}
        if "parameterType" not in cliArgs and optTestBase.isArgument and optTestBase.isParameter:
            cliArgs["parameterType"] = ParameterType.ARGUMENT

        helper = MockCliTestHelper(mock=MagicMock(),
                                   expectedArgs=expectedArgs,
                                   expectedKwargs=expectedKwargs)

        @click.command()
        @optTestBase.optionClass(**cliArgs)
        def cli(*args, **kwargs):
            helper.mock(*args, **kwargs)
        helper.cli = cli
        return helper


class OptTestBase(abc.ABC):
    """A test case base that is used with Opt...Test mixin classes to test
    supported click option behaviors.
    """

    def setUp(self):
        self.runner = LogCliRunner()

    @property
    def valueType(self):
        """The value `type` of the click.Option."""
        return str

    @property
    @abc.abstractmethod
    def optionClass(self):
        """The option class being tested"""
        pass

    @property
    def optionKey(self):
        """The option name as it appears as a function argument and in
        subsequent uses (e.g. kwarg dicts); dashes are replaced by underscores.
        """
        return self.optionName.replace("-", "_")

    @property
    def optionFlag(self):
        """The flag that is used on the command line for the option."""
        return f"--{self.optionName}"

    @property
    def shortOptionFlag(self):
        """The abbreviated flag that is used on the command line for the
        option.
        """
        return f"-{self.shortOptionName}" if self.shortOptionName else None

    @property
    @abc.abstractmethod
    def optionName(self):
        """The option name, matches the option flag that appears on the command
        line."""
        pass

    @property
    def shortOptionName(self):
        """The short option flag that can be used on the command line.

        Returns
        -------
        shortOptionName : `str` or `None`
            The short option, or None if a short option is not used for this
            option.
        """
        return None

    @property
    def optionValue(self):
        """The value to pass for the option flag when calling the test
        command. If the option class restricts option values, by default
        returns the first item from `self.optionClass.choices`, otherwise
        returns a nonsense string.
        """
        if self.isChoice:
            return self.choices[0]
        return "foobarbaz"

    @property
    def optionMultipleValues(self):
        """The value(s) to pass for the option flag when calling a test command
        with multiple inputs.

        Returns
        -------
        values : `list` [`str`]
            A list of values, each item in the list will be passed to the
            command with an option flag. Items in the list may be
            comma-separated, e.g. ["foo", "bar,baz"].
        """
        # This return value matches the value returned by
        # expectedMultipleValues.
        return ("foo", "bar,baz")

    @property
    def optionMultipleKeyValues(self):
        """The values to pass for the option flag when calling a test command
        with multiple key-value inputs."""
        return ["one=two,three=four", "five=six"]

    @property
    def expectedVal(self):
        """The expected value to receive in the command function. Typically
        that value is printed to stdout and compared with this value. By
        default returns the same value as `self.optionValue`."""
        if self.isChoice:
            return self.expectedChoiceValues[0]
        return self.optionValue

    @property
    def expectedValDefault(self):
        """When the option is not required and not passed to the test command,
        this is the expected default value to appear in the command function.
        """
        return None

    @property
    def expectedMultipleValues(self):
        """The expected values to receive in the command function when a test
        command is called with multiple inputs.

        Returns
        -------
        expectedValues : `list` [`str`]
            A list of expected values, e.g. ["foo", "bar", "baz"].
        """
        # This return value matches the value returned by optionMultipleValues.
        return ("foo", "bar", "baz")

    @property
    def expectedMultipleKeyValues(self):
        """The expected valuse to receive in the command function when a test
        command is called with multiple key - value inputs. """
        # These return values matches the values returned by
        # optionMultipleKeyValues
        return dict(one="two", three="four", five="six")

    @property
    def choices(self):
        """Return the list of valid choices for the option."""
        return self.optionClass.choices

    @property
    def expectedChoiceValues(self):
        """Return the list of expected values for the option choices. Must
        match the size and order of the list returned by `choices`."""
        return self.choices

    @property
    def metavar(self):
        """Return the metavar expected to be printed in help text after the
        option flag(s). If `None`, won't run a test for the metavar value."""
        return None

    @property
    def isArgument(self):
        """True if the Parameter under test is an Argument, False if it is an
        Option."""
        return False

    @property
    def isParameter(self):
        """True if the Parameter under test can be set to an Option or an
        Argument, False if it only supports one or the other."""
        return False

    @property
    def isChoice(self):
        """True if the parameter accepts a limited set of input values. Default
        implementation is to see if the option class has an attribute called
        choices, which should be of type `list` [`str`]."""
        return hasattr(self.optionClass, "choices")

    @property
    def isBool(self):
        """True if the option only accepts bool inputs."""
        return False

    def makeInputs(self, optionFlag, values=None):
        """Make the input arguments for a CLI invocation, taking into account
        if the parameter is an Option (use option flags) or an Argument (do not
        use option flags).

        Parameters
        ----------
        optionFlag : `str` or `None`
            The option flag to use if this is an Option. May be None if it is
            known that the parameter will never be an Option.
        optionValues : `str`, `list` [`str`], or None
            The values to use as inputs. If `None`; for an Argument returns an
            empty list, or for an Option returns a single-item list containing
            the option flag.

        Returns
        -------
        inputValues : `list` [`str`]
            The list of values to use as the input parameters to a CLI function
            invocation.
        """
        inputs = []
        if values is None:
            self.assertFalse(self.isArgument, "Arguments can not be flag-only; a value is required.")
            # if there are no values and this is an Option (not an
            # Argument) then treat it as a click.Flag; the Option will be
            # True for present, False for not present.
            inputs.append(optionFlag)
            return inputs
        for value in iterable(values):
            if not self.isArgument:
                inputs.append(optionFlag)
            inputs.append(value)
        return inputs

    def run_command(self, cmd, args):
        """Run a command.

        Parameters
        ----------
        cmd : click.Command
            The command function to call.
        args : [`str`]
            The arguments to pass to the function call.

        Returns
        -------
        result : `click.Result`
            The Result instance that contains the results of the executed
            command.
        """
        return self.runner.invoke(cmd, args)

    def run_test(self, cmd, cmdArgs, verifyFunc, verifyArgs=None):
        result = self.run_command(cmd, cmdArgs)
        verifyFunc(result, verifyArgs)

    def verifyCalledWith(self, result, mockInfo):
        """Verify the command function has been called with specified
        arguments."""
        self.assertEqual(result.exit_code, 0, clickResultMsg(result))
        mockInfo.mock.assert_called_with(*mockInfo.expectedArgs, **mockInfo.expectedKwargs)

    def verifyError(self, result, expectedMsg):
        """Verify the command failed with a non-zero exit code and an expected
        output message."""
        self.assertNotEqual(result.exit_code, 0, clickResultMsg(result))
        self.assertIn(expectedMsg, result.stdout)

    def verifyMissing(self, result, verifyArgs):
        """Verify there was a missing argument; that the expected error message
        has been written to stdout, and that the command exit code is not 0.
        """
        self.assertNotEqual(result.exit_code, 0, clickResultMsg(result))
        self.assertRegex(result.stdout, verifyArgs)


class OptFlagTest(OptTestBase):
    """A mixin that tests an option behaves as a flag, instead of accepting
    a value."""

    def test_forFlag_true(self):
        helper = CliFactory.mocked(self,
                                   expectedKwargs={self.optionKey: True})
        self.run_test(helper.cli,
                      self.makeInputs(self.optionFlag),
                      self.verifyCalledWith,
                      helper)

    def test_forFlag_false(self):
        helper = CliFactory.mocked(self,
                                   expectedKwargs={self.optionKey: False})
        self.run_test(helper.cli,
                      [],
                      self.verifyCalledWith,
                      helper)


class OptChoiceTest(OptTestBase):
    """A mixin that tests an option specifies and accepts a list of acceptable
    choices and rejects choices that are not in that list."""

    def test_forChoices_validValue(self):
        """Verify that each valid choice can be passed as a value and is
        printed to the command line.
        """
        for choice, expectedValue in zip(self.choices, self.expectedChoiceValues):
            helper = CliFactory.mocked(self,
                                       expectedKwargs={self.optionKey: expectedValue})
            self.run_test(helper.cli,
                          self.makeInputs(self.optionFlag, choice),
                          self.verifyCalledWith,
                          helper)

    def test_forChoices_invalidValue(self):
        """Verify that an invalid value fails with an expected error message.
        """
        cli = CliFactory.noOp(self)
        choice = self.choices[0]
        while choice in self.choices:
            choice += "foo"
        if self.shortOptionFlag:
            expected = fr"Invalid value for ['\"]{self.shortOptionFlag}['\"] / ['\"]{self.optionFlag}['\"]"
        else:
            expected = fr"Invalid value for ['\"]{self.optionFlag}['\"]"
        self.run_test(cli, [self.optionFlag, choice], self.verifyMissing, expected)


class OptCaseInsensitiveTest(OptTestBase):
    """A mixin that tests an option accepts values in a case-insensitive way.
    """

    def test_forCaseInsensitive_upperLower(self):
        """Verify case insensitivity by making an argument all upper case and
        all lower case and verifying expected output in both cases."""
        helper = CliFactory.mocked(self,
                                   expectedKwargs={self.optionKey: self.expectedVal})
        self.run_test(helper.cli,
                      self.makeInputs(self.optionFlag, self.optionValue.upper()),
                      self.verifyCalledWith,
                      helper)
        self.run_test(helper.cli,
                      self.makeInputs(self.optionFlag, self.optionValue.lower()),
                      self.verifyCalledWith,
                      helper)


class OptMultipleTest(OptTestBase):
    """A mixin that tests an option accepts multiple inputs which may be
    comma separated."""

    # no need to test multiple=False, this gets tested with the "required"
    # test case.

    def test_forMultiple(self):
        """Test that an option class accepts the 'multiple' keyword and that
        the command can accept multiple flag inputs for the option, and inputs
        accept comma-separated values within a single flag argument."""
        helper = CliFactory.mocked(self,
                                   expectedKwargs={self.optionKey: self.expectedMultipleValues},
                                   parameterKwargs=dict(multiple=True))
        self.run_test(helper.cli,
                      self.makeInputs(self.optionFlag, self.optionMultipleValues),
                      self.verifyCalledWith,
                      helper)

    def test_forMultiple_defaultSingle(self):
        """Test that the option's 'multiple' argument defaults to False."""
        helper = CliFactory.mocked(self,
                                   expectedKwargs={self.optionKey: self.optionValue})

        self.run_test(helper.cli,
                      [self.optionValue] if self.isArgument else[self.optionFlag, self.optionValue],
                      self.verifyCalledWith,
                      helper)


class OptSplitKeyValueTest(OptTestBase):
    """A mixin that tests that an option that accepts key-value inputs parses
    those inputs correctly.
    """

    def test_forKeyValue(self):
        """Test multiple key-value inputs and comma separation."""
        helper = CliFactory.mocked(self,
                                   expectedKwargs={self.optionKey: self.expectedMultipleKeyValues},
                                   parameterKwargs=dict(multiple=True, split_kv=True))
        self.run_test(helper.cli,
                      self.makeInputs(self.optionFlag, self.optionMultipleKeyValues),
                      self.verifyCalledWith,
                      helper)

    def test_forKeyValue_withoutMultiple(self):
        """Test comma-separated key-value inputs with a parameter that accepts
        only a single key-value pair."""

        def verify(result, args):
            self.assertNotEqual(result.exit_code, 0, clickResultMsg(result))

        values = ",".join(self.optionMultipleKeyValues)

        self.run_test(CliFactory.noOp(self, parameterKwargs=dict(split_kv=True)),
                      self.makeInputs(self.optionFlag, values),
                      self.verifyError,
                      f"Error: Could not parse key-value pair '{values}' using separator "
                      f"'{split_kv_separator}', with multiple values not allowed.")


class OptRequiredTest(OptTestBase):
    """A mixin that tests that an option that accepts a "required" argument
    and handles that argument correctly.
    """

    def test_required_missing(self):
        if self.isArgument:
            expected = fr"Missing argument ['\"]{self.optionName.upper()}['\"]"
        else:
            if self.shortOptionFlag:
                expected = fr"Missing option ['\"]{self.shortOptionFlag}['\"] / ['\"]{self.optionFlag}['\"]"
            else:
                expected = fr"Missing option ['\"]\-\-{self.optionName}['\"]"

        self.run_test(CliFactory.noOp(self, parameterKwargs=dict(required=True)),
                      [],
                      self.verifyMissing,
                      expected)

    def _test_forRequired_provided(self, required):
        def doTest(self):
            helper = CliFactory.mocked(self,
                                       expectedKwargs={self.optionKey: self.expectedVal},
                                       parameterKwargs=dict(required=required))
            self.run_test(helper.cli,
                          self.makeInputs(self.optionFlag, self.optionValue),
                          self.verifyCalledWith,
                          helper)

        if type(self.valueType) == click.Path:
            OptPathTypeTest.runForPathType(self, doTest)
        else:
            doTest(self)

    def test_required_provided(self):
        self._test_forRequired_provided(required=True)

    def test_required_notRequiredProvided(self):
        self._test_forRequired_provided(required=False)

    def test_required_notRequiredDefaultValue(self):
        """Verify that the expected default value is passed for a paramter when
        it is not used on the command line."""
        helper = CliFactory.mocked(self,
                                   expectedKwargs={self.optionKey: self.expectedValDefault})
        self.run_test(helper.cli,
                      [],
                      self.verifyCalledWith,
                      helper)


class OptPathTypeTest(OptTestBase):
    """A mixin that tests options that have `type=click.Path`.
    """

    @staticmethod
    def runForPathType(testObj, testFunc):
        """Function to execute the path type test, sets up directories and a
        file as needed by the test.

        Parameters
        ----------
        testObj : `OptTestBase` instance
            The `OptTestBase` subclass that is running the test.
        testFunc : callable
            The function that executes the test, takes no arguments.
        """
        with testObj.runner.isolated_filesystem():
            if testObj.valueType.exists:
                # If the file or dir is expected to exist, create it since it's
                # it doesn't exist because we're running in a temporary
                # directory.
                if testObj.valueType.dir_okay:
                    os.makedirs(testObj.optionValue)
                elif testObj.valueType.file_okay:
                    with open(testObj.optionValue, "w") as _:
                        pass
                else:
                    testObj.assertTrue(False,
                                       "Unexpected; at least one of file_okay or dir_okay should be True.")
            testFunc(testObj)

    def test_pathType(self):
        helper = CliFactory.mocked(self,
                                   expectedKwargs={self.optionKey: self.expectedVal})

        def doTest(self):
            self.run_test(helper.cli,
                          self.makeInputs(self.optionFlag, self.optionValue),
                          self.verifyCalledWith,
                          helper)

        OptPathTypeTest.runForPathType(self, doTest)


class OptHelpTest(OptTestBase):
    """A mixin that tests that an option has a defaultHelp parameter, accepts
    a custom help paramater, and prints the help message correctly.
    """
    # Specifying a very wide terminal prevents Click from wrapping text when
    # rendering output, which causes issues trying to compare expected strings.
    wideTerminal = dict(context_settings=dict(terminal_width=1000000))

    def _verify_forHelp(self, result, expectedHelpText):
        self.assertEqual(result.exit_code, 0, clickResultMsg(result))
        self.assertIn(expectedHelpText, result.output)

    def test_help_default(self):
        self.run_test(CliFactory.noOp(self, cmdInitKwArgs=self.wideTerminal),
                      ["--help"],
                      self._verify_forHelp,
                      self.optionClass.defaultHelp)

    def test_help_custom(self):
        helpText = "foobarbaz"
        self.run_test(CliFactory.noOp(self,
                                      parameterKwargs=dict(help=helpText),
                                      cmdInitKwArgs=self.wideTerminal),
                      ["--help"],
                      self._verify_forHelp,
                      helpText)

    def test_help_optionMetavar(self):
        """Test that a specified metavar prints correctly in the help output
        for Options."""

        # For now only run on test cases that define the metavar to test.
        # This could be expanded to get the raw metavar out of the parameter
        # and test for expected formatting of all shared option metavars.
        if self.metavar is None:
            return

        def getMetavar(isRequired):
            return self.metavar if isRequired else f"[{self.metavar}]"

        parameters = inspect.signature(self.optionClass.__init__).parameters.values()
        supportedInitArgs = [parameter.name for parameter in parameters]

        def doTest(required, multiple):
            """Test for the expected parameter flag(s), metavar, and muliptle
            indicator in the --help output.

            Parameters
            ----------
            required : `bool` or None
                True if the parameter is required, False if it is not required,
                or None if the parameter initializer does not take a required
                argument, in which case it is treated as not required.
            multiple : `bool` or None
                True if the parameter accepts multiple inputs, False if it does
                not, or None if the parameter initializer does not take a
                multiple argument, in which case it is treated as not multiple.
            """
            if self.isArgument:
                expected = f"{getMetavar(required)}{' ...' if multiple else ''}"
            else:
                if self.shortOptionFlag is not None and self.optionFlag is not None:
                    expected = ", ".join([self.shortOptionFlag, self.optionFlag])
                elif (self.shortOptionFlag is not None):
                    expected = self.shortOptionFlag
                else:
                    expected = self.optionFlag
                expected = f"{expected} {self.metavar}{' ...' if multiple else ''}"
            parameterKwargs = parameterKwargs = dict(required=required)
            if multiple is not None:
                parameterKwargs["multiple"] = multiple
            if "metavar" in supportedInitArgs:
                parameterKwargs["metavar"] = self.metavar
            self.run_test(CliFactory.noOp(self, parameterKwargs=parameterKwargs),
                          ["--help"],
                          self._verify_forHelp,
                          expected)

        for required in (False, True) if "required" in supportedInitArgs else (None,):
            for multiple in (False, True) if "multiple" in supportedInitArgs else (None,):
                doTest(required, multiple)
