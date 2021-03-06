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

import unittest

from lsst.daf.butler.tests import CliCmdTestBase
from lsst.daf.butler.cli.cmd import create


class CreateTest(CliCmdTestBase, unittest.TestCase):

    defaultExpected = dict(repo=None,
                           seed_config=None,
                           standalone=False,
                           override=False,
                           outfile=None)

    command = create

    def test_minimal(self):
        """Test only required parameters.
        """
        self.run_test(["create", "here"],
                      self.makeExpected(repo="here"))

    def test_requiredMissing(self):
        """Test that if the required parameter is missing it fails"""
        self.run_missing(["create"], r"Error: Missing argument ['\"]REPO['\"].")

    def test_all(self):
        """Test all parameters."""
        self.run_test(["create", "here",
                       "--seed-config", "foo",
                       "--standalone",
                       "--override",
                       "--outfile", "bar"],
                      self.makeExpected(repo="here",
                                        seed_config="foo",
                                        standalone=True,
                                        override=True,
                                        outfile="bar"))


if __name__ == "__main__":
    unittest.main()
