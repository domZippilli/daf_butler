# This file is part of daf_butler.
#
# Developed for the LSST Data Management System.
# This product includes software developed by the LSST Project
# (https://www.lsst.org).
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
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""Unit tests for methods in butler.time module.
"""

import unittest

from astropy.time import Time, TimeDelta
from lsst.daf.butler import time_utils


class TimeTestCase(unittest.TestCase):
    """A test case for time module
    """

    def test_time_before_epoch(self):
        """Tests for before-the-epoch time.
        """
        time = Time("1950-01-01T00:00:00", format="isot", scale="tai")
        value = time_utils.astropy_to_nsec(time)
        self.assertEqual(value, 0)

        value = time_utils.nsec_to_astropy(value)
        self.assertEqual(value, time_utils.EPOCH)

    def test_max_time(self):
        """Tests for after-the-end-of-astronomy time.
        """
        # there are rounding issues, need more complex comparison
        time = Time("2101-01-01T00:00:00", format="isot", scale="tai")
        value = time_utils.astropy_to_nsec(time)

        value_max = time_utils.astropy_to_nsec(time_utils.MAX_TIME)
        self.assertEqual(value, value_max)

    def test_round_trip(self):
        """Test precision of round-trip conversion.
        """
        # do tests at random points between epoch and max. time
        times = [
            "1970-01-01T12:00:00.123",
            "1999-12-31T23:59:57.123456",
            "2000-01-01T12:00:00.123456",
            "2030-01-01T12:00:00.123456789",
            "2075-08-17T00:03:45",
            "2099-12-31T23:00:50",
        ]
        for time in times:
            atime = Time(time, format="isot", scale="tai")
            for sec in range(7):
                # loop over few seconds to add to each time
                for i in range(100):
                    # loop over additional fractions of seconds
                    delta = sec + 0.3e-9 * i
                    in_time = atime + TimeDelta(delta, format="sec")
                    # do round-trip conversion to nsec and back
                    value = time_utils.astropy_to_nsec(in_time)
                    value = time_utils.nsec_to_astropy(value)
                    delta2 = value - in_time
                    delta2_sec = delta2.to_value("sec")
                    # absolute precision should be better than half
                    # nanosecond, but there are rounding errors too
                    self.assertLess(abs(delta2_sec), 0.51e-9)

    def test_times_equal(self):
        """Test for times_equal method
        """
        # time == time should always work
        time1 = Time("2000-01-01T00:00:00.123456789", format="isot", scale="tai")
        self.assertTrue(time_utils.times_equal(time1, time1))

        # one nsec difference
        time1 = Time("2000-01-01T00:00:00.123456789", format="isot", scale="tai")
        time2 = Time("2000-01-01T00:00:00.123456788", format="isot", scale="tai")
        self.assertTrue(time_utils.times_equal(time1, time2, 2.))
        self.assertTrue(time_utils.times_equal(time2, time1, 2.))
        self.assertFalse(time_utils.times_equal(time1, time2, .5))
        self.assertFalse(time_utils.times_equal(time2, time1, .5))

        # one nsec difference, times in UTC
        time1 = Time("2000-01-01T00:00:00.123456789", format="isot", scale="utc")
        time2 = Time("2000-01-01T00:00:00.123456788", format="isot", scale="utc")
        self.assertTrue(time_utils.times_equal(time1, time2, 2.))
        self.assertTrue(time_utils.times_equal(time2, time1, 2.))
        self.assertFalse(time_utils.times_equal(time1, time2, .5))
        self.assertFalse(time_utils.times_equal(time2, time1, .5))

        # 1/2 nsec difference
        time1 = Time("2000-01-01T00:00:00.123456789", format="isot", scale="tai")
        time2 = time1 + TimeDelta(0.5e-9, format="sec")
        self.assertTrue(time_utils.times_equal(time1, time2))
        self.assertTrue(time_utils.times_equal(time2, time1))
        self.assertFalse(time_utils.times_equal(time1, time2, .25))
        self.assertFalse(time_utils.times_equal(time2, time1, .25))

        # 1/2 microsec difference
        time1 = Time("2000-01-01T00:00:00.123456789", format="isot", scale="tai")
        time2 = time1 + TimeDelta(0.5e-6, format="sec")
        self.assertTrue(time_utils.times_equal(time1, time2, 501))
        self.assertTrue(time_utils.times_equal(time2, time1, 501))
        self.assertFalse(time_utils.times_equal(time1, time2, 499))
        self.assertFalse(time_utils.times_equal(time2, time1, 499))


if __name__ == "__main__":
    unittest.main()
