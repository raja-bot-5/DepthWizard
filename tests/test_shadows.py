import math
import unittest
from datetime import datetime, timedelta, timezone

import numpy as np

from depthwizard.calibration.shadows import (shadow_azimuth_from_footprints, shadow_heights, solar_position,
                                             time_for_azimuth)


class TestSolarPosition(unittest.TestCase):
    """Physical identities (no external oracle is installed)."""

    def _day_max(self, lon, lat, day):
        best = (-99, None, None)
        for m in range(0, 24 * 60, 2):
            t = day + timedelta(minutes=m)
            el, az = solar_position(lon, lat, t)
            if el > best[0]:
                best = (el, az, t)
        return best

    def test_june_solstice_noon_elevation_and_azimuth(self):
        lat, lon = 40.0, -105.27
        el, az, _ = self._day_max(lon, lat, datetime(2021, 6, 21, 6, tzinfo=timezone.utc))
        self.assertAlmostEqual(el, 90 - (lat - 23.44), delta=0.1)      # max elevation = 90 - (lat - obliquity)
        self.assertAlmostEqual(az, 180.0, delta=1.5)                    # due south at solar noon

    def test_equinox_equator_overhead_and_sunrise_east(self):
        el, _, t = self._day_max(0.0, 0.0, datetime(2021, 3, 20, 0, tzinfo=timezone.utc))
        self.assertGreater(el, 89.5)
        self.assertLess(abs((t - datetime(2021, 3, 20, 12, 7, tzinfo=timezone.utc)).total_seconds()), 20 * 60)
        el, az = solar_position(0.0, 0.0, t - timedelta(hours=6))      # 6 h before solar noon = sunrise
        self.assertLess(abs(el), 0.6)                                   # sun moves 0.25 deg/min; noon found to 2 min
        self.assertAlmostEqual(az, 90.0, delta=1.0)

    def test_naive_datetime_rejected(self):
        with self.assertRaises(ValueError):
            solar_position(0, 0, datetime(2021, 1, 1))

    def test_time_for_azimuth_round_trip(self):
        t0 = datetime(2021, 7, 26, 17, 30, tzinfo=timezone.utc)
        el0, az0 = solar_position(-105.27, 40.0, t0)
        t, el = time_for_azimuth(-105.27, 40.0, t0, az0)
        self.assertLess(abs((t - t0).total_seconds()), 120)
        self.assertAlmostEqual(el, el0, delta=0.3)


class TestShadowHeights(unittest.TestCase):
    def test_synthetic_box_building(self):
        # 20 m tall box, sun at azimuth 135 (SE), elevation 45 deg -> shadow 20 m towards NW (azimuth 315)
        gsd, h, el, az = 0.5, 20.0, 45.0, 135.0
        H = W = 300
        fp = np.zeros((H, W), np.int32)
        fp[140:180, 140:180] = 1
        shadow = np.zeros((H, W), bool)
        L = h / math.tan(math.radians(el)) / gsd                        # 40 px
        dr, dc = -math.cos(math.radians(315)), math.sin(math.radians(315))
        rr, cc = np.nonzero(fp == 1)
        for s in np.linspace(0, L, 200):
            shadow[np.round(rr + dr * s).astype(int), np.round(cc + dc * s).astype(int)] = True
        shadow &= fp == 0
        az_m = shadow_azimuth_from_footprints(shadow, fp, gsd)
        self.assertAlmostEqual((az_m["sun_azimuth_deg"] - az + 180) % 360 - 180, 0, delta=3)
        res = shadow_heights(shadow, fp, gsd, az, el)[1]
        self.assertIsNotNone(res.height_m)
        self.assertAlmostEqual(res.height_m, h, delta=0.1 * h)

    def test_no_shadow_reported_not_guessed(self):
        fp = np.zeros((100, 100), np.int32)
        fp[40:60, 40:60] = 1
        res = shadow_heights(np.zeros((100, 100), bool), fp, 0.5, 135.0, 45.0)[1]
        self.assertIsNone(res.height_m)
        self.assertTrue(res.reasons)
