import unittest

import numpy as np
from pyproj import Transformer
from rasterio.transform import from_origin

from depthwizard.geo.osm import parse_osm_buildings, rasterize
from helpers import DEHRADUN_UTM, UTM44N


def square_xml(lon0, lat0, lon1, lat1) -> bytes:
    return f"""<osm version="0.6">
 <node id="1" lon="{lon0}" lat="{lat0}"/><node id="2" lon="{lon1}" lat="{lat0}"/>
 <node id="3" lon="{lon1}" lat="{lat1}"/><node id="4" lon="{lon0}" lat="{lat1}"/>
 <way id="10"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="4"/><nd ref="1"/><tag k="building" v="yes"/></way>
 <way id="11"><nd ref="1"/><nd ref="2"/><tag k="building" v="yes"/></way>
 <way id="12"><nd ref="1"/><nd ref="2"/><nd ref="3"/><nd ref="1"/><tag k="highway" v="road"/></way>
 <relation id="20"><tag k="building" v="yes"/></relation>
</osm>""".encode()


class TestOSM(unittest.TestCase):
    def setUp(self):
        to_ll = Transformer.from_crs(UTM44N, "EPSG:4326", always_xy=True)
        x0, y0 = DEHRADUN_UTM
        # a 30 m x 30 m square, 30 m inside the image's upper-left corner
        self.ll0 = to_ll.transform(x0 + 30, y0 - 60)
        self.ll1 = to_ll.transform(x0 + 60, y0 - 30)

    def test_parse_counts(self):
        b, c = parse_osm_buildings(square_xml(*self.ll0, *self.ll1))
        self.assertEqual(len(b), 1)
        self.assertEqual(b[0]["id"], 10)
        self.assertEqual(c["skipped_open_or_incomplete"], 1)
        self.assertEqual(c["skipped_multipolygon_relations"], 1)

    def test_rasterize_area(self):
        b, _ = parse_osm_buildings(square_xml(*self.ll0, *self.ll1))
        r, ids = rasterize(b, from_origin(*DEHRADUN_UTM, 0.6, 0.6), UTM44N, (200, 200))
        self.assertEqual(ids, {1: 10})
        px = int((r == 1).sum())
        self.assertAlmostEqual(px * 0.36, 900.0, delta=60.0)       # 30 x 30 m, edge pixels +-
        rows, cols = np.nonzero(r)
        self.assertAlmostEqual(rows.min() * 0.6, 30.0, delta=1.5)
        self.assertAlmostEqual(cols.min() * 0.6, 30.0, delta=1.5)

    def test_entity_expansion_is_refused(self):
        bomb = b"""<?xml version="1.0"?><!DOCTYPE lolz [<!ENTITY lol "lol"><!ENTITY lol2 "&lol;&lol;&lol;">]>
<osm><node id="1" lon="0" lat="0"><tag k="x" v="&lol2;"/></node></osm>"""
        with self.assertRaises(Exception):
            parse_osm_buildings(bomb)


if __name__ == "__main__":
    unittest.main()
