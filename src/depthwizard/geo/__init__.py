"""Geo spine: image footprint -> DEM fetch -> DEM on the image grid, with datum metadata."""
from depthwizard.geo.bounds import footprint_wgs84, copernicus_tile_names
from depthwizard.geo.dem import COPERNICUS_GLO30, COPERNICUS_HEM, COPERNICUS_WBM, TILEZEN_SKADI, DEMSource, fetch_dem_window
from depthwizard.geo.reproject import dem_to_image_grid

__all__ = ["footprint_wgs84", "copernicus_tile_names", "COPERNICUS_GLO30", "COPERNICUS_HEM", "COPERNICUS_WBM", "TILEZEN_SKADI", "DEMSource",
           "fetch_dem_window", "dem_to_image_grid"]
