"""Metric calibration: relative model output -> absolute DSM (m, EGM2008) and derived object heights.

core.py holds the array-level methods used since Phase 4 (kept unchanged for existing callers):
dem_only, robust_affine, dem_plus_residual, dem_plus_smooth_residual, lowpass, checkerboard, ...
"""
from depthwizard.calibration.core import *  # noqa: F401,F403
from depthwizard.calibration.core import (Calibrated, anchor_pixels, cell_mean, checkerboard, dem_cell_ids,  # noqa: F401
                                          dem_only, dem_plus_residual, dem_plus_smooth_residual, lowpass,
                                          robust_affine, slope_deg)
