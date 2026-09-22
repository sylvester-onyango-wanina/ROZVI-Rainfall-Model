"""Basic tests for hydrology helpers."""
import numpy as np
from src.hydrology import scs_cn_runoff, compute_slope_degrees


def test_scs_cn_no_runoff_below_ia():
    precip = np.array([5.0, 10.0, 50.0])
    cn = np.array([70.0, 70.0, 70.0])
    q = scs_cn_runoff(precip, cn)
    assert q[0] == 0.0 or q[0] < 1e-6
    assert q[2] > 0


def test_slope_non_negative():
    dem = np.array([[10, 12, 15], [11, 13, 16], [12, 14, 17]], dtype=float)
    slope = compute_slope_degrees(dem, cellsize_m=30)
    assert slope.shape == dem.shape
    assert np.all(slope >= 0)
