"""Basic unit tests for rainfall indices."""
import numpy as np
import pytest
from src.rainfall_indices import rx1day, rx5day, mean_annual_precip, compute_all_indices


def test_rx1day_shape_and_value():
    data = np.random.rand(100, 5, 5) * 50
    out = rx1day(data)
    assert out.shape == (5, 5)
    assert np.all(out >= 0)
    assert np.all(out <= 50)


def test_rx5day_requires_enough_days():
    data = np.random.rand(3, 4, 4)
    with pytest.raises(ValueError):
        rx5day(data)


def test_compute_all_indices_keys():
    data = np.random.rand(400, 6, 6) * 30
    indices = compute_all_indices(data, n_years=400 / 365)
    expected = {"mean_annual_precip", "rx1day", "rx5day", "r95p", "r99p",
                "r10mm", "r20mm", "cdd", "cwd", "sdii"}
    assert expected.issubset(set(indices.keys()))
