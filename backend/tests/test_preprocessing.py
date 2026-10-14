import numpy as np
import pytest

from app.data.preprocessing import (
    FlowScaler,
    WindowDataset,
    chronological_split,
    count_windows,
    gather_windows,
    pairs_within_bounds,
    sample_window_pairs,
    window_starts,
)

L, H = 12, 12


def test_chronological_split_is_contiguous_and_ordered():
    b = chronological_split(1000, 0.7, 0.1, 0.2)
    assert b.train == (0, 700)
    assert b.val == (700, 800)
    assert b.test == (800, 1000)
    assert b.train[1] == b.val[0] and b.val[1] == b.test[0]
    with pytest.raises(ValueError):
        chronological_split(1000, 0.5, 0.1, 0.2)


def test_windows_never_cross_split_boundaries():
    b = chronological_split(600)
    for bounds in (b.train, b.val, b.test):
        starts = window_starts(bounds, L, H)
        assert starts.min() >= bounds[0]
        assert starts.max() + L + H <= bounds[1]
        assert (starts.max() + 1) + L + H > bounds[1]  # last valid start is used
    # a window ending exactly at the boundary is allowed, one step later is not
    assert pairs_within_bounds(np.array([[b.val[1] - L - H, 0]]), b.val, L, H)
    assert not pairs_within_bounds(np.array([[b.val[1] - L - H + 1, 0]]), b.val, L, H)


def test_no_leakage_between_sampled_splits():
    b = chronological_split(600)
    train, _ = sample_window_pairs(b.train, L, H, 10, 500, seed=1)
    val, _ = sample_window_pairs(b.val, L, H, 10, 100, seed=2)
    test, _ = sample_window_pairs(b.test, L, H, 10, 100, seed=3)
    assert pairs_within_bounds(train, b.train, L, H)
    assert pairs_within_bounds(val, b.val, L, H)
    assert pairs_within_bounds(test, b.test, L, H)
    # the last time index touched by any train window is before the validation period
    assert (train[:, 0] + L + H).max() <= b.val[0]
    assert (val[:, 0] + L + H).max() <= b.test[0]
    assert test[:, 0].min() >= b.test[0]


def test_sampling_is_deterministic_and_without_duplicates():
    b = chronological_split(600)
    a, total = sample_window_pairs(b.train, L, H, 10, 300, seed=42)
    c, _ = sample_window_pairs(b.train, L, H, 10, 300, seed=42)
    d, _ = sample_window_pairs(b.train, L, H, 10, 300, seed=43)
    assert total == count_windows(b.train, L, H, 10)
    assert np.array_equal(a, c)
    assert not np.array_equal(a, d)
    keys = a[:, 0] * 10 + a[:, 1]
    assert np.unique(keys).size == keys.size
    full, total_full = sample_window_pairs(b.train, L, H, 10, None, seed=0)
    assert full.shape[0] == total_full


def test_scaler_fitted_on_training_period_only(tiny_array):
    b = chronological_split(tiny_array.shape[0])
    scaler = FlowScaler(flow_channel=0).fit(tiny_array, b.train)
    expected_mean = tiny_array[: b.train[1]].mean(axis=(0, 1))
    assert np.allclose(scaler.mean, expected_mean, atol=1e-3)
    norm = scaler.transform(tiny_array)
    assert norm.dtype == np.float32
    restored = scaler.inverse_flow(norm[..., 0])
    assert np.allclose(restored, tiny_array[..., 0], atol=1e-2)


def test_window_dataset_shapes(tiny_array):
    b = chronological_split(tiny_array.shape[0])
    pairs, _ = sample_window_pairs(b.train, L, H, tiny_array.shape[1], 16, seed=0)
    ds = WindowDataset(tiny_array, pairs, L, H, 0)
    x, y = ds[0]
    assert x.shape == (L, 3) and y.shape == (H,)
    xb, yb = gather_windows(tiny_array, pairs, L, H, 0)
    assert xb.shape == (16, L, 3) and yb.shape == (16, H)
    t, s = pairs[3]
    assert np.allclose(xb[3], tiny_array[t : t + L, s, :])
    assert np.allclose(yb[3], tiny_array[t + L : t + L + H, s, 0])
