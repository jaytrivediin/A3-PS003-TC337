import numpy as np
import pytest

from app.data.partitioning import has_duplicate_examples, iid_partition, noniid_sensor_partition, sensor_train_statistics
from app.data.preprocessing import chronological_split, pairs_within_bounds, sample_window_pairs

L, H = 12, 12


@pytest.mark.parametrize("num_clients", [3, 4, 5])
def test_iid_partition_is_disjoint_balanced_and_reproducible(tiny_array, num_clients):
    b = chronological_split(tiny_array.shape[0])
    pool, _ = sample_window_pairs(b.train, L, H, tiny_array.shape[1], 400, seed=42)
    clients = iid_partition(pool, num_clients, 42, tiny_array, b.train)
    assert len(clients) == num_clients
    sizes = [c.num_examples for c in clients]
    assert sum(sizes) == 400 and max(sizes) - min(sizes) <= 1
    assert not has_duplicate_examples(clients)
    union = np.concatenate([c.pairs for c in clients])
    assert set(map(tuple, union.tolist())) == set(map(tuple, pool.tolist()))
    again = iid_partition(pool, num_clients, 42, tiny_array, b.train)
    assert all(np.array_equal(a.pairs, c.pairs) for a, c in zip(again, clients))
    # IID clients see a representative mix of sensors
    assert all(c.sensor_ids.size >= tiny_array.shape[1] // 2 for c in clients)


@pytest.mark.parametrize("num_clients", [3, 4, 5])
def test_noniid_partition_groups_sensors_by_training_flow(tiny_array, num_clients):
    b = chronological_split(tiny_array.shape[0])
    clients = noniid_sensor_partition(tiny_array, b.train, num_clients, 100, L, H, seed=42)
    assert len(clients) == num_clients
    all_sensors = np.concatenate([c.sensor_ids for c in clients])
    assert np.array_equal(np.sort(all_sensors), np.arange(tiny_array.shape[1]))  # disjoint cover
    for c in clients:
        assert c.num_examples <= 100
        assert set(np.unique(c.pairs[:, 1]).tolist()) <= set(c.sensor_ids.tolist())
        assert pairs_within_bounds(c.pairs, b.train, L, H)
    means = [c.mean_flow for c in clients]
    assert means == sorted(means) and means[0] < means[-1]  # distributions really differ
    assert not has_duplicate_examples(clients)


def test_sensor_statistics_use_training_period_only(tiny_array):
    b = chronological_split(tiny_array.shape[0])
    mean, std = sensor_train_statistics(tiny_array, b.train, 0)
    assert mean.shape == (tiny_array.shape[1],)
    assert np.allclose(mean, tiny_array[: b.train[1], :, 0].mean(axis=0))
    modified = tiny_array.copy()
    modified[b.test[0] :, :, 0] += 1000  # changing the test period must not change the statistics
    mean2, _ = sensor_train_statistics(modified, b.train, 0)
    assert np.allclose(mean, mean2)
