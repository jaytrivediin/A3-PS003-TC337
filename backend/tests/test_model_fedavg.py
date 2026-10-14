import numpy as np
import pytest

torch = pytest.importorskip("torch")

from app.data.preprocessing import FlowScaler, chronological_split, sample_window_pairs  # noqa: E402
from app.models.gru import GRUForecaster, count_parameters  # noqa: E402
from app.services.communication import estimate_centralized_payload, round_communication, summarize_federated  # noqa: E402
from app.services.fedavg import fedavg_aggregate, raw_tensor_bytes, state_dict_num_bytes  # noqa: E402
from app.services.training import evaluate_on_pairs, set_global_seed, train_on_pairs  # noqa: E402
from app.utils.metrics import mae, rmse  # noqa: E402


def test_gru_input_output_dimensions():
    model = GRUForecaster(input_size=3, hidden_size=32, horizon=12)
    out = model(torch.zeros(8, 12, 3))
    assert out.shape == (8, 12)
    assert count_parameters(model) == sum(p.numel() for p in model.parameters())
    with pytest.raises(ValueError):
        model(torch.zeros(12, 3))


def test_fedavg_weighted_average_simple_tensors():
    a = {"w": torch.ones(2, 2) * 1.0, "b": torch.tensor([1.0, 1.0]), "steps": torch.tensor(5)}
    b = {"w": torch.ones(2, 2) * 3.0, "b": torch.tensor([3.0, 3.0]), "steps": torch.tensor(9)}
    agg = fedavg_aggregate([a, b], [1, 3])  # weights 0.25 / 0.75 -> 2.5
    assert torch.allclose(agg["w"], torch.full((2, 2), 2.5))
    assert torch.allclose(agg["b"], torch.tensor([2.5, 2.5]))
    assert agg["steps"].item() == 9  # integer buffer taken from the largest client
    assert agg["w"].dtype == torch.float32
    with pytest.raises(ValueError):
        fedavg_aggregate([a, {"w": torch.ones(2, 2)}], [1, 1])
    with pytest.raises(ValueError):
        fedavg_aggregate([a, b], [0, 0])


def test_fedavg_on_real_gru_state_dicts_matches_manual_average():
    set_global_seed(0)
    m1, m2 = GRUForecaster(), GRUForecaster()
    agg = fedavg_aggregate([m1.state_dict(), m2.state_dict()], [100, 300])
    for k, v in agg.items():
        expected = 0.25 * m1.state_dict()[k] + 0.75 * m2.state_dict()[k]
        assert torch.allclose(v, expected, atol=1e-6)
    model = GRUForecaster()
    model.load_state_dict(agg)  # aggregated dict is loadable


def test_payload_bytes_are_measured_and_consistent():
    model = GRUForecaster()
    sd = model.state_dict()
    serialized = state_dict_num_bytes(sd)
    raw = raw_tensor_bytes(sd)
    assert raw == count_parameters(model) * 4
    assert serialized > raw > 0
    assert state_dict_num_bytes(GRUForecaster().state_dict()) == serialized


def test_round_communication_accounting():
    r1 = round_communication(1, 1000, [1000, 1000, 1000], 0, 250, 1000)
    assert r1["total_download_bytes"] == 3000
    assert r1["total_upload_bytes"] == 3000
    assert r1["round_bytes"] == 6000 and r1["cumulative_bytes"] == 6000
    r2 = round_communication(2, 1000, [1000, 1000, 1000], r1["cumulative_bytes"], 250, 1000)
    assert r2["cumulative_bytes"] == 12000
    summary = summarize_federated([r1, r2], 1000, 3)
    assert summary["total_bytes"] == 12000 and summary["total_download_bytes"] == 6000


def test_centralized_payload_estimate_from_selected_arrays():
    pairs = np.array([[0, 0], [1, 0], [50, 1]])
    est = estimate_centralized_payload(pairs, 12, 12, 3, 100, 2)
    assert est["values_per_example"] == 12 * 3 + 12
    assert est["estimated_raw_data_upload_bytes"] == 3 * 48 * 4
    assert est["unique_observations_covered"] == 25 + 24  # overlapping windows for sensor 0 + one for sensor 1
    assert est["measured_model_transfer_bytes"] == 0


def test_mae_rmse():
    a = np.array([1.0, 2.0, 3.0])
    p = np.array([1.0, 4.0, 1.0])
    assert mae(a, p) == pytest.approx(4 / 3)
    assert rmse(a, p) == pytest.approx(np.sqrt(8 / 3))


def test_training_reduces_loss_on_tiny_synthetic_data(tiny_array):
    set_global_seed(1)
    b = chronological_split(tiny_array.shape[0])
    scaler = FlowScaler().fit(tiny_array, b.train)
    norm = scaler.transform(tiny_array)
    pairs, _ = sample_window_pairs(b.train, 12, 12, tiny_array.shape[1], 256, seed=1)
    model = GRUForecaster()
    device = torch.device("cpu")
    losses = train_on_pairs(
        model, norm, pairs, lookback=12, horizon=12, target_channel=0, epochs=3, learning_rate=5e-3,
        batch_size=32, seed=1, device=device,
    )
    assert len(losses) == 3 and losses[-1] < losses[0]
    val_pairs, _ = sample_window_pairs(b.val, 12, 12, tiny_array.shape[1], 64, seed=2)
    metrics = evaluate_on_pairs(model, norm, val_pairs, lookback=12, horizon=12, target_channel=0, scaler=scaler, device=device)
    assert metrics["mae"] > 0 and metrics["rmse"] >= metrics["mae"]
