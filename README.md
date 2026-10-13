# FedTraffic — Federated Traffic-Flow Prediction with Communication-Cost Evaluation

**Subtitle:** Federated Traffic Forecasting & Communication Analyzer
**Category:** Smart Cities · Machine Learning · Federated Learning · Traffic Analytics

FedTraffic is a locally runnable hackathon prototype that trains a compact GRU traffic-flow
forecaster on the **real PEMS04 (PeMSD4) dataset** in two ways — a conventional **centralized
baseline** and **federated learning with FedAvg** across 3–5 *simulated* clients (IID or
sensor-based non-IID) — and measures the trade-off between prediction accuracy
(MAE / RMSE) and the **bytes exchanged** during federated training.

> Clients are logical participants inside one Python process. Nothing here is a live
> intersection, and the prototype provides **no formal privacy guarantees** (no differential
> privacy, no secure aggregation). Communication figures are serialised model payloads, not
> network packet captures.

---

## 1. Main features

| Area | What is implemented |
| --- | --- |
| Dataset | Download, validation and lazy loading of the real `pems04.npz` (16992 × 307 × 3); dataset status, dimensions, feature mapping, derived time range, missing-value counts, sensor explorer |
| Preprocessing | Chronological 70 / 10 / 20 split, leakage-free window indexing (`(t, sensor)` pairs — no materialised window copies), z-score normalisation fitted on the training period only, deterministic window sampling |
| Models | PyTorch GRU (3 inputs → hidden 32 → linear 12-step flow forecast), configurable |
| Centralized baseline | Independent run on the pooled windows, same architecture / epoch budget / seed / test windows |
| Federated learning | Genuine FedAvg on PyTorch state dicts weighted by local example counts; IID and non-IID (sensor groups ranked by training-period mean flow) partitioning |
| Communication accounting | `torch.save` payload bytes for every download (server → client) and upload (client → server), per round and cumulative; estimated raw-data upload for the centralized baseline (clearly labelled as an estimate) |
| Evaluation | MAE / RMSE in original flow units, validation history per round, fixed test set shared by all methods, three-seed evaluation (42, 123, 2026) with mean ± std |
| Web app | Overview dashboard, Dataset Explorer, Experiment configuration, live Federated Simulator (driven by real backend progress), Results & Analytics (10 chart types + CSV export + config inspection), Methodology, Project Information |
| Persistence | JSON record per run in `backend/results/experiments/`, saved global model in `backend/results/models/`, CSV export endpoints |
| Tests | 32 pytest tests covering dataset loading/validation, splits & leakage, partitioning, GRU shapes, FedAvg maths, communication accounting, metrics, serialisation and the API end-to-end |

## 2. Technology stack

* **Frontend:** React 19 + TypeScript + Tailwind CSS 4 + shadcn-style components + Recharts + Lucide icons, served by **Next.js (App Router)**. The Next.js server also hosts `/api/*` proxy routes to the Python backend so the browser stays on one origin.
  *Note:* the hosting platform for this prototype runs a Next.js server, so Next.js replaces a separate Vite dev server. All UI code is plain React/TypeScript under `src/` (components, pages, charts, services, types, hooks) and could be moved to a Vite project unchanged.
* **Backend:** Python 3.10+ · FastAPI · PyTorch (CPU) · NumPy · Pandas · scikit-learn (listed for analysis utilities) · pydantic-settings.
* **Storage:** JSON (experiment records), CSV (exports), `.pt` (models), NPZ (dataset). No database, no authentication, no cloud services, no API keys.

## 3. Architecture

```
Browser ──► Next.js (React UI + /api/* proxy, port 3000) ──► FastAPI backend (port 8000)
                                                                │
                                                                ├── PEMS04 loader (backend/data/raw/pems04.npz)
                                                                ├── Experiment runner (GRU, centralized, FedAvg)
                                                                ├── Job manager (background thread, live progress)
                                                                └── JSON store (backend/results/)
```

```
FedTraffic/
├── src/                        # frontend (React + TypeScript)
│   ├── app/                    # pages: /, /dataset, /experiments, /simulator, /results, /methodology, /about
│   │   └── api/                # health route + [...path] proxy to the backend
│   ├── components/             # layout (sidebar/header), ui primitives, charts, dataset explorer, app state
│   ├── hooks/use-api.ts        # fetch + polling hook with abort-on-unmount
│   ├── services/api.ts         # typed API client
│   ├── lib/                    # utils, backend URL, optional backend auto-launcher
│   └── types/api.ts            # request / response interfaces
├── backend/
│   ├── app/main.py             # FastAPI app factory
│   ├── app/api/                # routes_health.py, routes_dataset.py, routes_experiments.py
│   ├── app/core/config.py      # settings (FEDTRAFFIC_* env vars)
│   ├── app/data/               # pems04.py (loader/downloader), preprocessing.py, partitioning.py
│   ├── app/models/gru.py       # GRUForecaster
│   ├── app/services/           # training.py, fedavg.py, communication.py
│   ├── app/experiments/        # config.py (presets), runner.py, jobs.py, store.py
│   ├── app/utils/metrics.py    # MAE / RMSE
│   ├── scripts/download_pems04.py
│   ├── data/raw/               # pems04.npz lives here (git-ignored)
│   ├── results/                # experiments/*.json, models/*.pt (git-ignored)
│   ├── tests/                  # pytest suite
│   ├── requirements.txt
│   └── .env.example
├── .env.example                # BACKEND_URL, FEDTRAFFIC_AUTOSTART_BACKEND
└── README.md
```

## 4. Dataset: PEMS04

* **Source used:** PeMSD4 release in the ASTGCN repository —
  <https://github.com/Davidham3/ASTGCN/tree/master/data/PEMS04>
  (direct file: `https://raw.githubusercontent.com/Davidham3/ASTGCN/master/data/PEMS04/pems04.npz`, ≈ 33 MB).
* **Local path:** `backend/data/raw/pems04.npz` (override with `FEDTRAFFIC_DATA_DIR`).
* **Format:** NPZ with key `data`, array shape `(16992, 307, 3)` = time steps × sensors × features,
  stored as float64, loaded as float32. 5-minute sampling, 59 days (documented as January–February 2018).
  Timestamps are *derived* from that documented period only when the array has 16992 steps; the archive
  contains no timestamps and no sensor coordinates (none are invented).
* **Channels (configurable defaults):** 0 = traffic flow (vehicles / 5 min, forecast target),
  1 = occupancy, 2 = speed (`FEDTRAFFIC_FLOW_CHANNEL`, `..._OCCUPANCY_CHANNEL`, `..._SPEED_CHANNEL`).
* **Validation:** archive keys and dimensions are inspected, the 3-D array is validated (numeric, ≥ 3 dims,
  configured channels present, NaN fraction) before use. Missing / invalid files are reported — never
  replaced by synthetic data. PEMS04 encodes missing observations as zeros (counted in the UI).

### Getting the dataset
1. **From the UI:** Dataset Explorer → *Download dataset* (background download with progress, validated before being moved into place).
2. **Script:** `cd backend && python scripts/download_pems04.py` (`--force` to re-download).
3. **Manual:** download `pems04.npz` from the repository above and place it at `backend/data/raw/pems04.npz`.

The file is downloaded once and reused; it is git-ignored.

## 5. Setup

### Backend (Python 3.10+)
```bash
cd backend
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
cp .env.example .env                                    # optional overrides
python scripts/download_pems04.py                       # or download via the UI later
uvicorn app.main:app --reload --port 8000               # API docs: http://127.0.0.1:8000/docs
```

### Frontend (Node 20+)
```bash
npm install
cp .env.example .env            # BACKEND_URL=http://127.0.0.1:8000
npm run dev                     # http://localhost:3000
# production: npm run build && npm run start
```

Optional single-command start: set `FEDTRAFFIC_AUTOSTART_BACKEND=1` in `.env` and the Next.js server
spawns `python -m uvicorn app.main:app` from `./backend` (uses `backend/.venv/bin/python` if present,
else `python3`, override with `FEDTRAFFIC_PYTHON`). If a backend already answers at `BACKEND_URL`,
nothing is spawned.

### Environment variables
* Web: `BACKEND_URL` (default `http://127.0.0.1:8000`), `FEDTRAFFIC_AUTOSTART_BACKEND`, `FEDTRAFFIC_PYTHON`,
  `NEXT_PUBLIC_API_BASE_URL` (only if the browser should call FastAPI directly; CORS is enabled for
  localhost:3000/5173). `DATABASE_URL` is a leftover of the starter template and is **not used**.
* Backend (`backend/.env.example`): host/port, CORS origins, data/results directories, dataset URL/filename,
  documented start date, channel mapping, split ratios, evaluation sample seed, torch threads.

## 6. Using the application

1. Open <http://localhost:3000>. The header shows **Backend online/offline** and the **dataset status**.
2. **Dataset Explorer:** download PEMS04 if needed; inspect dimensions, features, the chronological split
   and any sensor's historical flow / occupancy / speed.
3. **Experiments → Run Quick Experiment:** 4 clients · 5 rounds · 1 local epoch · lr 1e-3 · batch 64 ·
   seed 42 · 4 000 training windows · 2 000 validation / test windows · complete comparison
   (centralized + FedAvg IID + FedAvg non-IID). On a laptop CPU this takes well under a minute.
4. **Federated Simulator:** watch the real job: server state (initializing → distributing → waiting for
   clients → aggregating → evaluating), per-client states (waiting → downloading → training → uploading →
   uploaded), epoch/batch progress, training loss, payload bytes, round history and convergence.
5. **Results & Analytics:** MAE / RMSE comparison, convergence, communication per round, cumulative
   communication, IID vs non-IID, client distribution, accuracy vs communication, three-seed summary,
   actual vs predicted flow, results table with CSV export and configuration inspection.
6. **Run Full Three-Seed Evaluation:** 20 rounds · 20 000 windows · 5 000 evaluation windows · seeds
   42, 123, 2026 → 9 runs (3 methods × 3 seeds). Mean ± std per method appears in chart I.
   Every field can also be customised and submitted with *Run with current configuration*.

Only one job runs at a time (the UI disables duplicate submissions; the API answers 409). Jobs can be cancelled.

## 7. How the federated simulator works

For each round `r = 1..R`:
1. The server snapshots the global GRU state dict and **measures its serialised size** (download payload).
2. For each client (sequentially): load the global parameters, train `E` local epochs with Adam on the
   client's own `(t, sensor)` windows, snapshot and **measure the local state dict** (upload payload).
3. The server aggregates with FedAvg: `W = Σ_k (n_k / Σ_j n_j) · W_k` over the real state dicts
   (float tensors averaged, integer buffers taken from the largest client) and loads the result.
4. The global model is evaluated on the fixed validation windows; metrics and bytes are appended to the
   round history and pushed to the UI as live progress.

Centralized rounds use the same epoch budget (`E` epochs over the pooled windows per round) so the
convergence charts are comparable. Validation / test windows are sampled with a fixed seed
(`FEDTRAFFIC_EVAL_SAMPLE_SEED=7`) and are identical for every method and every training seed.
Normalisation statistics come from the global training period and are shared with all clients —
a benchmarking convenience, not a privacy mechanism.

## 8. How communication bytes are calculated

* `payload = len(torch.save(state_dict, BytesIO()))` — measured, consistent, includes PyTorch container
  overhead, excludes any network-layer overhead. Raw tensor bytes (`Σ numel × 4`) are reported too.
* Per round: `download = payload × clients`, `upload = Σ client payloads`,
  `round_bytes = download + upload`, `cumulative = Σ round_bytes`.
  Per-client download/upload bytes, model parameter count and FedAvg weights are stored per round.
* Centralized: **estimated** raw-data upload `= windows × (lookback × features + horizon) × 4 bytes`,
  computed from the actually selected training windows; a deduplicated series estimate is also stored.
  The baseline has **no** per-round model traffic (`measured_model_transfer_bytes = 0`).

## 9. How results are stored

* `backend/results/experiments/<experiment_id>.json` — full record: ids, timestamps, status, config,
  dataset/split/normalisation info, example counts (and whether windows were sampled), partitioning
  (strategy, per-client examples/sensors/mean flow), per-round history (validation MAE/RMSE, losses,
  bytes), test MAE/RMSE, communication summary, prediction samples, duration, error/traceback.
* `backend/results/models/<experiment_id>.pt` — final global model state dict (used for the
  prediction comparison chart).
* CSV: `GET /api/experiments/{id}/export` (per-round) and `GET /api/export/experiments.csv` (summaries);
  the Results page also exports the on-screen table.

## 10. Tests

```bash
cd backend && pytest -q          # 32 tests, ~15 s on CPU
```
Covers: NPZ loading / shape validation / invalid & missing archives, chronological split, window-boundary
leakage prevention, deterministic sampling without duplicates, scaler fitted on train only, IID and
non-IID partitions (3–5 clients, disjoint, reproducible, ordered by flow), GRU input/output dimensions,
FedAvg weighted averaging (simple tensors + real GRU state dicts), payload byte measurement, round /
cumulative communication accounting, centralized payload estimate, MAE/RMSE, training loss decrease,
and the API end-to-end (health, dataset status, sensor endpoint, invalid configs, comparison job →
persisted JSON + model + CSV + predictions).

Tests use **tiny synthetic arrays** for isolation only; they are never presented as PEMS04 results.
Real-dataset behaviour was verified manually (status, sensor series, partition preview and a complete
quick comparison on the actual archive).

## 11. Limitations

* Prototype training budgets sample windows deterministically; each record states `train_sampled` /
  `eval_sampled`. Full-window runs are possible but slow on CPU.
* Clients are simulated sequentially in one process (no real network, stragglers or device heterogeneity).
* No differential privacy, secure aggregation or encryption — model updates are exchanged in the clear.
* Communication figures are payload sizes, not packet captures; no compression or protocol overhead.
* Jobs live in backend memory (records and models are persisted); restarting the backend clears the
  live job list but not the saved results.
* Derived timestamps rely on the documented collection period; the NPZ provides no timestamps/coordinates.
* GPU is optional and untested here; everything runs on CPU.
