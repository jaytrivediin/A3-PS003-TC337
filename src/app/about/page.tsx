"use client";

import { Code2, Database, Link2, Server, Shield, TerminalSquare } from "lucide-react";
import { useAppState } from "@/components/app-state";
import { Badge, Card, CardContent, CardDescription, CardHeader, CardTitle, Stat } from "@/components/ui/primitives";

const ENDPOINTS: [string, string, string][] = [
  ["GET", "/api/health", "Backend status, PyTorch version, device, dataset flags"],
  ["GET", "/api/dataset/status", "Availability, dimensions, features, split, load status"],
  ["POST", "/api/dataset/download", "Download + validate the real PEMS04 archive (background)"],
  ["GET", "/api/dataset/sensor/{id}", "Historical measurements and statistics for one sensor"],
  ["GET", "/api/dataset/partitions/preview", "IID / non-IID client partition preview (training period only)"],
  ["GET", "/api/config/defaults", "Quick / full presets, limits, split ratios"],
  ["POST", "/api/experiments", "Create and start an experiment or comparison job"],
  ["GET", "/api/experiments", "Saved experiment history (summaries)"],
  ["GET", "/api/experiments/{id}", "Status, live progress, configuration and results"],
  ["GET", "/api/experiments/{id}/results", "Metrics, rounds, communication, partitions, samples"],
  ["GET", "/api/experiments/{id}/predictions", "Actual vs. predicted flow over a test interval"],
  ["GET", "/api/experiments/{id}/export", "Per-round CSV export"],
  ["GET", "/api/export/experiments.csv", "CSV export of all experiment summaries"],
  ["GET", "/api/jobs · /api/jobs/{id}", "Job list / live job status; POST /api/jobs/{id}/cancel"],
];

export default function AboutPage() {
  const { health, dataset } = useAppState();
  const b = health?.backend.detail;
  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">Project Information</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-500">FedTraffic — Federated Traffic-Flow Prediction with Communication-Cost Evaluation. Smart Cities · Machine Learning · Federated Learning · Traffic Analytics.</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader><CardTitle>Objective</CardTitle></CardHeader>
          <CardContent className="space-y-2 text-sm leading-relaxed text-slate-700">
            <p>An interactive prototype that demonstrates and compares centralized machine learning with federated learning (FedAvg) for traffic-flow forecasting on the real PEMS04 dataset. Multiple simulated traffic intersections train a shared GRU model without sending their raw training windows to the aggregation server.</p>
            <p>The application evaluates prediction accuracy (MAE, RMSE), bytes transmitted during federated training, total communication across rounds, convergence over rounds, IID vs. non-IID client data, and variability across three random seeds (42, 123, 2026).</p>
            <div className="flex flex-wrap gap-2 pt-1">
              {["React", "Next.js", "TypeScript", "Tailwind CSS", "shadcn-style UI", "Recharts", "Lucide", "Python", "FastAPI", "PyTorch", "NumPy", "Pandas", "scikit-learn"].map((t) => <Badge key={t} tone="neutral">{t}</Badge>)}
            </div>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle className="flex items-center gap-2"><Server className="h-4 w-4 text-blue-600" /> Live environment</CardTitle></CardHeader>
          <CardContent className="grid gap-2">
            <Stat label="Backend" value={health?.backend.reachable ? "online" : "offline"} sub={health?.backend.url} />
            <Stat label="PyTorch" value={b?.torch_version ?? "—"} sub={b ? `device ${b.device} · CUDA ${b.cuda_available ? "available" : "not available"}` : ""} />
            <Stat label="Dataset" value={dataset?.load_status ?? "—"} sub={dataset?.shape ? dataset.shape.join(" × ") : ""} />
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle className="flex items-center gap-2"><Code2 className="h-4 w-4 text-blue-600" /> Architecture</CardTitle><CardDescription>Separate frontend and backend; JSON files for experiment records; CSV export; no database or external services.</CardDescription></CardHeader>
          <CardContent className="text-sm text-slate-700">
            <pre className="overflow-auto rounded-lg bg-slate-900 p-3 text-[11px] leading-relaxed text-slate-100">{`FedTraffic/
├── src/                     React + TypeScript frontend (Next.js App Router)
│   ├── app/                 pages: overview, dataset, experiments, simulator,
│   │                        results, methodology, about; /api/* proxy routes
│   ├── components/          layout, ui primitives, charts, dataset explorer
│   ├── services/api.ts      typed API client
│   ├── hooks/use-api.ts     polling hook with abort-on-unmount
│   └── types/api.ts         request / response interfaces
├── backend/                 Python FastAPI + PyTorch
│   ├── app/api              routes (health, dataset, experiments, jobs)
│   ├── app/data             PEMS04 loader, preprocessing, partitioning
│   ├── app/models/gru.py    compact GRU forecaster
│   ├── app/services         training, FedAvg, communication accounting
│   ├── app/experiments      config presets, runner, job manager, JSON store
│   ├── data/raw/pems04.npz  real dataset (downloaded, git-ignored)
│   ├── results/             experiments/*.json, models/*.pt
│   └── tests/               pytest suite (synthetic arrays only)
└── README.md`}</pre>
            <p className="mt-2 text-xs text-slate-500">The browser talks to <code>/api/*</code> on the web server, which proxies to the FastAPI backend configured by <code>BACKEND_URL</code> (default http://127.0.0.1:8000).</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle className="flex items-center gap-2"><TerminalSquare className="h-4 w-4 text-blue-600" /> Run locally</CardTitle></CardHeader>
          <CardContent className="space-y-3 text-sm text-slate-700">
            <div>
              <p className="text-xs font-semibold text-slate-900">Backend</p>
              <pre className="mt-1 overflow-auto rounded-lg bg-slate-900 p-3 text-[11px] text-slate-100">{`cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
python scripts/download_pems04.py      # or use the Download button in the UI
uvicorn app.main:app --reload --port 8000`}</pre>
            </div>
            <div>
              <p className="text-xs font-semibold text-slate-900">Frontend</p>
              <pre className="mt-1 overflow-auto rounded-lg bg-slate-900 p-3 text-[11px] text-slate-100">{`npm install
cp .env.example .env     # BACKEND_URL=http://127.0.0.1:8000
npm run dev              # http://localhost:3000`}</pre>
            </div>
            <div>
              <p className="text-xs font-semibold text-slate-900">Tests</p>
              <pre className="mt-1 overflow-auto rounded-lg bg-slate-900 p-3 text-[11px] text-slate-100">{`cd backend && pytest`}</pre>
            </div>
          </CardContent>
        </Card>
      </div>

      <Card>
        <CardHeader><CardTitle className="flex items-center gap-2"><Database className="h-4 w-4 text-blue-600" /> Backend API</CardTitle><CardDescription>Interactive documentation is served by FastAPI at <code>http://127.0.0.1:8000/docs</code>.</CardDescription></CardHeader>
        <CardContent className="px-0 pt-0">
          <table className="w-full text-xs">
            <tbody className="divide-y divide-slate-100">
              {ENDPOINTS.map(([m, p, d]) => (
                <tr key={p}><td className="w-16 px-4 py-2"><Badge tone={m === "GET" ? "info" : "success"}>{m}</Badge></td><td className="px-2 py-2 font-mono text-slate-800">{p}</td><td className="px-4 py-2 text-slate-600">{d}</td></tr>
              ))}
            </tbody>
          </table>
        </CardContent>
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader><CardTitle className="flex items-center gap-2"><Link2 className="h-4 w-4 text-blue-600" /> Dataset source</CardTitle></CardHeader>
          <CardContent className="space-y-2 text-sm text-slate-700">
            <p>PeMSD4 / PEMS04 as released with ASTGCN (Guo et al., AAAI 2019): <a className="text-blue-600 hover:underline" href="https://github.com/Davidham3/ASTGCN/tree/master/data/PEMS04" target="_blank" rel="noreferrer">github.com/Davidham3/ASTGCN · data/PEMS04</a>.</p>
            <p>Archive <code>pems04.npz</code>, key <code>data</code>, shape (16992, 307, 3): 5-minute observations of 307 detectors over January–February 2018 with channels flow, occupancy, speed. Expected local path: <code>backend/data/raw/pems04.npz</code>. The file is downloaded once and never silently replaced by synthetic data.</p>
          </CardContent>
        </Card>
        <Card>
          <CardHeader><CardTitle className="flex items-center gap-2"><Shield className="h-4 w-4 text-orange-500" /> Honest scope</CardTitle></CardHeader>
          <CardContent>
            <ul className="list-disc space-y-1 pl-5 text-sm text-slate-700">
              <li>Clients are simulated participants in one process, not real intersections or live feeds.</li>
              <li>Communication numbers are serialised model payloads (federated) and an estimated raw-data payload (centralized), not network captures.</li>
              <li>No differential privacy or cryptographic protection; model updates can leak information.</li>
              <li>CPU execution; GPU is optional and not required.</li>
              <li>Prototype defaults use sampled windows; every record states whether sampling was used.</li>
            </ul>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
