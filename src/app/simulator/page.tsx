"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { ArrowDownToLine, ArrowUpFromLine, Ban, BrainCircuit, CheckCircle2, Server, Users, Waypoints, XCircle } from "lucide-react";
import { useAppState } from "@/components/app-state";
import { ConvergenceChart } from "@/components/charts/charts";
import { Alert, Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle, EmptyState, Progress, Select, Skeleton, Stat, Table, TBody, TD, TH, THead, TR } from "@/components/ui/primitives";
import { useApi } from "@/hooks/use-api";
import { api } from "@/services/api";
import { cn, formatBytes, formatInt, formatNumber, METHOD_COLORS, METHOD_LABELS, statusTone } from "@/lib/utils";
import type { ClientProgress, ExperimentRecord, Job, RoundRecord, RunProgress } from "@/types/api";

const WORKFLOW = [
  { key: "initialize", label: "Initialize global model" },
  { key: "distribute", label: "Distribute parameters" },
  { key: "local_training", label: "Local training" },
  { key: "local_update", label: "Updated parameters" },
  { key: "upload", label: "Upload updates" },
  { key: "aggregate", label: "FedAvg aggregation" },
  { key: "evaluate", label: "Evaluate global model" },
  { key: "next_round", label: "Next round" },
];

const CLIENT_STATUS_LABEL: Record<string, string> = {
  waiting: "Waiting",
  downloading: "Downloading global model",
  training: "Training locally",
  uploading: "Uploading model update",
  uploaded: "Update received",
  completed: "Completed",
  failed: "Failed",
};

function clientTone(status: string) {
  if (status === "completed" || status === "uploaded") return "success" as const;
  if (status === "training" || status === "downloading" || status === "uploading") return "info" as const;
  if (status === "failed") return "danger" as const;
  return "warning" as const;
}

/** Build a view model from either live progress or a finished record. */
function deriveView(record: ExperimentRecord | null): RunProgress | null {
  if (!record) return null;
  if (record.progress && record.progress.server) return record.progress;
  const history = record.history ?? [];
  const last = history[history.length - 1];
  const clients: ClientProgress[] = (record.partitioning?.clients ?? []).map((c) => ({
    client_id: c.client_id,
    name: c.name,
    num_examples: c.num_examples,
    num_sensors: c.num_sensors,
    mean_flow: c.mean_flow,
    status: record.status === "completed" ? "completed" : record.status === "failed" ? "failed" : "waiting",
    epoch: record.config.local_epochs,
    total_epochs: record.config.local_epochs,
    batch: 0,
    total_batches: 0,
    train_loss: last?.client_train_losses?.[c.client_id - 1] ?? null,
    download_bytes: last?.download_bytes_per_client?.[c.client_id - 1] ?? 0,
    upload_bytes: last?.upload_bytes_per_client?.[c.client_id - 1] ?? 0,
    upload_status: record.status === "completed" ? "uploaded" : "pending",
  }));
  return {
    status: record.status,
    phase: record.status,
    message: record.error ?? undefined,
    percent: record.status === "completed" ? 100 : 0,
    current_round: history.length,
    total_rounds: record.config.rounds,
    workflow_step: record.status === "completed" ? "evaluate" : "initialize",
    server: {
      status: record.status === "completed" ? "completed" : record.status,
      round: history.length,
      val_mae: last?.val_mae ?? null,
      val_rmse: last?.val_rmse ?? null,
      parameter_count: record.model?.parameter_count,
      payload_bytes: record.model?.state_dict_bytes,
      cumulative_bytes: last?.cumulative_bytes ?? 0,
      aggregation: "FedAvg weighted by local example count",
    },
    clients,
    history,
  };
}

export default function SimulatorPage() {
  const { backendOnline, backendError, activeJob } = useAppState();
  const jobs = useApi((signal) => api.listJobs(signal), [backendOnline], { enabled: backendOnline, pollMs: (d) => (d?.jobs.some((j) => j.status === "running" || j.status === "queued") ? 2000 : 10000) });
  const experiments = useApi((signal) => api.listExperiments(signal), [backendOnline], { enabled: backendOnline });
  const [jobId, setJobId] = useState<string | null>(null);
  const [runId, setRunId] = useState<string | null>(null);
  const [cancelling, setCancelling] = useState(false);

  const jobList = useMemo(() => jobs.data?.jobs ?? [], [jobs.data]);
  const job: Job | null = useMemo(() => jobList.find((j) => j.job_id === jobId) ?? activeJob ?? jobList[0] ?? null, [jobList, jobId, activeJob]);

  // Follow the currently running run automatically while a job is active.
  useEffect(() => {
    if (!job) return;
    const running = job.runs.find((r) => r.status === "running") ?? job.runs[Math.max(0, Math.min(job.current_index, job.runs.length - 1))];
    const chosen = job.runs.find((r) => r.experiment_id === runId);
    if (!chosen || (job.status === "running" && running && running.experiment_id !== runId && chosen.status !== "running")) {
      setRunId(running?.experiment_id ?? job.runs[0]?.experiment_id ?? null);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [job?.job_id, job?.current_index, job?.status]);

  const record = useApi((signal) => api.getExperiment(runId as string, signal), [runId], {
    enabled: Boolean(runId) && backendOnline,
    pollMs: (d) => (d && (d.status === "running" || d.status === "queued") ? 1000 : d ? null : 3000),
  });
  const view = deriveView(record.data);
  const isFederated = record.data ? record.data.method !== "centralized" : true;
  const stepIndex = view?.workflow_step ? WORKFLOW.findIndex((w) => w.key === view.workflow_step) : -1;

  const cancel = async () => {
    if (!job) return;
    setCancelling(true);
    try {
      await api.cancelJob(job.job_id);
      jobs.refresh();
    } finally {
      setCancelling(false);
    }
  };

  const convergence = view?.history?.length && record.data ? [{ key: record.data.experiment_id, label: `${METHOD_LABELS[record.data.method]} (seed ${record.data.seed})`, color: METHOD_COLORS[record.data.method], history: view.history }] : [];

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">Federated Learning Simulator</h1>
          <p className="mt-1 max-w-3xl text-sm text-slate-500">
            Live view of the real training job: one logical server and <span className="font-semibold text-slate-700">SIMULATED CLIENTS</span> inside a single Python process. Every state shown below is reported by the backend while the PyTorch model is actually training.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {jobList.length ? (
            <Select className="w-72" value={job?.job_id ?? ""} onChange={(e) => { setJobId(e.target.value); setRunId(null); }}>
              {jobList.map((j) => (
                <option key={j.job_id} value={j.job_id}>{j.label ?? j.config.method} · {j.status} · {new Date(j.created_at).toLocaleTimeString()}</option>
              ))}
            </Select>
          ) : null}
          {experiments.data?.experiments.length ? (
            <Select className="w-64" value={job ? "" : runId ?? ""} onChange={(e) => { if (e.target.value) { setJobId("__saved__"); setRunId(e.target.value); } }}>
              <option value="">Inspect a saved run…</option>
              {experiments.data.experiments.map((e) => (
                <option key={e.experiment_id} value={e.experiment_id}>{METHOD_LABELS[e.method]} · seed {e.seed} · {e.status}</option>
              ))}
            </Select>
          ) : null}
        </div>
      </div>

      {!backendOnline ? <Alert tone="danger" title="Backend offline">{backendError ?? "Waiting for the Python backend."}</Alert> : null}

      {!job && !runId ? (
        <EmptyState icon={Waypoints} title="No training job to display" description="Jobs live in backend memory for the current session. Start a quick experiment to watch the server and simulated clients work round by round." action={<Link href="/experiments"><Button size="sm">Start an experiment</Button></Link>} />
      ) : null}

      {job && job.job_id !== "__saved__" ? (
        <Card>
          <CardContent className="flex flex-col gap-3 py-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div className="flex flex-wrap items-center gap-2">
                <Badge tone={statusTone(job.status === "completed_with_errors" ? "failed" : job.status)}>{job.status}</Badge>
                <span className="text-sm font-semibold text-slate-900">{job.label ?? "Experiment job"}</span>
                <span className="text-xs text-slate-500">{job.runs_total} run(s) · seeds {job.seeds.join(", ")} · {job.config.rounds} rounds · {job.config.local_epochs} local epoch(s) · {job.config.num_clients} clients</span>
              </div>
              {job.status === "running" || job.status === "queued" ? (
                <Button variant="outline" size="sm" onClick={cancel} loading={cancelling} disabled={job.cancel_requested}><Ban className="h-3.5 w-3.5" /> {job.cancel_requested ? "Cancelling…" : "Cancel job"}</Button>
              ) : null}
            </div>
            <Progress value={job.percent} tone={job.status === "failed" ? "danger" : job.status === "completed" ? "success" : "info"} />
            <div className="flex flex-wrap gap-2">
              {job.runs.map((r) => (
                <button key={r.experiment_id} type="button" onClick={() => setRunId(r.experiment_id)} className={cn("flex items-center gap-2 rounded-lg border px-3 py-1.5 text-xs transition-colors", runId === r.experiment_id ? "border-blue-500 bg-blue-50 text-blue-800" : "border-slate-200 bg-white text-slate-600 hover:border-slate-300")}>
                  <span className="h-2 w-2 rounded-full" style={{ background: METHOD_COLORS[r.method] }} />
                  {METHOD_LABELS[r.method]} · seed {r.seed}
                  <Badge tone={statusTone(r.status)}>{r.status}</Badge>
                </button>
              ))}
            </div>
            {job.error ? <Alert tone="danger">{job.error}</Alert> : null}
          </CardContent>
        </Card>
      ) : null}

      {runId && record.loading && !record.data ? <Skeleton className="h-64 w-full" /> : null}
      {record.error ? <Alert tone="danger" title="Could not load run">{record.error.message}</Alert> : null}

      {record.data && view ? (
        <>
          {record.data.status === "failed" ? <Alert tone="danger" title="Training failed">{record.data.error}</Alert> : null}
          {record.data.status === "cancelled" ? <Alert tone="warning" title="Run cancelled">{record.data.error}</Alert> : null}

          {isFederated ? (
            <Card>
              <CardHeader>
                <CardTitle>Training workflow</CardTitle>
                <CardDescription>{view.message ?? record.data.round_definition}</CardDescription>
              </CardHeader>
              <CardContent>
                <ol className="grid gap-2 sm:grid-cols-4 xl:grid-cols-8">
                  {WORKFLOW.map((w, i) => {
                    const active = i === stepIndex && record.data?.status === "running";
                    const done = record.data?.status === "completed" || (stepIndex >= 0 && i < stepIndex);
                    return (
                      <li key={w.key} className={cn("rounded-lg border px-2.5 py-2 text-[11px] leading-tight", active ? "border-blue-500 bg-blue-50 text-blue-800" : done ? "border-emerald-200 bg-emerald-50 text-emerald-800" : "border-slate-200 bg-white text-slate-500")}>
                        <span className="block text-[10px] font-semibold uppercase tracking-wide opacity-70">Step {i + 1}</span>
                        {w.label}
                      </li>
                    );
                  })}
                </ol>
              </CardContent>
            </Card>
          ) : null}

          <div className="grid gap-6 xl:grid-cols-3">
            <Card className="xl:col-span-1">
              <CardHeader className="flex-row items-start justify-between gap-2">
                <div>
                  <CardTitle className="flex items-center gap-2"><Server className="h-4 w-4 text-blue-600" /> {isFederated ? "Central server" : "Centralized trainer"}</CardTitle>
                  <CardDescription>Global GRU model · {METHOD_LABELS[record.data.method]} · seed {record.data.seed}</CardDescription>
                </div>
                <Badge tone={statusTone(view.server.status === "round_complete" ? "completed" : view.server.status === "waiting_for_clients" ? "waiting" : view.server.status)}>
                  {view.server.status.replace(/_/g, " ")}
                </Badge>
              </CardHeader>
              <CardContent className="flex flex-col gap-3">
                <div>
                  <div className="mb-1 flex items-center justify-between text-xs text-slate-600">
                    <span>Round {view.current_round} / {view.total_rounds}</span>
                    <span>{formatNumber(view.percent, 0)}%</span>
                  </div>
                  <Progress value={view.percent} tone={record.data.status === "failed" ? "danger" : record.data.status === "completed" ? "success" : "info"} />
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <Stat label="Validation MAE" value={formatNumber(view.server.val_mae ?? null, 2)} sub="veh / 5 min" />
                  <Stat label="Validation RMSE" value={formatNumber(view.server.val_rmse ?? null, 2)} sub="veh / 5 min" />
                  <Stat label="Parameters" value={formatInt(view.server.parameter_count ?? record.data.model?.parameter_count)} sub={`payload ${formatBytes(view.server.payload_bytes ?? record.data.model?.state_dict_bytes)}`} />
                  {isFederated ? <Stat label="Cumulative traffic" value={formatBytes(view.server.cumulative_bytes)} sub="download + upload" /> : <Stat label="Epoch" value={`${view.server.epoch ?? 0} / ${view.server.total_epochs ?? record.data.config.local_epochs}`} sub={view.server.train_loss != null ? `loss ${formatNumber(view.server.train_loss, 4)}` : "per round"} />}
                </div>
                {isFederated ? <p className="text-[11px] text-slate-500"><BrainCircuit className="mr-1 inline h-3 w-3" />{view.server.aggregation ?? "FedAvg weighted by local example count"} · W = Σ (n_k / Σn) · W_k</p> : null}
                {record.data.status === "completed" && record.data.test_metrics ? (
                  <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-3 text-xs text-emerald-900">
                    <p className="flex items-center gap-1 font-semibold"><CheckCircle2 className="h-3.5 w-3.5" /> Final test metrics (chronological test period)</p>
                    <p className="mt-1">MAE {formatNumber(record.data.test_metrics.mae, 3)} · RMSE {formatNumber(record.data.test_metrics.rmse, 3)} · {formatInt(record.data.test_metrics.num_examples)} windows</p>
                    <Link href="/results" className="mt-1 inline-block font-medium underline">Open results &amp; analytics</Link>
                  </div>
                ) : null}
              </CardContent>
            </Card>

            <Card className="xl:col-span-2">
              <CardHeader>
                <CardTitle className="flex items-center gap-2"><Users className="h-4 w-4 text-cyan-600" /> Simulated clients <Badge tone="cyan">SIMULATED CLIENTS</Badge></CardTitle>
                <CardDescription>Logical participants inside one application — not independent traffic installations. {record.data.partitioning?.description}</CardDescription>
              </CardHeader>
              <CardContent>
                {isFederated && view.clients.length ? (
                  <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-2 2xl:grid-cols-4">
                    {view.clients.map((c) => {
                      const epochPct = c.total_epochs ? (100 * ((c.epoch - 1 >= 0 ? c.epoch - 1 : 0) + (c.total_batches ? c.batch / c.total_batches : c.status === "completed" || c.status === "uploaded" ? 1 : 0))) / c.total_epochs : 0;
                      return (
                        <div key={c.client_id} className={cn("rounded-lg border p-3", c.status === "training" ? "border-blue-300 bg-blue-50/40" : "border-slate-200 bg-white")}>
                          <div className="flex items-center justify-between">
                            <p className="text-sm font-semibold text-slate-900">{c.name}</p>
                            <Badge tone={clientTone(c.status)}>{CLIENT_STATUS_LABEL[c.status] ?? c.status}</Badge>
                          </div>
                          <p className="mt-1 text-[11px] text-slate-500">{formatInt(c.num_examples)} local windows · {c.num_sensors} sensors · mean flow {formatNumber(c.mean_flow, 1)}</p>
                          <div className="mt-2">
                            <div className="mb-1 flex justify-between text-[11px] text-slate-600">
                              <span>Epoch {Math.min(c.epoch, c.total_epochs)} / {c.total_epochs}</span>
                              <span>{c.total_batches ? `batch ${c.batch}/${c.total_batches}` : ""}</span>
                            </div>
                            <Progress value={c.status === "completed" || c.status === "uploaded" || c.status === "uploading" ? 100 : epochPct} tone={c.status === "failed" ? "danger" : "info"} />
                          </div>
                          <div className="mt-2 grid grid-cols-3 gap-1 text-[11px] text-slate-600">
                            <span>loss {c.train_loss != null ? formatNumber(c.train_loss, 4) : "—"}</span>
                            <span className="flex items-center gap-1"><ArrowDownToLine className="h-3 w-3 text-blue-500" />{formatBytes(c.download_bytes)}</span>
                            <span className="flex items-center gap-1"><ArrowUpFromLine className="h-3 w-3 text-orange-500" />{c.upload_bytes ? formatBytes(c.upload_bytes) : c.upload_status}</span>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                ) : isFederated ? (
                  <EmptyState icon={Users} title="Clients are being prepared" description="Partitions are created from the training period as soon as the run starts." />
                ) : (
                  <EmptyState icon={Server} title="Centralized baseline has no clients" description="All selected training windows are pooled on one trainer. Communication is reported as an estimated one-time raw-data upload." />
                )}
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-6 xl:grid-cols-2">
            <Card>
              <CardHeader>
                <CardTitle>Round history</CardTitle>
                <CardDescription>Validation metrics and measured bytes after each completed round.</CardDescription>
              </CardHeader>
              <CardContent className="px-0 pt-0">
                {view.history.length ? (
                  <Table className="text-xs">
                    <THead><TR><TH>Round</TH><TH>Val MAE</TH><TH>Val RMSE</TH><TH>Train loss</TH>{isFederated ? <><TH>Download</TH><TH>Upload</TH><TH>Cumulative</TH></> : <TH>Epochs</TH>}<TH>Time</TH></TR></THead>
                    <TBody>
                      {[...view.history].reverse().map((h: RoundRecord) => (
                        <TR key={h.round}>
                          <TD>{h.round}</TD><TD>{formatNumber(h.val_mae, 3)}</TD><TD>{formatNumber(h.val_rmse, 3)}</TD><TD>{formatNumber(h.train_loss_mean, 4)}</TD>
                          {isFederated ? <><TD>{formatBytes(h.total_download_bytes)}</TD><TD>{formatBytes(h.total_upload_bytes)}</TD><TD>{formatBytes(h.cumulative_bytes)}</TD></> : <TD>{h.epochs_completed}</TD>}
                          <TD>{formatNumber(h.duration_seconds, 1)} s</TD>
                        </TR>
                      ))}
                    </TBody>
                  </Table>
                ) : (
                  <div className="p-4"><EmptyState icon={XCircle} title="No rounds completed yet" /></div>
                )}
              </CardContent>
            </Card>
            <Card>
              <CardHeader>
                <CardTitle>Convergence</CardTitle>
                <CardDescription>Validation MAE of the global model after each round.</CardDescription>
              </CardHeader>
              <CardContent>
                {convergence.length ? <ConvergenceChart series={convergence} height={260} /> : <EmptyState icon={BrainCircuit} title="Waiting for the first evaluation" />}
              </CardContent>
            </Card>
          </div>
        </>
      ) : null}
    </div>
  );
}
