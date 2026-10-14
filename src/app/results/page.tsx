"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { BarChart3, ChevronDown, ChevronUp, Download, FileJson, Scale, Trash2 } from "lucide-react";
import { useAppState } from "@/components/app-state";
import {
  AccuracyVsCommunicationChart,
  ClientDistributionChart,
  CommunicationRoundChart,
  ConvergenceChart,
  CumulativeCommunicationChart,
  MetricBarChart,
  PredictionChart,
  PredictionSampleChart,
  type SeriesSpec,
  type TradeoffPoint,
} from "@/components/charts/charts";
import { Alert, Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle, EmptyState, Field, Input, Select, Skeleton, Stat, Table, TBody, TD, TH, THead, TR } from "@/components/ui/primitives";
import { useApi, useDebouncedValue } from "@/hooks/use-api";
import { api } from "@/services/api";
import { downloadTextFile, formatBytes, formatDateTime, formatDuration, formatInt, formatNumber, mean, METHOD_COLORS, METHOD_LABELS, METHOD_ORDER, statusTone, std, toCsv } from "@/lib/utils";
import type { ExperimentRecord, ExperimentSummary, RunMethod } from "@/types/api";

const UNIT = "vehicles / 5 min";

function DetailsPanel({ id }: { id: string }) {
  const rec = useApi((signal) => api.getExperiment(id, signal), [id]);
  if (rec.loading && !rec.data) return <Skeleton className="h-32 w-full" />;
  if (rec.error) return <Alert tone="danger">{rec.error.message}</Alert>;
  if (!rec.data) return null;
  const r = rec.data;
  const shown = { experiment_id: r.experiment_id, job_id: r.job_id, status: r.status, method: r.method, seed: r.seed, config: r.config, data: r.data, model: r.model, partitioning: r.partitioning ? { ...r.partitioning, clients: r.partitioning.clients.map((c) => ({ ...c, sensor_ids_preview: `${c.sensor_ids_preview.length} ids…` })) } : undefined, communication: r.communication, test_metrics: r.test_metrics, best_validation: r.best_validation, dataset: r.dataset, error: r.error };
  return (
    <div className="grid gap-3 lg:grid-cols-2">
      <pre className="max-h-96 overflow-auto rounded-lg bg-slate-900 p-3 text-[11px] leading-relaxed text-slate-100">{JSON.stringify(shown, null, 2)}</pre>
      <div className="flex flex-col gap-3">
        {r.data ? (
          <Alert tone={r.data.train_sampled ? "warning" : "success"} title={r.data.train_sampled ? "Sampled training windows" : "All training windows used"}>
            {formatInt(r.data.train_examples)} of {formatInt(r.data.train_examples_available)} available training windows; {formatInt(r.data.test_examples)} of {formatInt(r.data.test_examples_available)} test windows (fixed sample seed {r.data.eval_sample_seed}).
          </Alert>
        ) : null}
        {r.communication ? <Alert tone="info" title={r.communication.type === "federated" ? "Measured model payloads" : "Estimated raw-data payload"}>{r.communication.note}</Alert> : null}
        {r.predictions_sample?.length ? (
          <div>
            <p className="mb-1 text-xs font-semibold text-slate-700">Prediction examples (test windows)</p>
            <div className="grid gap-2 sm:grid-cols-2">
              {r.predictions_sample.slice(0, 4).map((s) => (
                <div key={`${s.sensor_id}-${s.window_start}`} className="rounded-lg border border-slate-200 p-2">
                  <p className="text-[11px] text-slate-500">Sensor {s.sensor_id} · origin {s.forecast_origin_timestamp ? formatDateTime(s.forecast_origin_timestamp) : s.forecast_origin} · MAE {formatNumber(s.mae, 2)}</p>
                  <PredictionSampleChart sample={s} height={130} />
                </div>
              ))}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

export default function ResultsPage() {
  const { backendOnline, backendError, activeJob, dataset } = useAppState();
  const list = useApi((signal) => api.listExperiments(signal), [backendOnline], { enabled: backendOnline, pollMs: activeJob ? 5000 : 30000 });
  const summaries = useMemo(() => list.data?.experiments ?? [], [list.data]);
  const completed = useMemo(() => summaries.filter((e) => e.status === "completed"), [summaries]);

  const groups = useMemo(() => {
    const options: { id: string; label: string; ids: string[] }[] = [];
    const latestIds = METHOD_ORDER.map((m) => completed.find((e) => e.method === m)?.experiment_id).filter((x): x is string => Boolean(x));
    if (latestIds.length) options.push({ id: "latest", label: "Latest completed run per method", ids: latestIds });
    const byJob = new Map<string, ExperimentSummary[]>();
    completed.forEach((e) => {
      if (e.job_id) byJob.set(e.job_id, [...(byJob.get(e.job_id) ?? []), e]);
    });
    byJob.forEach((runs, jobId) => options.push({ id: jobId, label: `${runs[0].label ?? "Job"} · ${runs.length} run(s) · ${formatDateTime(runs[0].created_at)}`, ids: runs.map((r) => r.experiment_id) }));
    return options;
  }, [completed]);
  const [groupId, setGroupId] = useState<string>("latest");
  const group = groups.find((g) => g.id === groupId) ?? groups[0];
  const idsKey = group?.ids.join(",") ?? "";
  const records = useApi(async (signal) => Promise.all(idsKey.split(",").map((id) => api.getResults(id, signal))), [idsKey], { enabled: backendOnline && idsKey.length > 0 });
  const runs: ExperimentRecord[] = useMemo(() => (records.data ?? []).filter((r) => r.status === "completed"), [records.data]);

  const perMethod = useMemo(
    () =>
      METHOD_ORDER.map((m) => {
        const rs = runs.filter((r) => r.method === m && r.test_metrics);
        const maes = rs.map((r) => r.test_metrics!.mae);
        const rmses = rs.map((r) => r.test_metrics!.rmse);
        const bytes = rs.map((r) => r.communication?.total_bytes ?? 0);
        return { method: m, n: rs.length, seeds: rs.map((r) => r.seed), mae: mean(maes), maeStd: std(maes), rmse: mean(rmses), rmseStd: std(rmses), bytes: mean(bytes), runs: rs };
      }).filter((x) => x.n > 0),
    [runs],
  );

  const seriesFor = (filter: (r: ExperimentRecord) => boolean): SeriesSpec[] => {
    const counters: Partial<Record<RunMethod, number>> = {};
    return runs.filter(filter).map((r) => {
      const idx = counters[r.method] ?? 0;
      counters[r.method] = idx + 1;
      const sameMethod = runs.filter((x) => x.method === r.method).length;
      return {
        key: r.experiment_id,
        label: `${METHOD_LABELS[r.method]} · seed ${r.seed}`,
        color: sameMethod > 1 ? shade(METHOD_COLORS[r.method], idx) : METHOD_COLORS[r.method],
        history: r.history,
        dashed: r.method === "centralized",
      };
    });
  };

  const fedRuns = runs.filter((r) => r.method !== "centralized");
  const [fedId, setFedId] = useState<string>("");
  useEffect(() => {
    if (!fedRuns.find((r) => r.experiment_id === fedId)) setFedId(fedRuns[0]?.experiment_id ?? "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idsKey, records.data]);
  const fedRun = fedRuns.find((r) => r.experiment_id === fedId) ?? null;

  const tradeoff: TradeoffPoint[] = runs.filter((r) => r.test_metrics && r.communication).map((r) => ({ label: `${METHOD_LABELS[r.method]} · seed ${r.seed}`, method: r.method, bytes: Math.max(1, r.communication!.total_bytes), mae: r.test_metrics!.mae, estimated: r.communication!.type === "centralized", seed: r.seed }));

  // Predictions
  const [predRunId, setPredRunId] = useState<string>("");
  useEffect(() => {
    if (!completed.find((e) => e.experiment_id === predRunId)) setPredRunId(runs[0]?.experiment_id ?? completed[0]?.experiment_id ?? "");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [idsKey, completed.length, runs.length]);
  const [predSensor, setPredSensor] = useState(0);
  const [predHorizon, setPredHorizon] = useState(1);
  const [predDay, setPredDay] = useState(0);
  const testStart = dataset?.split?.partitions.test.start_index ?? 0;
  const testSteps = dataset?.split?.partitions.test.num_timesteps ?? 0;
  const maxPredDay = Math.max(0, Math.floor(testSteps / 288) - 1);
  const predParams = useDebouncedValue({ id: predRunId, sensor_id: predSensor, start: testStart + Math.min(predDay, maxPredDay) * 288, steps: 288, horizon_step: predHorizon }, 400);
  const predictions = useApi((signal) => api.predictions(predParams.id, { sensor_id: predParams.sensor_id, start: predParams.start, steps: predParams.steps, horizon_step: predParams.horizon_step }, signal), [JSON.stringify(predParams)], { enabled: backendOnline && Boolean(predParams.id) });

  // Table
  const [expanded, setExpanded] = useState<string | null>(null);
  const [deleting, setDeleting] = useState<string | null>(null);
  const exportTable = () => {
    const rows = summaries.map((e) => ({
      experiment_id: e.experiment_id, label: e.label, method: e.method, seed: e.seed, status: e.status, num_clients: e.config.num_clients, rounds: e.config.rounds, local_epochs: e.config.local_epochs,
      learning_rate: e.config.learning_rate, batch_size: e.config.batch_size, hidden_size: e.config.hidden_size, partition: e.partition_strategy, train_examples: e.train_examples, train_sampled: e.train_sampled,
      test_examples: e.test_examples, test_mae: e.test_mae, test_rmse: e.test_rmse, final_val_mae: e.final_val_mae, total_bytes: e.communication.total_bytes, download_bytes: e.communication.total_download_bytes,
      upload_bytes: e.communication.total_upload_bytes, estimated_raw_data_upload_bytes: e.communication.estimated_raw_data_upload_bytes, duration_seconds: e.duration_seconds, created_at: e.created_at,
    }));
    downloadTextFile("fedtraffic_results.csv", toCsv(rows));
  };
  const remove = async (id: string) => {
    if (!confirm("Delete this experiment record and its saved model?")) return;
    setDeleting(id);
    try {
      await api.deleteExperiment(id);
      list.refresh();
    } catch (e) {
      alert((e as Error).message);
    } finally {
      setDeleting(null);
    }
  };

  const noData = backendOnline && list.data && completed.length === 0;

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">Results &amp; Analytics</h1>
          <p className="mt-1 max-w-3xl text-sm text-slate-500">
            Every chart reads saved experiment outputs. Lower MAE / RMSE means better forecasts. Federated learning is not assumed to be more accurate or cheaper — the measured trade-offs are shown as they are.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {groups.length ? (
            <Select className="w-80" value={group?.id ?? ""} onChange={(e) => setGroupId(e.target.value)}>
              {groups.map((g) => <option key={g.id} value={g.id}>{g.label}</option>)}
            </Select>
          ) : null}
          <a href={api.exportAllUrl()} download>
            <Button variant="outline" size="sm"><Download className="h-3.5 w-3.5" /> Export all (CSV)</Button>
          </a>
        </div>
      </div>

      {!backendOnline ? <Alert tone="danger" title="Backend offline">{backendError ?? "Waiting for the Python backend."}</Alert> : null}
      {list.error && backendOnline ? <Alert tone="danger">{list.error.message}</Alert> : null}
      {noData ? <EmptyState icon={BarChart3} title="No completed experiments yet" description="Run a quick experiment or the three-seed evaluation; results appear here automatically." action={<Link href="/experiments"><Button size="sm">Configure an experiment</Button></Link>} /> : null}
      {records.loading && !records.data && idsKey ? <Skeleton className="h-72 w-full" /> : null}

      {runs.length ? (
        <>
          <div className="grid gap-6 xl:grid-cols-2">
            <Card>
              <CardHeader><CardTitle>A · Test MAE comparison</CardTitle><CardDescription>Mean over the runs in the selected group (error bars = sample std across seeds when n &gt; 1).</CardDescription></CardHeader>
              <CardContent><MetricBarChart unit={`MAE (${UNIT})`} items={perMethod.map((p) => ({ label: METHOD_LABELS[p.method], value: p.mae as number, error: p.n > 1 ? p.maeStd : null, color: METHOD_COLORS[p.method], n: p.n }))} /></CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle>B · Test RMSE comparison</CardTitle><CardDescription>RMSE penalises large errors more strongly than MAE.</CardDescription></CardHeader>
              <CardContent><MetricBarChart unit={`RMSE (${UNIT})`} items={perMethod.map((p) => ({ label: METHOD_LABELS[p.method], value: p.rmse as number, error: p.n > 1 ? p.rmseStd : null, color: METHOD_COLORS[p.method], n: p.n }))} /></CardContent>
            </Card>
          </div>

          <Card>
            <CardHeader><CardTitle>C · Convergence</CardTitle><CardDescription>Validation MAE versus training round (centralized: one round = the same number of epochs as a federated client&apos;s local epochs). Dashed = centralized.</CardDescription></CardHeader>
            <CardContent><ConvergenceChart series={seriesFor(() => true)} metric="val_mae" height={300} /></CardContent>
          </Card>

          <div className="grid gap-6 xl:grid-cols-2">
            <Card>
              <CardHeader className="flex-row items-start justify-between gap-2">
                <div><CardTitle>D · Communication per round</CardTitle><CardDescription>Measured serialised state_dict bytes: download (server → clients) and upload (clients → server).</CardDescription></div>
                {fedRuns.length ? (
                  <Select className="w-56" value={fedId} onChange={(e) => setFedId(e.target.value)}>
                    {fedRuns.map((r) => <option key={r.experiment_id} value={r.experiment_id}>{METHOD_LABELS[r.method]} · seed {r.seed}</option>)}
                  </Select>
                ) : null}
              </CardHeader>
              <CardContent>
                {fedRun ? (
                  <>
                    <CommunicationRoundChart history={fedRun.history} />
                    <div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-4">
                      <Stat label="Payload / transfer" value={formatBytes(fedRun.communication?.model_payload_bytes_per_transfer)} sub={`${formatInt(fedRun.model?.parameter_count)} params`} />
                      <Stat label="Total download" value={formatBytes(fedRun.communication?.total_download_bytes)} />
                      <Stat label="Total upload" value={formatBytes(fedRun.communication?.total_upload_bytes)} />
                      <Stat label="Total" value={formatBytes(fedRun.communication?.total_bytes)} sub={`${fedRun.communication?.rounds} rounds × ${fedRun.communication?.num_clients} clients`} />
                    </div>
                  </>
                ) : <EmptyState icon={BarChart3} title="No federated run in this group" />}
              </CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle>E · Cumulative communication</CardTitle><CardDescription>Running total of transferred bytes across rounds for every federated run in the group.</CardDescription></CardHeader>
              <CardContent>{fedRuns.length ? <CumulativeCommunicationChart series={seriesFor((r) => r.method !== "centralized")} /> : <EmptyState icon={BarChart3} title="No federated run in this group" />}</CardContent>
            </Card>
          </div>

          <div className="grid gap-6 xl:grid-cols-2">
            <Card>
              <CardHeader><CardTitle>F · IID versus non-IID</CardTitle><CardDescription>Same model, rounds and total window budget — only the client data distribution differs.</CardDescription></CardHeader>
              <CardContent>
                {perMethod.some((p) => p.method === "fed_iid") && perMethod.some((p) => p.method === "fed_noniid") ? (
                  <Table className="text-xs">
                    <THead><TR><TH>Metric</TH><TH>FedAvg IID</TH><TH>FedAvg non-IID</TH><TH>Δ (non-IID − IID)</TH></TR></THead>
                    <TBody>
                      {(() => {
                        const iid = perMethod.find((p) => p.method === "fed_iid")!;
                        const non = perMethod.find((p) => p.method === "fed_noniid")!;
                        const bestRound = (rs: ExperimentRecord[]) => mean(rs.map((r) => r.best_validation?.round ?? r.history.length));
                        const finalVal = (rs: ExperimentRecord[]) => mean(rs.map((r) => r.history[r.history.length - 1]?.val_mae ?? NaN).filter((v) => !Number.isNaN(v)));
                        const rows = [
                          ["Test MAE", iid.mae, non.mae],
                          ["Test RMSE", iid.rmse, non.rmse],
                          ["Final validation MAE", finalVal(iid.runs), finalVal(non.runs)],
                          ["Round of best validation MAE", bestRound(iid.runs), bestRound(non.runs)],
                          ["Total communication (bytes)", iid.bytes, non.bytes],
                        ] as [string, number | null, number | null][];
                        return rows.map(([label, a, b]) => (
                          <TR key={label}>
                            <TD className="font-medium text-slate-900">{label}</TD>
                            <TD>{label.includes("bytes") ? formatBytes(a) : formatNumber(a, 3)}</TD>
                            <TD>{label.includes("bytes") ? formatBytes(b) : formatNumber(b, 3)}</TD>
                            <TD className={a != null && b != null && b > a ? "text-orange-700" : "text-emerald-700"}>{a != null && b != null ? (label.includes("bytes") ? formatBytes(b - a) : formatNumber(b - a, 3)) : "—"}</TD>
                          </TR>
                        ));
                      })()}
                    </TBody>
                  </Table>
                ) : <EmptyState icon={Scale} title="Need both an IID and a non-IID run" description="Run the complete comparison to populate this table." />}
                <ConvergenceChart series={seriesFor((r) => r.method !== "centralized")} metric="val_mae" height={220} />
              </CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle>G · Client data distribution</CardTitle><CardDescription>{fedRun ? fedRun.partitioning?.description : "Select a federated run."}</CardDescription></CardHeader>
              <CardContent>
                {fedRun?.partitioning?.clients.length ? (
                  <>
                    <ClientDistributionChart clients={fedRun.partitioning.clients} />
                    <Table className="mt-2 text-xs">
                      <THead><TR><TH>Client</TH><TH>Windows</TH><TH>Sensors</TH><TH>Mean flow</TH><TH>σ flow</TH><TH>FedAvg weight</TH></TR></THead>
                      <TBody>
                        {fedRun.partitioning.clients.map((c) => (
                          <TR key={c.client_id}><TD>{c.name}</TD><TD>{formatInt(c.num_examples)}</TD><TD>{c.num_sensors}</TD><TD>{formatNumber(c.mean_flow, 1)}</TD><TD>{formatNumber(c.std_flow, 1)}</TD><TD>{formatNumber((fedRun.history[0]?.aggregation_weights ?? [])[c.client_id - 1] ?? null, 3)}</TD></TR>
                        ))}
                      </TBody>
                    </Table>
                  </>
                ) : <EmptyState icon={BarChart3} title="No client partitions in this group" />}
              </CardContent>
            </Card>
          </div>

          <div className="grid gap-6 xl:grid-cols-2">
            <Card>
              <CardHeader><CardTitle>H · Accuracy versus communication</CardTitle><CardDescription>Test MAE against total communication. Centralized points use the <em>estimated</em> one-time raw-data upload; federated points use measured model payloads.</CardDescription></CardHeader>
              <CardContent><AccuracyVsCommunicationChart points={tradeoff} /></CardContent>
            </Card>
            <Card>
              <CardHeader><CardTitle>I · Three-seed summary</CardTitle><CardDescription>Mean ± sample standard deviation across independent seeds (42, 123, 2026). Rounds are never treated as seeds.</CardDescription></CardHeader>
              <CardContent>
                <Table className="text-xs">
                  <THead><TR><TH>Method</TH><TH>Seeds</TH><TH>Test MAE</TH><TH>Test RMSE</TH><TH>Communication</TH></TR></THead>
                  <TBody>
                    {perMethod.map((p) => (
                      <TR key={p.method}>
                        <TD className="font-medium text-slate-900">{METHOD_LABELS[p.method]}</TD>
                        <TD>{p.seeds.join(", ")}</TD>
                        <TD>{formatNumber(p.mae, 3)}{p.n > 1 ? ` ± ${formatNumber(p.maeStd, 3)}` : ""}</TD>
                        <TD>{formatNumber(p.rmse, 3)}{p.n > 1 ? ` ± ${formatNumber(p.rmseStd, 3)}` : ""}</TD>
                        <TD>{formatBytes(p.bytes)}{p.method === "centralized" ? " (est.)" : ""}</TD>
                      </TR>
                    ))}
                  </TBody>
                </Table>
                {perMethod.every((p) => p.n < 2) ? <Alert tone="info" className="mt-3">This group contains a single seed per method. Run the full three-seed evaluation to obtain variability estimates.</Alert> : null}
              </CardContent>
            </Card>
          </div>
        </>
      ) : null}

      {completed.length ? (
        <Card>
          <CardHeader><CardTitle>J · Prediction comparison</CardTitle><CardDescription>Actual versus predicted traffic flow of a saved model over one day of the chronological test period (input windows never leave the test period).</CardDescription></CardHeader>
          <CardContent className="flex flex-col gap-4">
            <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Field label="Model (completed run)">
                <Select value={predRunId} onChange={(e) => setPredRunId(e.target.value)}>
                  {completed.map((e) => <option key={e.experiment_id} value={e.experiment_id}>{METHOD_LABELS[e.method]} · seed {e.seed} · MAE {formatNumber(e.test_mae, 1)}</option>)}
                </Select>
              </Field>
              <Field label={`Sensor (0 – ${(dataset?.num_sensors ?? 1) - 1})`}><Input type="number" min={0} max={(dataset?.num_sensors ?? 1) - 1} value={predSensor} onChange={(e) => setPredSensor(Math.max(0, Number(e.target.value) || 0))} /></Field>
              <Field label="Forecast step (× 5 min ahead)"><Input type="number" min={1} max={12} value={predHorizon} onChange={(e) => setPredHorizon(Math.max(1, Number(e.target.value) || 1))} /></Field>
              <Field label={`Test day offset (0 – ${maxPredDay})`}><Input type="number" min={0} max={maxPredDay} value={predDay} onChange={(e) => setPredDay(Math.max(0, Number(e.target.value) || 0))} /></Field>
            </div>
            {predictions.error ? <Alert tone="danger">{predictions.error.message}</Alert> : null}
            {predictions.loading && !predictions.data ? <Skeleton className="h-[280px] w-full" /> : predictions.data ? (
              <>
                <PredictionChart indices={predictions.data.indices} timestamps={predictions.data.timestamps} actual={predictions.data.actual} predicted={predictions.data.predicted} />
                <p className="text-[11px] text-slate-500">{predictions.data.horizon_minutes}-minute-ahead forecasts for sensor {predictions.data.sensor_id} · interval MAE {formatNumber(predictions.data.mae, 2)} · RMSE {formatNumber(predictions.data.rmse, 2)} ({predictions.data.steps} forecasts, {METHOD_LABELS[predictions.data.method]}).</p>
              </>
            ) : null}
          </CardContent>
        </Card>
      ) : null}

      {summaries.length ? (
        <Card>
          <CardHeader className="flex-row flex-wrap items-center justify-between gap-2">
            <div><CardTitle>Results table</CardTitle><CardDescription>All saved experiment records. Expand a row to inspect its exact configuration and prediction examples.</CardDescription></div>
            <Button variant="outline" size="sm" onClick={exportTable}><Download className="h-3.5 w-3.5" /> Export table (CSV)</Button>
          </CardHeader>
          <CardContent className="px-0 pt-0">
            <Table className="text-xs">
              <THead><TR><TH></TH><TH>Run</TH><TH>Method</TH><TH>Seed</TH><TH>Status</TH><TH>Clients</TH><TH>Rounds</TH><TH>Epochs</TH><TH>LR</TH><TH>Batch</TH><TH>Train windows</TH><TH>Test MAE</TH><TH>Test RMSE</TH><TH>Communication</TH><TH>Duration</TH><TH>Created</TH><TH></TH></TR></THead>
              <TBody>
                {summaries.map((e) => (
                  <RowGroup key={e.experiment_id} e={e} expanded={expanded === e.experiment_id} onToggle={() => setExpanded(expanded === e.experiment_id ? null : e.experiment_id)} onDelete={() => remove(e.experiment_id)} deleting={deleting === e.experiment_id} />
                ))}
              </TBody>
            </Table>
          </CardContent>
        </Card>
      ) : null}
    </div>
  );
}

function RowGroup({ e, expanded, onToggle, onDelete, deleting }: { e: ExperimentSummary; expanded: boolean; onToggle: () => void; onDelete: () => void; deleting: boolean }) {
  return (
    <>
      <TR>
        <TD><button type="button" onClick={onToggle} className="rounded p-1 hover:bg-slate-100" aria-label="Toggle details">{expanded ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}</button></TD>
        <TD className="font-mono text-[11px]">{e.experiment_id.slice(-10)}{e.label ? <span className="block text-[10px] text-slate-400">{e.label}</span> : null}</TD>
        <TD><span className="mr-1 inline-block h-2 w-2 rounded-full" style={{ background: METHOD_COLORS[e.method as RunMethod] }} />{METHOD_LABELS[e.method]}</TD>
        <TD>{e.seed}</TD>
        <TD><Badge tone={statusTone(e.status)}>{e.status}</Badge></TD>
        <TD>{e.method === "centralized" ? "—" : e.config.num_clients}</TD>
        <TD>{e.config.rounds}</TD>
        <TD>{e.config.local_epochs}</TD>
        <TD>{e.config.learning_rate}</TD>
        <TD>{e.config.batch_size}</TD>
        <TD>{formatInt(e.train_examples)}{e.train_sampled ? <span className="ml-1 text-[10px] text-orange-600">sampled</span> : e.train_examples ? <span className="ml-1 text-[10px] text-emerald-600">full</span> : null}</TD>
        <TD className="font-semibold text-slate-900">{formatNumber(e.test_mae, 2)}</TD>
        <TD>{formatNumber(e.test_rmse, 2)}</TD>
        <TD>{e.method === "centralized" ? `${formatBytes(e.communication.estimated_raw_data_upload_bytes)} est.` : formatBytes(e.communication.total_bytes)}</TD>
        <TD>{formatDuration(e.duration_seconds)}</TD>
        <TD className="whitespace-nowrap text-slate-500">{formatDateTime(e.created_at)}</TD>
        <TD>
          <div className="flex items-center gap-1">
            <a href={api.exportUrl(e.experiment_id)} download title="Export CSV" className="rounded p-1 hover:bg-slate-100"><Download className="h-3.5 w-3.5 text-slate-600" /></a>
            <button type="button" onClick={onToggle} title="Configuration" className="rounded p-1 hover:bg-slate-100"><FileJson className="h-3.5 w-3.5 text-slate-600" /></button>
            <button type="button" onClick={onDelete} disabled={deleting || e.status === "running" || e.status === "queued"} title="Delete" className="rounded p-1 hover:bg-red-50 disabled:opacity-40"><Trash2 className="h-3.5 w-3.5 text-red-600" /></button>
          </div>
        </TD>
      </TR>
      {expanded ? (
        <TR>
          <TD colSpan={17} className="bg-slate-50/80 p-4"><DetailsPanel id={e.experiment_id} /></TD>
        </TR>
      ) : null}
    </>
  );
}

/** Slightly vary a hex colour for multiple seeds of the same method. */
function shade(hex: string, index: number): string {
  const n = parseInt(hex.slice(1), 16);
  const f = 1 - Math.min(0.45, index * 0.18);
  const r = Math.round(((n >> 16) & 255) * f);
  const g = Math.round(((n >> 8) & 255) * f);
  const b = Math.round((n & 255) * f);
  return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, "0")}`;
}
