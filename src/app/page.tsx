"use client";

import Link from "next/link";
import { useMemo } from "react";
import { ArrowRight, Cpu, FlaskConical, Gauge, Network, Radio, Sigma } from "lucide-react";
import { useAppState } from "@/components/app-state";
import { ConvergenceChart, CumulativeCommunicationChart, MetricBarChart, type SeriesSpec } from "@/components/charts/charts";
import { SensorExplorer } from "@/components/dataset/sensor-explorer";
import { Alert, Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle, EmptyState, Skeleton, Table, TBody, TD, TH, THead, TR } from "@/components/ui/primitives";
import { useApi } from "@/hooks/use-api";
import { api } from "@/services/api";
import { formatBytes, formatDateTime, formatDuration, formatInt, formatNumber, METHOD_COLORS, METHOD_LABELS, METHOD_ORDER, statusTone } from "@/lib/utils";
import type { ExperimentRecord, ExperimentSummary, RunMethod } from "@/types/api";

function KpiCard({ icon: Icon, label, value, sub, loading }: { icon: React.ComponentType<{ className?: string }>; label: string; value: React.ReactNode; sub?: React.ReactNode; loading?: boolean }) {
  return (
    <Card>
      <CardContent className="flex items-start gap-3 py-4">
        <span className="flex h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-blue-50 text-blue-600">
          <Icon className="h-4 w-4" />
        </span>
        <div className="min-w-0">
          <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">{label}</p>
          {loading ? <Skeleton className="mt-1 h-6 w-20" /> : <p className="truncate text-xl font-semibold text-slate-900">{value}</p>}
          {sub ? <p className="truncate text-[11px] text-slate-500">{sub}</p> : null}
        </div>
      </CardContent>
    </Card>
  );
}

export default function OverviewPage() {
  const { backendOnline, backendError, healthLoading, dataset, activeJob } = useAppState();
  const experiments = useApi((signal) => api.listExperiments(signal), [backendOnline], {
    enabled: backendOnline,
    pollMs: (d) => (d?.experiments.some((e) => e.status === "running" || e.status === "queued") || activeJob ? 4000 : 20000),
  });

  const summaries = experiments.data?.experiments ?? [];
  const completed = useMemo(() => summaries.filter((e) => e.status === "completed"), [summaries]);
  const latestByMethod = useMemo(() => {
    const out: Partial<Record<RunMethod, ExperimentSummary>> = {};
    completed.forEach((e) => {
      if (!out[e.method]) out[e.method] = e; // list is newest-first
    });
    return out;
  }, [completed]);
  const latestIds = METHOD_ORDER.map((m) => latestByMethod[m]?.experiment_id).filter(Boolean).join(",");
  const details = useApi(
    async (signal) => {
      const ids = latestIds ? latestIds.split(",") : [];
      const recs = await Promise.all(ids.map((id) => api.getResults(id, signal)));
      return recs as ExperimentRecord[];
    },
    [latestIds],
    { enabled: backendOnline && latestIds.length > 0 },
  );

  const latest = completed[0];
  const latestFed = completed.find((e) => e.method !== "centralized");
  const fedTotalBytes = completed.filter((e) => e.method !== "centralized").reduce((acc, e) => acc + (e.communication.total_bytes ?? 0), 0);

  const convergenceSeries: SeriesSpec[] = (details.data ?? []).map((r) => ({ key: r.experiment_id, label: `${METHOD_LABELS[r.method]} (seed ${r.seed})`, color: METHOD_COLORS[r.method], history: r.history, dashed: r.method === "centralized" }));
  const commSeries = convergenceSeries.filter((s) => s.history.some((h) => h.round_bytes > 0));
  const compareItems = METHOD_ORDER.filter((m) => latestByMethod[m]?.test_mae != null).map((m) => ({ label: METHOD_LABELS[m], value: latestByMethod[m]!.test_mae as number, color: METHOD_COLORS[m] }));

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight text-slate-900">Overview</h1>
          <p className="mt-1 max-w-2xl text-sm text-slate-500">
            Centralized vs. federated (FedAvg) GRU traffic-flow forecasting on the real PEMS04 dataset, with measured model-payload communication costs.
          </p>
        </div>
        <Link href="/experiments">
          <Button>
            Configure experiment <ArrowRight className="h-4 w-4" />
          </Button>
        </Link>
      </div>

      {!backendOnline && !healthLoading ? (
        <Alert tone="danger" title="Python backend is offline">
          {backendError} Start it from the project root with <code className="rounded bg-white/70 px-1">cd backend &amp;&amp; uvicorn app.main:app --port 8000</code>. The dashboard will reconnect automatically.
        </Alert>
      ) : null}
      {backendOnline && dataset && dataset.load_status !== "loaded" ? (
        <Alert tone="warning" title="PEMS04 has not been downloaded yet">
          Experiments need the real dataset. Go to the <Link href="/dataset" className="font-medium underline">Dataset Explorer</Link> and press “Download dataset”, or follow the manual instructions there.
        </Alert>
      ) : null}

      <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-6">
        <KpiCard icon={Radio} label="Traffic sensors" value={dataset?.num_sensors != null ? formatInt(dataset.num_sensors) : "—"} sub={dataset?.shape ? `PEMS04 · ${dataset.shape.join(" × ")}` : "from the loaded dataset"} loading={backendOnline && !dataset} />
        <KpiCard icon={Network} label="Training clients" value={latestFed?.config.num_clients ?? "—"} sub={latestFed ? `latest federated run (${latestFed.partition_strategy})` : "simulated clients"} />
        <KpiCard icon={Gauge} label="Latest test MAE" value={latest ? formatNumber(latest.test_mae, 2) : "—"} sub={latest ? `${METHOD_LABELS[latest.method]} · veh / 5 min` : "no completed run"} />
        <KpiCard icon={Sigma} label="Latest test RMSE" value={latest ? formatNumber(latest.test_rmse, 2) : "—"} sub={latest ? `seed ${latest.seed}` : "no completed run"} />
        <KpiCard icon={Cpu} label="Federated communication" value={latestFed ? formatBytes(latestFed.communication.total_bytes) : "—"} sub={latestFed ? `latest run · all runs ${formatBytes(fedTotalBytes)}` : "measured model payloads"} />
        <KpiCard icon={FlaskConical} label="Completed runs" value={experiments.data ? formatInt(completed.length) : "—"} sub={summaries.length ? `${summaries.length} total records` : "none yet"} loading={backendOnline && experiments.loading && !experiments.data} />
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Historical traffic flow</CardTitle>
          <CardDescription>Raw PEMS04 observations for one sensor (5-minute intervals). Select a sensor and a time window supported by the data.</CardDescription>
        </CardHeader>
        <CardContent>
          <SensorExplorer />
        </CardContent>
      </Card>

      <div className="grid gap-6 xl:grid-cols-2">
        <Card>
          <CardHeader>
            <CardTitle>Model performance</CardTitle>
            <CardDescription>Validation MAE per round for the latest completed run of each method (lower is better).</CardDescription>
          </CardHeader>
          <CardContent>
            {convergenceSeries.length ? (
              <ConvergenceChart series={convergenceSeries} metric="val_mae" />
            ) : details.loading ? (
              <Skeleton className="h-[280px] w-full" />
            ) : (
              <EmptyState icon={Gauge} title="Run your first experiment to see results" description="Validation MAE across training rounds will appear here once an experiment completes." action={<Link href="/experiments"><Button size="sm" variant="outline">Open experiment configuration</Button></Link>} />
            )}
          </CardContent>
        </Card>
        <Card>
          <CardHeader>
            <CardTitle>Communication cost</CardTitle>
            <CardDescription>Cumulative bytes (download + upload of serialised model state) over federated rounds.</CardDescription>
          </CardHeader>
          <CardContent>
            {commSeries.length ? (
              <CumulativeCommunicationChart series={commSeries} />
            ) : (
              <EmptyState icon={Cpu} title="No federated communication measured yet" description="Federated rounds record the serialised model payload exchanged between the server and every simulated client." />
            )}
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 xl:grid-cols-5">
        <Card className="xl:col-span-2">
          <CardHeader>
            <CardTitle>Method comparison</CardTitle>
            <CardDescription>Test MAE of the latest completed centralized, IID and non-IID runs. Compare runs with identical settings for a fair comparison.</CardDescription>
          </CardHeader>
          <CardContent>
            {compareItems.length ? (
              <>
                <MetricBarChart items={compareItems} unit="MAE (veh / 5 min)" height={220} />
                <Table className="mt-3 text-xs">
                  <THead>
                    <TR><TH>Method</TH><TH>Test MAE</TH><TH>Test RMSE</TH><TH>Communication</TH></TR>
                  </THead>
                  <TBody>
                    {METHOD_ORDER.filter((m) => latestByMethod[m]).map((m) => {
                      const e = latestByMethod[m]!;
                      return (
                        <TR key={m}>
                          <TD><span className="inline-block h-2 w-2 rounded-full" style={{ background: METHOD_COLORS[m] }} /> {METHOD_LABELS[m]}</TD>
                          <TD>{formatNumber(e.test_mae, 3)}</TD>
                          <TD>{formatNumber(e.test_rmse, 3)}</TD>
                          <TD>{m === "centralized" ? `${formatBytes(e.communication.estimated_raw_data_upload_bytes)} (est. raw data)` : formatBytes(e.communication.total_bytes)}</TD>
                        </TR>
                      );
                    })}
                  </TBody>
                </Table>
              </>
            ) : (
              <EmptyState icon={Sigma} title="No completed runs to compare" description="Run the complete comparison (centralized + IID + non-IID) to populate this chart." />
            )}
          </CardContent>
        </Card>
        <Card className="xl:col-span-3">
          <CardHeader className="flex-row items-center justify-between">
            <div>
              <CardTitle>Experiment activity</CardTitle>
              <CardDescription>Most recent runs, their configuration and status.</CardDescription>
            </div>
            <Link href="/results" className="text-xs font-medium text-blue-600 hover:underline">All results →</Link>
          </CardHeader>
          <CardContent className="px-0 pb-2 pt-0">
            {experiments.error && backendOnline ? <Alert tone="danger" className="m-4">{experiments.error.message}</Alert> : null}
            {summaries.length ? (
              <Table className="text-xs">
                <THead>
                  <TR><TH>Run</TH><TH>Method</TH><TH>Seed</TH><TH>Config</TH><TH>Status</TH><TH>Test MAE</TH><TH>Finished</TH></TR>
                </THead>
                <TBody>
                  {summaries.slice(0, 8).map((e) => (
                    <TR key={e.experiment_id}>
                      <TD className="font-mono text-[11px]">{e.experiment_id.slice(-10)}</TD>
                      <TD>{METHOD_LABELS[e.method]}</TD>
                      <TD>{e.seed}</TD>
                      <TD className="text-slate-500">{e.method === "centralized" ? "" : `${e.config.num_clients} clients · `}{e.config.rounds} rounds · lr {e.config.learning_rate}</TD>
                      <TD><Badge tone={statusTone(e.status)}>{e.status}</Badge></TD>
                      <TD>{formatNumber(e.test_mae, 2)}</TD>
                      <TD className="text-slate-500">{e.finished_at ? `${formatDateTime(e.finished_at)} · ${formatDuration(e.duration_seconds)}` : "—"}</TD>
                    </TR>
                  ))}
                </TBody>
              </Table>
            ) : (
              <div className="p-4">
                <EmptyState icon={FlaskConical} title="No experiments yet" description="Start a quick experiment to see activity here." />
              </div>
            )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}
