"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { FlaskConical, Play, Rocket, Users, Zap } from "lucide-react";
import { useAppState } from "@/components/app-state";
import { ClientDistributionChart, SensorAssignmentChart } from "@/components/charts/charts";
import { Alert, Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle, EmptyState, Field, Input, Select, Skeleton, Table, TBody, TD, TH, THead, TR } from "@/components/ui/primitives";
import { useApi, useDebouncedValue } from "@/hooks/use-api";
import { api } from "@/services/api";
import { cn, formatInt, formatNumber } from "@/lib/utils";
import type { ExperimentConfig, Method, SeedMode } from "@/types/api";

const FALLBACK_QUICK: ExperimentConfig = {
  dataset: "PEMS04", method: "comparison", num_clients: 4, rounds: 5, local_epochs: 1, learning_rate: 0.001, batch_size: 64,
  seed_mode: "single", seed: 42, model_type: "gru", hidden_size: 32, num_layers: 1, lookback: 12, horizon: 12,
  max_train_examples: 4000, max_eval_examples: 2000, label: "Quick comparison",
};

const METHOD_CARDS: { value: Method; title: string; text: string }[] = [
  { value: "centralized", title: "Centralized baseline", text: "All selected training windows pooled on one server." },
  { value: "fed_iid", title: "Federated IID", text: "Windows shuffled and split evenly across simulated clients." },
  { value: "fed_noniid", title: "Federated non-IID", text: "Clients own disjoint sensor groups ranked by mean flow." },
  { value: "comparison", title: "Run complete comparison", text: "Centralized + IID + non-IID with identical settings." },
];

function validate(c: ExperimentConfig): Record<string, string> {
  const errors: Record<string, string> = {};
  const int = (v: number) => Number.isInteger(v);
  if (!int(c.num_clients) || c.num_clients < 3 || c.num_clients > 5) errors.num_clients = "Between 3 and 5 clients.";
  if (!int(c.rounds) || c.rounds < 1 || c.rounds > 500) errors.rounds = "1 – 500 rounds.";
  if (!int(c.local_epochs) || c.local_epochs < 1 || c.local_epochs > 50) errors.local_epochs = "1 – 50 epochs.";
  if (!(c.learning_rate > 0 && c.learning_rate <= 1)) errors.learning_rate = "Must be in (0, 1].";
  if (!int(c.batch_size) || c.batch_size < 4 || c.batch_size > 4096) errors.batch_size = "4 – 4096.";
  if (!int(c.seed) || c.seed < 0) errors.seed = "Non-negative integer.";
  if (!int(c.hidden_size) || c.hidden_size < 4 || c.hidden_size > 512) errors.hidden_size = "4 – 512.";
  if (!int(c.num_layers) || c.num_layers < 1 || c.num_layers > 4) errors.num_layers = "1 – 4 layers.";
  if (!int(c.lookback) || c.lookback < 1 || c.lookback > 288) errors.lookback = "1 – 288 steps.";
  if (!int(c.horizon) || c.horizon < 1 || c.horizon > 288) errors.horizon = "1 – 288 steps.";
  if (c.max_train_examples !== null && (!int(c.max_train_examples) || c.max_train_examples < 50)) errors.max_train_examples = "≥ 50 or empty for all windows.";
  if (c.max_eval_examples !== null && (!int(c.max_eval_examples) || c.max_eval_examples < 50)) errors.max_eval_examples = "≥ 50 or empty for all windows.";
  if (c.max_train_examples !== null && c.max_train_examples < c.num_clients * 4) errors.max_train_examples = "Too few windows for the number of clients.";
  return errors;
}

export default function ExperimentsPage() {
  const router = useRouter();
  const { backendOnline, backendError, dataset, activeJob, refreshActiveJob } = useAppState();
  const defaults = useApi((signal) => api.configDefaults(signal), [backendOnline], { enabled: backendOnline });
  const [config, setConfig] = useState<ExperimentConfig>(FALLBACK_QUICK);
  const [touchedPreset, setTouchedPreset] = useState<"quick" | "full" | "custom">("quick");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [previewStrategy, setPreviewStrategy] = useState<"iid" | "noniid">("noniid");

  useEffect(() => {
    if (defaults.data && touchedPreset === "quick") setConfig(defaults.data.quick);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [defaults.data]);

  const errors = useMemo(() => validate(config), [config]);
  const datasetReady = dataset?.load_status === "loaded";
  const canSubmit = backendOnline && datasetReady && !activeJob && Object.keys(errors).length === 0 && !submitting;

  const seeds = config.seed_mode === "three_seed" ? defaults.data?.three_seeds ?? [42, 123, 2026] : [config.seed];
  const methods = config.method === "comparison" ? ["centralized", "fed_iid", "fed_noniid"] : [config.method];
  const totalRuns = seeds.length * methods.length;

  const update = <K extends keyof ExperimentConfig>(key: K, value: ExperimentConfig[K]) => {
    setTouchedPreset("custom");
    setConfig((c) => ({ ...c, [key]: value }));
  };
  const num = (key: keyof ExperimentConfig, allowEmpty = false) => (e: React.ChangeEvent<HTMLInputElement>) => {
    const raw = e.target.value;
    if (allowEmpty && raw === "") return update(key, null as never);
    update(key, Number(raw) as never);
  };

  const applyPreset = (which: "quick" | "full") => {
    const preset = which === "quick" ? defaults.data?.quick ?? FALLBACK_QUICK : defaults.data?.full ?? { ...FALLBACK_QUICK, rounds: 20, seed_mode: "three_seed" as SeedMode, max_train_examples: 20000, max_eval_examples: 5000, label: "Three-seed evaluation" };
    setConfig(preset);
    setTouchedPreset(which);
  };

  const submit = async (override?: ExperimentConfig) => {
    const payload = override ?? config;
    const errs = validate(payload);
    if (Object.keys(errs).length) return;
    setSubmitting(true);
    setSubmitError(null);
    try {
      await api.createExperiment(payload);
      refreshActiveJob();
      router.push("/simulator");
    } catch (e) {
      setSubmitError((e as Error).message);
      refreshActiveJob();
    } finally {
      setSubmitting(false);
    }
  };

  const effectiveStrategy: "iid" | "noniid" = config.method === "fed_iid" ? "iid" : config.method === "fed_noniid" ? "noniid" : previewStrategy;
  const previewParams = useDebouncedValue({ strategy: effectiveStrategy, num_clients: config.num_clients, seed: config.seed, max_train_examples: config.max_train_examples, lookback: config.lookback, horizon: config.horizon }, 500);
  const previewEnabled = backendOnline && datasetReady && config.method !== "centralized" && !errors.num_clients && !errors.max_train_examples && !errors.lookback && !errors.horizon;
  const preview = useApi((signal) => api.partitionPreview(previewParams, signal), [JSON.stringify(previewParams)], { enabled: previewEnabled });

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">Experiment configuration</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-500">
          Configure a centralized baseline, a federated run or the complete comparison. Every run trains a real PyTorch GRU on PEMS04 windows; nothing is simulated except the client boundaries.
        </p>
      </div>

      {!backendOnline ? <Alert tone="danger" title="Backend offline">{backendError ?? "Waiting for the Python backend."}</Alert> : null}
      {backendOnline && dataset && !datasetReady ? (
        <Alert tone="warning" title="PEMS04 not loaded">Download the dataset in the <Link href="/dataset" className="font-medium underline">Dataset Explorer</Link> before running experiments.</Alert>
      ) : null}
      {activeJob ? (
        <Alert tone="info" title={`A job is already running (${activeJob.runs_finished}/${activeJob.runs_total} runs finished)`}>
          New submissions are disabled until it finishes. <Link href="/simulator" className="font-medium underline">Watch it in the simulator</Link>.
        </Alert>
      ) : null}

      <div className="grid gap-6 xl:grid-cols-3">
        <div className="flex flex-col gap-6 xl:col-span-2">
          <Card>
            <CardHeader className="flex-row flex-wrap items-center justify-between gap-3">
              <div>
                <CardTitle>Presets</CardTitle>
                <CardDescription>Quick: 4 clients · 5 rounds · 1 local epoch · 4 000 training windows. Full: 20 rounds · 20 000 windows · seeds 42, 123, 2026.</CardDescription>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button variant={touchedPreset === "quick" ? "default" : "outline"} size="sm" onClick={() => applyPreset("quick")}><Zap className="h-3.5 w-3.5" /> Quick preset</Button>
                <Button variant={touchedPreset === "full" ? "default" : "outline"} size="sm" onClick={() => applyPreset("full")}><Rocket className="h-3.5 w-3.5" /> Three-seed preset</Button>
              </div>
            </CardHeader>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Training method</CardTitle>
              <CardDescription>Dataset: <Badge tone={datasetReady ? "success" : "warning"}>PEMS04 {datasetReady ? `· ${dataset?.num_sensors} sensors` : "· not loaded"}</Badge></CardDescription>
            </CardHeader>
            <CardContent className="grid gap-3 sm:grid-cols-2">
              {METHOD_CARDS.map((m) => (
                <button
                  key={m.value}
                  type="button"
                  onClick={() => update("method", m.value)}
                  className={cn("rounded-lg border p-3 text-left transition-colors", config.method === m.value ? "border-blue-500 bg-blue-50 ring-2 ring-blue-500/20" : "border-slate-200 bg-white hover:border-slate-300")}
                >
                  <p className="text-sm font-semibold text-slate-900">{m.title}</p>
                  <p className="mt-0.5 text-xs text-slate-500">{m.text}</p>
                </button>
              ))}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Federated & training settings</CardTitle>
              <CardDescription>Centralized rounds use the same epoch budget (one round = local epochs over the pooled windows).</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
              <Field label="Simulated clients (3 – 5)" error={errors.num_clients} hint={config.method === "centralized" ? "Not used by the centralized baseline" : undefined}>
                <Input type="number" min={3} max={5} value={config.num_clients} onChange={num("num_clients")} />
              </Field>
              <Field label="Federated rounds" error={errors.rounds}>
                <Input type="number" min={1} max={500} value={config.rounds} onChange={num("rounds")} />
              </Field>
              <Field label="Local epochs per round" error={errors.local_epochs}>
                <Input type="number" min={1} max={50} value={config.local_epochs} onChange={num("local_epochs")} />
              </Field>
              <Field label="Learning rate (Adam)" error={errors.learning_rate}>
                <Input type="number" step="0.0001" min={0.00001} max={1} value={config.learning_rate} onChange={num("learning_rate")} />
              </Field>
              <Field label="Local batch size" error={errors.batch_size}>
                <Input type="number" min={4} max={4096} value={config.batch_size} onChange={num("batch_size")} />
              </Field>
              <Field label="Random seed mode">
                <Select value={config.seed_mode} onChange={(e) => update("seed_mode", e.target.value as SeedMode)}>
                  <option value="single">Single seed (quick experiment)</option>
                  <option value="three_seed">Three-seed evaluation (42, 123, 2026)</option>
                </Select>
              </Field>
              <Field label="Seed" error={errors.seed} hint={config.seed_mode === "three_seed" ? "Ignored: seeds 42, 123 and 2026 are used." : undefined}>
                <Input type="number" min={0} value={config.seed} disabled={config.seed_mode === "three_seed"} onChange={num("seed")} />
              </Field>
              <Field label="Max training windows (total)" error={errors.max_train_examples} hint="Empty = all available windows (slow). Split evenly across clients.">
                <Input type="number" min={50} value={config.max_train_examples ?? ""} placeholder="all" onChange={num("max_train_examples", true)} />
              </Field>
              <Field label="Max evaluation windows (val / test)" error={errors.max_eval_examples} hint="Fixed sample shared by every compared method.">
                <Input type="number" min={50} value={config.max_eval_examples ?? ""} placeholder="all" onChange={num("max_eval_examples", true)} />
              </Field>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Model</CardTitle>
              <CardDescription>Compact GRU: inputs flow + occupancy + speed, lookback 12 steps (1 h), linear head forecasting 12 steps (1 h) of flow.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <Field label="Architecture"><Input value="GRU + Linear" disabled /></Field>
              <Field label="Hidden size" error={errors.hidden_size}><Input type="number" min={4} max={512} value={config.hidden_size} onChange={num("hidden_size")} /></Field>
              <Field label="GRU layers" error={errors.num_layers}><Input type="number" min={1} max={4} value={config.num_layers} onChange={num("num_layers")} /></Field>
              <Field label="Run label"><Input value={config.label ?? ""} maxLength={120} placeholder="optional" onChange={(e) => update("label", e.target.value || null)} /></Field>
              <Field label="Lookback (steps × 5 min)" error={errors.lookback}><Input type="number" min={1} max={288} value={config.lookback} onChange={num("lookback")} /></Field>
              <Field label="Horizon (steps × 5 min)" error={errors.horizon}><Input type="number" min={1} max={288} value={config.horizon} onChange={num("horizon")} /></Field>
            </CardContent>
          </Card>
        </div>

        <div className="flex flex-col gap-6">
          <Card>
            <CardHeader>
              <CardTitle>Run</CardTitle>
              <CardDescription>{totalRuns} run{totalRuns > 1 ? "s" : ""}: {methods.length} method{methods.length > 1 ? "s" : ""} × {seeds.length} seed{seeds.length > 1 ? "s" : ""} ({seeds.join(", ")}).</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-2">
              <Button onClick={() => submit(defaults.data?.quick ?? FALLBACK_QUICK)} disabled={!backendOnline || !datasetReady || Boolean(activeJob) || submitting} loading={submitting}>
                <Zap className="h-4 w-4" /> Run Quick Experiment
              </Button>
              <Button variant="secondary" onClick={() => submit(defaults.data?.full ?? { ...config, seed_mode: "three_seed", method: "comparison" })} disabled={!backendOnline || !datasetReady || Boolean(activeJob) || submitting} loading={submitting}>
                <Rocket className="h-4 w-4" /> Run Full Three-Seed Evaluation
              </Button>
              <Button variant="outline" onClick={() => submit()} disabled={!canSubmit} loading={submitting}>
                <Play className="h-4 w-4" /> Run with current configuration
              </Button>
              {Object.keys(errors).length ? <p className="text-xs text-red-600">Fix {Object.keys(errors).length} validation issue(s) before running the current configuration.</p> : null}
              {submitError ? <Alert tone="danger" title="Submission failed">{submitError}</Alert> : null}
              <p className="text-[11px] text-slate-500">The quick and full buttons use the backend presets exactly as shown; the third button sends the form above. Jobs run sequentially in the backend.</p>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Exact configuration sent</CardTitle>
              <CardDescription>JSON payload for POST /api/experiments (current form).</CardDescription>
            </CardHeader>
            <CardContent>
              <pre className="max-h-72 overflow-auto rounded-lg bg-slate-900 p-3 text-[11px] leading-relaxed text-slate-100">{JSON.stringify(config, null, 2)}</pre>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="flex-row items-start justify-between gap-2">
              <div>
                <CardTitle>Client data distribution preview</CardTitle>
                <CardDescription>Computed on the training period only, before any training.</CardDescription>
              </div>
              {config.method === "comparison" ? (
                <Select className="w-28" value={previewStrategy} onChange={(e) => setPreviewStrategy(e.target.value as "iid" | "noniid")}>
                  <option value="noniid">non-IID</option>
                  <option value="iid">IID</option>
                </Select>
              ) : null}
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              {config.method === "centralized" ? (
                <EmptyState icon={Users} title="No clients in the centralized baseline" />
              ) : !previewEnabled ? (
                <EmptyState icon={Users} title="Preview unavailable" description="Needs a loaded dataset and a valid client configuration." />
              ) : preview.loading && !preview.data ? (
                <Skeleton className="h-56 w-full" />
              ) : preview.error ? (
                <Alert tone="danger">{preview.error.message}</Alert>
              ) : preview.data ? (
                <>
                  <ClientDistributionChart clients={preview.data.clients} height={200} />
                  {preview.data.sensor_client_assignment ? <SensorAssignmentChart meanFlow={preview.data.sensor_train_mean_flow} assignment={preview.data.sensor_client_assignment} height={150} /> : null}
                  <Table className="text-xs">
                    <THead><TR><TH>Client</TH><TH>Windows</TH><TH>Sensors</TH><TH>Mean flow</TH><TH>σ flow</TH></TR></THead>
                    <TBody>
                      {preview.data.clients.map((c) => (
                        <TR key={c.client_id}><TD>{c.name}</TD><TD>{formatInt(c.num_examples)}</TD><TD>{c.num_sensors}</TD><TD>{formatNumber(c.mean_flow, 1)}</TD><TD>{formatNumber(c.std_flow, 1)}</TD></TR>
                      ))}
                    </TBody>
                  </Table>
                  <p className="text-[11px] text-slate-500">{preview.data.description}</p>
                </>
              ) : null}
            </CardContent>
          </Card>
          <div className="text-[11px] text-slate-500">
            <FlaskConical className="mr-1 inline h-3 w-3" /> Validation/test windows are sampled with a fixed seed ({defaults.data?.eval_sample_seed ?? 7}) so all methods and seeds share the identical test set. Device: {defaults.data?.device ?? "cpu"}.
          </div>
        </div>
      </div>
    </div>
  );
}
