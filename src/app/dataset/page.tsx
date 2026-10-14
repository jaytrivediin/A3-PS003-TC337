"use client";

import { useState } from "react";
import { CheckCircle2, Database, Download, FolderOpen, RefreshCw } from "lucide-react";
import { useAppState } from "@/components/app-state";
import { SplitBar } from "@/components/charts/charts";
import { SensorExplorer } from "@/components/dataset/sensor-explorer";
import { Alert, Badge, Button, Card, CardContent, CardDescription, CardHeader, CardTitle, EmptyState, Progress, Skeleton, Stat, Table, TBody, TD, TH, THead, TR } from "@/components/ui/primitives";
import { api } from "@/services/api";
import { formatBytes, formatDateTime, formatInt } from "@/lib/utils";

export default function DatasetPage() {
  const { backendOnline, backendError, dataset, datasetLoading, datasetError, refreshDataset } = useAppState();
  const [starting, setStarting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);

  const startDownload = async (force = false) => {
    setStarting(true);
    setMessage(null);
    try {
      const res = await api.startDownload(force);
      setMessage(res.status === "already_present" ? "Archive already present." : res.status === "in_progress" ? "Download already in progress." : "Download started.");
      refreshDataset();
    } catch (e) {
      setMessage((e as Error).message);
    } finally {
      setStarting(false);
    }
  };

  const loaded = dataset?.load_status === "loaded";
  const dl = dataset?.download;

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold tracking-tight text-slate-900">Dataset Explorer</h1>
        <p className="mt-1 max-w-3xl text-sm text-slate-500">
          Inspect the real PeMSD4 / PEMS04 archive used for all experiments. Nothing on this page is synthetic: dimensions, statistics and series come from the loaded NPZ array.
        </p>
      </div>

      {!backendOnline ? <Alert tone="danger" title="Backend offline">{backendError ?? "Waiting for the Python backend."}</Alert> : null}
      {datasetError && backendOnline ? <Alert tone="danger" title="Could not read dataset status">{datasetError.message}</Alert> : null}

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-2">
          <CardHeader className="flex-row items-start justify-between gap-3">
            <div>
              <CardTitle>Dataset status</CardTitle>
              <CardDescription>Source, availability and load state of the archive on the backend machine.</CardDescription>
            </div>
            {dataset ? (
              <Badge tone={loaded ? "success" : dataset.load_status === "error" ? "danger" : "warning"} className="px-2.5 py-1">
                {loaded ? "Loaded" : dataset.load_status === "error" ? "Invalid archive" : "Not downloaded"}
              </Badge>
            ) : null}
          </CardHeader>
          <CardContent className="flex flex-col gap-4">
            {datasetLoading && !dataset ? (
              <Skeleton className="h-32 w-full" />
            ) : dataset ? (
              <>
                <div className="grid gap-3 sm:grid-cols-2">
                  <Stat label="Dataset" value={dataset.name} sub={dataset.source} />
                  <Stat label="Source" value={<a href={dataset.source_repository} target="_blank" rel="noreferrer" className="text-blue-600 hover:underline">Davidham3/ASTGCN · data/PEMS04</a>} sub={dataset.citation} />
                  <Stat label="Local path" value={<span className="break-all font-mono text-xs">{dataset.local_path}</span>} sub={dataset.file_exists ? `${formatBytes(dataset.file_size_bytes)} on disk` : "file missing"} />
                  <Stat label="Sampling interval" value={`${dataset.sampling_interval_minutes} minutes`} sub="288 observations per day · lookback 12 = 1 h, horizon 12 = 1 h" />
                </div>
                {dataset.load_status === "error" ? <Alert tone="danger" title="The archive exists but failed validation">{dataset.error}</Alert> : null}
                {!dataset.file_exists || dataset.load_status === "error" ? (
                  <div className="rounded-lg border border-orange-200 bg-orange-50 p-4 text-sm text-orange-900">
                    <p className="font-semibold">Download or set up the dataset</p>
                    <p className="mt-1 text-[13px]">
                      Press the button to download the real archive ({`≈ 33 MB`}) from the ASTGCN repository. The backend validates the NPZ before accepting it. Manual option: {dataset.manual_instructions}
                    </p>
                    <div className="mt-3 flex flex-wrap items-center gap-2">
                      <Button onClick={() => startDownload(dataset.load_status === "error")} loading={starting || Boolean(dl?.in_progress)} disabled={!backendOnline}>
                        <Download className="h-4 w-4" /> {dl?.in_progress ? "Downloading…" : "Download dataset"}
                      </Button>
                      <Button variant="outline" onClick={refreshDataset}><RefreshCw className="h-4 w-4" /> Refresh status</Button>
                    </div>
                    {dl?.in_progress ? (
                      <div className="mt-3">
                        <Progress value={dl.percent ?? 0} tone="warning" />
                        <p className="mt-1 text-xs">{formatBytes(dl.bytes_downloaded)} / {formatBytes(dl.total_bytes)} ({dl.percent ?? 0}%)</p>
                      </div>
                    ) : null}
                    {dl?.error ? <p className="mt-2 text-xs text-red-700">Download failed: {dl.error}</p> : null}
                    {message ? <p className="mt-2 text-xs">{message}</p> : null}
                  </div>
                ) : (
                  <div className="flex flex-wrap items-center gap-2 text-xs text-emerald-700">
                    <CheckCircle2 className="h-4 w-4" /> Archive validated: key <code className="rounded bg-slate-100 px-1 text-slate-800">{dataset.array_key}</code> ({dataset.archive_keys?.join(", ")}), stored as {dataset.original_dtype}, loaded as {dataset.in_memory_dtype} ({formatBytes(dataset.in_memory_bytes)} in memory).
                    {dataset.shape_matches_documentation ? " Shape matches the documented (16992, 307, 3)." : " Shape differs from the documented (16992, 307, 3)."}
                    <Button variant="ghost" size="sm" onClick={() => startDownload(true)} disabled={dl?.in_progress}><Download className="h-3.5 w-3.5" /> Re-download</Button>
                  </div>
                )}
              </>
            ) : (
              <EmptyState icon={Database} title="Dataset status unavailable" description="Connect the backend to read the dataset status." />
            )}
          </CardContent>
        </Card>

        <Card>
          <CardHeader>
            <CardTitle>Array dimensions</CardTitle>
            <CardDescription>Actual dimensions of the loaded NPZ array.</CardDescription>
          </CardHeader>
          <CardContent className="flex flex-col gap-3">
            {loaded && dataset ? (
              <>
                <Stat label="Shape (time × sensors × features)" value={dataset.shape?.join(" × ")} />
                <div className="grid grid-cols-3 gap-2">
                  <Stat label="Time steps" value={formatInt(dataset.num_timesteps)} />
                  <Stat label="Sensors" value={formatInt(dataset.num_sensors)} />
                  <Stat label="Features" value={formatInt(dataset.num_features)} />
                </div>
                <Stat label="Covered period" value={dataset.time_range?.derived ? `${formatDateTime(dataset.time_range.start)} → ${formatDateTime(dataset.time_range.end)}` : "Not derivable"} sub={`${dataset.duration_days} days · ${dataset.time_range?.basis}`} />
                <Stat label="Missing values" value={`${formatInt(dataset.missing_values?.nan_count)} NaN · ${formatInt(dataset.missing_values?.zero_flow_count)} zero-flow cells`} sub={dataset.missing_values?.note} />
                <p className="text-[11px] text-slate-500">The archive provides no sensor coordinates; FedTraffic therefore shows no map or geographic locations.</p>
              </>
            ) : (
              <EmptyState icon={FolderOpen} title="No array loaded" />
            )}
          </CardContent>
        </Card>
      </div>

      {loaded && dataset ? (
        <div className="grid gap-6 lg:grid-cols-2">
          <Card>
            <CardHeader>
              <CardTitle>Feature channels</CardTitle>
              <CardDescription>Default channel mapping (configurable via FEDTRAFFIC_*_CHANNEL). Verified against the loaded array&apos;s third dimension.</CardDescription>
            </CardHeader>
            <CardContent className="px-0 pt-0">
              <Table className="text-xs">
                <THead><TR><TH>Channel</TH><TH>Feature</TH><TH>Unit</TH><TH>Role in model</TH></TR></THead>
                <TBody>
                  {dataset.features.map((f) => (
                    <TR key={f.index}><TD className="font-mono">{f.index}</TD><TD className="font-medium text-slate-900">{f.name}</TD><TD>{f.unit}</TD><TD>{f.role}</TD></TR>
                  ))}
                </TBody>
              </Table>
            </CardContent>
          </Card>
          <Card>
            <CardHeader>
              <CardTitle>Chronological train / validation / test split</CardTitle>
              <CardDescription>{dataset.split?.strategy}. Forecast windows never cross a split boundary; normalisation statistics come from the training period only.</CardDescription>
            </CardHeader>
            <CardContent className="flex flex-col gap-3">
              {dataset.split ? (
                <>
                  <SplitBar train={dataset.split.partitions.train.num_timesteps} val={dataset.split.partitions.val.num_timesteps} test={dataset.split.partitions.test.num_timesteps} />
                  <Table className="text-xs">
                    <THead><TR><TH>Partition</TH><TH>Index range</TH><TH>Steps</TH><TH>Derived period</TH></TR></THead>
                    <TBody>
                      {(["train", "val", "test"] as const).map((k) => {
                        const p = dataset.split!.partitions[k];
                        return (
                          <TR key={k}>
                            <TD className="font-medium capitalize text-slate-900">{k === "val" ? "Validation" : k}</TD>
                            <TD className="font-mono">[{p.start_index}, {p.end_index})</TD>
                            <TD>{formatInt(p.num_timesteps)}</TD>
                            <TD>{p.start_timestamp ? `${formatDateTime(p.start_timestamp)} → ${formatDateTime(p.end_timestamp)}` : "—"}</TD>
                          </TR>
                        );
                      })}
                    </TBody>
                  </Table>
                </>
              ) : null}
            </CardContent>
          </Card>
        </div>
      ) : null}

      <Card>
        <CardHeader>
          <CardTitle>Traffic data explorer</CardTitle>
          <CardDescription>Historical series of one sensor with per-feature statistics, zero counts and missing values.</CardDescription>
        </CardHeader>
        <CardContent>
          <SensorExplorer showFeatureToggle showStats height={300} />
        </CardContent>
      </Card>
    </div>
  );
}
