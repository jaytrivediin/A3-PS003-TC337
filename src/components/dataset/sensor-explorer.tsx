"use client";

import { useMemo, useState } from "react";
import { Activity } from "lucide-react";
import { useAppState } from "@/components/app-state";
import { TrafficFlowChart } from "@/components/charts/charts";
import { Alert, Badge, EmptyState, Field, Input, Select, Skeleton, Stat, Tabs } from "@/components/ui/primitives";
import { useApi } from "@/hooks/use-api";
import { api } from "@/services/api";
import { formatInt, formatNumber } from "@/lib/utils";

type Feature = "flow" | "occupancy" | "speed";
const STEPS_PER_DAY = 288;
const FEATURE_META: Record<Feature, { label: string; unit: string; color: string }> = {
  flow: { label: "Traffic flow", unit: "vehicles / 5 min", color: "#2563eb" },
  occupancy: { label: "Occupancy", unit: "ratio", color: "#0891b2" },
  speed: { label: "Speed", unit: "mph", color: "#7c3aed" },
};

export function SensorExplorer({ showFeatureToggle = false, showStats = false, height = 280 }: { showFeatureToggle?: boolean; showStats?: boolean; height?: number }) {
  const { dataset, backendOnline } = useAppState();
  const loaded = backendOnline && dataset?.load_status === "loaded";
  const numSensors = dataset?.num_sensors ?? 0;
  const total = dataset?.num_timesteps ?? 0;
  const [sensor, setSensor] = useState(0);
  const [rangeDays, setRangeDays] = useState(1);
  const [dayOffset, setDayOffset] = useState(0);
  const [feature, setFeature] = useState<Feature>("flow");

  const maxDay = Math.max(0, Math.ceil(total / STEPS_PER_DAY) - 1);
  const start = Math.min(dayOffset, maxDay) * STEPS_PER_DAY;
  const end = rangeDays === 0 ? total : Math.min(total, start + rangeDays * STEPS_PER_DAY);

  const series = useApi(
    (signal) => api.sensorSeries(sensor, { start, end, max_points: 2016 }, signal),
    [sensor, start, end],
    { enabled: loaded && total > 0 },
  );

  const dayLabel = useMemo(() => {
    const ts = dataset?.time_range?.derived && dataset.time_range.start ? new Date(dataset.time_range.start) : null;
    if (!ts) return `day index ${dayOffset}`;
    const d = new Date(ts.getTime() + dayOffset * 24 * 3600 * 1000);
    return d.toLocaleDateString(undefined, { year: "numeric", month: "short", day: "numeric" });
  }, [dataset, dayOffset]);

  if (!loaded) {
    return (
      <EmptyState
        icon={Activity}
        title={backendOnline ? "PEMS04 is not loaded" : "Backend offline"}
        description={backendOnline ? "Download the dataset from the Dataset Explorer page to plot historical traffic flow." : "Start the Python backend to load PEMS04 and explore sensor data."}
      />
    );
  }

  const values = series.data ? (series.data[feature] as number[] | undefined) ?? [] : [];
  const meta = FEATURE_META[feature];

  return (
    <div className="flex flex-col gap-4">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Field label={`Sensor (0 – ${numSensors - 1})`}>
          <Input type="number" min={0} max={numSensors - 1} value={sensor} onChange={(e) => setSensor(Math.max(0, Math.min(numSensors - 1, Number(e.target.value) || 0)))} />
        </Field>
        <Field label="Window length">
          <Select value={rangeDays} onChange={(e) => setRangeDays(Number(e.target.value))}>
            <option value={1}>1 day (288 steps)</option>
            <option value={3}>3 days</option>
            <option value={7}>1 week</option>
            <option value={0}>Entire period (downsampled)</option>
          </Select>
        </Field>
        <Field label={`Start day (${dayLabel})`} hint={dataset?.time_range?.derived ? "Dates derived from the documented Jan–Feb 2018 period" : "No timestamps available; index based"}>
          <Input type="number" min={0} max={maxDay} value={dayOffset} disabled={rangeDays === 0} onChange={(e) => setDayOffset(Math.max(0, Math.min(maxDay, Number(e.target.value) || 0)))} />
        </Field>
        {showFeatureToggle ? (
          <Field label="Feature">
            <Tabs value={feature} onChange={setFeature} options={[{ value: "flow", label: "Flow" }, { value: "occupancy", label: "Occupancy" }, { value: "speed", label: "Speed" }]} />
          </Field>
        ) : (
          <div className="flex items-end">
            <Badge tone="neutral">5-minute observations · {formatInt(end - start)} steps</Badge>
          </div>
        )}
      </div>

      {series.error ? <Alert tone="danger" title="Could not load sensor data">{series.error.message}</Alert> : null}
      {series.loading && !series.data ? (
        <Skeleton className="w-full" style={{ height }} />
      ) : series.data ? (
        <>
          <TrafficFlowChart indices={series.data.indices} timestamps={series.data.timestamps} values={values} unit={meta.unit} color={meta.color} height={height} label={meta.label} />
          <p className="text-[11px] text-slate-500">
            Sensor {series.data.sensor_id}, steps {series.data.start}–{series.data.end - 1}
            {series.data.downsampled ? ` · downsampled by stride ${series.data.stride} for display` : ""} · {series.data.num_points} points.
            {series.data.timestamps ? " Timestamps derived from the documented collection period." : " No timestamps in the archive; index-based axis."}
          </p>
        </>
      ) : null}

      {showStats && series.data ? (
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          {series.data.statistics.features.map((f) => (
            <Stat key={f.key} label={`${f.name} (${f.unit})`} value={`μ ${formatNumber(f.mean, 2)} · σ ${formatNumber(f.std, 2)}`} sub={`min ${formatNumber(f.min, 1)} · max ${formatNumber(f.max, 1)} · zeros ${formatInt(f.zero_count)} · NaN ${formatInt(f.nan_count)}`} />
          ))}
          <Stat label="Training-period flow" value={`μ ${formatNumber(series.data.statistics.train_period_flow.mean, 2)}`} sub={`σ ${formatNumber(series.data.statistics.train_period_flow.std, 2)} · used for non-IID grouping`} />
        </div>
      ) : null}
    </div>
  );
}
