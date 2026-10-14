"use client";

import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  ErrorBar,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
  ZAxis,
} from "recharts";
import { formatBytes, formatClock, formatNumber, METHOD_COLORS, METHOD_LABELS } from "@/lib/utils";
import type { ClientSummary, PredictionSample, RoundRecord, RunMethod } from "@/types/api";

const GRID = "#e2e8f0";
const AXIS = { fontSize: 11, fill: "#64748b" };
const TOOLTIP_STYLE = { borderRadius: 8, border: "1px solid #e2e8f0", fontSize: 12, boxShadow: "0 4px 12px rgba(15,23,42,0.08)" };

export const CLIENT_COLORS = ["#2563eb", "#0891b2", "#7c3aed", "#ea580c", "#059669"];

/* ------------------------------------------------------ Traffic flow */
export function TrafficFlowChart({
  indices,
  timestamps,
  values,
  unit,
  color = "#2563eb",
  height = 280,
  label = "Flow",
}: {
  indices: number[];
  timestamps: string[] | null;
  values: number[];
  unit: string;
  color?: string;
  height?: number;
  label?: string;
}) {
  const data = values.map((v, i) => ({ i, x: formatClock(timestamps?.[i], indices[i]), v }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <AreaChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
        <defs>
          <linearGradient id="flowFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="5%" stopColor={color} stopOpacity={0.25} />
            <stop offset="95%" stopColor={color} stopOpacity={0.02} />
          </linearGradient>
        </defs>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="x" tick={AXIS} minTickGap={40} tickLine={false} axisLine={{ stroke: GRID }} />
        <YAxis tick={AXIS} tickLine={false} axisLine={false} width={48} label={{ value: unit, angle: -90, position: "insideLeft", fontSize: 11, fill: "#64748b" }} />
        <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v) => [formatNumber(Number(v), 1), label]} />
        <Area type="monotone" dataKey="v" stroke={color} strokeWidth={1.6} fill="url(#flowFill)" dot={false} isAnimationActive={false} />
      </AreaChart>
    </ResponsiveContainer>
  );
}

/* --------------------------------------------------------- Convergence */
export interface SeriesSpec {
  key: string;
  label: string;
  color: string;
  history: RoundRecord[];
  dashed?: boolean;
}

export function ConvergenceChart({
  series,
  metric = "val_mae",
  height = 280,
  unit = "vehicles / 5 min",
}: {
  series: SeriesSpec[];
  metric?: "val_mae" | "val_rmse" | "train_loss_mean";
  height?: number;
  unit?: string;
}) {
  const maxRound = Math.max(0, ...series.map((s) => s.history.length));
  const rows = Array.from({ length: maxRound }, (_, i) => {
    const row: Record<string, number | null> = { round: i + 1 };
    series.forEach((s) => {
      const entry = s.history[i];
      row[s.key] = entry ? (entry[metric] as number) : null;
    });
    return row;
  });
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="round" tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }} label={{ value: "Round", position: "insideBottomRight", offset: -2, fontSize: 11, fill: "#64748b" }} allowDecimals={false} />
        <YAxis tick={AXIS} tickLine={false} axisLine={false} width={52} domain={["auto", "auto"]} label={{ value: metric === "train_loss_mean" ? "MSE (normalised)" : unit, angle: -90, position: "insideLeft", fontSize: 11, fill: "#64748b" }} />
        <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v) => formatNumber(Number(v), 3)} labelFormatter={(l) => `Round ${l}`} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {series.map((s) => (
          <Line key={s.key} type="monotone" dataKey={s.key} name={s.label} stroke={s.color} strokeWidth={2} strokeDasharray={s.dashed ? "5 4" : undefined} dot={{ r: 2.5 }} connectNulls isAnimationActive={false} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------------------------- Communication / round */
export function CommunicationRoundChart({ history, height = 260 }: { history: RoundRecord[]; height?: number }) {
  const data = history.map((h) => ({ round: h.round, download: h.total_download_bytes, upload: h.total_upload_bytes }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="round" tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }} label={{ value: "Round", position: "insideBottomRight", offset: -2, fontSize: 11, fill: "#64748b" }} />
        <YAxis tick={AXIS} tickLine={false} axisLine={false} width={64} tickFormatter={(v) => formatBytes(Number(v), 0)} label={{ value: "Bytes", angle: -90, position: "insideLeft", fontSize: 11, fill: "#64748b" }} />
        <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v) => formatBytes(Number(v))} labelFormatter={(l) => `Round ${l}`} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Bar dataKey="download" name="Download (server → clients)" stackId="a" fill="#2563eb" radius={[0, 0, 0, 0]} isAnimationActive={false} />
        <Bar dataKey="upload" name="Upload (clients → server)" stackId="a" fill="#f97316" radius={[3, 3, 0, 0]} isAnimationActive={false} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function CumulativeCommunicationChart({ series, height = 260 }: { series: SeriesSpec[]; height?: number }) {
  const maxRound = Math.max(0, ...series.map((s) => s.history.length));
  const rows = Array.from({ length: maxRound }, (_, i) => {
    const row: Record<string, number | null> = { round: i + 1 };
    series.forEach((s) => {
      row[s.key] = s.history[i] ? s.history[i].cumulative_bytes : null;
    });
    return row;
  });
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={rows} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="round" tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }} allowDecimals={false} label={{ value: "Round", position: "insideBottomRight", offset: -2, fontSize: 11, fill: "#64748b" }} />
        <YAxis tick={AXIS} tickLine={false} axisLine={false} width={64} tickFormatter={(v) => formatBytes(Number(v), 0)} label={{ value: "Cumulative bytes", angle: -90, position: "insideLeft", fontSize: 11, fill: "#64748b" }} />
        <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v) => formatBytes(Number(v))} labelFormatter={(l) => `Round ${l}`} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {series.map((s) => (
          <Line key={s.key} type="monotone" dataKey={s.key} name={s.label} stroke={s.color} strokeWidth={2} dot={{ r: 2.5 }} connectNulls isAnimationActive={false} strokeDasharray={s.dashed ? "5 4" : undefined} />
        ))}
      </LineChart>
    </ResponsiveContainer>
  );
}

/* --------------------------------------------------- Metric comparison */
export interface MetricBarItem {
  label: string;
  value: number;
  error?: number | null;
  color: string;
  n?: number;
}

export function MetricBarChart({ items, unit, height = 240 }: { items: MetricBarItem[]; unit: string; height?: number }) {
  const data = items.map((i) => ({ ...i, err: i.error ?? 0 }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 12, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="label" tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }} />
        <YAxis tick={AXIS} tickLine={false} axisLine={false} width={52} label={{ value: unit, angle: -90, position: "insideLeft", fontSize: 11, fill: "#64748b" }} />
        <Tooltip
          contentStyle={TOOLTIP_STYLE}
          formatter={(v, _n, p) => {
            const item = p.payload as MetricBarItem;
            const err = item.error ? ` ± ${formatNumber(item.error, 3)}` : "";
            return [`${formatNumber(Number(v), 3)}${err}${item.n && item.n > 1 ? ` (n=${item.n})` : ""}`, "value"];
          }}
        />
        <Bar dataKey="value" radius={[4, 4, 0, 0]} isAnimationActive={false} maxBarSize={72}>
          {data.map((d) => (
            <Cell key={d.label} fill={d.color} />
          ))}
          <ErrorBar dataKey="err" width={6} strokeWidth={1.5} stroke="#334155" />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/* --------------------------------------------- Accuracy vs communication */
export interface TradeoffPoint {
  label: string;
  method: RunMethod;
  bytes: number;
  mae: number;
  estimated?: boolean;
  seed?: number;
}

export function AccuracyVsCommunicationChart({ points, height = 280 }: { points: TradeoffPoint[]; height?: number }) {
  const groups: Record<string, TradeoffPoint[]> = {};
  points.forEach((p) => {
    (groups[p.method] ||= []).push(p);
  });
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 12, right: 20, bottom: 12, left: 0 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" />
        <XAxis dataKey="bytes" type="number" name="Communication" tick={AXIS} tickFormatter={(v) => formatBytes(Number(v), 0)} tickLine={false} axisLine={{ stroke: GRID }} scale="log" domain={["auto", "auto"]} label={{ value: "Total communication (log scale)", position: "insideBottom", offset: -6, fontSize: 11, fill: "#64748b" }} />
        <YAxis dataKey="mae" type="number" name="Test MAE" tick={AXIS} tickLine={false} axisLine={false} width={52} domain={["auto", "auto"]} label={{ value: "Test MAE (veh / 5 min)", angle: -90, position: "insideLeft", fontSize: 11, fill: "#64748b" }} />
        <ZAxis range={[80, 80]} />
        <Tooltip
          contentStyle={TOOLTIP_STYLE}
          cursor={{ strokeDasharray: "3 3" }}
          formatter={(v, name) => (name === "Communication" ? formatBytes(Number(v)) : formatNumber(Number(v), 3))}
          labelFormatter={() => ""}
          content={({ payload }) => {
            const p = payload?.[0]?.payload as TradeoffPoint | undefined;
            if (!p) return null;
            return (
              <div className="rounded-lg border border-slate-200 bg-white px-3 py-2 text-xs shadow-md">
                <p className="font-semibold text-slate-800">{p.label}</p>
                <p>Test MAE: {formatNumber(p.mae, 3)}</p>
                <p>
                  {p.estimated ? "Estimated raw-data upload" : "Measured model payloads"}: {formatBytes(p.bytes)}
                </p>
              </div>
            );
          }}
        />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        {Object.entries(groups).map(([method, pts]) => (
          <Scatter key={method} name={`${METHOD_LABELS[method as RunMethod]}${method === "centralized" ? " (estimate)" : ""}`} data={pts} fill={METHOD_COLORS[method as RunMethod]} shape={method === "centralized" ? "diamond" : "circle"} isAnimationActive={false} />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
}

/* ----------------------------------------------------- Client distribution */
export function ClientDistributionChart({ clients, height = 240 }: { clients: ClientSummary[]; height?: number }) {
  const data = clients.map((c) => ({ name: c.name, examples: c.num_examples, meanFlow: c.mean_flow, sensors: c.num_sensors }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="name" tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }} />
        <YAxis yAxisId="left" tick={AXIS} tickLine={false} axisLine={false} width={52} label={{ value: "Training windows", angle: -90, position: "insideLeft", fontSize: 11, fill: "#64748b" }} />
        <YAxis yAxisId="right" orientation="right" tick={AXIS} tickLine={false} axisLine={false} width={52} label={{ value: "Mean flow", angle: 90, position: "insideRight", fontSize: 11, fill: "#64748b" }} />
        <Tooltip contentStyle={TOOLTIP_STYLE} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Bar yAxisId="left" dataKey="examples" name="Local training windows" radius={[4, 4, 0, 0]} isAnimationActive={false} maxBarSize={56}>
          {data.map((_, i) => (
            <Cell key={i} fill={CLIENT_COLORS[i % CLIENT_COLORS.length]} />
          ))}
        </Bar>
        <Bar yAxisId="right" dataKey="meanFlow" name="Mean training flow (veh / 5 min)" fill="#cbd5e1" radius={[4, 4, 0, 0]} isAnimationActive={false} maxBarSize={56} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function SensorAssignmentChart({ meanFlow, assignment, height = 200 }: { meanFlow: number[]; assignment: number[] | null; height?: number }) {
  const order = meanFlow.map((v, i) => ({ v, i })).sort((a, b) => a.v - b.v);
  const data = order.map(({ v, i }, rank) => ({ rank, sensor: i, flow: v, client: assignment ? assignment[i] : 0 }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }} barCategoryGap={0}>
        <XAxis dataKey="rank" tick={AXIS} tickLine={false} axisLine={{ stroke: GRID }} label={{ value: "Sensors ranked by training-period mean flow", position: "insideBottom", offset: -2, fontSize: 11, fill: "#64748b" }} />
        <YAxis tick={AXIS} tickLine={false} axisLine={false} width={48} label={{ value: "veh / 5 min", angle: -90, position: "insideLeft", fontSize: 11, fill: "#64748b" }} />
        <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v, _n, p) => [`${formatNumber(Number(v), 1)} (sensor ${(p.payload as { sensor: number }).sensor}${assignment ? `, client ${(p.payload as { client: number }).client}` : ""})`, "Mean flow"]} labelFormatter={() => ""} />
        <Bar dataKey="flow" isAnimationActive={false}>
          {data.map((d) => (
            <Cell key={d.sensor} fill={assignment ? CLIENT_COLORS[(d.client - 1 + CLIENT_COLORS.length) % CLIENT_COLORS.length] : "#94a3b8"} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

/* ------------------------------------------------------------ Predictions */
export function PredictionChart({
  indices,
  timestamps,
  actual,
  predicted,
  height = 280,
}: {
  indices: number[];
  timestamps: string[] | null;
  actual: number[];
  predicted: number[];
  height?: number;
}) {
  const data = actual.map((a, i) => ({ x: formatClock(timestamps?.[i], indices[i]), actual: a, predicted: predicted[i] }));
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 12, bottom: 4, left: 0 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="x" tick={AXIS} minTickGap={40} tickLine={false} axisLine={{ stroke: GRID }} />
        <YAxis tick={AXIS} tickLine={false} axisLine={false} width={48} label={{ value: "vehicles / 5 min", angle: -90, position: "insideLeft", fontSize: 11, fill: "#64748b" }} />
        <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v) => formatNumber(Number(v), 1)} />
        <Legend wrapperStyle={{ fontSize: 11 }} />
        <Line type="monotone" dataKey="actual" name="Actual flow" stroke="#0f172a" strokeWidth={1.6} dot={false} isAnimationActive={false} />
        <Line type="monotone" dataKey="predicted" name="Predicted flow" stroke="#2563eb" strokeWidth={1.8} dot={false} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export function PredictionSampleChart({ sample, height = 160 }: { sample: PredictionSample; height?: number }) {
  const L = sample.input_flow.length;
  const data = [
    ...sample.input_flow.map((v, i) => ({ step: i - L + 1, history: v, actual: null as number | null, predicted: null as number | null })),
    ...sample.actual.map((v, i) => ({ step: i + 1, history: null as number | null, actual: v, predicted: sample.predicted[i] })),
  ];
  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 6, right: 8, bottom: 0, left: -12 }}>
        <CartesianGrid stroke={GRID} strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="step" tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} axisLine={{ stroke: GRID }} />
        <YAxis tick={{ fontSize: 10, fill: "#64748b" }} tickLine={false} axisLine={false} />
        <ReferenceLine x={0} stroke="#94a3b8" strokeDasharray="4 3" />
        <Tooltip contentStyle={TOOLTIP_STYLE} formatter={(v) => formatNumber(Number(v), 1)} labelFormatter={(l) => `${Number(l) <= 0 ? "History" : "Forecast"} step ${l} (${Number(l) * 5} min)`} />
        <Line type="monotone" dataKey="history" name="History" stroke="#94a3b8" strokeWidth={1.5} dot={false} isAnimationActive={false} />
        <Line type="monotone" dataKey="actual" name="Actual" stroke="#0f172a" strokeWidth={1.5} dot={{ r: 2 }} isAnimationActive={false} />
        <Line type="monotone" dataKey="predicted" name="Predicted" stroke="#2563eb" strokeWidth={1.8} dot={{ r: 2 }} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

/* ---------------------------------------------------------- Split bar */
export function SplitBar({ train, val, test }: { train: number; val: number; test: number }) {
  const total = train + val + test || 1;
  const seg = (n: number, color: string, label: string) => (
    <div className={`${color} flex h-8 items-center justify-center text-[11px] font-medium text-white`} style={{ width: `${(100 * n) / total}%` }} title={`${label}: ${n} steps`}>
      {label} · {Math.round((100 * n) / total)}%
    </div>
  );
  return (
    <div className="flex w-full overflow-hidden rounded-md">
      {seg(train, "bg-blue-600", "Train")}
      {seg(val, "bg-cyan-500", "Validation")}
      {seg(test, "bg-slate-500", "Test")}
    </div>
  );
}
