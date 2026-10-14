import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";
import type { RunMethod, RunStatus } from "@/types/api";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatBytes(bytes: number | null | undefined, digits = 1): string {
  if (bytes === null || bytes === undefined || Number.isNaN(bytes)) return "—";
  if (bytes < 1024) return `${bytes} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let value = bytes / 1024;
  let i = 0;
  while (value >= 1024 && i < units.length - 1) {
    value /= 1024;
    i += 1;
  }
  return `${value.toFixed(digits)} ${units[i]}`;
}

export function formatNumber(value: number | null | undefined, digits = 2): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toLocaleString(undefined, { maximumFractionDigits: digits, minimumFractionDigits: digits });
}

export function formatInt(value: number | null | undefined): string {
  if (value === null || value === undefined || Number.isNaN(value)) return "—";
  return value.toLocaleString();
}

export function formatDuration(seconds: number | null | undefined): string {
  if (seconds === null || seconds === undefined) return "—";
  if (seconds < 60) return `${seconds.toFixed(1)} s`;
  const m = Math.floor(seconds / 60);
  const s = Math.round(seconds % 60);
  return `${m}m ${s}s`;
}

export function formatDateTime(iso: string | null | undefined): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}

export function formatClock(iso: string | null | undefined, index?: number): string {
  if (iso) {
    const d = new Date(iso);
    if (!Number.isNaN(d.getTime())) {
      return `${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
    }
    return iso;
  }
  return index !== undefined ? `t=${index}` : "";
}

export const METHOD_LABELS: Record<RunMethod, string> = {
  centralized: "Centralized",
  fed_iid: "FedAvg IID",
  fed_noniid: "FedAvg non-IID",
};

export const METHOD_COLORS: Record<RunMethod, string> = {
  centralized: "#4f46e5",
  fed_iid: "#2563eb",
  fed_noniid: "#0891b2",
};

export const METHOD_ORDER: RunMethod[] = ["centralized", "fed_iid", "fed_noniid"];

export const SEED_COLORS = ["#2563eb", "#0891b2", "#7c3aed", "#ea580c", "#059669", "#dc2626"];

export function statusTone(status: RunStatus | string | undefined): "neutral" | "info" | "success" | "warning" | "danger" {
  switch (status) {
    case "completed":
      return "success";
    case "running":
    case "training":
    case "downloading":
    case "uploading":
    case "aggregating":
    case "evaluating":
    case "distributing":
      return "info";
    case "queued":
    case "waiting":
    case "cancelled":
      return "warning";
    case "failed":
      return "danger";
    default:
      return "neutral";
  }
}

export function mean(values: number[]): number | null {
  if (!values.length) return null;
  return values.reduce((a, b) => a + b, 0) / values.length;
}

export function std(values: number[]): number | null {
  if (values.length < 2) return values.length === 1 ? 0 : null;
  const m = mean(values) as number;
  return Math.sqrt(values.reduce((acc, v) => acc + (v - m) ** 2, 0) / (values.length - 1));
}

export function downloadTextFile(filename: string, text: string, type = "text/csv") {
  const blob = new Blob([text], { type });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

export function toCsv(rows: Record<string, unknown>[]): string {
  if (!rows.length) return "";
  const headers = Object.keys(rows[0]);
  const escape = (v: unknown) => {
    const s = v === null || v === undefined ? "" : String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  return [headers.join(","), ...rows.map((r) => headers.map((h) => escape(r[h])).join(","))].join("\n");
}
