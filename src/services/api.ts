import type {
  ConfigDefaults,
  CreateExperimentResponse,
  DatasetStatus,
  ExperimentConfig,
  ExperimentRecord,
  ExperimentSummary,
  Job,
  PartitionPreview,
  PredictionSeries,
  SensorSeries,
  SensorsOverview,
  WebHealth,
} from "@/types/api";

/** Base URL for API calls. Defaults to the same-origin Next.js proxy (/api -> FastAPI). */
export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "/api").replace(/\/$/, "");

export class ApiError extends Error {
  status: number;
  code?: string;
  detail?: unknown;
  constructor(message: string, status: number, code?: string, detail?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.detail = detail;
  }
}

export class BackendUnreachableError extends ApiError {
  constructor(message: string) {
    super(message, 503, "backend_unreachable");
    this.name = "BackendUnreachableError";
  }
}

async function safeJson(res: Response): Promise<Record<string, unknown> | null> {
  try {
    return (await res.json()) as Record<string, unknown>;
  } catch {
    return null;
  }
}

function describeDetail(detail: unknown, fallback: string): string {
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) {
    return detail
      .map((d) => {
        const item = d as { loc?: unknown[]; msg?: string };
        const loc = Array.isArray(item.loc) ? item.loc.filter((l) => l !== "body").join(".") : "";
        return loc ? `${loc}: ${item.msg ?? ""}` : item.msg ?? "";
      })
      .join("; ");
  }
  if (detail && typeof detail === "object" && "message" in detail) {
    return String((detail as { message: unknown }).message);
  }
  return fallback;
}

export async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers: { Accept: "application/json", ...(init.body ? { "Content-Type": "application/json" } : {}), ...(init.headers ?? {}) },
      cache: "no-store",
    });
  } catch (error) {
    if ((error as Error).name === "AbortError") throw error;
    throw new BackendUnreachableError("The FedTraffic web server could not be reached.");
  }
  if (res.status === 503) {
    const body = await safeJson(res);
    if (body && body.error === "backend_unreachable") {
      throw new BackendUnreachableError(String(body.message ?? "Python backend unreachable"));
    }
  }
  if (!res.ok) {
    const body = await safeJson(res);
    const detail = body?.detail ?? body?.error ?? body?.message;
    const code = detail && typeof detail === "object" && "code" in detail ? String((detail as { code: unknown }).code) : undefined;
    throw new ApiError(describeDetail(detail, `${res.status} ${res.statusText}`), res.status, code, detail);
  }
  if (res.status === 204) return undefined as T;
  return (await res.json()) as T;
}

const qs = (params: Record<string, string | number | boolean | null | undefined>) => {
  const search = new URLSearchParams();
  Object.entries(params).forEach(([k, v]) => {
    if (v !== undefined && v !== null && v !== "") search.set(k, String(v));
  });
  const s = search.toString();
  return s ? `?${s}` : "";
};

export const api = {
  health: (signal?: AbortSignal) => request<WebHealth>("/health", { signal }),
  datasetStatus: (signal?: AbortSignal) => request<DatasetStatus>("/dataset/status", { signal }),
  startDownload: (force = false) => request<{ status: string }>(`/dataset/download${qs({ force })}`, { method: "POST" }),
  sensorSeries: (sensorId: number, params: { start?: number; end?: number; max_points?: number }, signal?: AbortSignal) =>
    request<SensorSeries>(`/dataset/sensor/${sensorId}${qs(params)}`, { signal }),
  sensorsOverview: (signal?: AbortSignal) => request<SensorsOverview>("/dataset/sensors", { signal }),
  partitionPreview: (
    params: { strategy: "iid" | "noniid"; num_clients: number; seed: number; max_train_examples: number | null; lookback: number; horizon: number },
    signal?: AbortSignal,
  ) => request<PartitionPreview>(`/dataset/partitions/preview${qs(params)}`, { signal }),
  configDefaults: (signal?: AbortSignal) => request<ConfigDefaults>("/config/defaults", { signal }),
  createExperiment: (config: Partial<ExperimentConfig>) =>
    request<CreateExperimentResponse>("/experiments", { method: "POST", body: JSON.stringify(config) }),
  listExperiments: (signal?: AbortSignal) => request<{ experiments: ExperimentSummary[]; count: number }>("/experiments", { signal }),
  getExperiment: (id: string, signal?: AbortSignal) => request<ExperimentRecord>(`/experiments/${id}`, { signal }),
  getResults: (id: string, signal?: AbortSignal) => request<ExperimentRecord>(`/experiments/${id}/results`, { signal }),
  predictions: (
    id: string,
    params: { sensor_id: number; start?: number | null; steps: number; horizon_step: number },
    signal?: AbortSignal,
  ) => request<PredictionSeries>(`/experiments/${id}/predictions${qs(params)}`, { signal }),
  deleteExperiment: (id: string) => request<{ deleted: string }>(`/experiments/${id}`, { method: "DELETE" }),
  listJobs: (signal?: AbortSignal) => request<{ jobs: Job[] }>("/jobs", { signal }),
  activeJob: (signal?: AbortSignal) => request<{ job: Job | null }>("/jobs/active", { signal }),
  getJob: (id: string, signal?: AbortSignal) => request<Job>(`/jobs/${id}`, { signal }),
  cancelJob: (id: string) => request<Job>(`/jobs/${id}/cancel`, { method: "POST" }),
  exportUrl: (id: string) => `${API_BASE}/experiments/${id}/export`,
  exportAllUrl: () => `${API_BASE}/export/experiments.csv`,
};
