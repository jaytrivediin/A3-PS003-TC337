"use client";

import { createContext, useContext, type ReactNode } from "react";
import { api, BackendUnreachableError } from "@/services/api";
import { useApi } from "@/hooks/use-api";
import type { DatasetStatus, Job, WebHealth } from "@/types/api";

interface AppState {
  health: WebHealth | null;
  healthLoading: boolean;
  backendOnline: boolean;
  backendError: string | null;
  dataset: DatasetStatus | null;
  datasetLoading: boolean;
  datasetError: Error | null;
  refreshDataset: () => void;
  activeJob: Job | null;
  refreshActiveJob: () => void;
}

const Ctx = createContext<AppState | null>(null);

export function AppStateProvider({ children }: { children: ReactNode }) {
  const health = useApi<WebHealth>((signal) => api.health(signal), [], { pollMs: 10000 });
  const backendOnline = Boolean(health.data?.backend.reachable);

  const dataset = useApi<DatasetStatus>((signal) => api.datasetStatus(signal), [backendOnline], {
    enabled: backendOnline,
    pollMs: (d) => (d?.download.in_progress ? 1500 : 30000),
  });

  const active = useApi<{ job: Job | null }>((signal) => api.activeJob(signal), [backendOnline], {
    enabled: backendOnline,
    pollMs: (d) => (d?.job ? 2000 : 8000),
  });

  let backendError: string | null = null;
  if (health.error) {
    backendError = health.error instanceof BackendUnreachableError ? health.error.message : health.error.message;
  } else if (health.data && !health.data.backend.reachable) {
    backendError = `Python backend not reachable at ${health.data.backend.url}${health.data.backend.error ? ` (${health.data.backend.error})` : ""}.`;
  }

  return (
    <Ctx.Provider
      value={{
        health: health.data,
        healthLoading: health.loading,
        backendOnline,
        backendError,
        dataset: dataset.data,
        datasetLoading: dataset.loading,
        datasetError: dataset.error,
        refreshDataset: dataset.refresh,
        activeJob: active.data?.job ?? null,
        refreshActiveJob: active.refresh,
      }}
    >
      {children}
    </Ctx.Provider>
  );
}

export function useAppState(): AppState {
  const ctx = useContext(Ctx);
  if (!ctx) throw new Error("useAppState must be used within AppStateProvider");
  return ctx;
}
