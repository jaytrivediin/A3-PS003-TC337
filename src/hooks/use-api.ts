"use client";

import { useCallback, useEffect, useRef, useState } from "react";

type Fetcher<T> = (signal: AbortSignal) => Promise<T>;

interface Options<T> {
  enabled?: boolean;
  /** Poll interval in ms, or a function deriving the interval from the latest data (null = stop). */
  pollMs?: number | ((data: T | null) => number | null);
  /** Keep previous data while re-fetching (default true). */
  keepPrevious?: boolean;
}

/**
 * Generic data hook with abort-on-unmount and optional polling.
 * Polling stops automatically when the component unmounts or when `pollMs` returns null.
 */
export function useApi<T>(fetcher: Fetcher<T>, deps: unknown[], options: Options<T> = {}) {
  const { enabled = true, pollMs, keepPrevious = true } = options;
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState<boolean>(enabled);
  const [tick, setTick] = useState(0);
  const fetcherRef = useRef(fetcher);
  fetcherRef.current = fetcher;
  const pollRef = useRef(pollMs);
  pollRef.current = pollMs;

  useEffect(() => {
    if (!enabled) {
      setLoading(false);
      return;
    }
    const controller = new AbortController();
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | undefined;
    if (!keepPrevious) setData(null);
    setLoading(true);

    const run = async () => {
      let latest: T | null = null;
      try {
        latest = await fetcherRef.current(controller.signal);
        if (cancelled) return;
        setData(latest);
        setError(null);
      } catch (err) {
        if (cancelled || (err as Error).name === "AbortError") return;
        setError(err as Error);
      } finally {
        if (!cancelled) setLoading(false);
      }
      if (cancelled) return;
      const p = pollRef.current;
      const interval = typeof p === "function" ? p(latest) : p;
      if (interval && interval > 0) timer = setTimeout(run, interval);
    };
    void run();
    return () => {
      cancelled = true;
      controller.abort();
      if (timer) clearTimeout(timer);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [...deps, enabled, tick]);

  const refresh = useCallback(() => setTick((t) => t + 1), []);
  return { data, error, loading, refresh, setData };
}

export function useDebouncedValue<T>(value: T, delay = 400): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const t = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(t);
  }, [value, delay]);
  return debounced;
}
