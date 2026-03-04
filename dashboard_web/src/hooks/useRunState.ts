import { useCallback, useEffect, useRef, useState } from "react";
import type { RunItem, StatePayload } from "../types";

/* Global injection point for offline/exported dashboards */
declare global {
  interface Window {
    __BNNR_STATE__?: StatePayload;
  }
}

async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return (await res.json()) as T;
}

/**
 * Central hook — manages run list, active run, state fetching and WebSocket.
 * 
 * Data sources (tried in order):
 * 1. window.__BNNR_STATE__  — injected inline by export (works with file://)
 * 2. /api/runs + /api/run/:id/state — live dashboard (FastAPI backend)
 * 3. ./data/state.json — offline export served via HTTP
 */
export function useRunState() {
  const LIVE_REFRESH_MS = 1500;
  const [runs, setRuns] = useState<RunItem[]>([]);
  const [activeRun, setActiveRun] = useState("");
  const [state, setState] = useState<StatePayload | null>(null);
  const [offline, setOffline] = useState(false);
  const [dashboardMode, setDashboardMode] = useState<"live" | "serve">("live");
  // Start as loading=true unless we have inline state
  const [loading, setLoading] = useState(!window.__BNNR_STATE__);
  const runsRefreshInFlight = useRef(false);

  const refreshRuns = useCallback(async () => {
    if (offline || runsRefreshInFlight.current) return;
    runsRefreshInFlight.current = true;
    try {
      const payload = await fetchJson<{ runs: RunItem[] }>("/api/runs");
      setRuns(payload.runs);
      if (payload.runs.length > 0) {
        // Auto-select the newest run when none is selected yet.
        if (!activeRun) {
          setActiveRun(payload.runs[0].id);
        } else {
          // If current run disappeared, switch to newest available run.
          const exists = payload.runs.some((r) => r.id === activeRun);
          if (!exists) setActiveRun(payload.runs[0].id);
        }
      }
    } catch {
      // Keep existing UI state; caller controls retry cadence.
    } finally {
      runsRefreshInFlight.current = false;
    }
  }, [activeRun, offline]);

  /* ---- initial data loading ---- */
  useEffect(() => {
    // 1) Check for inline injected state (export opened via file://)
    if (window.__BNNR_STATE__) {
      setOffline(true);
      setState(window.__BNNR_STATE__);
      setLoading(false);
      return;
    }

    // 2) Try live API
    // Fetch dashboard mode (serve vs live)
    fetchJson<{ mode: string }>("/api/mode")
      .then((m) => { if (m.mode === "serve") setDashboardMode("serve"); })
      .catch(() => {});

    setLoading(true);
    fetchJson<{ runs: RunItem[] }>("/api/runs")
      .then((p) => {
        setRuns(p.runs);
        if (p.runs.length > 0) {
          setActiveRun(p.runs[0].id);
        } else {
          setLoading(false);
        }
      })
      .catch(async () => {
        // 3) Try offline state.json (export served via HTTP)
        try {
          const replay = await fetchJson<StatePayload>("./data/state.json");
          setOffline(true);
          setState(replay);
        } catch {
          /* no data at all */
        }
        setLoading(false);
      });
  }, []);

  /* ---- refresh run list while waiting for first live run ---- */
  useEffect(() => {
    if (offline) return;
    // Always do one immediate refresh to catch runs created after page load.
    void refreshRuns();

    // If no run is selected yet, poll run list until one appears.
    if (activeRun) return;
    const timer = setInterval(() => {
      void refreshRuns();
    }, 2000);
    return () => clearInterval(timer);
  }, [activeRun, offline, refreshRuns]);

  /* ---- fetch state when run changes ---- */
  const abortRef = useRef<AbortController | null>(null);
  useEffect(() => {
    if (!activeRun || offline) return;
    // Cancel any in-flight request
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    setLoading(true);
    fetchJson<StatePayload>(`/api/run/${activeRun}/state`, controller.signal)
      .then((s) => {
        if (!controller.signal.aborted) setState(s);
      })
      .catch((err) => {
        if (err instanceof DOMException && err.name === "AbortError") return;
        console.error("Failed to fetch run state:", err);
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });

    return () => controller.abort();
  }, [activeRun, offline]);

  /* ---- WebSocket with low-latency throttle and auto-reconnect ---- */
  const lastRefresh = useRef(0);
  const refreshTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const refreshInFlight = useRef(false);

  const refreshState = useCallback(() => {
    if (!activeRun || offline || refreshInFlight.current) return;
    refreshInFlight.current = true;
    fetchJson<StatePayload>(`/api/run/${activeRun}/state`)
      .then(setState)
      .catch(() => undefined)
      .finally(() => { refreshInFlight.current = false; });
  }, [activeRun, offline]);

  useEffect(() => {
    if (!activeRun || offline) return;

    let reconnectDelay = 1000;
    let reconnectTimer: ReturnType<typeof setTimeout> | null = null;
    let cancelled = false;

    function connect() {
      if (cancelled) return;

      // Use wss:// when page is served over HTTPS, ws:// otherwise
      const wsProtocol = window.location.protocol === "https:" ? "wss:" : "ws:";
      const ws = new WebSocket(`${wsProtocol}//${window.location.host}/ws/run/${activeRun}`);

      ws.onopen = () => {
        // Reset backoff on successful connection
        reconnectDelay = 1000;
      };

      ws.onmessage = (ev) => {
        // Only trigger refresh on "new_events" messages, ignore heartbeats
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type !== "new_events") return;
        } catch {
          // Not JSON — ignore
          return;
        }

        const now = Date.now();
        const elapsed = now - lastRefresh.current;
        if (elapsed >= LIVE_REFRESH_MS) {
          lastRefresh.current = now;
          refreshState();
        } else if (!refreshTimer.current) {
          refreshTimer.current = setTimeout(() => {
            lastRefresh.current = Date.now();
            refreshTimer.current = null;
            refreshState();
          }, LIVE_REFRESH_MS - elapsed);
        }
      };

      ws.onclose = () => {
        if (cancelled) return;
        // Exponential backoff reconnect (max 30s)
        reconnectTimer = setTimeout(() => {
          reconnectDelay = Math.min(reconnectDelay * 2, 30_000);
          connect();
        }, reconnectDelay);
      };

      ws.onerror = () => {
        // onerror is always followed by onclose, which handles reconnect
      };

      // Store reference for cleanup
      wsRef.current = ws;
    }

    const wsRef = { current: null as WebSocket | null };
    connect();

    return () => {
      cancelled = true;
      wsRef.current?.close();
      if (refreshTimer.current) clearTimeout(refreshTimer.current);
      if (reconnectTimer) clearTimeout(reconnectTimer);
    };
  }, [activeRun, offline, refreshState, LIVE_REFRESH_MS]);

  return { runs, activeRun, setActiveRun, state, offline, loading, dashboardMode };
}

/** Resolve an artifact path to a URL the browser can fetch. */
export function resolveArtifact(
  path: string | null | undefined,
  activeRun: string,
  offline: boolean,
): string {
  if (!path) return "";
  if (path.startsWith("http")) return path;
  if (offline)
    return path.startsWith("artifacts/") ? `./${path}` : path;
  return path.startsWith("artifacts/")
    ? `/artifacts/${activeRun}/${path}`
    : path;
}
