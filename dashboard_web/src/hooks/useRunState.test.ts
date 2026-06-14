import { act, renderHook, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { resolveArtifact, useRunState } from "./useRunState";

/** Minimal controllable WebSocket double. */
class MockWebSocket {
  static instances: MockWebSocket[] = [];
  url: string;
  onopen: (() => void) | null = null;
  onmessage: ((ev: { data: string }) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  readyState = 0;
  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
  }
  close() {
    this.readyState = 3;
  }
}

function jsonResponse(body: unknown): Response {
  return {
    ok: true,
    status: 200,
    statusText: "OK",
    json: async () => body,
  } as Response;
}

let stateFetchCount = 0;

beforeEach(() => {
  MockWebSocket.instances = [];
  stateFetchCount = 0;
  vi.stubGlobal("WebSocket", MockWebSocket as unknown as typeof WebSocket);
  vi.stubGlobal(
    "fetch",
    vi.fn((input: RequestInfo | URL) => {
      const url = String(input);
      if (url.includes("/api/mode")) return Promise.resolve(jsonResponse({ mode: "live" }));
      if (url.endsWith("/api/runs")) {
        return Promise.resolve(jsonResponse({ runs: [{ id: "run1", path: "/p" }] }));
      }
      if (url.includes("/state")) {
        stateFetchCount += 1;
        return Promise.resolve(jsonResponse({ branches: [], metrics: [] }));
      }
      return Promise.resolve(jsonResponse({}));
    }),
  );
  // Ensure no inline export state.
  (window as unknown as { __BNNR_STATE__?: unknown }).__BNNR_STATE__ = undefined;
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("useRunState WebSocket", () => {
  it("opens a WebSocket for the active run", async () => {
    renderHook(() => useRunState());
    await waitFor(() => expect(MockWebSocket.instances.length).toBeGreaterThanOrEqual(1));
    expect(MockWebSocket.instances[0].url).toContain("/ws/run/run1");
  });

  it("refetches run state when a new_events message arrives", async () => {
    renderHook(() => useRunState());
    await waitFor(() => expect(MockWebSocket.instances.length).toBe(1));

    const before = stateFetchCount;
    const ws = MockWebSocket.instances[0];
    act(() => {
      ws.onopen?.();
      ws.onmessage?.({ data: JSON.stringify({ type: "new_events" }) });
    });
    await waitFor(() => expect(stateFetchCount).toBeGreaterThan(before));
  });

  it("ignores non-new_events messages (no refetch)", async () => {
    renderHook(() => useRunState());
    await waitFor(() => expect(MockWebSocket.instances.length).toBe(1));

    const before = stateFetchCount;
    const ws = MockWebSocket.instances[0];
    act(() => {
      ws.onopen?.();
      ws.onmessage?.({ data: JSON.stringify({ type: "heartbeat" }) });
    });
    // Give any (unexpected) async refetch a chance to run.
    await act(async () => {
      await Promise.resolve();
    });
    expect(stateFetchCount).toBe(before);
  });

  it("reconnects after the socket closes", async () => {
    renderHook(() => useRunState());
    await waitFor(() => expect(MockWebSocket.instances.length).toBe(1));

    act(() => {
      MockWebSocket.instances[0].onopen?.();
      MockWebSocket.instances[0].onclose?.();
    });
    // First reconnect backoff is 1000 ms; allow real-timer slack.
    await waitFor(() => expect(MockWebSocket.instances.length).toBe(2), { timeout: 4000 });
    expect(MockWebSocket.instances[1].url).toContain("/ws/run/run1");
  });
});

describe("resolveArtifact", () => {
  it("returns empty string for missing path", () => {
    expect(resolveArtifact(null, "run1", false)).toBe("");
    expect(resolveArtifact(undefined, "run1", false)).toBe("");
  });

  it("passes through absolute http URLs", () => {
    expect(resolveArtifact("http://x/y.png", "run1", false)).toBe("http://x/y.png");
  });

  it("namespaces artifact paths by run when online", () => {
    expect(resolveArtifact("artifacts/a.png", "run1", false)).toBe("/artifacts/run1/artifacts/a.png");
  });

  it("uses relative artifact paths when offline", () => {
    expect(resolveArtifact("artifacts/a.png", "run1", true)).toBe("./artifacts/a.png");
  });
});
