import { useCallback, useMemo, useState } from "react";
import ReactFlow, {
  Background,
  Controls,
  MarkerType,
  MiniMap,
  type Edge,
  type Node,
} from "reactflow";
import "reactflow/dist/style.css";
import dagre from "dagre";
import type { BranchEdge, BranchNode, DecisionRecord, StatePayload } from "../types";
import { useChartColors, useTheme } from "../ThemeContext";
import { NodeTooltip } from "./NodeTooltip";
import { NodeDetailPanel } from "./NodeDetailPanel";

/* ------------------------------------------------------------------ */
/*  dagre layout                                                       */
/* ------------------------------------------------------------------ */
function layoutGraph(
  nodes: BranchNode[],
  edges: BranchEdge[],
  selectedPath: Set<string>,
) {
  const g = new dagre.graphlib.Graph();
  g.setGraph({ rankdir: "LR", ranksep: 140, nodesep: 50, marginx: 40, marginy: 40 });
  g.setDefaultEdgeLabel(() => ({}));

  nodes.forEach((n) => {
    g.setNode(n.id, { width: 170, height: 56 });
  });
  edges.forEach((e) => {
    g.setEdge(e.from, e.to);
  });

  dagre.layout(g);

  const positions = new Map<string, { x: number; y: number }>();
  nodes.forEach((n) => {
    const pos = g.node(n.id);
    if (pos) positions.set(n.id, { x: pos.x - 85, y: pos.y - 28 });
  });
  return positions;
}

/* ------------------------------------------------------------------ */
/*  Component                                                          */
/* ------------------------------------------------------------------ */
interface Props {
  state: StatePayload;
  activeRun: string;
  offline: boolean;
}

export function BranchGraph({ state, activeRun, offline }: Props) {
  const { theme } = useTheme();
  const cc = useChartColors();
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [hoverNodeId, setHoverNodeId] = useState<string | null>(null);
  const [hoverPos, setHoverPos] = useState<{ x: number; y: number } | null>(null);

  const graph = state.branch_graph;
  const selectedPathSet = useMemo(
    () => new Set(state.selected_path ?? []),
    [state.selected_path],
  );
  const selectedEdgeSet = useMemo(() => {
    const out = new Set<string>();
    (state.selected_path_edges ?? []).forEach((e) => out.add(`${e.from}->${e.to}`));
    return out;
  }, [state.selected_path_edges]);

  const positions = useMemo(
    () => layoutGraph(graph.nodes, graph.edges, selectedPathSet),
    [graph.nodes, graph.edges, selectedPathSet],
  );

  const flowNodes: Node[] = useMemo(() => {
    // Detect duplicate labels under the same parent so we can disambiguate
    const parentOf = new Map<string, string>();
    graph.edges.forEach((e) => {
      // last edge wins (closest parent in graph)
      parentOf.set(e.to, e.from);
    });
    // Count label occurrences per parent
    const labelCountByParent = new Map<string, Map<string, number>>();
    graph.nodes.forEach((n) => {
      const parent = parentOf.get(n.id) ?? "__root__";
      if (!labelCountByParent.has(parent)) labelCountByParent.set(parent, new Map());
      const map = labelCountByParent.get(parent)!;
      map.set(n.label, (map.get(n.label) ?? 0) + 1);
    });

    return graph.nodes.map((n) => {
      const pos = positions.get(n.id) ?? { x: 0, y: 0 };
      const onBestPath = selectedPathSet.has(n.id);
      const isSelected = n.id === selectedNodeId;
      const isRejected = n.status === "rejected";
      const metrics = state.branches[n.id]?.metrics;
      const acc = metrics?.accuracy;
      const accLabel = acc !== undefined ? `${(acc * 100).toFixed(1)}%` : "";
      const bestEpochLabel = n.best_epoch != null ? `e${n.best_epoch}/${n.total_epochs ?? "?"}` : "";

      // Show iteration number when there are duplicate labels under the same parent
      const parent = parentOf.get(n.id) ?? "__root__";
      const siblingLabelCount = labelCountByParent.get(parent)?.get(n.label) ?? 0;
      const showIteration = siblingLabelCount > 1 && n.iteration != null && n.iteration > 0;
      const displayLabel = showIteration ? `${n.label} [iter ${n.iteration}]` : n.label;

      return {
        id: n.id,
        position: pos,
        data: {
          label: (
            <div style={{ textAlign: "center", lineHeight: 1.3 }}>
              <div style={{ fontWeight: 700, fontSize: 12 }}>{displayLabel}</div>
              {accLabel && (
                <div style={{ fontSize: 11, color: onBestPath ? "var(--green)" : isRejected ? "var(--red)" : "var(--muted)" }}>
                  acc {accLabel}{bestEpochLabel ? ` @${bestEpochLabel}` : ""}
                </div>
              )}
              {isRejected && (
                <div style={{ fontSize: 9, color: "var(--red)", fontWeight: 700, textTransform: "uppercase" as const, letterSpacing: 0.5 }}>
                  no improvement
                </div>
              )}
            </div>
          ),
        },
        style: {
          borderRadius: 10,
          border: isSelected
            ? "3px solid var(--accent)"
            : onBestPath
              ? "2px solid var(--green)"
              : isRejected
                ? `1px solid ${theme === "dark" ? "rgba(239,68,68,0.4)" : "#fca5a5"}`
                : `1px solid ${theme === "dark" ? "rgba(148,163,184,0.3)" : "#94a3b8"}`,
          background: onBestPath
            ? (theme === "dark" ? "rgba(34,197,94,0.12)" : "#dcfce7")
            : isRejected
              ? (theme === "dark" ? "rgba(239,68,68,0.08)" : "#fef2f2")
              : n.status === "evaluated"
                ? (theme === "dark" ? "rgba(30,30,50,0.9)" : "#f1f5f9")
                : (theme === "dark" ? "rgba(20,20,35,0.9)" : "#f8fafc"),
          color: theme === "dark" ? "#f1f5f9" : "#1a1a2e",
          padding: "6px 10px",
          minWidth: 150,
          cursor: "pointer",
          opacity: isRejected ? 0.75 : 1,
        },
      };
    });
  }, [graph.nodes, graph.edges, positions, selectedPathSet, selectedNodeId, state.branches]);

  const flowEdges: Edge[] = useMemo(
    () =>
      graph.edges.map((e, i) => {
        const key = `${e.from}->${e.to}`;
        const isBest = selectedEdgeSet.has(key);
        return {
          id: `e${i}`,
          source: e.from,
          target: e.to,
          animated: isBest,
          markerEnd: { type: MarkerType.ArrowClosed, color: isBest ? "#22c55e" : (theme === "dark" ? "#475569" : "#cbd5e1") },
          style: {
            stroke: isBest ? "#22c55e" : (theme === "dark" ? "#475569" : "#cbd5e1"),
            strokeWidth: isBest ? 3 : 1.5,
            strokeDasharray: isBest ? undefined : "6 3",
          },
        };
      }),
    [graph.edges, selectedEdgeSet],
  );

  const handleNodeClick = useCallback((_: unknown, node: Node) => {
    setSelectedNodeId((prev) => (prev === node.id ? null : node.id));
  }, []);

  const handleNodeMouseEnter = useCallback(
    (event: React.MouseEvent, node: Node) => {
      setHoverNodeId(node.id);
      setHoverPos({ x: event.clientX, y: event.clientY });
    },
    [],
  );
  const handleNodeMouseLeave = useCallback(() => {
    setHoverNodeId(null);
    setHoverPos(null);
  }, []);

  // find node data for tooltip / panel
  const hoverNode = hoverNodeId
    ? graph.nodes.find((n) => n.id === hoverNodeId) ?? null
    : null;
  const selectedNode = selectedNodeId
    ? graph.nodes.find((n) => n.id === selectedNodeId) ?? null
    : null;

  // find decision for this node
  const decisionForNode = useCallback(
    (nodeId: string): DecisionRecord | undefined => {
      return (state.decision_history ?? []).find(
        (d) => d.selected_branch_id === nodeId,
      );
    },
    [state.decision_history],
  );

  return (
    <div style={{ position: "relative" }}>
      <div className="flow-box">
        <ReactFlow
          nodes={flowNodes}
          edges={flowEdges}
          fitView
          proOptions={{ hideAttribution: true }}
          onNodeClick={handleNodeClick}
          onNodeMouseEnter={handleNodeMouseEnter}
          onNodeMouseLeave={handleNodeMouseLeave}
          minZoom={0.2}
          maxZoom={2}
        >
          <Background />
          <MiniMap
            nodeColor={(n) =>
              selectedPathSet.has(n.id) ? "#86efac" : (theme === "dark" ? "#2a2a3d" : "#e2e8f0")
            }
            maskColor={theme === "dark" ? "rgba(8,8,16,0.7)" : undefined}
          />
          <Controls />
        </ReactFlow>
      </div>

      {/* Hover tooltip */}
      {hoverNode && hoverPos && (
        <NodeTooltip
          node={hoverNode}
          metrics={state.branches[hoverNode.id]?.metrics}
          position={hoverPos}
          onBestPath={selectedPathSet.has(hoverNode.id)}
          decision={decisionForNode(hoverNode.id)}
        />
      )}

      {/* Click detail panel */}
      {selectedNode && (
        <NodeDetailPanel
          node={selectedNode}
          state={state}
          activeRun={activeRun}
          offline={offline}
          onClose={() => setSelectedNodeId(null)}
        />
      )}
    </div>
  );
}
