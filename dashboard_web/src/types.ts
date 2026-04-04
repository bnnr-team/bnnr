/* Shared TypeScript types for the BNNR Dashboard */

export type RunItem = { id: string; path: string };

export type BranchNode = {
  id: string;
  label: string;
  depth: number;
  best: boolean;
  status?: string;
  augmentation?: string;
  iteration?: number;
  best_epoch?: number;
  total_epochs?: number;
};

export type BranchEdge = { from: string; to: string };

export type MetricPoint = {
  iteration: number;
  epoch: number;
  branch: string;
  loss: number;
  accuracy: number;
  f1_macro: number;
  // Multilabel metrics (present when task=multilabel)
  f1_samples?: number;
  is_best_epoch?: boolean;
};

export type SamplePoint = {
  sample_id: string;
  branch_id?: string;
  iteration: number;
  epoch: number;
  branch: string;
  true_class: number;
  predicted_class: number;
  confidence: number;
  artifacts?: {
    original?: string | null;
    augmented?: string | null;
    xai?: string | null;
    xai_gt?: string | null;
    xai_saliency?: string | null;
    xai_pred?: string | null;
  };
  detection_details?: {
    image_size?: [number, number];
    gt?: Array<{ box: [number, number, number, number]; label: number; saliency_mean?: number; saliency_max?: number }>;
    pred?: Array<{ box: [number, number, number, number]; label: number; score?: number; saliency_mean?: number; saliency_max?: number }>;
  };
};

export type ProbeItem = {
  sample_id: string;
  class_id: number;
  index: number;
};

export type DecisionRecord = {
  iteration: number;
  selected_branch_id: string;
  selected_branch_label: string;
  decision_reason: string;
  baseline_accuracy: number | null;
  results: Record<string, { accuracy?: number; f1_macro?: number; loss?: number }>;
};

export type ConfusionEntry = {
  matrix: number[][];
  labels: number[];
  iteration: number;
  epoch: number;
  branch: string;
};

export type ConfusedWith = {
  class: string;
  count: number;
};

export type PerClassRow = {
  iteration: number;
  epoch: number;
  branch: string;
  accuracy: number;
  support: number;
  precision?: number;
  recall?: number;
  f1?: number;
  xai_insight?: string;
  severity?: "ok" | "warning" | "critical" | "";
  quality_score?: number;
  trend?: "improving" | "stable" | "declining" | "new" | "";
  confused_with?: ConfusedWith[];
  short_text?: string;
  quality_breakdown?: Record<string, number>;
  augmentation_impact?: string;
  baseline_delta?: Record<string, number>;
};

export type XAIDiagnosis = {
  text: string;
  severity: "ok" | "warning" | "critical";
  quality_score: number;
  confused_with: ConfusedWith[];
  trend: "improving" | "stable" | "declining" | "new";
  short_text: string;
  mean_confidence?: number;
  quality_breakdown?: Record<string, number>;
  augmentation_impact?: string;
  baseline_delta?: Record<string, number>;
};

export type XAIInsightEntry = {
  iteration: number;
  epoch: number;
  branch: string;
  insights: Record<string, string>;
  diagnoses?: Record<string, XAIDiagnosis>;
};

export type DataQualityWarning = {
  type: string;
  severity: "critical" | "warning" | "info";
  count: number;
  message: string;
  groups?: number;
  indices?: number[];
  image_paths?: string[];
};

export type DuplicateGroup = {
  indices: number[];
  size: number;
  image_paths?: string[];
};

export type DataQuality = {
  scanned_samples: number;
  warnings: DataQualityWarning[];
  duplicate_groups: DuplicateGroup[];
  total_duplicate_pairs: number;
  total_flagged_images: number;
  summary: string;
};

export type DatasetProfile = {
  num_classes: number;
  class_distribution: Record<string, number>;
  val_class_distribution: Record<string, number>;
  total_train_samples: number;
  total_val_samples: number;
  imbalance_ratio: number;
  image_shape: number[];
  class_names: string[];
  data_quality?: DataQuality;
};

export type StatePayload = {
  run: Record<string, unknown>;
  task?: "classification" | "multilabel";
  selected_path: string[];
  selected_path_edges?: BranchEdge[];
  branches: Record<string, { metrics?: Record<string, number>; best_epoch?: number; total_epochs?: number }>;
  metrics_timeline: MetricPoint[];
  sample_timelines: Record<string, SamplePoint[]>;
  per_class_timeline: Record<string, PerClassRow[]>;
  confusion_timeline: ConfusionEntry[];
  branch_graph: { nodes: BranchNode[]; edges: BranchEdge[] };
  probe_set: ProbeItem[];
  decision_history?: DecisionRecord[];
  dataset_profile?: DatasetProfile;
  sample_branch_snapshots?: Record<
    string,
    { sample_pairs?: string[][]; probe_labels?: number[] }
  >;
  xai_insights_timeline?: XAIInsightEntry[];
  pipeline_phase?: { phase: string; message: string; timestamp?: string } | null;
  pipeline_complete?: boolean;
};

/* Tabs visible in the dashboard */
export type TabId = "overview" | "tree" | "metrics" | "samples" | "analysis" | "insight";
