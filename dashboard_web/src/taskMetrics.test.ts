import { describe, expect, it } from "vitest";
import type { MetricPoint } from "./types";
import {
  normalizeRunTask,
  primaryMetricValue,
  secondaryMetricValue,
} from "./taskMetrics";

const point = (over: Partial<MetricPoint>): MetricPoint =>
  ({ epoch: 1, ...over }) as MetricPoint;

describe("normalizeRunTask", () => {
  it("passes through multilabel and detection", () => {
    expect(normalizeRunTask("multilabel")).toBe("multilabel");
    expect(normalizeRunTask("detection")).toBe("detection");
  });

  it("defaults unknown/undefined to classification", () => {
    expect(normalizeRunTask("classification")).toBe("classification");
    expect(normalizeRunTask(undefined)).toBe("classification");
    expect(normalizeRunTask("something-else")).toBe("classification");
  });
});

describe("primaryMetricValue", () => {
  it("classification uses accuracy (0 when absent)", () => {
    expect(primaryMetricValue(point({ accuracy: 0.8 }), "classification")).toBe(0.8);
    expect(primaryMetricValue(point({}), "classification")).toBe(0);
  });

  it("multilabel prefers f1_samples, falls back to f1_macro", () => {
    expect(primaryMetricValue(point({ f1_samples: 0.7, f1_macro: 0.5 }), "multilabel")).toBe(0.7);
    expect(primaryMetricValue(point({ f1_macro: 0.5 }), "multilabel")).toBe(0.5);
  });

  it("detection prefers map_50, falls back to map_50_95", () => {
    expect(primaryMetricValue(point({ map_50: 0.6, map_50_95: 0.4 }), "detection")).toBe(0.6);
    expect(primaryMetricValue(point({ map_50_95: 0.4 }), "detection")).toBe(0.4);
  });
});

describe("secondaryMetricValue", () => {
  it("classification/multilabel uses f1_macro", () => {
    expect(secondaryMetricValue(point({ f1_macro: 0.55 }), "classification")).toBe(0.55);
    expect(secondaryMetricValue(point({}), "multilabel")).toBe(0);
  });

  it("detection uses map_50_95, falls back to map_50", () => {
    expect(secondaryMetricValue(point({ map_50_95: 0.3, map_50: 0.6 }), "detection")).toBe(0.3);
    expect(secondaryMetricValue(point({ map_50: 0.6 }), "detection")).toBe(0.6);
  });
});
