import { describe, expect, it } from "vitest";
import { calibrate, MIN_EACH } from "../src/threshold.js";

const items = (sims: number[], isNew: boolean) => sims.map((bestSim) => ({ bestSim, isNew }));

describe("calibrate", () => {
  it("needs enough reviews of both kinds", () => {
    expect(calibrate(items(Array(50).fill(0.3), true))).toBeNull();
    expect(calibrate([...items(Array(MIN_EACH).fill(0.3), true), ...items(Array(MIN_EACH - 1).fill(0.8), false)])).toBeNull();
  });

  it("finds a threshold between cleanly separated groups", () => {
    const c = calibrate([...items(Array(20).fill(0.3), true), ...items(Array(20).fill(0.7), false)]);
    expect(c!.balancedAccuracy).toBe(1);
    expect(c!.threshold).toBeGreaterThan(0.3);
    expect(c!.threshold).toBeLessThanOrEqual(0.7);
  });

  it("weights both decisions equally however many there are", () => {
    // 100 matches all at 0.6, 10 news at 0.55: plain accuracy would ignore the news.
    const c = calibrate([...items(Array(10).fill(0.55), true), ...items(Array(100).fill(0.6), false)]);
    expect(c!.threshold).toBeGreaterThan(0.55);
    expect(c!.threshold).toBeLessThanOrEqual(0.6);
  });
});
