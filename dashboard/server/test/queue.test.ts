import { describe, expect, it } from "vitest";
import { interleave } from "../src/queue.js";

describe("interleave", () => {
  it("puts an audit item in every fifth slot", () => {
    const out = interleave(["u1", "u2", "u3", "u4", "u5", "u6"], ["a1", "a2"], 10);
    expect(out).toEqual(["u1", "u2", "u3", "u4", "a1", "u5", "u6", "a2"]);
  });

  it("fills from whichever list still has items, up to the limit", () => {
    expect(interleave([], ["a1", "a2"], 5)).toEqual(["a1", "a2"]);
    expect(interleave(["u1", "u2"], [], 5)).toEqual(["u1", "u2"]);
    expect(interleave(["u1", "u2", "u3"], ["a1"], 2)).toEqual(["u1", "u2"]);
  });
});
