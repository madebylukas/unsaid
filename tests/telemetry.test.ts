import { describe, expect, it } from "vitest";
import { computeQuality, formatResolution } from "../src/telemetry";

describe("capture telemetry", () => {
  it("refuses to score a missing face", () => {
    expect(computeQuality({
      mouthWidthPx: 120,
      fps: 30,
      luminance: 60,
      rollDegrees: 0,
      faceVisible: false,
    })).toEqual({ score: 0, label: "no face" });
  });

  it("rates a clear, level capture as good", () => {
    const result = computeQuality({
      mouthWidthPx: 130,
      fps: 30,
      luminance: 62,
      rollDegrees: 2,
      faceVisible: true,
    });
    expect(result.label).toBe("good");
    expect(result.score).toBeGreaterThanOrEqual(76);
  });

  it("formats only real camera dimensions", () => {
    expect(formatResolution(1280, 720)).toBe("1280×720");
    expect(formatResolution(0, 720)).toBe("—");
  });
});
