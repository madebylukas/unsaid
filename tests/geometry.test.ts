import { describe, expect, it } from "vitest";
import { mapCoverPoint, polygonArea } from "../src/geometry";

describe("object-fit cover geometry", () => {
  it("maps points directly when aspect ratios match", () => {
    expect(mapCoverPoint(.25, .75, 1280, 720, 1280, 720)).toEqual({ x: 320, y: 540 });
  });

  it("accounts for horizontal cropping and mirroring", () => {
    const point = mapCoverPoint(.25, .5, 800, 600, 1280, 720, true);
    expect(point.x).toBeCloseTo(666.67, 1);
    expect(point.y).toBeCloseTo(300, 1);
  });

  it("accounts for vertical cropping", () => {
    const point = mapCoverPoint(.5, .25, 1000, 400, 1280, 720);
    expect(point.x).toBeCloseTo(500, 1);
    expect(point.y).toBeCloseTo(59.38, 1);
  });
});

describe("polygonArea", () => {
  it("measures an axis-aligned mouth box", () => {
    expect(polygonArea([
      { x: 10, y: 10 },
      { x: 50, y: 10 },
      { x: 50, y: 30 },
      { x: 10, y: 30 },
    ])).toBe(800);
  });
});
