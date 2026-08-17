import { describe, expect, it } from "vitest";
import {
  BASE_FEATURE_SIZE,
  FEATURE_SIZE,
  SEQUENCE_LENGTH,
  extractMouthFrame,
  prepareSequence,
  resampleFrames,
} from "../src/features";
import type { Point3D } from "../src/features";

function face(scale = 1, offsetX = 0, offsetY = 0): Point3D[] {
  const points = Array.from({ length: 468 }, () => ({ x: offsetX, y: offsetY, z: 0 }));
  const shape: Record<number, [number, number, number]> = {
    61: [-1, 0, 0], 291: [1, 0, 0], 13: [0, -.16, -.02], 14: [0, .17, .02],
    0: [0, -.31, -.03], 17: [0, .32, .03], 78: [-.78, -.06, 0], 308: [.78, -.06, 0],
    82: [-.32, -.14, 0], 312: [.32, -.14, 0], 87: [-.32, .14, 0], 317: [.32, .14, 0],
    40: [-.58, -.25, 0], 270: [.58, -.25, 0], 146: [-.84, .18, 0], 375: [.84, .18, 0],
    181: [-.58, .27, 0], 405: [.58, .27, 0], 91: [-.78, .08, 0], 321: [.78, .08, 0],
  };
  for (const [index, [x, y, z]] of Object.entries(shape)) {
    points[Number(index)] = { x: offsetX + x * scale, y: offsetY + y * scale, z: z * scale };
  }
  return points;
}

describe("mouth features", () => {
  it("is invariant to translation and scale", () => {
    const original = extractMouthFrame(face(.08, .5, .5));
    const shifted = extractMouthFrame(face(.14, .2, .7));
    expect(original).toHaveLength(BASE_FEATURE_SIZE);
    original.forEach((value, index) => expect(value).toBeCloseTo(shifted[index], 5));
  });

  it("resamples endpoints and adds temporal velocity", () => {
    const frames = [[0, 2], [1, 4], [2, 6]];
    const resampled = resampleFrames(frames, 5);
    expect(resampled).toEqual([[0, 2], [.5, 3], [1, 4], [1.5, 5], [2, 6]]);

    const fullFrames = [extractMouthFrame(face(.08)), extractMouthFrame(face(.08, 0, .01))];
    const sequence = prepareSequence(fullFrames);
    expect(sequence).toHaveLength(SEQUENCE_LENGTH);
    expect(sequence[0]).toHaveLength(FEATURE_SIZE);
  });
});
