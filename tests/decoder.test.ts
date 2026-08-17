import { describe, expect, it } from "vitest";
import { fusePredictions, normalisedEntropy } from "../src/decoder";

const tied = [
  { label: "pat", probability: .34 },
  { label: "bat", probability: .33 },
  { label: "mat", probability: .33 },
];

describe("ambiguity decoder", () => {
  it("uses context without destroying the visual evidence", () => {
    const baseball = fusePredictions(tied, "The baseball player swung the ___");
    const yoga = fusePredictions(tied, "Put the yoga ___ on the floor");
    const dog = fusePredictions(tied, "Gently ___ the dog");
    expect(baseball[0].label).toBe("bat");
    expect(yoga[0].label).toBe("mat");
    expect(dog[0].label).toBe("pat");
    expect(baseball.every((candidate) => candidate.fusedScore > 0)).toBe(true);
  });

  it("preserves the visual winner when no context exists", () => {
    expect(fusePredictions(tied, "")[0].label).toBe("pat");
  });

  it("reports maximum uncertainty for a uniform distribution", () => {
    expect(normalisedEntropy([1, 1, 1])).toBeCloseTo(1, 6);
    expect(normalisedEntropy([1, 0, 0])).toBeCloseTo(0, 5);
  });
});
