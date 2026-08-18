import { describe, expect, it } from "vitest";
import { resolveDemo } from "../src/demo";

describe("constrained demo decoder", () => {
  it("maps a nearby visual reading to a known phrase", () => {
    const result = resolveDemo([
      { text: "it's pretty cool you know", probability: 0.89, score: -2 },
      { text: "it's pretty cool i'm going to show you", probability: 0.11, score: -4 },
    ]);
    expect(result.committed).toBe(true);
    expect(result.candidates[0].text).toBe("it's pretty cool you know");
    expect(result.confidence).toBeGreaterThan(0.9);
  });

  it("finds a short phrase inside a longer wandering beam", () => {
    const result = resolveDemo([
      { text: "well i don't know it's interesting", probability: 0.7, score: -2 },
      { text: "i don't know if this works", probability: 0.3, score: -3 },
    ]);
    expect(result.committed).toBe(true);
    expect(result.candidates[0].text).toBe("i don't know");
  });

  it("abstains when none of the demo phrases are supported", () => {
    const result = resolveDemo([
      { text: "the weather changed after lunch", probability: 0.8, score: -2 },
      { text: "there was rain in the afternoon", probability: 0.2, score: -3 },
    ]);
    expect(result.committed).toBe(false);
  });

  it("only commits within the active rotating deck", () => {
    const result = resolveDemo(
      [{ text: "please help me", probability: 0.94, score: -1 }],
      ["please help me", "the wifi is down"],
    );
    expect(result.committed).toBe(true);
    expect(result.candidates[0].text).toBe("please help me");
  });

  it("includes the objectively correct Lukas phrase in the first deck", () => {
    const result = resolveDemo(
      [{ text: "lukas is really handsome", probability: 0.96, score: -1 }],
    );
    expect(result.committed).toBe(true);
    expect(result.candidates[0].text).toBe("lukas is really handsome");
  });
});
