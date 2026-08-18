export interface QualityInput {
  mouthWidthPx: number;
  fps: number;
  luminance: number;
  rollDegrees: number;
  faceVisible: boolean;
}

export interface QualityResult {
  score: number;
  label: "good" | "fair" | "poor" | "no face";
}

function bandScore(value: number, low: number, idealLow: number, idealHigh: number, high: number): number {
  if (value <= low || value >= high) return 0;
  if (value >= idealLow && value <= idealHigh) return 1;
  if (value < idealLow) return (value - low) / (idealLow - low);
  return (high - value) / (high - idealHigh);
}

export function computeQuality(input: QualityInput): QualityResult {
  if (!input.faceVisible) return { score: 0, label: "no face" };
  const size = Math.min(1, Math.max(0, (input.mouthWidthPx - 42) / 82));
  const frameRate = Math.min(1, Math.max(0, (input.fps - 12) / 13));
  const light = bandScore(input.luminance, 12, 38, 88, 98);
  const roll = Math.min(1, Math.max(0, 1 - Math.abs(input.rollDegrees) / 22));
  const score = Math.round((size * 0.42 + frameRate * 0.24 + light * 0.2 + roll * 0.14) * 100);
  return { score, label: score >= 76 ? "good" : score >= 50 ? "fair" : "poor" };
}

export function formatResolution(width: number, height: number): string {
  return width > 0 && height > 0 ? `${width}×${height}` : "—";
}
