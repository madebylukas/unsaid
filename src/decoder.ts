export interface RawPrediction {
  label: string;
  probability: number;
}

export interface FusedPrediction extends RawPrediction {
  visualScore: number;
  contextScore: number;
  fusedScore: number;
}

interface LexicalEntry {
  cues: string[];
  phrases?: string[];
}

const LEXICON: Record<string, LexicalEntry> = {
  pat: { cues: ["dog", "cat", "head", "shoulder", "gently", "praise", "pet"] },
  bat: { cues: ["baseball", "cricket", "cave", "wing", "swing", "vampire", "ball"] },
  mat: { cues: ["floor", "yoga", "door", "exercise", "welcome", "wipe", "feet"] },
  pack: { cues: ["bag", "trip", "luggage", "suitcase", "travel", "boxes"] },
  back: { cues: ["return", "behind", "again", "spine", "home", "come"] },
  map: { cues: ["route", "directions", "location", "geography", "road", "navigate"] },
  mac: { cues: ["apple", "computer", "laptop", "software", "desktop", "book"] },
  pin: { cues: ["code", "number", "needle", "badge", "location", "security"] },
  bin: { cues: ["trash", "rubbish", "garbage", "recycle", "waste"] },
  fan: { cues: ["air", "cool", "ceiling", "supporter", "heat", "sports"] },
  van: { cues: ["drive", "vehicle", "delivery", "road", "moving", "car"] },
  park: { cues: ["car", "outside", "trees", "bench", "vehicle", "grass"] },
  bark: { cues: ["dog", "tree", "woof", "loud", "wood"] },
  mark: { cues: ["score", "grade", "pen", "target", "name", "check"] },
};

export const DEMO_CONTEXTS = [
  "The baseball player swung the ___",
  "Leave your shoes on the ___",
  "Gently ___ the dog",
];

function tokenize(text: string): string[] {
  return text.toLowerCase().match(/[a-z']+/g) ?? [];
}

export function contextPrior(label: string, context: string): number {
  if (!context.trim()) return 1;
  const tokens = tokenize(context);
  const entry = LEXICON[label.toLowerCase()];
  let score = 1;

  if (entry) {
    for (const cue of entry.cues) {
      if (tokens.includes(cue)) score += 4;
    }
    for (const phrase of entry.phrases ?? []) {
      if (context.toLowerCase().includes(phrase)) score += 7;
    }
  }

  // A word explicitly present in surrounding text is a modest personal-style cue.
  const occurrences = tokens.filter((token) => token === label.toLowerCase()).length;
  score += occurrences * 1.5;
  return score;
}

function normalise(values: number[]): number[] {
  const total = values.reduce((sum, value) => sum + value, 0) || 1;
  return values.map((value) => value / total);
}

export function fusePredictions(
  predictions: RawPrediction[],
  context: string,
  contextWeight = 0.48,
): FusedPrediction[] {
  if (predictions.length === 0) return [];
  const visual = normalise(predictions.map((prediction) => Math.max(prediction.probability, 1e-6)));
  const contextScores = normalise(
    predictions.map((prediction) => contextPrior(prediction.label, context)),
  );
  const fused = normalise(
    predictions.map((_, index) =>
      Math.pow(visual[index], 1 - contextWeight) *
      Math.pow(contextScores[index], contextWeight),
    ),
  );

  return predictions
    .map((prediction, index) => ({
      ...prediction,
      visualScore: visual[index],
      contextScore: contextScores[index],
      fusedScore: fused[index],
    }))
    .sort((a, b) => b.fusedScore - a.fusedScore);
}

export function normalisedEntropy(probabilities: number[]): number {
  if (probabilities.length <= 1) return 0;
  const values = normalise(probabilities.map((value) => Math.max(value, 1e-9)));
  const entropy = -values.reduce((sum, value) => sum + value * Math.log(value), 0);
  return entropy / Math.log(values.length);
}
