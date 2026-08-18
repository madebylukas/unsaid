import type { Candidate } from "./api";

export const DEMO_DECKS = [
  [
    "it's pretty cool you know",
    "hello how are you",
    "thank you very much",
    "i don't know",
    "see you tomorrow",
    "send the message",
    "lukas is really handsome",
  ],
  [
    "please help me",
    "this is not a drill",
    "you look suspicious",
    "we have a problem",
    "meet me outside",
    "the wifi is down",
  ],
] as const;

export const DEMO_PHRASES = DEMO_DECKS.flat();

export interface DemoResolution {
  committed: boolean;
  candidates: Candidate[];
  confidence: number;
  margin: number;
}

function normalize(text: string): string {
  return text
    .toLowerCase()
    .replace(/[’']/g, "")
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function levenshtein(a: string, b: string): number {
  if (!a.length) return b.length;
  if (!b.length) return a.length;
  const previous = Array.from({ length: b.length + 1 }, (_, index) => index);
  for (let row = 1; row <= a.length; row += 1) {
    const current = [row];
    for (let column = 1; column <= b.length; column += 1) {
      current[column] = Math.min(
        current[column - 1] + 1,
        previous[column] + 1,
        previous[column - 1] + (a[row - 1] === b[column - 1] ? 0 : 1),
      );
    }
    previous.splice(0, previous.length, ...current);
  }
  return previous[b.length];
}

function textSimilarity(a: string, b: string): number {
  const left = normalize(a);
  const right = normalize(b);
  if (!left || !right) return 0;
  const characterScore = 1 - levenshtein(left, right) / Math.max(left.length, right.length);
  const leftWords = new Set(left.split(" "));
  const rightWords = new Set(right.split(" "));
  const overlap = [...leftWords].filter((word) => rightWords.has(word)).length;
  const wordScore = (2 * overlap) / (leftWords.size + rightWords.size);
  return characterScore * 0.72 + wordScore * 0.28;
}

function phraseSimilarity(hypothesis: string, phrase: string): number {
  const hypothesisWords = normalize(hypothesis).split(" ").filter(Boolean);
  const phraseWords = normalize(phrase).split(" ").filter(Boolean);
  if (!hypothesisWords.length || !phraseWords.length) return 0;

  let best = textSimilarity(hypothesis, phrase);
  const shortest = Math.max(1, phraseWords.length - 1);
  const longest = Math.min(hypothesisWords.length, phraseWords.length + 2);
  for (let size = shortest; size <= longest; size += 1) {
    for (let start = 0; start + size <= hypothesisWords.length; start += 1) {
      best = Math.max(best, textSimilarity(hypothesisWords.slice(start, start + size).join(" "), phrase));
    }
  }
  return best;
}

export function resolveDemo(candidates: Candidate[], phrases: readonly string[] = DEMO_PHRASES): DemoResolution {
  const ranked = phrases.map((phrase) => {
    let bestMatch = 0;
    let sourceProbability = 0;
    for (const candidate of candidates) {
      const match = phraseSimilarity(candidate.text, phrase);
      if (match > bestMatch) {
        bestMatch = match;
        sourceProbability = candidate.probability;
      }
    }
    const confidence = Math.min(1, bestMatch * 0.92 + sourceProbability * 0.08);
    return { text: phrase, probability: confidence, score: bestMatch };
  }).sort((a, b) => b.probability - a.probability);

  const confidence = ranked[0]?.probability ?? 0;
  const margin = confidence - (ranked[1]?.probability ?? 0);
  return {
    committed: confidence >= 0.86 || (confidence >= 0.68 && margin >= 0.08),
    candidates: ranked.slice(0, 5),
    confidence,
    margin,
  };
}
