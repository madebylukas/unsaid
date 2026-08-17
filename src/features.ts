export interface Point3D {
  x: number;
  y: number;
  z: number;
}

export const SEQUENCE_LENGTH = 32;

const LIP_POINTS = [
  61, 291, 13, 14, 0, 17, 78, 308, 82, 312,
  87, 317, 40, 270, 146, 375, 181, 405, 91, 321,
] as const;

const OPENING_PAIRS = [
  [13, 14],
  [0, 17],
  [82, 87],
  [312, 317],
] as const;

export const BASE_FEATURE_SIZE = LIP_POINTS.length * 2 + OPENING_PAIRS.length + 2;
export const FEATURE_SIZE = BASE_FEATURE_SIZE * 2;

function rotate(x: number, y: number, angle: number): [number, number] {
  const cosine = Math.cos(angle);
  const sine = Math.sin(angle);
  return [x * cosine - y * sine, x * sine + y * cosine];
}

function distance(a: Point3D, b: Point3D): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

export function extractMouthFrame(landmarks: Point3D[]): number[] {
  if (landmarks.length < 406) {
    throw new Error("Face mesh does not contain the required mouth landmarks.");
  }

  const left = landmarks[61];
  const right = landmarks[291];
  const centerX = (left.x + right.x) / 2;
  const centerY = (left.y + right.y) / 2;
  const mouthWidth = Math.max(distance(left, right), 1e-5);
  const correction = -Math.atan2(right.y - left.y, right.x - left.x);

  const features: number[] = [];
  for (const index of LIP_POINTS) {
    const point = landmarks[index];
    const [x, y] = rotate(point.x - centerX, point.y - centerY, correction);
    features.push(x / mouthWidth, y / mouthWidth);
  }

  for (const [upper, lower] of OPENING_PAIRS) {
    features.push(distance(landmarks[upper], landmarks[lower]) / mouthWidth);
  }

  // Depth is noisier than x/y on a webcam, but the two relative cues help with
  // lip protrusion while remaining scale-normalised.
  features.push((landmarks[13].z - landmarks[14].z) / mouthWidth);
  features.push((landmarks[0].z - landmarks[17].z) / mouthWidth);

  return features;
}

function interpolate(a: number[], b: number[], amount: number): number[] {
  return a.map((value, index) => value + (b[index] - value) * amount);
}

export function resampleFrames(
  frames: number[][],
  targetLength = SEQUENCE_LENGTH,
): number[][] {
  if (frames.length === 0) {
    throw new Error("Cannot resample an empty mouth sequence.");
  }
  if (frames.length === 1) {
    return Array.from({ length: targetLength }, () => [...frames[0]]);
  }

  return Array.from({ length: targetLength }, (_, targetIndex) => {
    const sourcePosition = (targetIndex / (targetLength - 1)) * (frames.length - 1);
    const before = Math.floor(sourcePosition);
    const after = Math.min(Math.ceil(sourcePosition), frames.length - 1);
    return interpolate(frames[before], frames[after], sourcePosition - before);
  });
}

export function prepareSequence(frames: number[][]): number[][] {
  const sequence = resampleFrames(frames);
  return sequence.map((frame, index) => {
    const previous = index === 0 ? frame : sequence[index - 1];
    const velocity = frame.map((value, featureIndex) =>
      (value - previous[featureIndex]) * 3,
    );
    return [...frame, ...velocity];
  });
}

export function sequenceMotion(sequence: number[][]): number {
  if (sequence.length < 2) return 0;
  let total = 0;
  for (let frame = 1; frame < sequence.length; frame += 1) {
    for (let feature = 0; feature < BASE_FEATURE_SIZE; feature += 1) {
      total += Math.abs(sequence[frame][feature] - sequence[frame - 1][feature]);
    }
  }
  return total / ((sequence.length - 1) * BASE_FEATURE_SIZE);
}

export const MOUTH_CONTOURS: readonly (readonly number[])[] = [
  [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 0, 39, 40, 185, 61],
  [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 13, 81, 80, 191, 78],
];
