import type { LayersModel, Sequential, Tensor } from "@tensorflow/tfjs";
import { FEATURE_SIZE, SEQUENCE_LENGTH } from "./features";
import type { RawPrediction } from "./decoder";

export interface TrainingSample {
  label: string;
  sequence: number[][];
}

export interface TrainingUpdate {
  epoch: number;
  totalEpochs: number;
  accuracy: number;
  loss: number;
}

const MODEL_URL = "localstorage://shh-type-mouth-model";
const LABELS_KEY = "shh-type-model-labels";
type TfModule = typeof import("@tensorflow/tfjs");
let tfModulePromise: Promise<TfModule> | null = null;

function loadTensorFlow(): Promise<TfModule> {
  tfModulePromise ??= import("@tensorflow/tfjs");
  return tfModulePromise;
}

function gaussianNoise(): number {
  const a = Math.max(Math.random(), 1e-9);
  const b = Math.random();
  return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
}

function augment(sequence: number[][], strength: number): number[][] {
  const stretch = 0.92 + Math.random() * 0.16;
  return sequence.map((_frame, frameIndex) => {
    const sourceIndex = Math.min(
      sequence.length - 1,
      Math.max(0, Math.round((frameIndex - sequence.length / 2) * stretch + sequence.length / 2)),
    );
    return sequence[sourceIndex].map((value, featureIndex) => {
      const noiseScale = featureIndex < FEATURE_SIZE / 2 ? strength : strength * 1.8;
      return value + gaussianNoise() * noiseScale;
    });
  });
}

function createModel(tf: TfModule, classCount: number): Sequential {
  const model = tf.sequential();
  model.add(
    tf.layers.conv1d({
      inputShape: [SEQUENCE_LENGTH, FEATURE_SIZE],
      filters: 24,
      kernelSize: 5,
      padding: "same",
      activation: "relu",
    }),
  );
  model.add(tf.layers.batchNormalization());
  model.add(tf.layers.maxPooling1d({ poolSize: 2 }));
  model.add(
    tf.layers.conv1d({
      filters: 48,
      kernelSize: 3,
      padding: "same",
      activation: "relu",
    }),
  );
  model.add(tf.layers.globalAveragePooling1d({}));
  model.add(tf.layers.dense({ units: 64, activation: "relu" }));
  model.add(tf.layers.dropout({ rate: 0.22 }));
  model.add(tf.layers.dense({ units: classCount, activation: "softmax" }));
  model.compile({
    optimizer: tf.train.adam(0.0025),
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
  return model;
}

export class MouthModel {
  private model: LayersModel | null = null;
  private labels: string[] = [];

  get ready(): boolean {
    return this.model !== null && this.labels.length > 1;
  }

  get trainedLabels(): string[] {
    return [...this.labels];
  }

  async initialise(): Promise<void> {
    const stored = localStorage.getItem(LABELS_KEY);
    if (!stored) return;
    try {
      const tf = await loadTensorFlow();
      await tf.ready();
      this.labels = JSON.parse(stored) as string[];
      this.model = await tf.loadLayersModel(MODEL_URL);
    } catch {
      this.labels = [];
      this.model = null;
      localStorage.removeItem(LABELS_KEY);
    }
  }

  async train(
    samples: TrainingSample[],
    onUpdate?: (update: TrainingUpdate) => void,
  ): Promise<void> {
    const tf = await loadTensorFlow();
    await tf.ready();
    this.labels = [...new Set(samples.map((sample) => sample.label))];
    if (this.labels.length < 2) {
      throw new Error("Record at least two different words before training.");
    }

    this.model?.dispose();
    this.model = createModel(tf, this.labels.length);
    const sequences: number[][][] = [];
    const targets: number[] = [];
    const variantsPerSample = Math.max(8, Math.ceil(40 / samples.length));

    for (const sample of samples) {
      const target = this.labels.indexOf(sample.label);
      sequences.push(sample.sequence);
      targets.push(target);
      for (let index = 0; index < variantsPerSample; index += 1) {
        sequences.push(augment(sample.sequence, 0.004 + Math.random() * 0.006));
        targets.push(target);
      }
    }

    const xs = tf.tensor3d(sequences, [sequences.length, SEQUENCE_LENGTH, FEATURE_SIZE]);
    const targetTensor = tf.tensor1d(targets, "int32");
    const ys = tf.oneHot(targetTensor, this.labels.length);
    const epochs = 72;

    try {
      await this.model.fit(xs, ys, {
        epochs,
        batchSize: Math.min(24, sequences.length),
        shuffle: true,
        validationSplit: sequences.length >= 30 ? 0.15 : 0,
        verbose: 0,
        callbacks: {
          onEpochEnd: async (epoch, logs) => {
            if (epoch % 3 === 0 || epoch === epochs - 1) {
              onUpdate?.({
                epoch: epoch + 1,
                totalEpochs: epochs,
                accuracy: Number(logs?.acc ?? logs?.accuracy ?? 0),
                loss: Number(logs?.loss ?? 0),
              });
              await tf.nextFrame();
            }
          },
        },
      });
      await this.model.save(MODEL_URL);
      localStorage.setItem(LABELS_KEY, JSON.stringify(this.labels));
    } finally {
      xs.dispose();
      targetTensor.dispose();
      ys.dispose();
    }
  }

  async predict(sequence: number[][]): Promise<RawPrediction[]> {
    if (!this.model || this.labels.length === 0) {
      throw new Error("Train the mouth model first.");
    }
    const tf = await loadTensorFlow();
    const input = tf.tensor3d([sequence], [1, SEQUENCE_LENGTH, FEATURE_SIZE]);
    try {
      const output = this.model.predict(input) as Tensor;
      const values = Array.from(await output.data());
      output.dispose();
      return this.labels
        .map((label, index) => ({ label, probability: values[index] ?? 0 }))
        .sort((a, b) => b.probability - a.probability);
    } finally {
      input.dispose();
    }
  }

  async reset(): Promise<void> {
    const hadStoredModel = localStorage.getItem(LABELS_KEY) !== null;
    this.model?.dispose();
    this.model = null;
    this.labels = [];
    localStorage.removeItem(LABELS_KEY);
    if (hadStoredModel) {
      const tf = await loadTensorFlow();
      try {
        await tf.io.removeModel(MODEL_URL);
      } catch {
        // The model may not exist yet.
      }
    }
  }
}
