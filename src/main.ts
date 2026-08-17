import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import "./style.css";
import {
  DEMO_CONTEXTS,
  fusePredictions,
  normalisedEntropy,
} from "./decoder";
import type { FusedPrediction, RawPrediction } from "./decoder";
import {
  extractMouthFrame,
  MOUTH_CONTOURS,
  prepareSequence,
  sequenceMotion,
} from "./features";
import type { Point3D } from "./features";
import { MouthModel } from "./model";
import type { TrainingSample, TrainingUpdate } from "./model";

const SAMPLE_KEY = "shh-type-training-samples-v1";
const DEFAULT_LABELS = ["pat", "bat", "mat"];
const CAPTURE_MS = 1900;
const MIN_CAPTURE_FRAMES = 16;

type CapturePurpose = "sample" | "prediction";

interface CaptureState {
  purpose: CapturePurpose;
  frames: number[][];
  startedAt: number;
  resolve: (sequence: number[][]) => void;
  reject: (error: Error) => void;
}

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("App root not found.");

app.innerHTML = `
  <main class="shell">
    <header class="masthead">
      <a class="brand" href="#" aria-label="SHH Type home">
        <span class="brand-mark" aria-hidden="true"></span>
        SHH<span>//</span>TYPE
      </a>
      <div class="privacy-pill"><i></i> camera stays local</div>
      <a class="repo-link" href="https://github.com/madebylukas/shh-type" target="_blank" rel="noreferrer">source ↗</a>
    </header>

    <section class="hero-copy">
      <p class="eyebrow">WEBCAM-ONLY SILENT SPEECH / EXPERIMENT 001</p>
      <h1>Say it.<br><em>Don’t.</em></h1>
      <p class="lede">A tiny neural network learns how <strong>you</strong> mouth words—then shows its doubts instead of laundering them into certainty.</p>
    </section>

    <section class="instrument" aria-label="Silent speech instrument">
      <div class="camera-column">
        <div class="camera-frame" id="camera-frame">
          <video id="webcam" autoplay muted playsinline></video>
          <canvas id="overlay"></canvas>
          <div class="scanline" aria-hidden="true"></div>
          <div class="camera-noise" aria-hidden="true"></div>
          <div class="camera-corners" aria-hidden="true"></div>
          <div class="camera-status top-left"><span id="live-dot"></span><b id="camera-label">BOOTING VISION</b></div>
          <div class="camera-status top-right"><b id="fps">-- FPS</b></div>
          <div class="mouth-readout">
            <span>MOUTH SIGNAL</span>
            <div class="signal-bars" id="signal-bars" aria-hidden="true"></div>
          </div>
          <div class="capture-overlay" id="capture-overlay" hidden>
            <span id="capture-kicker">GET READY</span>
            <strong id="capture-count">3</strong>
            <small id="capture-instruction">Face forward. Mouth it once.</small>
          </div>
          <div class="permission-card" id="permission-card">
            <span class="permission-icon">◉</span>
            <h2>Your face is the input.</h2>
            <p>No microphone. Frames are processed in this tab and never uploaded.</p>
            <button class="primary" id="start-camera">Enable webcam</button>
          </div>
        </div>

        <div class="telemetry">
          <div><span>TRACK</span><strong id="track-state">OFFLINE</strong></div>
          <div><span>QUALITY</span><strong id="quality-state">—</strong></div>
          <div><span>MODEL</span><strong id="model-state">UNTRAINED</strong></div>
          <div><span>FRAMES</span><strong id="frame-state">0</strong></div>
        </div>
      </div>

      <div class="control-column">
        <div class="mode-tabs" role="tablist">
          <button class="mode-tab active" data-mode="calibrate" role="tab">01 / TEACH</button>
          <button class="mode-tab" data-mode="decode" role="tab">02 / DECODE</button>
        </div>

        <section class="panel active" id="calibrate-panel">
          <div class="panel-heading">
            <div>
              <span class="section-number">PERSONAL MOUTHPRINT</span>
              <h2>Teach three ambiguous words.</h2>
            </div>
            <button class="text-button danger" id="reset-all">Reset</button>
          </div>
          <p class="panel-note">Start with <b>pat / bat / mat</b>. They are deliberately cruel: the opening consonants look almost identical.</p>
          <div class="word-list" id="word-list"></div>
          <form class="add-word" id="add-word-form">
            <input id="new-word" maxlength="18" autocomplete="off" placeholder="add another word" aria-label="Add another word" />
            <button type="submit">＋</button>
          </form>
          <div class="action-stack">
            <button class="capture-button" id="record-sample" disabled>
              <span class="capture-glyph">●</span>
              <span><b>Record example</b><small id="record-label">Select a word first</small></span>
              <kbd>R</kbd>
            </button>
            <button class="train-button" id="train-model" disabled>
              <span>Train local neural model</span>
              <strong id="training-progress">READY WHEN YOU ARE</strong>
            </button>
          </div>
          <div class="training-rule"><i id="training-fill"></i></div>
          <p class="microcopy" id="training-hint">Record at least 3 examples of 2 words. Four each is better.</p>
        </section>

        <section class="panel" id="decode-panel">
          <div class="panel-heading">
            <div>
              <span class="section-number">PROBABILITY, NOT PROPHECY</span>
              <h2>Watch context break the tie.</h2>
            </div>
          </div>
          <label class="context-label" for="context-input">Sentence context <span>use ___ for the missing word</span></label>
          <input class="context-input" id="context-input" value="The baseball player swung the ___" />
          <div class="context-presets" id="context-presets"></div>
          <button class="capture-button predict" id="predict-word" disabled>
            <span class="capture-glyph">●</span>
            <span><b>Mouth one trained word</b><small>1.9 seconds · no audio</small></span>
            <kbd>SPACE</kbd>
          </button>
          <div class="result-card">
            <div class="result-label"><span>COMPOSED OUTPUT</span><b id="confidence-label">WAITING</b></div>
            <p id="composed-output">Your decoded sentence will appear here.</p>
          </div>
          <div class="lattice-heading">
            <span>CANDIDATE LATTICE</span>
            <div><i class="visual-key"></i>vision <i class="context-key"></i>context</div>
          </div>
          <div class="candidate-list" id="candidate-list">
            <div class="empty-lattice">Train the model, then mouth a word. Its uncertainty will be shown here—warts, probabilities and all.</div>
          </div>
        </section>
      </div>
    </section>

    <footer>
      <span>NO AUDIO · NO CLOUD · NO TELEPATHY</span>
      <span>Built to expose the hard part.</span>
    </footer>
  </main>
  <div class="toast" id="toast" role="status" aria-live="polite"></div>
`;

const $ = <T extends HTMLElement>(selector: string): T => {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`Missing element: ${selector}`);
  return element;
};

const video = $("#webcam") as HTMLVideoElement;
const canvas = $("#overlay") as HTMLCanvasElement;
const canvasContext = canvas.getContext("2d");
if (!canvasContext) throw new Error("Canvas 2D context unavailable.");
const ctx: CanvasRenderingContext2D = canvasContext;

const mouthModel = new MouthModel();
let faceLandmarker: FaceLandmarker | null = null;
let currentLandmarks: NormalizedLandmark[] | null = null;
let labels = [...DEFAULT_LABELS];
let activeLabel = labels[0];
let samples: TrainingSample[] = loadSamples();
let capture: CaptureState | null = null;
let rawPredictions: RawPrediction[] = [];
let lastVideoTime = -1;
let frameCounter = 0;
let fpsWindowStart = performance.now();
let measuredFps = 0;
let lastQuality = 0;
let toastTimer = 0;
const trails = new Map<number, { x: number; y: number }[]>();

function loadSamples(): TrainingSample[] {
  try {
    const parsed = JSON.parse(localStorage.getItem(SAMPLE_KEY) ?? "[]") as TrainingSample[];
    return parsed.filter((sample) => sample.label && Array.isArray(sample.sequence));
  } catch {
    return [];
  }
}

function persistSamples(): void {
  localStorage.setItem(SAMPLE_KEY, JSON.stringify(samples));
}

function showToast(message: string, kind: "good" | "bad" = "good"): void {
  const toast = $("#toast");
  toast.textContent = message;
  toast.className = `toast visible ${kind}`;
  window.clearTimeout(toastTimer);
  toastTimer = window.setTimeout(() => {
    toast.className = "toast";
  }, 3600);
}

function sampleCount(label: string): number {
  return samples.filter((sample) => sample.label === label).length;
}

function canTrain(): boolean {
  return labels.filter((label) => sampleCount(label) >= 3).length >= 2;
}

function renderWords(): void {
  $("#word-list").innerHTML = labels
    .map((label) => {
      const count = sampleCount(label);
      return `
        <button class="word-row ${label === activeLabel ? "active" : ""}" data-label="${label}">
          <span class="radio"><i></i></span>
          <strong>${label}</strong>
          <span class="takes">${Array.from({ length: 4 }, (_, index) => `<i class="${index < count ? "filled" : ""}"></i>`).join("")}</span>
          <small>${count}/4 takes</small>
          ${labels.length > 2 ? `<span class="remove-word" data-remove="${label}" title="Remove ${label}">×</span>` : ""}
        </button>`;
    })
    .join("");

  const recordButton = $("#record-sample") as HTMLButtonElement;
  recordButton.disabled = !faceLandmarker || Boolean(capture);
  $("#record-label").textContent = `Mouth “${activeLabel}” once`;
  const trainButton = $("#train-model") as HTMLButtonElement;
  trainButton.disabled = !canTrain() || Boolean(capture);
  const qualifying = labels.filter((label) => sampleCount(label) >= 3).length;
  $("#training-hint").textContent = canTrain()
    ? `${samples.length} takes captured. Train now, or add another pass for resilience.`
    : `Record at least 3 examples of 2 words. ${qualifying}/2 words ready.`;
}

function setMode(mode: "calibrate" | "decode"): void {
  document.querySelectorAll<HTMLElement>(".mode-tab").forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.mode === mode);
  });
  $("#calibrate-panel").classList.toggle("active", mode === "calibrate");
  $("#decode-panel").classList.toggle("active", mode === "decode");
}

function renderPresets(): void {
  $("#context-presets").innerHTML = DEMO_CONTEXTS.map(
    (preset, index) => `<button data-context="${preset}">0${index + 1}</button>`,
  ).join("");
}

function resizeCanvas(): void {
  const rect = video.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function mouthPoint(landmarks: NormalizedLandmark[], index: number): { x: number; y: number } {
  const rect = video.getBoundingClientRect();
  return { x: (1 - landmarks[index].x) * rect.width, y: landmarks[index].y * rect.height };
}

function drawMouth(landmarks: NormalizedLandmark[]): void {
  const rect = video.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  const opening = Math.abs(landmarks[13].y - landmarks[14].y);
  const hue = 176 + Math.min(120, opening * 1500);

  ctx.save();
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.shadowBlur = 14;
  ctx.shadowColor = `hsla(${hue}, 100%, 65%, .8)`;

  MOUTH_CONTOURS.forEach((contour, contourIndex) => {
    ctx.beginPath();
    contour.forEach((index, pointIndex) => {
      const point = mouthPoint(landmarks, index);
      if (pointIndex === 0) ctx.moveTo(point.x, point.y);
      else ctx.lineTo(point.x, point.y);
    });
    ctx.strokeStyle = contourIndex === 0 ? `hsla(${hue}, 100%, 68%, .95)` : "rgba(255, 87, 214, .9)";
    ctx.lineWidth = contourIndex === 0 ? 2.4 : 1.5;
    ctx.stroke();
  });

  const trackedPoints = [61, 291, 13, 14, 0, 17, 78, 308, 87, 317];
  trackedPoints.forEach((index, pointIndex) => {
    const point = mouthPoint(landmarks, index);
    const history = trails.get(index) ?? [];
    history.push(point);
    if (history.length > 8) history.shift();
    trails.set(index, history);

    history.forEach((trailPoint, trailIndex) => {
      ctx.beginPath();
      ctx.arc(trailPoint.x, trailPoint.y, 1 + trailIndex * 0.18, 0, Math.PI * 2);
      ctx.fillStyle = `hsla(${hue + pointIndex * 7}, 100%, 70%, ${trailIndex / 13})`;
      ctx.fill();
    });
  });
  ctx.restore();
}

function updateTelemetry(landmarks: NormalizedLandmark[] | null): void {
  const trackState = $("#track-state");
  const qualityState = $("#quality-state");
  if (!landmarks) {
    trackState.textContent = faceLandmarker ? "SEARCHING" : "OFFLINE";
    trackState.className = "warn";
    qualityState.textContent = "—";
    return;
  }
  const width = Math.abs(landmarks[61].x - landmarks[291].x);
  lastQuality = Math.max(0, Math.min(1, (width - 0.055) / 0.09));
  trackState.textContent = "LOCKED";
  trackState.className = "good";
  qualityState.textContent = lastQuality > 0.68 ? "CLEAN" : lastQuality > 0.3 ? "FAIR" : "MOVE CLOSER";
  qualityState.className = lastQuality > 0.3 ? "good" : "warn";
}

function renderSignal(frame: number[]): void {
  const bars = $("#signal-bars");
  if (bars.children.length === 0) {
    bars.innerHTML = Array.from({ length: 22 }, () => "<i></i>").join("");
  }
  const opening = Math.min(1, Math.abs(frame[41] ?? 0) * 4.5);
  Array.from(bars.children).forEach((bar, index) => {
    const wave = 0.15 + Math.abs(Math.sin(performance.now() / 170 + index * 0.63)) * opening;
    (bar as HTMLElement).style.transform = `scaleY(${wave})`;
  });
}

function renderLoop(): void {
  if (faceLandmarker && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
    if (video.currentTime !== lastVideoTime) {
      lastVideoTime = video.currentTime;
      const result = faceLandmarker.detectForVideo(video, performance.now());
      currentLandmarks = result.faceLandmarks[0] ?? null;
      if (currentLandmarks) {
        drawMouth(currentLandmarks);
        const frame = extractMouthFrame(currentLandmarks as Point3D[]);
        renderSignal(frame);
        if (capture) {
          capture.frames.push(frame);
          $("#frame-state").textContent = String(capture.frames.length);
        }
      } else {
        const rect = video.getBoundingClientRect();
        ctx.clearRect(0, 0, rect.width, rect.height);
      }
      updateTelemetry(currentLandmarks);
      frameCounter += 1;
      const now = performance.now();
      if (now - fpsWindowStart >= 1000) {
        measuredFps = Math.round((frameCounter * 1000) / (now - fpsWindowStart));
        $("#fps").textContent = `${measuredFps} FPS`;
        frameCounter = 0;
        fpsWindowStart = now;
      }
    }
  }
  requestAnimationFrame(renderLoop);
}

async function createLandmarker(): Promise<FaceLandmarker> {
  const vision = await FilesetResolver.forVisionTasks("/mediapipe");
  const options = {
    baseOptions: {
      modelAssetPath: "/models/face_landmarker.task",
      delegate: "GPU" as const,
    },
    runningMode: "VIDEO" as const,
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  };
  try {
    return await FaceLandmarker.createFromOptions(vision, options);
  } catch {
    return FaceLandmarker.createFromOptions(vision, {
      ...options,
      baseOptions: { ...options.baseOptions, delegate: "CPU" },
    });
  }
}

async function startCamera(): Promise<void> {
  const button = $("#start-camera") as HTMLButtonElement;
  button.disabled = true;
  button.textContent = "Loading vision model…";
  try {
    const [landmarker, stream] = await Promise.all([
      createLandmarker(),
      navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
      }),
    ]);
    faceLandmarker = landmarker;
    video.srcObject = stream;
    await video.play();
    resizeCanvas();
    $("#permission-card").hidden = true;
    $("#camera-label").textContent = "LIVE / NO AUDIO";
    $("#live-dot").classList.add("live");
    renderWords();
    showToast("Vision online. Nothing leaves this tab.");
  } catch (error) {
    button.disabled = false;
    button.textContent = "Try webcam again";
    showToast(error instanceof Error ? error.message : "Could not start the webcam.", "bad");
  }
}

function delay(ms: number): Promise<void> {
  return new Promise((resolve) => window.setTimeout(resolve, ms));
}

async function runCaptureCountdown(label: string): Promise<void> {
  const overlay = $("#capture-overlay");
  overlay.hidden = false;
  $("#capture-kicker").textContent = `MOUTH “${label.toUpperCase()}”`;
  $("#capture-instruction").textContent = "Face forward. Mouth it once, naturally.";
  for (const count of [3, 2, 1]) {
    $("#capture-count").textContent = String(count);
    await delay(520);
  }
  $("#capture-count").textContent = "●";
  $("#capture-kicker").textContent = "CAPTURING";
  overlay.classList.add("recording");
}

async function captureSequence(purpose: CapturePurpose): Promise<number[][]> {
  if (!faceLandmarker || !currentLandmarks) {
    throw new Error("I need a clean face lock before recording.");
  }
  if (lastQuality < 0.2) {
    throw new Error("Move closer to the camera, then try again.");
  }

  await runCaptureCountdown(purpose === "sample" ? activeLabel : "ONE WORD");
  const sequencePromise = new Promise<number[][]>((resolve, reject) => {
    capture = { purpose, frames: [], startedAt: performance.now(), resolve, reject };
  });
  renderWords();
  window.setTimeout(() => finishCapture(), CAPTURE_MS);
  return sequencePromise;
}

function finishCapture(): void {
  const activeCapture = capture;
  if (!activeCapture) return;
  capture = null;
  const overlay = $("#capture-overlay");
  overlay.classList.remove("recording");
  overlay.hidden = true;
  $("#frame-state").textContent = String(activeCapture.frames.length);
  renderWords();

  if (activeCapture.frames.length < MIN_CAPTURE_FRAMES) {
    activeCapture.reject(new Error("Not enough frames captured. Keep the tab visible and retry."));
    return;
  }
  const sequence = prepareSequence(activeCapture.frames);
  if (sequenceMotion(sequence) < 0.0005) {
    activeCapture.reject(new Error("Barely any mouth movement detected. Exaggerate it slightly."));
    return;
  }
  activeCapture.resolve(sequence);
}

async function recordSample(): Promise<void> {
  try {
    const sequence = await captureSequence("sample");
    samples.push({ label: activeLabel, sequence });
    persistSamples();
    renderWords();
    showToast(`Take ${sampleCount(activeLabel)} for “${activeLabel}” captured.`);
  } catch (error) {
    showToast(error instanceof Error ? error.message : "Capture failed.", "bad");
  }
}

async function trainModel(): Promise<void> {
  const trainButton = $("#train-model") as HTMLButtonElement;
  trainButton.disabled = true;
  $("#training-progress").textContent = "BUILDING TENSORS";
  try {
    await mouthModel.train(samples.filter((sample) => sampleCount(sample.label) >= 3), (update: TrainingUpdate) => {
      const percent = Math.round((update.epoch / update.totalEpochs) * 100);
      $("#training-fill").style.width = `${percent}%`;
      $("#training-progress").textContent = `${percent}% · ACC ${(update.accuracy * 100).toFixed(0)}%`;
    });
    $("#model-state").textContent = "PERSONALISED";
    $("#model-state").className = "good";
    $("#training-progress").textContent = "MODEL SAVED LOCALLY";
    ($("#predict-word") as HTMLButtonElement).disabled = false;
    setMode("decode");
    showToast("Your mouthprint model is trained. Now try to confuse it.");
  } catch (error) {
    showToast(error instanceof Error ? error.message : "Training failed.", "bad");
  } finally {
    trainButton.disabled = !canTrain();
  }
}

function sentenceWith(label: string): string {
  const value = ($("#context-input") as HTMLInputElement).value.trim();
  if (!value) return label;
  return value.includes("___") ? value.replace("___", label) : `${value} ${label}`;
}

function renderPredictions(): void {
  if (rawPredictions.length === 0) return;
  const contextValue = ($("#context-input") as HTMLInputElement).value;
  const fused = fusePredictions(rawPredictions, contextValue);
  const entropy = normalisedEntropy(fused.map((candidate) => candidate.fusedScore));
  const winner = fused[0];
  $("#composed-output").textContent = sentenceWith(winner.label);
  const confidenceLabel = $("#confidence-label");
  confidenceLabel.textContent = entropy < 0.45 ? "HIGH CONFIDENCE" : entropy < 0.72 ? "AMBIGUOUS" : "VERY AMBIGUOUS";
  confidenceLabel.className = entropy < 0.45 ? "good" : "warn";

  $("#candidate-list").innerHTML = fused
    .slice(0, 5)
    .map((candidate: FusedPrediction, index) => `
      <button class="candidate ${index === 0 ? "winner" : ""}" data-candidate="${candidate.label}">
        <span class="candidate-rank">0${index + 1}</span>
        <strong>${candidate.label}</strong>
        <div class="score-stack">
          <i class="visual-score" style="width:${(candidate.visualScore * 100).toFixed(1)}%"></i>
          <i class="context-score" style="width:${(candidate.contextScore * 100).toFixed(1)}%"></i>
        </div>
        <b>${Math.round(candidate.fusedScore * 100)}%</b>
      </button>
    `)
    .join("");
}

async function predictWord(): Promise<void> {
  try {
    const sequence = await captureSequence("prediction");
    rawPredictions = await mouthModel.predict(sequence);
    renderPredictions();
    showToast("Prediction decoded. Change the context and watch the lattice move.");
  } catch (error) {
    showToast(error instanceof Error ? error.message : "Prediction failed.", "bad");
  }
}

async function resetAll(): Promise<void> {
  if (!window.confirm("Delete the locally stored examples and model?")) return;
  samples = [];
  labels = [...DEFAULT_LABELS];
  activeLabel = labels[0];
  rawPredictions = [];
  localStorage.removeItem(SAMPLE_KEY);
  await mouthModel.reset();
  $("#model-state").textContent = "UNTRAINED";
  $("#training-progress").textContent = "READY WHEN YOU ARE";
  $("#training-fill").style.width = "0%";
  ($("#predict-word") as HTMLButtonElement).disabled = true;
  renderWords();
  showToast("Local mouthprint erased.");
}

$("#start-camera").addEventListener("click", () => void startCamera());
$("#record-sample").addEventListener("click", () => void recordSample());
$("#train-model").addEventListener("click", () => void trainModel());
$("#predict-word").addEventListener("click", () => void predictWord());
$("#reset-all").addEventListener("click", () => void resetAll());

$("#word-list").addEventListener("click", (event) => {
  const target = event.target as HTMLElement;
  const remove = target.closest<HTMLElement>("[data-remove]");
  if (remove?.dataset.remove) {
    const label = remove.dataset.remove;
    labels = labels.filter((word) => word !== label);
    samples = samples.filter((sample) => sample.label !== label);
    if (activeLabel === label) activeLabel = labels[0];
    persistSamples();
    renderWords();
    return;
  }
  const row = target.closest<HTMLElement>("[data-label]");
  if (row?.dataset.label) {
    activeLabel = row.dataset.label;
    renderWords();
  }
});

$("#add-word-form").addEventListener("submit", (event) => {
  event.preventDefault();
  const input = $("#new-word") as HTMLInputElement;
  const label = input.value.trim().toLowerCase().replace(/[^a-z'-]/g, "");
  if (!label || labels.includes(label)) return;
  labels.push(label);
  activeLabel = label;
  input.value = "";
  renderWords();
});

document.querySelectorAll<HTMLElement>(".mode-tab").forEach((tab) => {
  tab.addEventListener("click", () => setMode(tab.dataset.mode as "calibrate" | "decode"));
});

$("#context-presets").addEventListener("click", (event) => {
  const target = (event.target as HTMLElement).closest<HTMLButtonElement>("[data-context]");
  if (!target?.dataset.context) return;
  ($("#context-input") as HTMLInputElement).value = target.dataset.context;
  renderPredictions();
});
$("#context-input").addEventListener("input", renderPredictions);
$("#candidate-list").addEventListener("click", (event) => {
  const target = (event.target as HTMLElement).closest<HTMLElement>("[data-candidate]");
  if (!target?.dataset.candidate) return;
  $("#composed-output").textContent = sentenceWith(target.dataset.candidate);
  showToast(`Selected “${target.dataset.candidate}”. Human veto retained.`);
});

window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement) return;
  if (event.key.toLowerCase() === "r" && !capture) void recordSample();
  if (event.code === "Space" && !capture && mouthModel.ready) {
    event.preventDefault();
    setMode("decode");
    void predictWord();
  }
});
window.addEventListener("resize", resizeCanvas);

async function initialise(): Promise<void> {
  const storedLabels = [...new Set(samples.map((sample) => sample.label))];
  labels = [...new Set([...DEFAULT_LABELS, ...storedLabels])];
  await mouthModel.initialise();
  if (mouthModel.ready) {
    $("#model-state").textContent = "PERSONALISED";
    $("#model-state").className = "good";
    ($("#predict-word") as HTMLButtonElement).disabled = false;
  }
  renderWords();
  renderPresets();
  renderLoop();
}

void initialise();
