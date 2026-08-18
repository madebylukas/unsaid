import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import type { NormalizedLandmark } from "@mediapipe/tasks-vision";
import "./style.css";
import { getEngineHealth, inferVideo } from "./api";
import type { Candidate, EngineHealth, InferenceResult } from "./api";
import { DEMO_DECKS, resolveDemo } from "./demo";
import { MOUTH_CONTOURS } from "./features";
import { mapCoverPoint, polygonArea } from "./geometry";
import { computeQuality, formatResolution } from "./telemetry";

const PUBLIC_DEMO = import.meta.env.PROD;

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) throw new Error("app root not found");

app.innerHTML = `
  <div class="app-shell">
    <header class="topbar">
      <a class="wordmark" href="#">unsaid</a>
      <span class="descriptor">visual speech / ${PUBLIC_DEMO ? "public demo" : "local"}</span>
      <a class="source" href="https://github.com/madebylukas/unsaid" target="_blank" rel="noreferrer">github ↗</a>
    </header>

    <main class="workspace">
      <section class="camera-panel" aria-label="webcam">
        <div class="camera-stage" id="camera-stage">
          <video id="webcam" autoplay muted playsinline></video>
          <canvas id="mouth-overlay"></canvas>
          <div class="camera-state"><span id="camera-state">camera off</span></div>

          <div class="camera-permission" id="camera-permission">
            <p>unsaid needs the webcam.<br>audio is never requested.</p>
            <button id="enable-camera">enable camera</button>
          </div>

          <div class="record-time" id="record-time" hidden>00:00.0</div>

          <div class="camera-data">
            <div class="camera-identity">
              <span id="camera-name">camera unavailable</span>
              <strong id="camera-spec">—</strong>
            </div>
            <dl>
              <div><dt>fps</dt><dd id="fps-value">—</dd></div>
              <div><dt>light</dt><dd id="light-value">—</dd></div>
              <div><dt>mouth area</dt><dd id="mouth-value">—</dd></div>
              <div><dt>quality</dt><dd id="quality-value">—</dd></div>
            </dl>
          </div>
        </div>
      </section>

      <aside class="output-panel">
        <div class="output-head">
          <span>transcript</span>
          <span id="inference-state">idle</span>
        </div>

        <div class="phrase-toolbar">
          <span>available sentences / <b id="deck-number">01</b></span>
          <button id="rotate-phrases">rotate set ↻</button>
        </div>

        <div class="phrase-bank" id="phrase-bank"></div>

        <div class="transcript-wrap">
          <p class="transcript" id="transcript">choose a phrase.</p>
          <p class="notice" id="notice">${PUBLIC_DEMO
            ? "enable the camera, then mouth one line from the demo set."
            : "start the local engine, then mouth one line from the demo set."}</p>
        </div>

        <div class="result-meta" id="result-meta">
          <span id="result-mode">demo / ${DEMO_DECKS[0].length} phrases</span><span>no result yet</span>
        </div>

        <div class="mode-control">
          <span>decode mode</span>
          <div>
            <button class="active" data-mode="demo">demo</button>
            <button data-mode="open">open</button>
          </div>
        </div>

        <button class="record-button" id="record-button" disabled>
          <span id="record-action">start reading</span>
          <kbd>space</kbd>
        </button>

        <section class="candidates-section">
          <div class="section-head"><span id="candidate-label">demo matches</span><span id="score-label">match</span></div>
          <div class="candidates" id="candidates">
            <p>the model keeps several readings when the image is ambiguous.</p>
          </div>
        </section>

        <section class="engine-section">
          <div class="section-head"><span>engine</span><span id="engine-state">checking</span></div>
          <div class="engine-row">
            <div>
              <strong id="engine-name">auto-avsr</strong>
              <span id="engine-detail">${PUBLIC_DEMO ? "checking the visual model" : "looking for localhost:8787"}</span>
            </div>
            <code id="engine-command">npm run vsr</code>
          </div>
        </section>
      </aside>
    </main>

    <footer class="footer">
      <span>webcam → mouth crop → visual transformer → beam search</span>
      <span>no microphone · clips deleted after reading · esc cancels</span>
    </footer>
  </div>
`;

const $ = <T extends HTMLElement>(selector: string): T => {
  const element = document.querySelector<T>(selector);
  if (!element) throw new Error(`missing element: ${selector}`);
  return element;
};

const video = $("#webcam") as HTMLVideoElement;
const overlay = $("#mouth-overlay") as HTMLCanvasElement;
const canvasContext = overlay.getContext("2d");
if (!canvasContext) throw new Error("canvas is unavailable");
const ctx: CanvasRenderingContext2D = canvasContext;
const lightCanvas = document.createElement("canvas");
lightCanvas.width = 64;
lightCanvas.height = 36;
const lightContext = lightCanvas.getContext("2d", { willReadFrequently: true });

let landmarker: FaceLandmarker | null = null;
let stream: MediaStream | null = null;
let engine: EngineHealth | null = null;
let recorder: MediaRecorder | null = null;
let recordedChunks: Blob[] = [];
let discardRecording = false;
let inferenceBusy = false;
let inferenceStatusTimer = 0;
let decodeMode: "demo" | "open" = "demo";
let lastResult: InferenceResult | null = null;
let lastDuration = 0;
let recordingStarted = 0;
let recordingTimer = 0;
let lastVideoTime = -1;
let frameCount = 0;
let fpsStarted = performance.now();
let measuredFps = 0;
let luminance = 0;
let mouthWidthPx = 0;
let mouthAreaPx = 0;
let rollDegrees = 0;
let faceVisible = false;
let lastLightSample = 0;
let activeDeckIndex = 0;

function setNotice(message: string, isError = false): void {
  const notice = $("#notice");
  notice.textContent = message;
  notice.classList.toggle("error", isError);
}

function escapeHtml(text: string): string {
  return text.replace(/[&<>"]/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
  })[character] ?? character);
}

function activePhrases(): readonly string[] {
  return DEMO_DECKS[activeDeckIndex];
}

function renderPhraseBank(selectedPhrase = ""): void {
  $("#deck-number").textContent = String(activeDeckIndex + 1).padStart(2, "0");
  $("#phrase-bank").innerHTML = activePhrases().map((phrase, index) => `
    <span class="${phrase === selectedPhrase ? "selected" : ""}">
      <b>${String(index + 1).padStart(2, "0")}</b>${escapeHtml(phrase)}
    </span>
  `).join("");
}

function renderMode(): void {
  document.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach((button) => {
    button.classList.toggle("active", button.dataset.mode === decodeMode);
  });
  $("#phrase-bank").hidden = decodeMode === "open";
  $(".phrase-toolbar").hidden = decodeMode === "open";
  $("#candidate-label").textContent = decodeMode === "demo" ? "demo matches" : "visual alternatives";
  $("#score-label").textContent = decodeMode === "demo" ? "match" : "beam share";
  $("#result-mode").textContent = decodeMode === "demo" ? `demo / ${activePhrases().length} phrases` : "open vocabulary";

  if (lastResult) {
    renderInferenceResult(lastResult, lastDuration);
    return;
  }
  renderPhraseBank();
  $("#transcript").textContent = decodeMode === "demo" ? "choose a phrase." : "mouth a sentence.";
  $("#candidates").innerHTML = `<p>${decodeMode === "demo"
    ? "the model may commit only to one of the phrases above."
    : "the model keeps several readings when the image is ambiguous."}</p>`;
  if (engine) {
    setNotice(decodeMode === "demo"
      ? "mouth one demo phrase exactly, then stop."
      : "face the camera and speak silently at a natural pace.");
  }
}

function resizeOverlay(): void {
  const rect = video.getBoundingClientRect();
  const ratio = window.devicePixelRatio || 1;
  overlay.width = Math.round(rect.width * ratio);
  overlay.height = Math.round(rect.height * ratio);
  overlay.style.width = `${rect.width}px`;
  overlay.style.height = `${rect.height}px`;
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
}

function point(landmarks: NormalizedLandmark[], index: number): { x: number; y: number } {
  const rect = video.getBoundingClientRect();
  return mapCoverPoint(
    landmarks[index].x,
    landmarks[index].y,
    rect.width,
    rect.height,
    video.videoWidth,
    video.videoHeight,
    true,
  );
}

function drawMouth(landmarks: NormalizedLandmark[] | null): void {
  const rect = video.getBoundingClientRect();
  ctx.clearRect(0, 0, rect.width, rect.height);
  if (!landmarks) return;

  ctx.save();
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  MOUTH_CONTOURS.forEach((contour, contourIndex) => {
    ctx.beginPath();
    contour.forEach((index, position) => {
      const current = point(landmarks, index);
      if (position === 0) ctx.moveTo(current.x, current.y);
      else ctx.lineTo(current.x, current.y);
    });
    ctx.strokeStyle = contourIndex === 0 ? "rgba(255,255,255,.88)" : "rgba(255,255,255,.46)";
    ctx.lineWidth = contourIndex === 0 ? 1.35 : 0.85;
    ctx.stroke();
  });

  for (const index of [61, 291, 13, 14]) {
    const current = point(landmarks, index);
    ctx.beginPath();
    ctx.arc(current.x, current.y, 1.7, 0, Math.PI * 2);
    ctx.fillStyle = "rgba(255,255,255,.92)";
    ctx.fill();
  }
  ctx.restore();
}

function sampleLight(): void {
  if (!lightContext || video.readyState < HTMLMediaElement.HAVE_CURRENT_DATA) return;
  lightContext.drawImage(video, 0, 0, lightCanvas.width, lightCanvas.height);
  const pixels = lightContext.getImageData(0, 0, lightCanvas.width, lightCanvas.height).data;
  let total = 0;
  for (let index = 0; index < pixels.length; index += 4) {
    total += pixels[index] * 0.2126 + pixels[index + 1] * 0.7152 + pixels[index + 2] * 0.0722;
  }
  luminance = Math.round((total / (pixels.length / 4) / 255) * 100);
}

function renderTelemetry(): void {
  const quality = computeQuality({ mouthWidthPx, fps: measuredFps, luminance, rollDegrees, faceVisible });
  $("#fps-value").textContent = measuredFps ? String(measuredFps) : "—";
  $("#light-value").textContent = luminance ? `${luminance}%` : "—";
  $("#mouth-value").textContent = mouthAreaPx ? `${(mouthAreaPx / 1000).toFixed(1)}k px²` : "—";
  $("#quality-value").textContent = faceVisible ? `${quality.score} / ${quality.label}` : quality.label;
  $("#quality-value").className = quality.score >= 76 ? "quality-good" : quality.score >= 50 ? "quality-fair" : "quality-poor";
  $("#camera-state").textContent = faceVisible ? "face locked" : "searching for face";
  updateRecordAvailability();
}

function renderLoop(): void {
  if (landmarker && video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const result = landmarker.detectForVideo(video, performance.now());
    const landmarks = result.faceLandmarks[0] ?? null;
    faceVisible = Boolean(landmarks);
    if (landmarks) {
      mouthWidthPx = Math.abs(landmarks[61].x - landmarks[291].x) * video.videoWidth;
      mouthAreaPx = polygonArea(MOUTH_CONTOURS[0].map((index) => ({
        x: landmarks[index].x * video.videoWidth,
        y: landmarks[index].y * video.videoHeight,
      })));
      rollDegrees = Math.atan2(
        landmarks[291].y - landmarks[61].y,
        landmarks[291].x - landmarks[61].x,
      ) * 180 / Math.PI;
    } else {
      mouthWidthPx = 0;
      mouthAreaPx = 0;
      rollDegrees = 0;
    }
    drawMouth(landmarks);
    frameCount += 1;
    const now = performance.now();
    if (now - lastLightSample >= 900) {
      sampleLight();
      lastLightSample = now;
    }
    if (now - fpsStarted >= 1000) {
      measuredFps = Math.round((frameCount * 1000) / (now - fpsStarted));
      frameCount = 0;
      fpsStarted = now;
      renderTelemetry();
    }
  }
  requestAnimationFrame(renderLoop);
}

async function createLandmarker(): Promise<FaceLandmarker> {
  const vision = await FilesetResolver.forVisionTasks("/mediapipe");
  const shared = {
    runningMode: "VIDEO" as const,
    numFaces: 1,
    minFaceDetectionConfidence: 0.5,
    minFacePresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  };
  try {
    return await FaceLandmarker.createFromOptions(vision, {
      ...shared,
      baseOptions: { modelAssetPath: "/models/face_landmarker.task", delegate: "GPU" },
    });
  } catch {
    return FaceLandmarker.createFromOptions(vision, {
      ...shared,
      baseOptions: { modelAssetPath: "/models/face_landmarker.task", delegate: "CPU" },
    });
  }
}

async function enableCamera(): Promise<void> {
  const button = $("#enable-camera") as HTMLButtonElement;
  button.disabled = true;
  button.textContent = "starting…";
  try {
    const [visionModel, cameraStream] = await Promise.all([
      createLandmarker(),
      navigator.mediaDevices.getUserMedia({
        audio: false,
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 }, facingMode: "user" },
      }),
    ]);
    landmarker = visionModel;
    stream = cameraStream;
    video.srcObject = stream;
    await video.play();
    resizeOverlay();
    $("#camera-permission").hidden = true;
    const track = stream.getVideoTracks()[0];
    const settings = track.getSettings();
    $("#camera-name").textContent = track.label || "webcam";
    $("#camera-spec").textContent = `${formatResolution(settings.width ?? video.videoWidth, settings.height ?? video.videoHeight)} · ${Math.round(settings.frameRate ?? 0)} fps · video only`;
    $("#camera-state").textContent = "searching for face";
    setNotice(decodeMode === "demo"
      ? "mouth one demo phrase exactly, then stop."
      : "mouth a sentence at a natural pace, then stop.");
    updateRecordAvailability();
  } catch (error) {
    button.disabled = false;
    button.textContent = "try again";
    setNotice(error instanceof Error ? error.message : "camera unavailable", true);
  }
}

function supportedMimeType(): string {
  for (const type of ["video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"]) {
    if (MediaRecorder.isTypeSupported(type)) return type;
  }
  return "";
}

function updateRecordAvailability(): void {
  const button = $("#record-button") as HTMLButtonElement;
  const quality = computeQuality({ mouthWidthPx, fps: measuredFps, luminance, rollDegrees, faceVisible });
  const recording = recorder?.state === "recording";
  button.disabled = !recording && (inferenceBusy || !stream || !engine || engine.status === "setup_required" || quality.score < 50);
  document.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach((modeButton) => {
    modeButton.disabled = inferenceBusy || recording;
  });
  ($("#rotate-phrases") as HTMLButtonElement).disabled = inferenceBusy || recording;
}

function renderClock(): void {
  if (!recorder || recorder.state !== "recording") return;
  const elapsed = performance.now() - recordingStarted;
  const seconds = elapsed / 1000;
  $("#record-time").textContent = `00:${seconds.toFixed(1).padStart(4, "0")}`;
  if (seconds >= 12) {
    stopRecording();
    return;
  }
  recordingTimer = requestAnimationFrame(renderClock);
}

function startRecording(): void {
  if (!stream || recorder?.state === "recording") return;
  const mimeType = supportedMimeType();
  recorder = new MediaRecorder(stream, mimeType ? { mimeType, videoBitsPerSecond: 2_000_000 } : undefined);
  recordedChunks = [];
  discardRecording = false;
  recorder.addEventListener("dataavailable", (event) => {
    if (event.data.size) recordedChunks.push(event.data);
  });
  recorder.addEventListener("stop", () => void submitRecording());
  recorder.start(250);
  recordingStarted = performance.now();
  $("#record-time").hidden = false;
  $("#record-button").classList.add("recording");
  $("#record-action").textContent = "stop + read";
  $("#inference-state").textContent = "recording";
  $("#transcript").textContent = "…";
  renderPhraseBank();
  setNotice(decodeMode === "demo" ? "mouth one demo phrase exactly, then stop." : "mouth a complete phrase, then stop.");
  renderClock();
}

function stopRecording(): void {
  if (!recorder || recorder.state !== "recording") return;
  recorder.stop();
  cancelAnimationFrame(recordingTimer);
  $("#record-button").classList.remove("recording");
  $("#record-action").textContent = "start reading";
  $("#inference-state").textContent = "processing";
}

function renderCandidates(candidates: Candidate[]): void {
  const container = $("#candidates");
  if (!candidates.length) {
    container.innerHTML = "<p>no alternative hypotheses returned.</p>";
    return;
  }
  container.innerHTML = candidates.map((candidate, index) => `
    <div class="candidate-row">
      <span>${String(index + 1).padStart(2, "0")}</span>
      <strong>${escapeHtml(candidate.text)}</strong>
      <div class="score"><i style="width:${Math.max(2, candidate.probability * 100).toFixed(1)}%"></i></div>
      <b>${Math.round(candidate.probability * 100)}%</b>
    </div>
  `).join("");
}

function renderInferenceResult(result: InferenceResult, duration: number): void {
  if (decodeMode === "demo") {
    const resolution = resolveDemo(result.candidates, activePhrases());
    renderCandidates(resolution.candidates);
    if (resolution.committed) {
      const winningPhrase = resolution.candidates[0].text;
      $("#transcript").textContent = winningPhrase;
      renderPhraseBank(winningPhrase);
      $("#inference-state").textContent = `${Math.round(resolution.confidence * 100)}% match`;
      setNotice(`raw visual reading: “${result.transcript}”. constrained to the demo set.`);
    } else {
      $("#transcript").textContent = "not confident.";
      renderPhraseBank();
      $("#inference-state").textContent = "abstained";
      setNotice(`raw visual reading: “${result.transcript}”. try one demo phrase more slowly.`, true);
    }
  } else {
    $("#transcript").textContent = result.transcript || "no reading";
    $("#inference-state").textContent = "complete";
    renderCandidates(result.candidates);
    setNotice("visual-only result. alternatives remain visible below.");
  }
  $("#result-meta").innerHTML = `<span id="result-mode">${decodeMode === "demo" ? `demo / ${activePhrases().length} phrases` : "open vocabulary"}</span><span>${duration.toFixed(1)} s clip · ${(result.latency_ms / 1000).toFixed(1)} s read · ${escapeHtml(result.device)}</span>`;
}

async function submitRecording(): Promise<void> {
  if (discardRecording) {
    discardRecording = false;
    recordedChunks = [];
    return;
  }
  const duration = (performance.now() - recordingStarted) / 1000;
  $("#record-time").hidden = true;
  const blob = new Blob(recordedChunks, { type: recorder?.mimeType || "video/webm" });
  if (!blob.size) {
    $("#inference-state").textContent = "idle";
    $("#transcript").textContent = "no video.";
    setNotice("the camera produced no video frames. try recording again.", true);
    return;
  }
  inferenceBusy = true;
  updateRecordAvailability();
  const inferenceStarted = performance.now();
  $("#inference-state").textContent = "reading 0s";
  setNotice(`${PUBLIC_DEMO ? "reading on the demo server" : "reading locally"}. the first clip can take 10–20 seconds.`);
  inferenceStatusTimer = window.setInterval(() => {
    const seconds = Math.floor((performance.now() - inferenceStarted) / 1000);
    $("#inference-state").textContent = `reading ${seconds}s`;
  }, 1000);
  try {
    const result: InferenceResult = await inferVideo(blob);
    lastResult = result;
    lastDuration = duration;
    renderInferenceResult(result, duration);
  } catch (error) {
    $("#transcript").textContent = "model error.";
    $("#inference-state").textContent = "failed";
    setNotice(error instanceof Error ? error.message : "inference failed", true);
  } finally {
    window.clearInterval(inferenceStatusTimer);
    inferenceBusy = false;
    updateRecordAvailability();
  }
}

function toggleRecording(): void {
  if (recorder?.state === "recording") stopRecording();
  else startRecording();
}

async function pollEngine(): Promise<void> {
  const nextEngine = await getEngineHealth();
  const state = $("#engine-state");
  const detail = $("#engine-detail");
  const command = $("#engine-command");
  if (!nextEngine && inferenceBusy && engine) {
    state.textContent = "decoding";
    detail.textContent = `${engine.engine} · ${engine.device} · request in progress`;
    return;
  }
  engine = nextEngine;
  if (!engine) {
    state.textContent = "offline";
    detail.textContent = PUBLIC_DEMO ? "the demo model is unavailable" : "localhost:8787 is not running";
    command.textContent = PUBLIC_DEMO ? "try again shortly" : "npm run vsr";
    if (!inferenceBusy) {
      $("#record-action").textContent = "engine offline";
      $("#inference-state").textContent = "blocked";
      setNotice(PUBLIC_DEMO
        ? "the demo model is waking up or unavailable. try again shortly."
        : "local model is offline. run npm run vsr in Terminal and leave it running.", true);
    }
  } else if (engine.status === "setup_required") {
    state.textContent = "setup required";
    detail.textContent = `${engine.engine} · weights missing`;
    command.textContent = "npm run setup:vsr";
    if (!inferenceBusy) {
      $("#record-action").textContent = "model missing";
      setNotice("model files are missing. run npm run setup:vsr once.", true);
    }
  } else if (engine.status === "loading") {
    state.textContent = "loading model";
    detail.textContent = `${engine.engine} · ${engine.device} · first read`;
    command.textContent = PUBLIC_DEMO ? "secure demo server" : "localhost:8787";
  } else {
    state.textContent = engine.model_loaded ? "ready" : "ready / cold";
    detail.textContent = `${engine.engine} · ${engine.device} · ${engine.model_loaded ? "loaded" : "loads on first read"}`;
    command.textContent = PUBLIC_DEMO ? "secure demo server" : "localhost:8787";
    if (!inferenceBusy && recorder?.state !== "recording") {
      $("#record-action").textContent = "start reading";
      if (!lastResult) {
        setNotice(decodeMode === "demo"
          ? "enable the camera, then mouth one demo phrase exactly."
          : "enable the camera, then mouth a sentence at a natural pace.");
        $("#inference-state").textContent = "idle";
      }
    }
  }
  updateRecordAvailability();
}

$("#enable-camera").addEventListener("click", () => void enableCamera());
$("#record-button").addEventListener("click", toggleRecording);
document.querySelectorAll<HTMLButtonElement>("[data-mode]").forEach((button) => {
  button.addEventListener("click", () => {
    decodeMode = button.dataset.mode === "open" ? "open" : "demo";
    renderMode();
  });
});
$("#rotate-phrases").addEventListener("click", () => {
  activeDeckIndex = (activeDeckIndex + 1) % DEMO_DECKS.length;
  lastResult = null;
  renderPhraseBank();
  $("#transcript").textContent = "choose a phrase.";
  $("#inference-state").textContent = "idle";
  $("#result-meta").innerHTML = `<span id="result-mode">demo / ${activePhrases().length} phrases</span><span>no result yet</span>`;
  $("#candidates").innerHTML = "<p>the model may commit only to one of the phrases above.</p>";
  setNotice("mouth one demo phrase exactly, then stop.");
});
window.addEventListener("resize", resizeOverlay);
window.addEventListener("keydown", (event) => {
  if (event.target instanceof HTMLInputElement) return;
  if (event.code === "Space" && !( $("#record-button") as HTMLButtonElement).disabled) {
    event.preventDefault();
    toggleRecording();
  }
  if (event.key === "Escape" && recorder?.state === "recording") {
    discardRecording = true;
    recorder.stop();
    recordedChunks = [];
    $("#record-time").hidden = true;
    $("#record-button").classList.remove("recording");
    $("#record-action").textContent = "start reading";
    $("#inference-state").textContent = "cancelled";
  }
});

renderLoop();
renderMode();
void pollEngine();
window.setInterval(() => void pollEngine(), 4000);
