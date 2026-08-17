import { copyFile, mkdir, access } from "node:fs/promises";
import { constants } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const publicDir = resolve(root, "public");
const wasmSource = resolve(root, "node_modules/@mediapipe/tasks-vision/wasm");
const wasmTarget = resolve(publicDir, "mediapipe");
const modelTarget = resolve(publicDir, "models/face_landmarker.task");
const modelUrl =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task";

await mkdir(wasmTarget, { recursive: true });
await mkdir(dirname(modelTarget), { recursive: true });

for (const filename of [
  "vision_wasm_internal.js",
  "vision_wasm_internal.wasm",
  "vision_wasm_nosimd_internal.js",
  "vision_wasm_nosimd_internal.wasm",
]) {
  await copyFile(resolve(wasmSource, filename), resolve(wasmTarget, filename));
}

try {
  await access(modelTarget, constants.R_OK);
} catch {
  process.stdout.write("Downloading MediaPipe Face Landmarker model...\n");
  const response = await fetch(modelUrl);
  if (!response.ok) {
    throw new Error(`Model download failed: ${response.status} ${response.statusText}`);
  }
  const bytes = Buffer.from(await response.arrayBuffer());
  await import("node:fs/promises").then(({ writeFile }) => writeFile(modelTarget, bytes));
}

process.stdout.write("Local vision assets ready.\n");
