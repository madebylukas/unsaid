import { createWriteStream } from "node:fs";
import { access, mkdir, rename, stat, unlink } from "node:fs/promises";
import { dirname, resolve } from "node:path";
import { Readable, Transform } from "node:stream";
import { pipeline } from "node:stream/promises";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const chaplin = resolve(root, ".cache/chaplin");
const commit = "7aee1f8fca776ce4f63690063310b53573b7d804";

function run(command, args) {
  const result = spawnSync(command, args, { cwd: root, stdio: "inherit" });
  if (result.status !== 0) throw new Error(`${command} failed with exit code ${result.status}`);
}

try {
  await access(resolve(chaplin, ".git"));
} catch {
  await mkdir(resolve(root, ".cache"), { recursive: true });
  run("git", ["clone", "https://github.com/amanvirparhar/chaplin.git", chaplin]);
}
run("git", ["-C", chaplin, "checkout", commit]);

const assets = [
  {
    path: "benchmarks/LRS3/models/LRS3_V_WER19.1/model.json",
    url: "https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.json",
    minimumBytes: 5_000,
  },
  {
    path: "benchmarks/LRS3/models/LRS3_V_WER19.1/model.pth",
    url: "https://huggingface.co/Amanvir/LRS3_V_WER19.1/resolve/main/model.pth",
    minimumBytes: 1_000_000_000,
  },
  {
    path: "benchmarks/LRS3/language_models/lm_en_subword/model.json",
    url: "https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.json",
    minimumBytes: 900,
  },
  {
    path: "benchmarks/LRS3/language_models/lm_en_subword/model.pth",
    url: "https://huggingface.co/Amanvir/lm_en_subword/resolve/main/model.pth",
    minimumBytes: 200_000_000,
  },
];

for (const asset of assets) {
  const target = resolve(chaplin, asset.path);
  try {
    const existing = await stat(target);
    if (existing.size >= asset.minimumBytes) {
      process.stdout.write(`ready  ${asset.path}\n`);
      continue;
    }
  } catch {
    // Download below.
  }
  await mkdir(dirname(target), { recursive: true });
  const partial = `${target}.part`;
  let resumeBytes = 0;
  try {
    resumeBytes = (await stat(partial)).size;
  } catch {
    try {
      resumeBytes = (await stat(target)).size;
      if (resumeBytes) await rename(target, partial);
    } catch {
      resumeBytes = 0;
    }
  }

  process.stdout.write(`${resumeBytes ? "resume" : "fetch "} ${asset.path}\n`);
  let response = await fetch(asset.url, {
    headers: resumeBytes ? { Range: `bytes=${resumeBytes}-` } : undefined,
  });
  if (!response.ok || !response.body) {
    throw new Error(`download failed: ${response.status} ${response.statusText}`);
  }
  const canResume = resumeBytes > 0 && response.status === 206;
  if (resumeBytes && !canResume) {
    await unlink(partial).catch(() => undefined);
    resumeBytes = 0;
    response = await fetch(asset.url);
    if (!response.ok || !response.body) {
      throw new Error(`download failed: ${response.status} ${response.statusText}`);
    }
  }
  const remainingBytes = Number(response.headers.get("content-length") ?? 0);
  const expectedBytes = remainingBytes ? resumeBytes + remainingBytes : 0;
  let receivedBytes = resumeBytes;
  let reportedPercent = -10;
  const progress = new Transform({
    transform(chunk, _encoding, callback) {
      receivedBytes += chunk.length;
      if (expectedBytes) {
        const percent = Math.floor((receivedBytes / expectedBytes) * 10) * 10;
        if (percent > reportedPercent) {
          reportedPercent = percent;
          process.stdout.write(`       ${percent}%\n`);
        }
      }
      callback(null, chunk);
    },
  });
  await pipeline(
    Readable.fromWeb(response.body),
    progress,
    createWriteStream(partial, { flags: canResume ? "a" : "w" }),
  );
  await rename(partial, target);
  const downloaded = await stat(target);
  if (downloaded.size < asset.minimumBytes) throw new Error(`download incomplete: ${asset.path}`);
}

process.stdout.write("\nvisual speech assets ready. start the engine with `npm run vsr`.\n");
