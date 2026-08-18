export interface Candidate {
  text: string;
  probability: number;
  score: number;
}

export interface InferenceResult {
  transcript: string;
  candidates: Candidate[];
  latency_ms: number;
  duration_seconds: number;
  device: string;
}

export interface EngineHealth {
  status: "ready" | "loading" | "setup_required";
  model_loaded: boolean;
  device: string;
  engine: string;
  detail?: string;
}

const ENGINE_ORIGIN = import.meta.env.PROD ? "/api" : "http://127.0.0.1:8787";

export async function getEngineHealth(): Promise<EngineHealth | null> {
  try {
    const response = await fetch(`${ENGINE_ORIGIN}/health`, { signal: AbortSignal.timeout(1500) });
    if (!response.ok) return null;
    return (await response.json()) as EngineHealth;
  } catch {
    return null;
  }
}

export async function inferVideo(blob: Blob): Promise<InferenceResult> {
  const form = new FormData();
  form.append("video", blob, "silent-speech.webm");
  const response = await fetch(`${ENGINE_ORIGIN}/infer`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    let detail = `visual model returned ${response.status}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      if (payload.detail) detail = payload.detail;
    } catch {
      // The response was not JSON.
    }
    throw new Error(detail);
  }
  return (await response.json()) as InferenceResult;
}
