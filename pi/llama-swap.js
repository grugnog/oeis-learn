// Pi >= 0.85.1. Loaded before model selection, including --list-models.
// The host catalog is authoritative; no per-model defaults or cached list here.
const object = (value) => value !== null && typeof value === "object" && !Array.isArray(value);
const positiveInteger = (value) => Number.isSafeInteger(value) && value > 0;

export function apiBaseUrl(value) {
  const url = new URL(value);
  if (!["http:", "https:"].includes(url.protocol) || url.username || url.password || url.search || url.hash) {
    throw new Error("LAB_BASE_URL must be an HTTP(S) URL without credentials, query or fragment");
  }
  const path = url.pathname.replace(/\/+$/, "");
  url.pathname = path.endsWith("/v1") ? path : `${path}/v1`;
  return url.toString().replace(/\/$/, "");
}

export function catalogModels(payload) {
  if (!Array.isArray(payload?.data) || payload.data.length === 0) {
    throw new Error("/v1/models returned an empty or invalid catalog");
  }
  const ids = new Set();
  return payload.data.map((entry) => {
    const meta = entry?.meta?.llamaswap;
    if (typeof entry?.id !== "string" || !entry.id.trim() || ids.has(entry.id)) {
      throw new Error("/v1/models contains a missing or duplicate model ID");
    }
    ids.add(entry.id);
    if (!object(meta) || meta.schemaVersion !== 1 || !positiveInteger(meta.contextWindow)
      || !positiveInteger(meta.maxTokens) || meta.maxTokens > meta.contextWindow
      || typeof meta.reasoning !== "boolean" || !Array.isArray(meta.input)
      || meta.input.length === 0 || !meta.input.every((input) => ["text", "image"].includes(input))
      || !object(meta.compat)) {
      throw new Error(`${entry.id}: missing/invalid meta.llamaswap v1 metadata; rebuild the host configuration`);
    }
    return {
      id: entry.id,
      name: entry.name || entry.description || entry.id,
      contextWindow: meta.contextWindow,
      maxTokens: meta.maxTokens,
      reasoning: meta.reasoning,
      input: meta.input,
      compat: meta.compat,
      cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
    };
  });
}

/** @param {import("@earendil-works/pi-coding-agent").ExtensionAPI} pi */
export default async function (pi) {
  if (!process.env.LAB_BASE_URL) return;
  try {
    const baseUrl = apiBaseUrl(process.env.LAB_BASE_URL);
    const refreshModels = async () => {
      const response = await fetch(`${baseUrl}/models`, {
        // Only catalog discovery is bounded; this is not a generation timeout.
        signal: AbortSignal.timeout(10000),
        redirect: "error",
        headers: { Authorization: `Bearer ${process.env.LAB_API_KEY || "local"}` },
      });
      if (!response.ok) throw new Error(`catalog returned HTTP ${response.status}`);
      return catalogModels(await response.json());
    };
    const models = await refreshModels();
    pi.registerProvider("lab", {
      name: "Lab (llama-swap)",
      baseUrl,
      api: "openai-completions",
      apiKey: process.env.LAB_API_KEY ? "$LAB_API_KEY" : "local",
      models,
      refreshModels,
    });
  } catch (error) {
    // Leave unrelated providers usable; selecting --provider lab still fails
    // explicitly rather than silently choosing an unrelated provider.
    const message = `Lab model discovery failed: ${error.message}. Check LAB_BASE_URL and the host catalog, then /reload.`;
    process.stderr.write(`${message}\n`);
    pi.on("session_start", async (_event, ctx) => {
      if (ctx.hasUI) ctx.ui.notify(message, "warning");
    });
  }
}
