/**
 * Minimal client for the text-embeddings-inference server the Python indexer used, needed
 * only to embed the eight domain phrases. Mirrors global_law_vector.embed: bearer key,
 * explicit User-Agent (the RunPod proxy's Cloudflare rejects anonymous clients), `truncate`.
 * Returns null instead of throwing so the caller can fall back to keyword centroids.
 */
import { env } from "./env";

export async function embedQueries(texts: string[], timeoutMs = 20_000): Promise<Float32Array[] | null> {
  const e = env();
  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "User-Agent": "global-law-vector-web/0.1.0",
  };
  if (e.teiApiKey) headers.Authorization = `Bearer ${e.teiApiKey}`;
  const ctl = new AbortController();
  const timer = setTimeout(() => ctl.abort(), timeoutMs);
  try {
    const res = await fetch(`${e.teiUrl}/embed`, {
      method: "POST",
      headers,
      body: JSON.stringify({ inputs: texts.map((t) => e.queryPrefix + t), truncate: true }),
      signal: ctl.signal,
      cache: "no-store",
    });
    if (!res.ok) {
      console.warn(`[glv-web] TEI /embed ${res.status}: ${(await res.text()).slice(0, 200)}`);
      return null;
    }
    const data = (await res.json()) as number[][];
    if (!Array.isArray(data) || data.length !== texts.length) return null;
    return data.map((v) => Float32Array.from(v));
  } catch (err) {
    console.warn(`[glv-web] TEI unreachable at ${e.teiUrl}: ${(err as Error).message}`);
    return null;
  } finally {
    clearTimeout(timer);
  }
}
