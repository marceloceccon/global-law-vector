/**
 * The cached UMAP layout written by `glv project` (src/global_law_vector/project.py).
 *
 * Location: `GLV_PROJECTION`, else `$GLV_CACHE_DIR/projection.json`, else
 * `~/.cache/global-law-vector/projection.json`, the same default the Python side writes to.
 * Re-read whenever the file's mtime changes, so re-running `glv project` is picked up by the
 * next request without restarting the server. The type documents the whole file format; the
 * viewer reads `ids`, `xyz`, `country_totals`, `params`, `created_at`, `model` and `dims`.
 */
import fs from "node:fs";
import os from "node:os";
import path from "node:path";

export type ProjectionFile = {
  version: 1;
  method: "umap";
  created_at: string;
  model: string | null;
  dims: number;
  params: { n_neighbors: number; min_dist: number; metric: string; seed: number; fit: number };
  sample: number;
  total_chunks: number;
  country_totals: Record<string, number>;
  countries: string[];
  corpora: string[];
  ids: number[];
  country: number[];
  corpus: number[];
  xyz: number[];
};

export function projectionPath(get: (k: string, d?: string) => string): string {
  const explicit = get("GLV_PROJECTION");
  if (explicit) return expand(explicit);
  const cacheDir = get("GLV_CACHE_DIR");
  return path.join(cacheDir ? expand(cacheDir) : path.join(os.homedir(), ".cache", "global-law-vector"), "projection.json");
}

function expand(p: string): string {
  return p.startsWith("~") ? path.join(os.homedir(), p.slice(1)) : p;
}

let cached: { mtimeMs: number; file: string; value: ProjectionFile } | undefined;

/** The projection, or null when none has been computed yet. */
export function loadProjection(file: string): ProjectionFile | null {
  let stat: fs.Stats;
  try {
    stat = fs.statSync(file);
  } catch {
    return null;
  }
  if (cached && cached.file === file && cached.mtimeMs === stat.mtimeMs) return cached.value;
  const value = JSON.parse(fs.readFileSync(file, "utf8")) as ProjectionFile;
  const consistent =
    value.version === 1 &&
    value.method === "umap" &&
    Array.isArray(value.ids) &&
    value.ids.length === value.sample &&
    Array.isArray(value.xyz) &&
    value.xyz.length === value.sample * 3;
  if (!consistent) throw new Error(`${file} is not a projection file this viewer understands; rerun \`glv project\``);
  cached = { mtimeMs: stat.mtimeMs, file, value };
  return value;
}
