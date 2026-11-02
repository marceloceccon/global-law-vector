/**
 * Settings, read the way the Python side reads them, so one `.env` configures both.
 *
 * Precedence: the process environment (which includes `web/.env*` under `next dev` and
 * `next start`, as Next.js loads those itself), then the `.env` at the repository root,
 * parsed with the rules of `global_law_vector.config` (KEY=VALUE, `#` comments, optional
 * quotes). The repository root is the nearest ancestor of the working directory holding a
 * `pyproject.toml`, which also covers the standalone server, whose working directory is
 * `web/.next/standalone`. Defaults match `.env.example`.
 */
import fs from "node:fs";
import path from "node:path";
import { projectionPath } from "./projection";

export type Env = {
  databaseUrl: string;
  schema: string;
  vectorType: "halfvec" | "vector";
  dims: number;
  teiUrl: string;
  teiApiKey: string;
  queryPrefix: string;
  /** Sample size of the PCA fallback; a UMAP layout fixes its own. */
  points: number;
  /** Embeddings used to fit the PCA fallback (a prefix of the sample). */
  fitSize: number;
  /** The UMAP layout written by `glv project`; PCA is used when it is missing. */
  projectionFile: string;
};

function parseDotenv(file: string): Record<string, string> {
  const out: Record<string, string> = {};
  for (const raw of fs.readFileSync(file, "utf8").split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#") || !line.includes("=")) continue;
    const i = line.indexOf("=");
    const key = line.slice(0, i).trim();
    let value = line.slice(i + 1).trim();
    if (/^(["']).*\1$/.test(value)) value = value.slice(1, -1);
    else value = value.replace(/\s+#.*$/, ""); // inline comment, as in .env.example
    if (key) out[key] = value;
  }
  return out;
}

/** The repository's `.env`, or nothing when not running inside a checkout (e.g. a container). */
function repositoryDotenv(): Record<string, string> {
  for (let dir = process.cwd(); ; dir = path.dirname(dir)) {
    if (fs.existsSync(path.join(dir, "pyproject.toml"))) {
      const file = path.join(dir, ".env");
      return fs.existsSync(file) ? parseDotenv(file) : {};
    }
    if (dir === path.dirname(dir)) return {};
  }
}

function integer(name: string, raw: string, min: number, max: number): number {
  const v = Number(raw);
  if (!Number.isInteger(v)) throw new Error(`${name} must be an integer, not "${raw}"`);
  return Math.min(Math.max(v, min), max);
}

let cached: Env | undefined;

export function env(): Env {
  if (cached) return cached;
  const fromFile = repositoryDotenv();
  const get = (k: string, d = "") => process.env[k] ?? fromFile[k] ?? d;

  // Interpolated into SQL as an identifier, so only plain lower-case names are accepted
  // (Postgres folds unquoted identifiers to lower case anyway).
  const schema = get("GLV_SCHEMA", "global_law");
  if (!/^[a-z_][a-z0-9_]*$/.test(schema)) throw new Error(`GLV_SCHEMA must be a plain lower-case identifier, not "${schema}"`);
  const vectorType = get("GLV_VECTOR_TYPE", "halfvec");
  if (vectorType !== "halfvec" && vectorType !== "vector") throw new Error(`GLV_VECTOR_TYPE must be halfvec or vector, not "${vectorType}"`);

  cached = {
    databaseUrl: get("DATABASE_URL", "postgresql://utility:utility@127.0.0.1:5432/utility"),
    schema,
    vectorType,
    dims: integer("GLV_DIMS", get("GLV_DIMS", "768"), 1, 16_000),
    teiUrl: get("TEI_URL", "http://127.0.0.1:8080").replace(/\/+$/, ""),
    teiApiKey: get("TEI_API_KEY"),
    queryPrefix: get("GLV_QUERY_PREFIX", "query: "),
    points: integer("GLV_WEB_POINTS", get("GLV_WEB_POINTS", "40000"), 500, 60_000),
    fitSize: integer("GLV_WEB_FIT", get("GLV_WEB_FIT", "2000"), 200, 5000),
    projectionFile: projectionPath(get),
  };
  return cached;
}
