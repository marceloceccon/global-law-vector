/**
 * Settings, read the way the Python side reads them.
 *
 * Order of precedence: the real environment, then `web/.env*` (which Next.js loads itself),
 * then the repository's `../.env`, parsed with the same rules as `global_law_vector.config`
 * (KEY=VALUE, `#` comments, optional quotes). Defaults match `.env.example`.
 */
import fs from "node:fs";
import path from "node:path";

export type Env = {
  databaseUrl: string;
  schema: string;
  vectorType: "halfvec" | "vector";
  dims: number;
  teiUrl: string;
  teiApiKey: string;
  queryPrefix: string;
  /** Points in the visualisation sample. */
  points: number;
  /** Embeddings used to fit the 3-D projection (a prefix of the sample). */
  fitSize: number;
};

function parseDotenv(file: string): Record<string, string> {
  const out: Record<string, string> = {};
  if (!fs.existsSync(file)) return out;
  for (const raw of fs.readFileSync(file, "utf8").split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || line.startsWith("#") || !line.includes("=")) continue;
    const i = line.indexOf("=");
    const key = line.slice(0, i).trim();
    let value = line.slice(i + 1).trim();
    const quoted = /^(["']).*\1$/.test(value);
    if (quoted) value = value.slice(1, -1);
    else value = value.replace(/\s+#.*$/, ""); // inline comment, as in .env.example
    if (key) out[key] = value;
  }
  return out;
}

let cached: Env | undefined;

export function env(): Env {
  if (cached) return cached;
  const repoEnv = parseDotenv(path.resolve(process.cwd(), "..", ".env"));
  const get = (k: string, d = "") => process.env[k] ?? repoEnv[k] ?? d;

  const schema = get("GLV_SCHEMA", "global_law");
  if (!/^[a-z_][a-z0-9_]*$/i.test(schema)) throw new Error(`GLV_SCHEMA is not a plain identifier: ${schema}`);
  const vectorType = get("GLV_VECTOR_TYPE", "halfvec");
  if (vectorType !== "halfvec" && vectorType !== "vector") throw new Error(`unsupported GLV_VECTOR_TYPE ${vectorType}`);

  cached = {
    databaseUrl: get("DATABASE_URL", "postgresql://utility:utility@127.0.0.1:5432/utility"),
    schema,
    vectorType,
    dims: Number(get("GLV_DIMS", "768")),
    teiUrl: get("TEI_URL", "http://127.0.0.1:8080").replace(/\/+$/, ""),
    teiApiKey: get("TEI_API_KEY"),
    queryPrefix: get("GLV_QUERY_PREFIX", "query: "),
    points: Math.min(Math.max(Number(get("GLV_WEB_POINTS", "12000")), 500), 20000),
    fitSize: Math.min(Math.max(Number(get("GLV_WEB_FIT", "2000")), 200), 5000),
  };
  return cached;
}
