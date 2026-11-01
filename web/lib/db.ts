/** One pg pool per process; kept on globalThis so Next's dev-mode hot reloads reuse it. */
import { Pool } from "pg";
import { env } from "./env";

declare global {
  // eslint-disable-next-line no-var
  var __glvPool: Pool | undefined;
}

export function pool(): Pool {
  if (!globalThis.__glvPool) {
    globalThis.__glvPool = new Pool({
      connectionString: env().databaseUrl,
      max: 4,
      statement_timeout: 120_000,
      application_name: "global-law-vector-web",
    });
  }
  return globalThis.__glvPool;
}

/** pgvector text literal: `[0.1,0.2,...]`. */
export function vectorLiteral(v: ArrayLike<number>): string {
  const parts = new Array<string>(v.length);
  for (let i = 0; i < v.length; i++) parts[i] = v[i].toFixed(7);
  return `[${parts.join(",")}]`;
}

/** `$n::halfvec(768)` — the cast every vector parameter needs. */
export function vectorCast(param: number): string {
  const e = env();
  return `$${param}::${e.vectorType}(${e.dims})`;
}
