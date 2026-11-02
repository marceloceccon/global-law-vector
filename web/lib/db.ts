/**
 * One pg pool per process. The viewer only reads, and every session is opened read-only so
 * that stays true whatever the database role is allowed to do.
 */
import { Pool } from "pg";
import { env } from "./env";

// Kept on globalThis so Next's dev-mode hot reloads reuse the pool instead of leaking one.
const g = globalThis as typeof globalThis & { __glvPool?: Pool };

export function pool(): Pool {
  if (!g.__glvPool) {
    const p = new Pool({
      connectionString: env().databaseUrl,
      max: 4,
      statement_timeout: 120_000,
      application_name: "global-law-vector-web",
      options: "-c default_transaction_read_only=on",
    });
    // An idle client losing its connection (a database restart) must not crash the server;
    // the pool replaces it on the next query.
    p.on("error", (err) => console.error(`[glv-web] idle database client: ${err.message}`));
    g.__glvPool = p;
  }
  return g.__glvPool;
}

/** pgvector text literal: `[0.1,0.2,...]`. */
export function vectorLiteral(v: ArrayLike<number>): string {
  const parts = new Array<string>(v.length);
  for (let i = 0; i < v.length; i++) parts[i] = v[i].toFixed(7);
  return `[${parts.join(",")}]`;
}

/** `$n::halfvec(768)`: the cast every vector parameter needs. */
export function vectorCast(param: number): string {
  const e = env();
  return `$${param}::${e.vectorType}(${e.dims})`;
}
