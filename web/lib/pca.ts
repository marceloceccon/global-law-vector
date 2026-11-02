/**
 * The fallback projection, used when no UMAP layout has been computed (see lib/points.ts):
 * principal component analysis from 768 dimensions to 3, fitted on a few thousand sampled
 * embeddings with orthogonal (subspace) iteration.
 *
 * PCA is linear, so the three axes are plain vectors and Postgres can project every sampled
 * chunk with one inner product per axis (`embedding <#> axis`); only the fit sample's vectors
 * leave the database. It keeps the global shape of the space but compresses local clusters,
 * which is why `glv project` and UMAP are preferred.
 *
 * Cost of a fit: k · iters passes over the fit matrix, each 2 · N · D multiply-adds, about
 * 2000 × 768 × 2 × 3 × 40 ≈ 370 M flops. No covariance matrix is formed.
 */
import { lcg } from "./random";

export type Projection = {
  mean: Float64Array;
  /** k unit vectors in embedding space, most variance first. */
  axes: Float64Array[];
  /** Standard deviation of the fit sample along each axis. */
  spread: number[];
};

export function parseVector(text: string): Float32Array {
  // pgvector text form: "[0.1,0.2,...]"
  const inner = text.startsWith("[") ? text.slice(1, -1) : text;
  const parts = inner.split(",");
  const out = new Float32Array(parts.length);
  for (let i = 0; i < parts.length; i++) out[i] = +parts[i];
  return out;
}

export function fitPCA(rows: Float32Array[], k = 3, iters = 40): Projection {
  const n = rows.length;
  if (n < k + 1) throw new Error(`need more than ${k} vectors to fit a ${k}-D projection`);
  const d = rows[0].length;

  const mean = new Float64Array(d);
  for (const r of rows) for (let j = 0; j < d; j++) mean[j] += r[j];
  for (let j = 0; j < d; j++) mean[j] /= n;

  // Centred copy as one flat matrix (row major).
  const x = new Float64Array(n * d);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) x[i * d + j] = rows[i][j] - mean[j];

  const rand = lcg(42);
  let v: Float64Array[] = Array.from({ length: k }, () => {
    const a = new Float64Array(d);
    for (let j = 0; j < d; j++) a[j] = rand() - 0.5;
    return a;
  });
  orthonormalise(v);

  const proj = new Float64Array(n);
  for (let it = 0; it < iters; it++) {
    const next: Float64Array[] = [];
    for (const a of v) {
      // w = Xᵀ (X a)
      for (let i = 0; i < n; i++) {
        let s = 0;
        const o = i * d;
        for (let j = 0; j < d; j++) s += x[o + j] * a[j];
        proj[i] = s;
      }
      const w = new Float64Array(d);
      for (let i = 0; i < n; i++) {
        const p = proj[i];
        if (p === 0) continue;
        const o = i * d;
        for (let j = 0; j < d; j++) w[j] += x[o + j] * p;
      }
      next.push(w);
    }
    orthonormalise(next);
    v = next;
  }

  const spread = v.map((a) => {
    let ss = 0;
    for (let i = 0; i < n; i++) {
      let s = 0;
      const o = i * d;
      for (let j = 0; j < d; j++) s += x[o + j] * a[j];
      ss += s * s;
    }
    return Math.sqrt(ss / n);
  });
  // Most variance first.
  const order = spread.map((s, i) => [s, i] as const).sort((p, q) => q[0] - p[0]).map((p) => p[1]);
  return { mean, axes: order.map((i) => v[i]), spread: order.map((i) => spread[i]) };
}

/** Modified Gram–Schmidt, in place. */
function orthonormalise(vs: Float64Array[]): void {
  for (let i = 0; i < vs.length; i++) {
    const a = vs[i];
    for (let p = 0; p < i; p++) {
      const b = vs[p];
      let dot = 0;
      for (let j = 0; j < a.length; j++) dot += a[j] * b[j];
      for (let j = 0; j < a.length; j++) a[j] -= dot * b[j];
    }
    let norm = 0;
    for (let j = 0; j < a.length; j++) norm += a[j] * a[j];
    norm = Math.sqrt(norm) || 1;
    for (let j = 0; j < a.length; j++) a[j] /= norm;
  }
}

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let s = 0;
  for (let j = 0; j < a.length; j++) s += a[j] * b[j];
  return s;
}
