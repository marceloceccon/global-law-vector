/**
 * Builds the point-cloud payload from the tables the Python indexer filled
 * (`{schema}.corpus`, `.document`, `.chunk`). Nothing is written to the database.
 *
 * Coordinates come from one of two places:
 *  - the UMAP layout `glv project` cached on disk (preferred, see lib/projection.ts): its
 *    sample of chunk ids and their 3-D positions are used as they are;
 *  - otherwise a PCA fitted here on the first `fitSize` of a fresh country-balanced sample,
 *    with Postgres projecting every sampled chunk by inner product (`embedding <#> axis`).
 *
 * Either way the domain colouring is computed on the fly: cosine similarity to the eight
 * domain vectors (TEI-embedded phrases, or keyword centroids when TEI is unreachable),
 * z-scored over the sample so a broad phrase cannot claim half the cloud. The best domain
 * wins; the margin to the runner-up becomes a confidence that fades uncertain points.
 *
 * The payload is cached in-process for CACHE_TTL_MS (one entry; see getPoints). A new
 * projection file invalidates it on the next request.
 */
import { countryColor } from "./colors";
import { pool, vectorCast, vectorLiteral } from "./db";
import { DOMAINS } from "./domains";
import { embedQueries } from "./embed";
import { env } from "./env";
import { dot, fitPCA, parseVector, type Projection } from "./pca";
import { loadProjection, type ProjectionFile } from "./projection";
import { lcg, shuffle } from "./random";

const CACHE_TTL_MS = 15 * 60_000;

export type PointsPayload = {
  generatedAt: string;
  totalChunks: number;
  sample: number;
  model: string | null;
  countries: { name: string; total: number; sampled: number; color: string }[];
  domains: { id: string; label: string; color: string }[];
  domainSource: "tei" | "keyword-centroids";
  projection:
    | { method: "umap"; fitSize: number; createdAt: string; params: ProjectionFile["params"] }
    | { method: "pca"; fitSize: number; spread: number[] };
  /** Parallel arrays, length `sample`; xyz has 3 entries per point. */
  xyz: number[];
  country: number[];
  domain: number[];
  confidence: number[];
};

/**
 * `centered` tells build() how to score: TEI phrase vectors are compared by cosine, as the
 * model intends; keyword centroids are unit directions from the sample mean, compared by
 * inner product with the mean-centred chunk vector (see keywordCentroids).
 */
type DomainVectors = {
  vectors: Float32Array[];
  source: PointsPayload["domainSource"];
  centered: boolean;
};

type CacheEntry = { key: string; at: number; settled: boolean; value: Promise<PointsPayload> };

// Kept on globalThis so Next's dev-mode hot reloads keep the cache.
const g = globalThis as typeof globalThis & {
  __glvCache?: { domains?: Promise<DomainVectors>; payload?: CacheEntry };
};
const cache = (g.__glvCache ??= {});

function t(name: string): string {
  return `${env().schema}.${name}`;
}

// --- 1. sample (PCA path only) -------------------------------------------------------------

type Sampled = { ids: number[]; totals: Map<string, number> };

async function sampleIds(n: number): Promise<Sampled> {
  // One pass over the join: every chunk gets a random rank within its country and the
  // country's total. We keep at most n per country (an upper bound for the allocation
  // below), which is a few MB of small integers even for the largest table.
  const sql = `
    WITH ranked AS (
      SELECT ch.id, co.country,
             row_number() OVER (PARTITION BY co.country ORDER BY random()) AS rn,
             count(*)     OVER (PARTITION BY co.country)                   AS total
      FROM ${t("chunk")} ch
      JOIN ${t("document")} d  ON d.id = ch.document_id
      JOIN ${t("corpus")}   co ON co.id = d.corpus_id
    )
    SELECT id, country, total::int AS total FROM ranked WHERE rn <= $1 ORDER BY country, rn`;
  const { rows } = await pool().query<{ id: string; country: string; total: number }>(sql, [n]);

  const byCountry = new Map<string, number[]>();
  const totals = new Map<string, number>();
  for (const r of rows) {
    totals.set(r.country, r.total);
    let list = byCountry.get(r.country);
    if (!list) byCountry.set(r.country, (list = []));
    list.push(Number(r.id));
  }

  // Water-filling allocation of n across countries, smallest first.
  const names = [...byCountry.keys()].sort((a, b) => byCountry.get(a)!.length - byCountry.get(b)!.length);
  let remaining = n;
  const ids: number[] = [];
  names.forEach((name, i) => {
    const share = Math.floor(remaining / (names.length - i));
    const take = Math.min(share, byCountry.get(name)!.length);
    ids.push(...byCountry.get(name)!.slice(0, take));
    remaining -= take;
  });
  // Shuffle so the PCA fit prefix is as balanced as the whole.
  shuffle(ids, lcg(42));
  return { ids, totals };
}

// --- 2. PCA fallback -----------------------------------------------------------------------

async function fitProjection(ids: number[]): Promise<Projection> {
  const { rows } = await pool().query<{ embedding: string }>(
    `SELECT embedding::text AS embedding FROM ${t("chunk")} WHERE id = ANY($1::bigint[])`,
    [ids],
  );
  return fitPCA(rows.map((r) => parseVector(r.embedding)), 3);
}

// --- 3. domain vectors -------------------------------------------------------------------

const MIN_CENTROID_MATCHES = 25;

/** `\m(word|word|...)`: a case-insensitive Postgres regex anchored at word starts. */
function keywordRegex(keywords: string[]): string {
  const esc = keywords.map((k) => k.trim().replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  return `\\m(${esc.join("|")})`;
}

async function keywordCentroids(sampleIds: number[], mean: Float64Array): Promise<Float32Array[]> {
  // Fallback when TEI is unreachable: a direction in embedding space per domain, built from
  // chunks whose text starts a word with one of the domain's keywords.
  //
  // Two corrections keep this about the topic rather than the language. Matching happens
  // inside the country-balanced sample and the per-country means are averaged, so a keyword
  // that is common in one language (Russian "налог") cannot turn the domain into "Russian".
  // The sample mean is then subtracted and the result normalised, so the shared component
  // every legal text has (and that cosine against a raw centroid would reward) drops out.
  // A domain too rare in the sample falls back to a bounded scan of the whole table.
  const out: Float32Array[] = [];
  for (const d of DOMAINS) {
    const re = keywordRegex(d.keywords);
    const { rows } = await pool().query<{ c: string; n: string }>(
      `SELECT avg(ch.embedding::vector)::text AS c, count(*) AS n
         FROM ${t("chunk")} ch
         JOIN ${t("document")} doc ON doc.id = ch.document_id
         JOIN ${t("corpus")}   co  ON co.id = doc.corpus_id
        WHERE ch.id = ANY($1::bigint[]) AND ch.text ~* $2
        GROUP BY co.country HAVING count(*) >= 3`,
      [sampleIds, re],
    );
    let means = rows.map((r) => parseVector(r.c));
    const matched = rows.reduce((s, r) => s + Number(r.n), 0);
    if (matched < MIN_CENTROID_MATCHES) {
      const fallback = await pool().query<{ c: string | null }>(
        `SELECT avg(embedding::vector)::text AS c
           FROM (SELECT embedding FROM ${t("chunk")} WHERE text ~* $1 LIMIT 300) q`,
        [re],
      );
      if (!fallback.rows[0]?.c) throw new Error(`no chunks match the keywords of domain "${d.id}"`);
      means = [parseVector(fallback.rows[0].c)];
    }
    const dir = new Float32Array(mean.length);
    for (const m of means) for (let j = 0; j < dir.length; j++) dir[j] += m[j] / means.length;
    let norm = 0;
    for (let j = 0; j < dir.length; j++) {
      dir[j] -= mean[j];
      norm += dir[j] * dir[j];
    }
    norm = Math.sqrt(norm) || 1;
    for (let j = 0; j < dir.length; j++) dir[j] /= norm;
    out.push(dir);
  }
  return out;
}

async function domainVectors(sampleIds: number[], mean: () => Promise<Float64Array>): Promise<DomainVectors> {
  // Phrase embeddings never change, so a successful TEI answer is kept for the process;
  // a failure is not, so a server that comes up later is used on the next build.
  if (!cache.domains) {
    const vectors = await embedQueries(DOMAINS.map((d) => d.phrase));
    if (vectors) cache.domains = Promise.resolve({ vectors, source: "tei", centered: false });
    else {
      console.warn("[glv-web] colouring domains from keyword centroids instead of TEI");
      return {
        vectors: await keywordCentroids(sampleIds, await mean()),
        source: "keyword-centroids",
        centered: true,
      };
    }
  }
  return cache.domains;
}

// --- assemble ----------------------------------------------------------------------------

type Layout = {
  ids: number[];
  /** xyz per id, aligned with `ids`; null means Postgres projects (PCA path). */
  xyz: Float32Array | null;
  totals: Map<string, number>;
  pca: Projection | null;
  meta: PointsPayload["projection"];
};

async function layout(n: number, proj: ProjectionFile | null): Promise<Layout> {
  const e = env();
  if (proj) {
    if (proj.dims !== e.dims) {
      throw new Error(`the projection was made from ${proj.dims}-dimensional embeddings but GLV_DIMS is ${e.dims}`);
    }
    const totals = new Map(Object.entries(proj.country_totals));
    return {
      ids: proj.ids,
      xyz: Float32Array.from(proj.xyz),
      totals,
      pca: null,
      meta: { method: "umap", fitSize: proj.params.fit, createdAt: proj.created_at, params: proj.params },
    };
  }
  console.warn(`[glv-web] no projection at ${e.projectionFile}; falling back to PCA (run \`glv project\`)`);
  const sampled = await sampleIds(n);
  const pca = await fitProjection(sampled.ids.slice(0, e.fitSize));
  return {
    ids: sampled.ids,
    xyz: null,
    totals: sampled.totals,
    pca,
    meta: { method: "pca", fitSize: Math.min(e.fitSize, sampled.ids.length), spread: pca.spread.map(round) },
  };
}

/** One row of the build query: `p0..p2` are PCA coordinates, `s0..s7` domain similarities. */
type Row = { id: string; country: string } & Record<`p${number}` | `s${number}`, number>;

async function build(n: number, proj: ProjectionFile | null): Promise<PointsPayload> {
  const e = env();
  const lay = await layout(n, proj);
  // The PCA fit's mean doubles as the sample mean the centroid fallback needs; on the UMAP
  // path it is fitted lazily, only if TEI turns out to be unreachable.
  const meanOf = async () => (lay.pca ?? (lay.pca = await fitProjection(lay.ids.slice(0, e.fitSize)))).mean;
  const domains = await domainVectors(lay.ids, meanOf);

  // Projection (PCA path) and similarities in SQL: 3 inner products + 8 similarities per chunk.
  const params: unknown[] = [lay.ids];
  const axisCols = (lay.xyz ? [] : lay.pca!.axes).map((a, i) => {
    params.push(vectorLiteral(a));
    return `-(ch.embedding <#> ${vectorCast(params.length)}) AS p${i}`;
  });
  const simCols = domains.vectors.map((v, i) => {
    params.push(vectorLiteral(v));
    const cast = vectorCast(params.length);
    return domains.centered
      ? `-(ch.embedding <#> ${cast}) AS s${i}` // (e − μ)·v, with μ·v subtracted below
      : `1 - (ch.embedding <=> ${cast}) AS s${i}`;
  });
  const simOffset = domains.vectors.map((v) => (domains.centered ? dot(lay.pca!.mean, v) : 0));
  const sql = `
    SELECT ch.id, co.country${axisCols.length ? ", " + axisCols.join(", ") : ""}, ${simCols.join(", ")}
    FROM ${t("chunk")} ch
    JOIN ${t("document")} d  ON d.id = ch.document_id
    JOIN ${t("corpus")}   co ON co.id = d.corpus_id
    WHERE ch.id = ANY($1::bigint[])`;
  const { rows } = await pool().query<Row>(sql, params);

  // Keep the layout's order (the cached xyz is aligned with it); chunks deleted since the
  // projection was made simply drop out.
  const byId = new Map(rows.map((r) => [Number(r.id), r]));
  const kept: { r: Row; i: number }[] = [];
  lay.ids.forEach((id, i) => {
    const r = byId.get(id);
    if (r) kept.push({ r, i });
  });

  const meanOffset = lay.pca && !lay.xyz ? lay.pca.axes.map((a) => dot(lay.pca!.mean, a)) : [];
  const scale = lay.pca && !lay.xyz ? 1 / (2.5 * lay.pca.spread[0]) : 1; // PCA cloud roughly in [-1, 1]³
  const countryIndex = new Map<string, number>();
  const countryNames: string[] = [];
  const xyz: number[] = [];
  const country: number[] = [];
  const domain: number[] = [];
  const margins: number[] = [];

  // Standardise each domain's similarity over the sample before picking the winner. A broad
  // phrase ("public administration…") is a little similar to almost every statute and would
  // otherwise claim half the cloud; z-scores make the colour mean "closer to this domain than
  // the typical chunk is", which is what the eye should read.
  const k = DOMAINS.length;
  const sims = new Float64Array(kept.length * k);
  kept.forEach(({ r }, p) => {
    for (let i = 0; i < k; i++) sims[p * k + i] = Number(r[`s${i}`]) - simOffset[i];
  });
  const zMean = new Float64Array(k), zStd = new Float64Array(k);
  for (let i = 0; i < k; i++) {
    let s = 0, ss = 0;
    for (let p = 0; p < kept.length; p++) { const v = sims[p * k + i]; s += v; ss += v * v; }
    zMean[i] = s / kept.length;
    zStd[i] = Math.sqrt(Math.max(ss / kept.length - zMean[i] * zMean[i], 1e-12));
  }

  kept.forEach(({ r, i: li }, p) => {
    let ci = countryIndex.get(r.country);
    if (ci === undefined) {
      ci = countryNames.length;
      countryIndex.set(r.country, ci);
      countryNames.push(r.country);
    }
    if (lay.xyz) xyz.push(lay.xyz[li * 3], lay.xyz[li * 3 + 1], lay.xyz[li * 3 + 2]);
    else for (let a = 0; a < 3; a++) xyz.push(round((Number(r[`p${a}`]) - meanOffset[a]) * scale));
    let best = 0, second = -Infinity, bestZ = -Infinity;
    for (let d = 0; d < k; d++) {
      const z = (sims[p * k + d] - zMean[d]) / zStd[d];
      if (z > bestZ) { second = bestZ; bestZ = z; best = d; }
      else if (z > second) second = z;
    }
    country.push(ci);
    domain.push(best);
    margins.push(bestZ - second);
  });

  // Confidence = percentile rank of the margin within the sample. Absolute margins depend on
  // the model's similarity range; ranks make the fade-to-neutral the same strength regardless.
  const order = margins.map((m, i) => [m, i] as const).sort((a, b) => a[0] - b[0]);
  const confidence = new Array<number>(margins.length);
  order.forEach(([, i], rank) => (confidence[i] = round(rank / Math.max(1, margins.length - 1))));

  // Countries alphabetical; colours follow that order (see lib/colors.ts).
  const sortedNames = [...countryNames].sort((a, b) => a.localeCompare(b));
  const remap = new Map(sortedNames.map((name, i) => [countryIndex.get(name)!, i]));
  for (let i = 0; i < country.length; i++) country[i] = remap.get(country[i])!;
  const sampledPer = new Array<number>(sortedNames.length).fill(0);
  for (const c of country) sampledPer[c]++;
  const countries = sortedNames.map((name, i) => ({
    name,
    total: lay.totals.get(name) ?? 0,
    sampled: sampledPer[i],
    color: countryColor(i),
  }));

  return {
    generatedAt: new Date().toISOString(),
    totalChunks: [...lay.totals.values()].reduce((a, b) => a + b, 0),
    sample: kept.length,
    model: proj?.model ?? null,
    countries,
    domains: DOMAINS.map(({ id, label, color }) => ({ id, label, color })),
    domainSource: domains.source,
    projection: lay.meta,
    xyz,
    country,
    domain,
    confidence,
  };
}

const round = (v: number) => Math.round(v * 1000) / 1000;

/**
 * The payload, built at most once at a time. One entry is cached: the layout file's
 * `created_at` (or the PCA sample size) is its key, so a new `glv project` run is served on
 * the next request without a restart, and `?n=` cannot fill memory with payloads.
 * `refresh` rebuilds unless a build is already running, which is then shared.
 */
export function getPoints(n: number, refresh = false): Promise<PointsPayload> {
  const proj = loadProjection(env().projectionFile);
  const key = proj ? `umap:${proj.created_at}` : `pca:${n}`;
  const hit = cache.payload;
  if (hit && hit.key === key && (!hit.settled || (!refresh && Date.now() - hit.at < CACHE_TTL_MS))) {
    return hit.value;
  }
  const entry: CacheEntry = { key, at: Date.now(), settled: false, value: build(n, proj) };
  cache.payload = entry;
  entry.value.then(
    () => (entry.settled = true),
    () => {
      if (cache.payload === entry) cache.payload = undefined;
    },
  );
  return entry.value;
}
