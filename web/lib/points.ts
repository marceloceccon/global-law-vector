/**
 * Builds the point-cloud payload from the tables the Python indexer filled
 * (`{schema}.corpus`, `.document`, `.chunk`). Nothing is written to the database.
 *
 * Steps, all cached in-process for CACHE_TTL_MS:
 *  1. Sample: up to `n` chunk ids, balanced across countries (water-filling: every country
 *     gets min(its size, an equal share), unused share flows to the larger ones), so small
 *     countries stay visible next to France's 800k chunks.
 *  2. Project: fit PCA on the first `fitSize` sampled embeddings (the only vectors that leave
 *     Postgres), then let Postgres project every sampled chunk with `embedding <#> axis`.
 *  3. Colour: cosine similarity to the eight domain vectors (TEI-embedded phrases, or keyword
 *     centroids when TEI is unreachable), also computed in SQL. The best domain wins; the
 *     margin to the runner-up becomes a confidence that fades uncertain points to neutral.
 */
import { pool, vectorCast, vectorLiteral } from "./db";
import { DOMAINS } from "./domains";
import { embedQueries } from "./embed";
import { env } from "./env";
import { dot, fitPCA, parseVector, type Projection } from "./pca";

const CACHE_TTL_MS = 15 * 60_000;

export type PointsPayload = {
  generatedAt: string;
  totalChunks: number;
  sample: number;
  countries: { name: string; total: number; sampled: number }[];
  corpora: string[];
  domains: { id: string; label: string; color: string }[];
  domainSource: "tei" | "keyword-centroids";
  projection: { method: "pca"; fitSize: number; spread: number[] };
  /** Parallel arrays, length `sample`; xyz has 3 entries per point. */
  xyz: number[];
  country: number[];
  corpus: number[];
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

declare global {
  // eslint-disable-next-line no-var
  var __glvCache: {
    domains?: Promise<DomainVectors>;
    payload?: Map<number, { at: number; value: Promise<PointsPayload> }>;
  };
}
const cache = (globalThis.__glvCache ??= {});

function t(name: string): string {
  return `${env().schema}.${name}`;
}

// --- 1. sample -------------------------------------------------------------------------

type Sampled = { ids: number[]; country: string[]; totals: Map<string, number> };

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

  // Water-filling allocation of n across countries.
  const names = [...byCountry.keys()].sort((a, b) => byCountry.get(a)!.length - byCountry.get(b)!.length);
  let remaining = n;
  const ids: number[] = [];
  const country: string[] = [];
  names.forEach((name, i) => {
    const share = Math.floor(remaining / (names.length - i));
    const take = Math.min(share, byCountry.get(name)!.length);
    for (const id of byCountry.get(name)!.slice(0, take)) {
      ids.push(id);
      country.push(name);
    }
    remaining -= take;
  });
  return { ids, country, totals };
}

// --- 2. projection -----------------------------------------------------------------------

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

async function domainVectors(sampleIds: number[], mean: Float64Array): Promise<DomainVectors> {
  // Phrase embeddings never change, so a successful TEI answer is kept for the process;
  // a failure is not, so a server that comes up later is used on the next build.
  if (!cache.domains) {
    const vectors = await embedQueries(DOMAINS.map((d) => d.phrase));
    if (vectors) cache.domains = Promise.resolve({ vectors, source: "tei", centered: false });
    else {
      console.warn("[glv-web] colouring domains from keyword centroids instead of TEI");
      return {
        vectors: await keywordCentroids(sampleIds, mean),
        source: "keyword-centroids",
        centered: true,
      };
    }
  }
  return cache.domains;
}

// --- assemble ----------------------------------------------------------------------------

async function build(n: number): Promise<PointsPayload> {
  const e = env();
  const sampled = await sampleIds(n);
  const projection = await fitProjection(sampled.ids.slice(0, e.fitSize));
  const domains = await domainVectors(sampled.ids, projection.mean);

  // Projection and similarities in SQL: 3 inner products + 8 similarities per chunk.
  const params: unknown[] = [sampled.ids];
  const axisCols = projection.axes.map((a, i) => {
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
  const simOffset = domains.vectors.map((v) => (domains.centered ? dot(projection.mean, v) : 0));
  const sql = `
    SELECT ch.id, co.country, co.path AS corpus, ${axisCols.join(", ")}, ${simCols.join(", ")}
    FROM ${t("chunk")} ch
    JOIN ${t("document")} d  ON d.id = ch.document_id
    JOIN ${t("corpus")}   co ON co.id = d.corpus_id
    WHERE ch.id = ANY($1::bigint[])`;
  const { rows } = await pool().query<Record<string, string | number>>(sql, params);

  const meanOffset = projection.axes.map((a) => dot(projection.mean, a));
  const scale = 1 / (2.5 * projection.spread[0]); // the cloud sits roughly in [-1, 1]³
  const countryIndex = new Map<string, number>();
  const corpusIndex = new Map<string, number>();
  const countries: PointsPayload["countries"] = [];
  const corpora: string[] = [];
  const xyz: number[] = [];
  const country: number[] = [];
  const corpus: number[] = [];
  const domain: number[] = [];
  const margins: number[] = [];

  // Standardise each domain's similarity over the sample before picking the winner. A broad
  // phrase ("public administration…") is a little similar to almost every statute and would
  // otherwise claim half the cloud; z-scores make the colour mean "closer to this domain than
  // the typical chunk is", which is what the eye should read.
  const k = DOMAINS.length;
  const sims = new Float64Array(rows.length * k);
  rows.forEach((r, p) => {
    for (let i = 0; i < k; i++) sims[p * k + i] = Number(r[`s${i}`]) - simOffset[i];
  });
  const zMean = new Float64Array(k), zStd = new Float64Array(k);
  for (let i = 0; i < k; i++) {
    let s = 0, ss = 0;
    for (let p = 0; p < rows.length; p++) { const v = sims[p * k + i]; s += v; ss += v * v; }
    zMean[i] = s / rows.length;
    zStd[i] = Math.sqrt(Math.max(ss / rows.length - zMean[i] * zMean[i], 1e-12));
  }

  rows.forEach((r, p) => {
    let ci = countryIndex.get(r.country as string);
    if (ci === undefined) {
      ci = countries.length;
      countryIndex.set(r.country as string, ci);
      countries.push({ name: r.country as string, total: sampled.totals.get(r.country as string) ?? 0, sampled: 0 });
    }
    countries[ci].sampled++;
    let ki = corpusIndex.get(r.corpus as string);
    if (ki === undefined) {
      ki = corpora.length;
      corpusIndex.set(r.corpus as string, ki);
      corpora.push(r.corpus as string);
    }
    for (let i = 0; i < 3; i++) xyz.push(round((Number(r[`p${i}`]) - meanOffset[i]) * scale));
    let best = 0, second = -Infinity, bestZ = -Infinity;
    for (let i = 0; i < k; i++) {
      const z = (sims[p * k + i] - zMean[i]) / zStd[i];
      if (z > bestZ) { second = bestZ; bestZ = z; best = i; }
      else if (z > second) second = z;
    }
    country.push(ci);
    corpus.push(ki);
    domain.push(best);
    margins.push(bestZ - second);
  });

  // Confidence = percentile rank of the margin within the sample. Absolute margins depend on
  // the model's similarity range; ranks make the fade-to-neutral the same strength regardless.
  const order = margins.map((m, i) => [m, i] as const).sort((a, b) => a[0] - b[0]);
  const confidence = new Array<number>(margins.length);
  order.forEach(([, i], rank) => (confidence[i] = round(rank / Math.max(1, margins.length - 1))));

  const totalChunks = [...sampled.totals.values()].reduce((a, b) => a + b, 0);
  countries.sort((a, b) => a.name.localeCompare(b.name));
  // Re-map country indices to the sorted order.
  const remap = new Map(countries.map((c, i) => [countryIndex.get(c.name)!, i]));
  for (let i = 0; i < country.length; i++) country[i] = remap.get(country[i])!;

  return {
    generatedAt: new Date().toISOString(),
    totalChunks,
    sample: rows.length,
    countries,
    corpora,
    domains: DOMAINS.map(({ id, label, color }) => ({ id, label, color })),
    domainSource: domains.source,
    projection: { method: "pca", fitSize: Math.min(e.fitSize, sampled.ids.length), spread: projection.spread.map(round) },
    xyz,
    country,
    corpus,
    domain,
    confidence,
  };
}

const round = (v: number) => Math.round(v * 1000) / 1000;

export function getPoints(n: number, refresh = false): Promise<PointsPayload> {
  const map = (cache.payload ??= new Map());
  const hit = map.get(n);
  if (hit && !refresh && Date.now() - hit.at < CACHE_TTL_MS) return hit.value;
  const value = build(n);
  map.set(n, { at: Date.now(), value });
  value.catch(() => map.delete(n));
  return value;
}
