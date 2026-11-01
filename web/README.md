# global-law-vector · embedding space

A Next.js prototype that draws the chunks already indexed by `glv` as a live, orbiting 3-D
point cloud. Read-only: it never embeds, re-indexes or writes anything.

```
cd web
npm install
npm run dev          # http://localhost:3000
```

## Settings

The app reads the same settings as the Python side, in this order: the process environment,
`web/.env*` (loaded by Next.js), then the repository's `../.env`. Defaults are those of
`.env.example`.

| Variable | Default | Used for |
|---|---|---|
| `DATABASE_URL` | `postgresql://utility:utility@127.0.0.1:5432/utility` | the pgvector database `glv index` filled |
| `GLV_SCHEMA` | `global_law` | schema holding `corpus`, `document`, `chunk` |
| `GLV_VECTOR_TYPE`, `GLV_DIMS` | `halfvec`, `768` | the cast for vector parameters |
| `TEI_URL`, `TEI_API_KEY` | `http://127.0.0.1:8080`, unset | embedding the eight domain phrases once per process |
| `GLV_QUERY_PREFIX` | `query: ` | prefix for those phrases |
| `GLV_WEB_POINTS` | `12000` | points in the sample (500–20000, also `?n=` on the API) |
| `GLV_WEB_FIT` | `2000` | embeddings used to fit the projection |

## How it works

`GET /api/points` builds the sample and caches it in the server process for fifteen minutes
(`?refresh=1` rebuilds). The first call takes a few seconds on a 1.6 M-chunk index; the rest
of the page is static.

1. **Sample.** One pass over the chunk/document/corpus join ranks every chunk randomly within
   its country. The sample is balanced across countries (each gets an equal share, unused share
   flows to the larger ones), so Malta's 1,280 chunks are as visible as France's 800,000.
2. **Project 768 → 3.** PCA, fitted on the first 2,000 sampled embeddings with orthogonal
   iteration (`lib/pca.ts`). The three axes are plain vectors, so Postgres projects every other
   point with one inner product per axis (`embedding <#> axis`). Only the fit sample's vectors
   ever leave the database. PCA keeps global structure (language and country separate
   clearly); it does not unfold local clusters the way UMAP would.
3. **Colour by domain.** Eight fixed legal domains (`lib/domains.ts`). Their phrases are
   embedded with the same model through TEI; if TEI is unreachable, the average embedding of
   up to 300 chunks mentioning each domain's keywords stands in. Cosine similarity to the eight
   vectors is computed in SQL, the best wins, and the margin to the runner-up (as a percentile)
   fades uncertain points towards neutral. The legend says which source was used.
4. **Render.** One `BufferGeometry` drawn twice with an additive soft-disc shader: a bright
   core and a wide faint halo, so density alone produces brightness and depth. Country toggles
   rewrite a per-vertex visibility attribute; nothing is re-fetched. OrbitControls with
   auto-rotate; no picking.

The palette is the dark-mode categorical set of the dataviz reference palette, validated for
colour-vision deficiency against the page background.
