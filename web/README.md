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
| `GLV_PROJECTION` | `$GLV_CACHE_DIR/projection.json`, i.e. `~/.cache/global-law-vector/projection.json` | the UMAP layout written by `glv project` |
| `GLV_WEB_POINTS` | `40000` | points in the PCA fallback sample (500–60000, also `?n=` on the API); the UMAP file fixes its own size |
| `GLV_WEB_FIT` | `2000` | embeddings used to fit the PCA fallback |

## Serving it to the network

The compose file in `../global-law-utility` has a `web` service that builds this directory
into a self-contained image, joins the `utility` network (so `utility-pgvector` and
`utility-embeddings` resolve) and publishes port 3000 on every interface of the host:

```sh
cd ../global-law-utility
docker compose up -d --build web
# then, from any machine on the LAN: http://<host-ip>:3000
```

Rebuild with the same command after changing the code. Without Docker, a production build
bound to all interfaces does the same job:

```sh
npm run build && HOSTNAME=0.0.0.0 PORT=3000 npm start
```

The image is read-only against the database and holds no secrets; `DATABASE_URL` and
`TEI_URL` come from the compose `environment` block.

## How it works

`GET /api/points` builds the payload and caches it in the server process for fifteen minutes
(`?refresh=1` rebuilds; a newer projection file also invalidates it). The first call takes a
second or two with the UMAP file, about ten seconds on the PCA fallback.

1. **Sample and layout: `glv project`** (`src/global_law_vector/project.py`). One pass over
   the chunk/document/corpus join ranks every chunk randomly within its country; the sample is
   balanced across countries (each gets an equal share, unused share flows to the larger
   ones), so Malta's 1,280 chunks are as visible as France's 800,000. The order is shuffled,
   the vectors are fetched, and UMAP (`umap-learn`) reduces them to three components:
   `n_neighbors=20`, `min_dist=0.1`, `metric=cosine`, `random_state=42`, fitted on the first
   20,000 points and the remaining 20,000 placed with `transform`. Coordinates are centred
   on the median and scaled so the 99th-percentile radius is 1. On a six-core desktop the fit
   takes about 50 s and the whole command about two minutes; the result is 1.4 MB of JSON in
   `~/.cache/global-law-vector/projection.json`. Rerun it after indexing more corpora.
2. **Fallback: PCA in the API.** Without that file the API draws its own balanced sample,
   fits PCA on 2,000 embeddings with orthogonal iteration (`lib/pca.ts`) and lets Postgres
   project the rest with one inner product per axis. The caption on the page says which
   method is on screen. PCA keeps the global shape but squashes local structure; with UMAP,
   84 % of a point's ten nearest neighbours share its country (measured on this index),
   which is the structure the viewer is meant to show.
3. **Colour.** The primary encoding is country: ten muted OKLCH hues (`lib/colors.ts`),
   assigned in alphabetical order of the countries present, validated against the page
   background with the dataviz palette validator (lightness, chroma, contrast and adjacent
   legend pairs pass; ten mutually colour-blind-safe hues do not exist, so the name always sits
   beside the swatch and the toggles are the exact way to isolate a country). The secondary
   encoding, on a toggle, is the nearest of eight fixed legal domains (`lib/domains.ts`):
   their phrases are embedded with the index model through TEI, or, if TEI is unreachable,
   keyword centroids from the sample stand in; cosine similarity to the eight vectors is
   computed in SQL and z-scored over the sample, the best wins, and the margin to the runner-up
   fades uncertain points towards neutral. Both colours travel to the GPU as vertex attributes,
   so switching is instant.
4. **Render.** Two additive layers from one `BufferGeometry`: small soft discs, and a wide
   faint halo whose strength follows local density (measured in the browser by
   `lib/spatial.ts`, a uniform-grid pass that takes well under a second). Blending is modest
   enough that dense regions brighten without turning white. A faint cube at the
   normalisation extent gives orientation while orbiting. Country toggles rewrite a per-vertex
   visibility attribute; nothing is re-fetched. OrbitControls with slow auto-rotate; no
   picking, no post-processing.

   Tuning knobs are the `size` and `alpha` props of the two layers near the bottom of
   `components/PointCloud.tsx`, and `--points` on `glv project` (40,000 by default; a laptop
   GPU is comfortable there, and 20,000 is a safe choice for integrated graphics).
