# global-law-vector viewer

A Next.js application that draws a sample of the `glv` index as a 3-D point cloud. Each point
is one chunk of a provision, placed so that chunks with similar embeddings sit close together.
Points are coloured by country or by the nearest of eight legal domains. The cloud turns slowly
and can be orbited, zoomed and panned, and countries and domains can be switched on and off.

The viewer only reads. It never embeds law text, never re-indexes and never writes to the
database.

![Coloured by country: 40,000 chunks from ten countries](../docs/images/viewer-by-country.png)

*Coloured by country. Russian law forms an island of its own at the bottom, and Canada's sits
apart at the top. Argentina, Spain, Brazil and Portugal overlap in the central mass, where
Spanish- and Portuguese-language law meets.*

![Coloured by legal domain, filtered to three domains and nine countries](../docs/images/viewer-by-domain-filtered.png)

*Coloured by legal domain, with Russia and five of the eight domains switched off. The counts
beside each checkbox are the points currently shown.*

## Running it

The viewer needs Node.js 20.9 or later, the database `glv index` filled and, for the
intended layout, a projection computed once from the repository root:

```sh
pip install -e '.[project]'
glv project                      # about two minutes; rerun after indexing more corpora
```

Then, in `web/`:

```sh
npm install
npm run dev                      # development server on http://localhost:3000
```

Without a projection file the viewer still works. It falls back to a PCA projection fitted
when the first request arrives, and says so in its caption.

## Settings

The viewer reads the same settings as `glv`, so one `.env` configures both. It looks at the
process environment first, then at the `.env` in the repository root. Under `npm run dev`
and `npm start`, Next.js also loads `web/.env*` files into the environment.

| Variable | Default | Used for |
| --- | --- | --- |
| `DATABASE_URL` | `postgresql://utility:utility@127.0.0.1:5432/utility` | The database `glv index` filled. Sessions are opened read-only. |
| `GLV_SCHEMA` | `global_law` | Schema holding the index. Plain lower-case identifiers only. |
| `GLV_VECTOR_TYPE`, `GLV_DIMS` | `halfvec`, `768` | Casting vector parameters. They must match the index. |
| `TEI_URL`, `TEI_API_KEY` | `http://127.0.0.1:8080`, unset | Embedding the eight domain phrases, once per server process. |
| `GLV_QUERY_PREFIX` | `query: ` | Prefix for those phrases. |
| `GLV_PROJECTION` | `$GLV_CACHE_DIR/projection.json` | The layout written by `glv project`. |
| `GLV_CACHE_DIR` | `~/.cache/global-law-vector` | Where the projection is looked for when `GLV_PROJECTION` is unset. |
| `GLV_WEB_POINTS` | `40000` | Sample size of the PCA fallback, from 500 to 60,000. A UMAP layout fixes its own size. |
| `GLV_WEB_FIT` | `2000` | Embeddings used to fit the PCA fallback, from 200 to 5,000. |

## Serving it on a network

The production build is a self-contained server, so serving it to other machines takes
two commands:

```sh
npm run build
HOSTNAME=0.0.0.0 PORT=3000 npm start
# from another machine: http://<this machine's address>:3000
```

Stop the old server before starting a rebuilt one, or it keeps serving the previous build.

The viewer has no authentication. Anyone who can reach the port can load the sample, which
holds coordinates, countries and domain labels but no law text, and can ask the server to
rebuild it. Keep it on a trusted network.

`Dockerfile` packages the same server as an image with no sources and no `node_modules`. Its
header shows how to run it with the database address, the embeddings server and the projection
file mounted in.

## How the picture is made

`GET /api/points` returns everything the page draws. The server builds it on the first request,
in about 1.5 seconds with a projection file. It keeps the result in memory for fifteen minutes,
or until `glv project` writes a newer file. `?refresh=1` rebuilds it.

**Sample.** `glv project` draws 40,000 chunks, balanced across countries. Each country gets an
equal share, and the share of a country with fewer chunks flows to the larger ones. Malta's
1,280 chunks are therefore all in the sample, while France contributes about 4,600 of its
812,000. The order within each country is a hash of the chunk id and a seed, so the same seed
draws the same sample from an unchanged index.

**Layout.** UMAP (`umap-learn`) reduces the 768-dimensional embeddings to three dimensions,
with `n_neighbors=20`, `min_dist=0.1`, cosine distance and seed 42. The model is fitted on half
of the sample, and `transform` places the other half. Coordinates are centred on the median and
scaled so that 99 % of points lie within a radius of 1. The faint cube marks that extent. The
axes themselves have no meaning. Distances are only meaningful locally: two points side by side
are similar chunks, while the gap between two islands says little about how different they are.
On the ten-country index, 84 % of a point's ten nearest neighbours in the layout come from the
same country. On a six-core desktop the fit takes about 50 seconds and the whole command about
two minutes. The result is 1.4 MB of JSON.

**PCA fallback.** Without a projection file the API draws its own balanced sample and fits a
3-component PCA on 2,000 of its embeddings (`lib/pca.ts`). Postgres then projects every
sampled chunk with one inner product per axis. PCA keeps the overall shape of the space but
compresses local structure, so the cloud is a much tighter blob.

**Country colours.** Ten muted hues (`lib/colors.ts`) are assigned to the countries in
alphabetical order and checked for contrast against the background. No set of ten hues stays
distinguishable under every kind of colour-vision deficiency. Every swatch is therefore
labelled, and the checkboxes isolate a country exactly. Countries beyond the tenth share a
neutral grey.

**Domain colours.** The domain of a chunk is the nearest of eight fixed descriptions
(`lib/domains.ts`): contract and commercial, criminal, administrative and public, tax, family,
labour, inheritance, and anti-money-laundering. Each description is embedded with the index
model. The server computes the cosine similarity of every sampled chunk to all eight in SQL. It
standardises each domain's scores over the sample, so that a broad description cannot claim
most of the cloud, and takes the highest. A point close to a tie between two domains is drawn
closer to grey. If the embeddings server is unreachable, keyword centroids from the sample stand in for
the descriptions. The panel says which source is in use.

This colouring is a reading aid, not a legal classification. Most statutes touch several of
these areas, and many fit none of them well.

**Rendering.** The page uses three.js through react-three-fiber. One buffer of points is drawn
twice with additive blending: once as small soft discs in the point's colour, and once as a
wide, very faint halo whose strength follows local density (`lib/spatial.ts`). Dense regions
therefore read brighter. Colours reach the screen exactly as the legend shows them. Both
colourings and the visibility of each point are vertex attributes, so switching colouring or
toggling a checkbox costs no network request and no new geometry.

## Tuning

The `size` and `alpha` props of the two layers at the bottom of `components/PointCloud.tsx`
control how heavy the cloud looks. `glv project --points N` sets the sample size. 40,000 runs
smoothly on a laptop GPU, and 20,000 suits integrated graphics.

## Development

```sh
npm run typecheck
npm run build
```

| Path | Role |
| --- | --- |
| `app/api/points/route.ts` | The single API route |
| `lib/points.ts` | Builds the payload: layout, domain scores, colours, cache |
| `lib/projection.ts` | Reads the file written by `glv project` |
| `lib/pca.ts` | The fallback projection |
| `lib/domains.ts`, `lib/colors.ts` | Domain descriptions and both palettes |
| `lib/spatial.ts` | Local density for the halo |
| `components/PointCloud.tsx` | Shaders and scene |
| `components/Overlay.tsx` | The control panel |
| `components/Viewer.tsx` | Fetching and state |
