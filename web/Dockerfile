# Production image for the viewer: Next.js's standalone server and its static assets only,
# no node_modules and no sources.
#
#   docker build -t glv-web web/
#   docker run --rm -p 3000:3000 \
#     -e DATABASE_URL=postgresql://user:password@db-host:5432/dbname \
#     -e TEI_URL=http://embeddings-host:80 \
#     -v ~/.cache/global-law-vector/projection.json:/data/projection.json:ro \
#     -e GLV_PROJECTION=/data/projection.json \
#     glv-web
#
# The container must reach the database (and, for TEI-embedded domain phrases, the
# embeddings server) by those names; on a Docker network, add --network <name>. Without the
# projection mount the viewer falls back to PCA.

FROM node:24-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

FROM node:24-alpine AS build
WORKDIR /app
ENV NEXT_TELEMETRY_DISABLED=1
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

FROM node:24-alpine AS run
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    HOSTNAME=0.0.0.0 \
    PORT=3000
COPY --from=build /app/.next/standalone ./
COPY --from=build /app/.next/static ./.next/static
USER node
EXPOSE 3000
CMD ["node", "server.js"]
