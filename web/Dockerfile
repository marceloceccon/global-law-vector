# Production image for the point-cloud viewer. Three stages so the final image carries only
# Next's standalone server (no node_modules, no sources): about 200 MB.
#
#   docker build -t utility-web web/
#   docker run --rm -p 3000:3000 --network utility \
#     -e DATABASE_URL=postgresql://utility:utility@utility-pgvector:5432/utility \
#     -e TEI_URL=http://utility-embeddings:80 utility-web
#
# or, with the rest of the stack, `docker compose up -d --build web` in global-law-utility.

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
