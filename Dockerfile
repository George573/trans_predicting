FROM ghcr.io/pnpm/pnpm:11.27.1@sha256:cc8a0551f7df20e46d6b089baf8f5f444fbd36de84ca72fd499171e474d8ec47 AS web
RUN pnpm_config_store_dir=/pnpm/runtime-store pnpm runtime set node 24.21.0 -g
WORKDIR /web
COPY dashboard/package.json dashboard/pnpm-lock.yaml ./
RUN --mount=type=cache,id=pnpm,target=/pnpm/store pnpm i --frozen-lockfile --trust-lockfile
COPY dashboard .
RUN pnpm build

FROM golang:1.24-bookworm AS build
WORKDIR /src
COPY service .
RUN bash scripts/fetch_libs.sh \
 && CGO_ENABLED=1 go build -trimpath -ldflags="-s -w" -o /server ./cmd/server

FROM debian:bookworm-slim
COPY --from=build /src/lib/libcatboostmodel.so /usr/local/lib/
RUN ldconfig
COPY --from=build /server /app/server
COPY artifacts/bundle /app/bundle
COPY input/calendar/2024.xml input/calendar/2025.xml input/calendar/2026.xml /app/calendar/
COPY web/geo/routes.geojson /app/geo/routes.geojson
COPY --from=web /web/dist /app/web
USER 65534:65534
EXPOSE 8080
ENTRYPOINT ["/app/server", "-bundle", "/app/bundle", "-calendar", "/app/calendar", "-geo", "/app/geo/routes.geojson", "-static", "/app/web"]
