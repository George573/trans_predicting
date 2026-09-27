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
USER 65534:65534
EXPOSE 8080
ENTRYPOINT ["/app/server", "-bundle", "/app/bundle", "-calendar", "/app/calendar", "-geo", "/app/geo/routes.geojson"]
