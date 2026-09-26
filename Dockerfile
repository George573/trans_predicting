FROM python:3.12-slim AS export
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy==2.5.3
WORKDIR /app
COPY src src
COPY dataset/labels/labels_day_test.csv dataset/labels/
ARG CHECKPOINT=outputs/runs/final/boarding_only/final.pt
COPY ${CHECKPOINT} model.pt
RUN PYTHONPATH=src python -m tram_forecast export-head --checkpoint model.pt \
    --labels dataset/labels/labels_day_test.csv --output head.json \
 && chmod a+r head.json

FROM golang:1.24-alpine AS build
WORKDIR /src
COPY service .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /server ./cmd/server

FROM scratch
COPY --from=build /server /server
COPY --from=export /app/head.json /head.json
USER 65534:65534
EXPOSE 8080
ENTRYPOINT ["/server", "-model", "/head.json"]
