FROM python:3.12-slim AS export
RUN pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir numpy==2.5.3 onnx==1.23.0
WORKDIR /app
COPY src src
ARG CHECKPOINT=outputs/boarding_only_61days_v5_final/final.pt
COPY ${CHECKPOINT} model.pt
RUN PYTHONPATH=src python -m tram_forecast export-onnx --checkpoint model.pt --output onnx \
 && chmod -R a+rX onnx

FROM python:3.12-slim AS runner
RUN pip install --no-cache-dir numpy==2.5.3 onnxruntime==1.30.0
WORKDIR /app
COPY src src
COPY dataset/labels/labels_day_test.csv dataset/labels/
COPY --from=export /app/onnx onnx
ENV PYTHONPATH=src
USER nobody
STOPSIGNAL SIGINT
EXPOSE 8000
CMD ["python", "-m", "tram_forecast", "serve", "--model", "onnx", "--labels", "dataset/labels/labels_day_test.csv"]

FROM golang:1.24-alpine AS server-build
WORKDIR /src
COPY bench/go-inference-stand .
RUN CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /server ./cmd/server

FROM scratch AS server
COPY --from=server-build /server /server
USER 65534:65534
EXPOSE 8080
ENTRYPOINT ["/server"]
