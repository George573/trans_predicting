import json
from datetime import date, datetime, time, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from time import perf_counter
from urllib.parse import parse_qs, urlparse

import numpy as np
import onnxruntime

from .config import ROUTES
from .schema import calendar, label_grid, request_calendar

FALLBACK_ROUTES = (5,)
REQUEST_CALENDAR = (
    "weekday_sin",
    "weekday_cos",
    "day_of_month_sin",
    "day_of_month_cos",
)


class Runner:
    def __init__(self, model, labels):
        model = Path(model)
        self.meta = json.loads((model / "model.json").read_text())
        self.cutoff = date.fromisoformat(self.meta["cutoff"])
        self.first = self.cutoff - timedelta(days=21)
        self.counts, present = label_grid(labels, ROUTES, self.first, self.cutoff)
        if not present.any(axis=1).all():
            raise ValueError(
                f"labels must cover {self.first}..{self.cutoff} for every route"
            )
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 1
        options.inter_op_num_threads = 1
        encoder = onnxruntime.InferenceSession(
            str(model / "encoder.onnx"), options, providers=["CPUExecutionProvider"]
        )
        start = datetime.combine(self.first, time.min)
        hours = calendar(start + timedelta(hours=i) for i in range(504))
        self.encoded = encoder.run(
            None,
            {
                "counts": self.counts[:, None, :],
                "calendar": np.repeat(hours[None], len(ROUTES), axis=0),
            },
        )[0]
        self.head = onnxruntime.InferenceSession(
            str(model / "head.onnx"), options, providers=["CPUExecutionProvider"]
        )

    def info(self):
        return {
            **self.meta,
            "engine": f"onnxruntime {onnxruntime.__version__}",
            "history": {
                "from": str(self.first),
                "to": str(self.cutoff - timedelta(days=1)),
                "hours": 504,
            },
            "fallback_routes": list(FALLBACK_ROUTES),
        }

    def predict(self, routes, start, days):
        last = self.cutoff + timedelta(days=self.meta["forecast_days"] - 1)
        if days < 1 or start < self.cutoff or start + timedelta(days=days - 1) > last:
            raise ValueError(f"forecast covers {self.cutoff}..{last}")
        unknown = set(routes) - set(ROUTES) - set(FALLBACK_ROUTES)
        if unknown:
            raise ValueError(f"unknown routes {sorted(unknown)}")
        neural = np.isin(routes, ROUTES)
        values = np.zeros((len(routes), days, 24), dtype=np.float32)
        if neural.any():
            index = np.repeat(
                np.array([ROUTES.index(r) for r in routes if r in ROUTES]), days
            )
            dates = [start + timedelta(days=d) for d in range(days)]
            lead = (start - self.cutoff).days + 1 + np.arange(days, dtype=np.float32)
            values[neural] = self.head.run(
                None,
                {
                    "encoded": self.encoded[index],
                    "route_indices": index.astype(np.int64) + 1,
                    "calendar": np.tile(request_calendar(dates), (neural.sum(), 1)),
                    "lead": np.tile(lead, neural.sum()),
                },
            )[0].reshape(-1, days, 24)
        return values

    def explain(self, route, day):
        values = self.predict([route], day, 1)[0, 0].astype(np.float64).round(3)
        if route not in ROUTES:
            return {"fallback": True, "values": values.tolist()}
        i = ROUTES.index(route)
        history = self.counts[i].reshape(21, 24)
        same = [
            (self.first + timedelta(days=d)).weekday() == day.weekday()
            for d in range(21)
        ]
        return {
            "fallback": False,
            "input": {
                "route_index": i + 1,
                "lead": (day - self.cutoff).days + 1,
                **dict(zip(REQUEST_CALENDAR, request_calendar([day])[0].tolist())),
            },
            "history": {
                "boardings": int(history.sum()),
                "same_weekday": history[same]
                .mean(axis=0, dtype=np.float64)
                .round(1)
                .tolist(),
            },
            "values": values.tolist(),
        }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    disable_nagle_algorithm = True
    runner = None

    def do_GET(self):
        url = urlparse(self.path)
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        started = perf_counter()
        try:
            if url.path == "/healthz":
                body = {"status": "ok"}
            elif url.path == "/model":
                body = self.runner.info()
            elif url.path == "/predict":
                routes = [int(r) for r in query["route"].split(",")]
                days = int(query.get("days", 1))
                values = self.runner.predict(
                    routes, date.fromisoformat(query["from"]), days
                )
                body = {
                    "values": values.reshape(len(routes), -1)
                    .astype(np.float64)
                    .round(3)
                    .tolist(),
                    "rows": int(np.isin(routes, ROUTES).sum()) * days,
                    "model_ms": (perf_counter() - started) * 1000,
                }
            elif url.path == "/explain":
                body = self.runner.explain(
                    int(query["route"]), date.fromisoformat(query["date"])
                )
                body["model_ms"] = (perf_counter() - started) * 1000
            else:
                return self.reply(404, {"error": f"unknown path {url.path}"})
        except KeyError as exc:
            return self.reply(400, {"error": f"missing parameter {exc}"})
        except ValueError as exc:
            return self.reply(400, {"error": str(exc)})
        self.reply(200, body)

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass


class Server(ThreadingHTTPServer):
    request_queue_size = 128


def serve(model, labels, host, port):
    Handler.runner = Runner(model, labels)
    server = Server((host, port), Handler)
    print(
        json.dumps({"listening": f"{host}:{port}", **Handler.runner.info()}), flush=True
    )
    server.serve_forever()
