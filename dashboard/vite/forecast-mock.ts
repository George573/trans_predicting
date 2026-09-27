import { readFileSync } from "node:fs";
import type { IncomingMessage, ServerResponse } from "node:http";
import type { Plugin } from "vite";
import type { components } from "../src/shared/api/schema/generated.ts";

type ForecastRequest = components["schemas"]["ForecastRequest"];
type ForecastResponse = components["schemas"]["ForecastResponse"];
type Series = components["schemas"]["Series"];
type ApiError = components["schemas"]["Error"];

function fixture(name: string) {
  return JSON.parse(readFileSync(new URL(`../../api/examples/${name}.json`, import.meta.url), "utf8")) as ForecastResponse;
}

function tile(values: number[], points: number) {
  return Array.from({ length: points }, (_, index) => values[index % values.length]);
}

function resized(series: Series, points: number): Series {
  const value = tile(series.value, points);
  return {
    ...series,
    base: tile(series.base, points),
    value,
    usual: tile(series.usual, points),
    ...(series.lo && { lo: tile(series.lo, points) }),
    ...(series.hi && { hi: tile(series.hi, points) }),
    base_total: tile(series.base, points).reduce((sum, item) => sum + item, 0),
    total: value.reduce((sum, item) => sum + item, 0),
    peak_index: value.indexOf(Math.max(...value)),
  };
}

function zeroed(series: Series): Series {
  const flat = series.value.map(() => 0);
  return { ...series, base: flat, value: flat, ...(series.lo && { lo: flat }), ...(series.hi && { hi: flat }), base_total: 0, total: 0, peak_index: 0 };
}

function failure(code: ApiError["error"]["code"], message: string, field?: string): ApiError {
  return { error: { code, message, ...(field && { field }) } };
}

function answer(request: ForecastRequest, scenario: string): [number, ForecastResponse | ApiError] {
  if (scenario === "400") return [400, failure("bad_request", "Поле granularity обязательно", "granularity")];
  if (scenario === "401") return [401, failure("unauthorized", "Нужна авторизация: войдите под учётной записью диспетчера")];
  if (scenario === "413") return [413, failure("too_many_conditions", "Условий не больше 16, задано 17", "conditions")];
  if (scenario === "500") return [500, failure("internal", "Модель не ответила: бандл не загружен")];

  const days = Math.round((Date.parse(request.to) - Date.parse(request.from)) / 86400000) + 1;
  if (request.from > request.to) return [400, failure("bad_request", `Начало периода ${request.from} позже конца ${request.to}`, "from")];
  if (request.from < "2025-01-01") return [400, failure("out_of_domain", `Дата ${request.from} вне области определения модели, доступно с 2025-01-01`, "from")];
  if (request.to > "2026-04-30") return [400, failure("out_of_domain", `Дата ${request.to} вне области определения модели, доступно до 2026-04-30`, "to")];
  if (request.granularity === "hour" && days > 31) return [413, failure("period_too_long", `По часам доступен период не больше 31 суток, запрошено ${days}`, "granularity")];
  if (request.conditions.length > 16) return [413, failure("too_many_conditions", `Условий не больше 16, задано ${request.conditions.length}`, "conditions")];

  const example = fixture(request.granularity === "hour" ? "forecast-day" : "forecast-month");
  const wanted = request.routes?.length ? example.series.filter((series) => request.routes?.includes(series.route)) : example.series;
  const points = request.granularity === "hour" ? 24 * days : days;
  const series = wanted.map((item) => (scenario === "zeros" ? zeroed(resized(item, points)) : resized(item, points)));
  return [200, { ...example, start: `${request.from}T00:00`, series: scenario === "empty" ? [] : series, meta: { ...example.meta, rows: series.length * points } }];
}

function reply(response: ServerResponse, status: number, payload: unknown) {
  response.statusCode = status;
  response.setHeader("Content-Type", "application/json");
  response.setHeader("Server-Timing", "parse;dur=0.1, features;dur=0.6, model;dur=1.2, corrections;dur=0.2, json;dur=0.4");
  response.end(JSON.stringify(payload));
}

function received(request: IncomingMessage) {
  return new Promise<string>((resolve) => {
    let text = "";
    request.on("data", (chunk: Buffer) => { text += chunk.toString(); });
    request.on("end", () => resolve(text));
  });
}

export function forecastMock(): Plugin {
  return {
    name: "forecast-mock",
    apply: "serve",
    configureServer(server) {
      const geometry = JSON.parse(readFileSync(new URL("../../factors/data/route_lines.geojson", import.meta.url), "utf8")) as { features: { properties: { route: number; names: string[] } }[] };
      const routes = geometry.features.map(({ properties }) => ({
        route: properties.route,
        name: properties.names[0].replace(/^Трамвай \d+: /, "").replace(" => ", " - "),
        has_geometry: true,
        has_history: properties.route !== 5,
        geometry_source: "osm",
      })) as components["schemas"]["Route"][];


      const catalog: components["schemas"]["CatalogEntry"][] = [
        { type: "rain", class: "operational", title: "Дождь", range: { min: 0, max: 10, step: 0.2 }, auto: false,
          passport: { unit: "мм/ч", step: 0.2, source: "Open-Meteo ERA5", source_url: "https://open-meteo.com", effect_pct: -5.4, ci_pct: [-10, -3.1], sample: 1953, horizons: ["day", "week"], class: "operational" } },
        { type: "temperature", class: "operational", title: "Отклонение температуры", range: { min: -20, max: 20, step: 0.5 }, auto: false,
          passport: { unit: "градусы от нормы", step: 0.5, source: "Open-Meteo ERA5", effect_pct: 0.6, ci_pct: [-0.2, 1.7], sample: 1953, horizons: ["day", "week"], class: "operational" } },
        { type: "event", class: "operational", title: "Событие рядом с линией", range: { min: 0, max: 5, step: 1 }, auto: false,
          passport: { unit: "события в час", step: 1, source: "KudaGo", effect_pct: 2.1, ci_pct: [-1.4, 5.6], sample: 412, horizons: ["day", "week"], class: "operational" } },
        { type: "delay", class: "operational", title: "Сообщение о задержке", range: { min: 0, max: 3, step: 1 }, auto: false,
          passport: { unit: "посты в час", step: 1, source: "Сообщения перевозчика", effect_pct: -29.6, ci_pct: [-35.1, -24.2], sample: 268, horizons: ["day", "week"], class: "operational" } },
        { type: "closure", class: "structural", title: "Перекрытие участка", range: { min: 0, max: 1, step: 1 }, auto: false,
          passport: { unit: "доля перекрытия", step: 1, source: "Замер по истории отмен", effect_pct: -97, ci_pct: [-99, -92], sample: 74, horizons: ["day", "week", "month", "season"], class: "structural" } },
        { type: "route_change", class: "structural", title: "Изменение маршрута", range: { min: 0, max: 1, step: 1 }, auto: false,
          passport: { unit: "доля изменения", step: 1, source: "Замер по истории изменений", effect_pct: -15, ci_pct: [-17, -13], sample: 91, horizons: ["day", "week", "month", "season"], class: "structural" } },
      ];
      const started = Date.now();
      let served = 0;
      let rows = 0;

      server.middlewares.use((request, response, next) => {
        const path = (request.url ?? "").split("?")[0];
        if (path === "/healthz") return reply(response, 200, { status: "ok", bundle: fixture("forecast-day").bundle });
        if (path === "/api/v1/routes") return reply(response, 200, { routes });
        if (path === "/api/v1/model") return reply(response, 200, { default: "catboost", models: [{ name: "catboost", bundle: fixture("forecast-day").bundle, model: "catboost_cyclic_service", describe: "CatBoost, мок", files: [], mode: "service", features: [], train_period: ["2025-01-01", "2025-10-31"], horizon: ["2025-01-01", "2026-04-30"] }] });
        if (path === "/geo/routes.geojson") return reply(response, 200, JSON.parse(readFileSync(new URL("../../web/geo/routes.geojson", import.meta.url), "utf8")));
        if (path === "/api/v1/conditions") return reply(response, 200, { conditions: catalog });
        if (path === "/api/v1/stats") return reply(response, 200, { requests: served, rows, rps_1m: served / Math.max(1, (Date.now() - started) / 60000), p50_ms: 1.1, p95_ms: 2.4, cpu_pct: 3.2, rss_mb: 48.5, uptime_s: (Date.now() - started) / 1000 });
        if (path === "/api/v1/explain") {
          void received(request).then((text) => {
            const asked = JSON.parse(text || "{}") as { route?: number; date?: string; hour?: number };
            const raw = 612;
            reply(response, 200, {
              route: asked.route ?? 7, date: asked.date ?? "2025-11-11", hour: asked.hour ?? 8,
              calendar: { weekday: 2, holiday: false, school_vacation: false },
              features: [ { name: "hour_sin", value: 0.866, kind: "float" }, { name: "hour_cos", value: -0.5, kind: "float" }, { name: "weekday", value: 2, kind: "categorical", hash: 2 }, { name: "route", value: asked.route ?? 7, kind: "categorical", hash: 7 } ],
              steps: [ { step: "сырой выход модели", factor: null, value: raw }, { step: "обрезка нуля", factor: null, value: raw }, { step: "калибровка по дню недели", factor: 1.02, value: Math.round(raw * 1.02) }, { step: "условие c1 - дождь", detail: "1.2 мм/ч", factor: 0.936, value: Math.round(raw * 1.02 * 0.936) } ],
              value: Math.round(raw * 1.02 * 0.936),
            });
          });
          return;
        }
        if (path === "/api/v1/forecast/export") {
          void received(request).then((text) => {
            const asked = JSON.parse(text || "{}") as ForecastRequest;
            const [, payload] = answer(asked, "");
            const body = payload as ForecastResponse;
            const lines = ["маршрут;время;посадки;обычный уровень;низ;верх"];
            body.series.forEach((series) => series.value.forEach((value, index) => {
              lines.push([series.route, new Date(Date.parse(`${body.start}Z`) + index * (body.step === "1h" ? 3600000 : 86400000)).toISOString().slice(0, 16), Math.round(value), Math.round(series.usual[index]), Math.round(series.lo?.[index] ?? 0), Math.round(series.hi?.[index] ?? 0)].join(";"));
            }));
            response.statusCode = 200;
            response.setHeader("Content-Type", "text/csv; charset=utf-8");
            response.setHeader("Content-Disposition", 'attachment; filename="forecast.csv"');
            response.end("\ufeff" + lines.join("\n"));
          });
          return;
        }
        if (path !== "/api/v1/forecast") return next();
        void received(request).then((text) => {
          served += 1;
          const [status, payload] = answer(JSON.parse(text || "{}") as ForecastRequest, String(request.headers["x-mock"] ?? ""));
          reply(response, status, payload);
        });
      });
    },
  };
}
