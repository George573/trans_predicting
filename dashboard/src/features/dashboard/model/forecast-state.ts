import { useEffect, useReducer, useRef, useState } from "react";
import { fetchClient } from "@/shared/api/instance";
import type { components } from "@/shared/api/schema/generated";
import { normalizeRequest, requestViolation } from "../forecast-request";
import { composeSeason, type ForecastResponse } from "../domain/season";
import type { ModelName } from "../domain/model";

export type { ForecastResponse };
export type ForecastRequest = components["schemas"]["ForecastRequest"];
export type Route = components["schemas"]["Route"];
export type Timings = components["schemas"]["Timings"] & { json?: number };
export type RequestTelemetry = { status: number | "загрузка" | "сеть" | "не отправлен"; browserMs: number | null; rows: number | null; stages: Timings };
export type ForecastFailure = { message: string; field?: string; source: "запрос" | "сервис" | "сеть" };
export type HistoryEntry = { id: number; at: number; request: ForecastRequest; status: string; durationMs: number | null };

function serverTimings(header: string | null, fallback: Timings = {}): Timings {
  if (!header) return fallback;
  const values: Timings = { ...fallback };
  for (const part of header.split(",")) {
    const match = part.trim().match(/^(parse|features|model|corrections|json)\s*;[^,]*?dur\s*=\s*([\d.]+)/i);
    if (match) values[match[1].toLowerCase() as keyof Timings] = Number(match[2]);
  }
  return values;
}

type Action = { type: "replace"; request: ForecastRequest } | { type: "routes"; routes: number[] }
  | { type: "conditions"; conditions: ForecastRequest["conditions"] } | { type: "period"; from: string; to: string; horizon: ForecastRequest["horizon"]; granularity: ForecastRequest["granularity"] }
  | { type: "model"; model: ModelName; from: string; to: string };
export const initialRequest: ForecastRequest = { routes: [], from: "2025-11-10", to: "2025-11-10", horizon: "day", granularity: "hour", corridor: true, conditions: [] };

function reducer(state: ForecastRequest, action: Action): ForecastRequest {
  if (action.type === "replace") return structuredClone(action.request);
  if (action.type === "conditions") return { ...state, conditions: action.conditions };
  if (action.type === "period") return { ...state, from: action.from, to: action.to, horizon: action.horizon, granularity: action.granularity };
  if (action.type === "model") return { ...state, model: action.model, from: action.from, to: action.to };
  return { ...state, routes: action.routes as ForecastRequest["routes"] };
}

const structuralPart = ({ model, routes, from, to, horizon, granularity, corridor }: ForecastRequest) => JSON.stringify([model, routes, from, to, horizon, granularity, corridor]);

export function useForecast(horizonOf: (model?: ModelName) => string[] | undefined) {
  const [request, dispatch] = useReducer(reducer, initialRequest);
  const [settled, setSettled] = useState(initialRequest);
  const [data, setData] = useState<ForecastResponse | null>(null);
  const [acceptedRequest, setAcceptedRequest] = useState<ForecastRequest | null>(null);
  const [loading, setLoading] = useState(true);
  const [failure, setFailure] = useState<ForecastFailure | null>(null);
  const [telemetry, setTelemetry] = useState<RequestTelemetry>({ status: "загрузка", browserMs: null, rows: null, stages: {} });
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const nextId = useRef(0);
  const modelHorizon = useRef(horizonOf);
  modelHorizon.current = horizonOf;
  const [revision, retry] = useReducer((value: number) => value + 1, 0);
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);

  useEffect(() => {
    if (request === settled) return;
    if (structuralPart(request) !== structuralPart(settled)) {
      setSettled(request);
      return;
    }
    const timer = window.setTimeout(() => setSettled(request), 175);
    return () => window.clearTimeout(timer);
  }, [request, settled]);

  useEffect(() => {
    const violation = requestViolation(settled, modelHorizon.current(settled.model));
    if (violation) {
      setLoading(false);
      setFailure({ message: violation.message, field: violation.field, source: "запрос" });
      setTelemetry({ status: "не отправлен", browserMs: null, rows: null, stages: {} });
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    setFailure(null);
    setTelemetry({ status: "загрузка", browserMs: null, rows: null, stages: {} });
    queueMicrotask(async () => {
      if (controller.signal.aborted) return;
      const started = performance.now();
      const id = ++nextId.current;
      const body = normalizeRequest(settled);
      setHistory((previous) => [{ id, at: Date.now(), request: body, status: "загрузка", durationMs: null }, ...previous.map((entry) => entry.status === "загрузка" ? { ...entry, status: "отменён" } : entry)].slice(0, 12));
      try {
        const result = await fetchClient.POST("/api/v1/forecast", { body, signal: controller.signal });
        if (controller.signal.aborted) return;
        const elapsed = performance.now() - started;
        setHistory((previous) => previous.map((entry) => entry.id === id ? { ...entry, status: String(result.response.status), durationMs: elapsed } : entry));
        setTelemetry({
          status: result.response.status,
          browserMs: elapsed,
          rows: result.data?.meta.rows ?? null,
          stages: serverTimings(result.response.headers.get("Server-Timing"), result.data?.meta.timings),
        });
        if (result.error) setFailure({ message: result.error.error.message, field: result.error.error.field, source: "сервис" });
        else if (!result.data) setFailure({ message: "Сервис ответил без данных прогноза", source: "сервис" });
        else {
          setData(body.horizon === "season" ? composeSeason(result.data) : result.data);
          setAcceptedRequest(body);
          setUpdatedAt(Date.now());
        }
      } catch {
        if (!controller.signal.aborted) {
          setFailure({ message: "Нет связи с сервисом: запрос не дошёл", source: "сеть" });
          setHistory((previous) => previous.map((entry) => entry.id === id && entry.status === "загрузка" ? { ...entry, status: "сеть", durationMs: performance.now() - started } : entry));
          setTelemetry((previous) => previous.status === "загрузка" ? { status: "сеть", browserMs: performance.now() - started, rows: null, stages: {} } : previous);
        }
      } finally { if (!controller.signal.aborted) setLoading(false); }
    });
    return () => controller.abort();
  }, [settled, revision]);

  return { request, dispatch, data, acceptedRequest, loading, failure, retry, telemetry, history, updatedAt, settling: request !== settled };
}

export function useComparison(request: ForecastRequest | null, model: ModelName | null) {
  const [answer, setAnswer] = useState<{ data: ForecastResponse | null; error: string; model: ModelName | null }>({ data: null, error: "", model: null });
  useEffect(() => {
    const controller = new AbortController();
    queueMicrotask(async () => {
      if (controller.signal.aborted) return;
      if (!request || !model) { setAnswer({ data: null, error: "", model: null }); return; }
      try {
        const result = await fetchClient.POST("/api/v1/forecast", { body: { ...request, model }, signal: controller.signal });
        if (controller.signal.aborted) return;
        if (result.data) setAnswer({ data: request.horizon === "season" ? composeSeason(result.data) : result.data, error: "", model });
        else setAnswer({ data: null, error: result.error?.error.message ?? "Сервис ответил без данных прогноза", model });
      } catch {
        if (!controller.signal.aborted) setAnswer({ data: null, error: "Нет связи с сервисом: сравнение не получено", model });
      }
    });
    return () => controller.abort();
  }, [request, model]);
  return answer;
}
