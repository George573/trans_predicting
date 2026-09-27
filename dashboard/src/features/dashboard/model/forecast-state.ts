import { useEffect, useReducer, useRef, useState } from "react";
import { fetchClient } from "@/shared/api/instance";
import type { components } from "@/shared/api/schema/generated";
import { normalizeRequest, requestViolation } from "../forecast-request";
import { composeSeason, splitSeasonRequest, type SeasonAnswer } from "../domain/season";

export type ForecastRequest = components["schemas"]["ForecastRequest"];
export type ForecastResponse = components["schemas"]["ForecastResponse"];
export type Route = components["schemas"]["Route"];
export type Timings = components["schemas"]["Timings"];
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
  | { type: "conditions"; conditions: ForecastRequest["conditions"] } | { type: "period"; from: string; to: string; horizon: ForecastRequest["horizon"]; granularity: ForecastRequest["granularity"] };
export const initialRequest: ForecastRequest = { routes: [], from: "2025-11-10", to: "2025-11-10", horizon: "day", granularity: "hour", corridor: true, conditions: [] };

function reducer(state: ForecastRequest, action: Action): ForecastRequest {
  if (action.type === "replace") return structuredClone(action.request);
  if (action.type === "conditions") return { ...state, conditions: action.conditions };
  if (action.type === "period") return { ...state, from: action.from, to: action.to, horizon: action.horizon, granularity: action.granularity };
  return { ...state, routes: action.routes as ForecastRequest["routes"] };
}

const structuralPart = ({ routes, from, to, horizon, granularity, corridor }: ForecastRequest) => JSON.stringify([routes, from, to, horizon, granularity, corridor]);

export function useForecast(horizon?: string[]) {
  const [request, dispatch] = useReducer(reducer, initialRequest);
  const [settled, setSettled] = useState(initialRequest);
  const [data, setData] = useState<ForecastResponse | null>(null);
  const [acceptedRequest, setAcceptedRequest] = useState<ForecastRequest | null>(null);
  const [seasonAnswers, setSeasonAnswers] = useState<SeasonAnswer[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [failure, setFailure] = useState<ForecastFailure | null>(null);
  const [telemetry, setTelemetry] = useState<RequestTelemetry>({ status: "загрузка", browserMs: null, rows: null, stages: {} });
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const nextId = useRef(0);
  const modelHorizon = useRef(horizon);
  modelHorizon.current = horizon;
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
    const violation = requestViolation(settled, modelHorizon.current);
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
      const parts = body.horizon === "season" ? splitSeasonRequest(body) : null;
      const sent = parts ? parts.map((part) => part.request) : [body];
      setHistory((previous) => [...sent.map((item, index) => ({ id: id * 10 + index, at: Date.now(), request: item, status: "загрузка", durationMs: null })).reverse(), ...previous.map((entry) => entry.status === "загрузка" ? { ...entry, status: "отменён" } : entry)].slice(0, 12));
      try {
        const results = await Promise.all(sent.map((item) => fetchClient.POST("/api/v1/forecast", { body: item, signal: controller.signal })));
        if (controller.signal.aborted) return;
        const elapsed = performance.now() - started;
        setHistory((previous) => previous.map((entry) => {
          const index = entry.id - id * 10;
          return index >= 0 && index < results.length && entry.id >= id * 10 ? { ...entry, status: String(results[index].response.status), durationMs: elapsed } : entry;
        }));
        const responses = results.flatMap((result) => result.data ? [result.data] : []);
        setTelemetry({
          status: results.find((result) => result.response.status !== 200)?.response.status ?? 200,
          browserMs: elapsed,
          rows: responses.reduce((total, response) => total + response.meta.rows, 0) || null,
          stages: serverTimings(results[0].response.headers.get("Server-Timing"), responses[0]?.meta.timings),
        });
        const rejected = results.find((result) => result.error)?.error;
        if (rejected) setFailure({ message: rejected.error.message, field: rejected.error.field, source: "сервис" });
        else if (responses.length !== sent.length) setFailure({ message: "Сервис ответил без данных прогноза", source: "сервис" });
        else if (new Set(responses.map((response) => response.bundle)).size > 1) setFailure({ message: `Части сезона собраны разными версиями модели: ${[...new Set(responses.map((response) => response.bundle))].join(", ")}. Повторите запрос целиком.`, source: "сервис" });
        else {
          const answers = parts ? parts.map((part, index) => ({ label: part.label, response: responses[index] })) : null;
          setSeasonAnswers(answers);
          setData(answers ? composeSeason(answers) : responses[0]);
          setAcceptedRequest(body);
          setUpdatedAt(Date.now());
        }
      } catch {
        if (!controller.signal.aborted) {
          setFailure({ message: "Нет связи с сервисом: запрос не дошёл", source: "сеть" });
          setHistory((previous) => previous.map((entry) => entry.id >= id * 10 && entry.status === "загрузка" ? { ...entry, status: "сеть", durationMs: performance.now() - started } : entry));
          setTelemetry((previous) => previous.status === "загрузка" ? { status: "сеть", browserMs: performance.now() - started, rows: null, stages: {} } : previous);
        }
      } finally { if (!controller.signal.aborted) setLoading(false); }
    });
    return () => controller.abort();
  }, [settled, revision]);

  return { request, dispatch, data, acceptedRequest, seasonAnswers, loading, failure, retry, telemetry, history, updatedAt, settling: request !== settled };
}
