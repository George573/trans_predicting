import type { components } from "@/shared/api/schema/generated";

export type ForecastRequest = components["schemas"]["ForecastRequest"];
export type RequestViolation = { message: string; field: keyof ForecastRequest };

export const forecastDomain: [string, string] = ["2025-01-01", "2026-04-30"];

export function normalizeRequest(request: ForecastRequest): ForecastRequest {
  return {
    routes: [...new Set(request.routes ?? [])].sort((first, second) => first - second),
    from: request.from,
    to: request.to,
    granularity: request.granularity,
    horizon: request.horizon,
    corridor: request.corridor,
    conditions: [...request.conditions]
      .sort((first, second) => first.id.localeCompare(second.id))
      .map((condition) => ({
        id: condition.id,
        type: condition.type,
        value: condition.value,
        ...(condition.scope && {
          scope: {
            routes: condition.scope.routes ?? null,
            dates: condition.scope.dates ?? null,
            weekdays: condition.scope.weekdays ?? null,
            hours: condition.scope.hours ?? null,
          },
        }),
      })),
  };
}

export function sameRequest(first: ForecastRequest, second: ForecastRequest) {
  return JSON.stringify(normalizeRequest(first)) === JSON.stringify(normalizeRequest(second));
}

export function periodDays(from: string, to: string) {
  return Math.round((Date.parse(to) - Date.parse(from)) / 86400000) + 1;
}

export function requestViolation(request: ForecastRequest, horizon?: string[]): RequestViolation | null {
  const [earliest, latest] = horizon?.length === 2 ? horizon : forecastDomain;
  if (request.from > request.to) return { message: `Начало периода ${request.from} позже конца ${request.to}`, field: "from" };
  if (request.from < earliest) return { message: `Дата ${request.from} вне области определения модели, доступно с ${earliest}`, field: "from" };
  if (request.to > latest) return { message: `Дата ${request.to} вне области определения модели, доступно до ${latest}`, field: "to" };
  const days = periodDays(request.from, request.to);
  if (request.granularity === "hour" && days > 31) return { message: `По часам доступен период не больше 31 суток, запрошено ${days}`, field: "granularity" };
  if (request.conditions.length > 16) return { message: `Условий не больше 16, задано ${request.conditions.length}`, field: "conditions" };
  return null;
}
