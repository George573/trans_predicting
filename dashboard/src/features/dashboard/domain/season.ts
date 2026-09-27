import dayjs from "dayjs";
import type { components } from "@/shared/api/schema/generated";

type ForecastRequest = components["schemas"]["ForecastRequest"];
type ForecastResponse = components["schemas"]["ForecastResponse"];
type Timings = components["schemas"]["Timings"];

export type SeasonRequest = Omit<ForecastRequest, "from" | "to" | "granularity" | "horizon">;

export type SeasonPart = {
  label: string;
  days: number;
  routeHours: number;
  request: ForecastRequest;
};

const seasonMonthPairs = [
  { label: "ноябрь - декабрь", from: "2025-11-01", to: "2025-12-31" },
  { label: "январь - февраль", from: "2026-01-01", to: "2026-02-28" },
  { label: "март - апрель", from: "2026-03-01", to: "2026-04-30" },
];

export const seasonRange: [string, string] = [seasonMonthPairs[0].from, seasonMonthPairs[seasonMonthPairs.length - 1].to];

function inclusiveDays(from: string, to: string) {
  return (Date.parse(to) - Date.parse(from)) / 86400000 + 1;
}

function requestedRouteCount(routes: ForecastRequest["routes"]) {
  return routes?.length || 10;
}

export function splitSeasonRequest(request: SeasonRequest): SeasonPart[] {
  const routeCount = requestedRouteCount(request.routes);
  return seasonMonthPairs.map(({ label, from, to }) => {
    const days = inclusiveDays(from, to);
    return {
      label,
      days,
      routeHours: days * 24 * routeCount,
      request: { ...request, from, to, granularity: "day", horizon: "season" },
    };
  });
}

export type SeasonAnswer = { label: string; response: ForecastResponse };

export const seasonMonths = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04"];

function sum(values: number[]) {
  return values.reduce((total, value) => total + value, 0);
}

function mergedTimings(answers: SeasonAnswer[]) {
  return (["parse", "features", "model", "corrections", "json"] as const).reduce((stages: Timings, stage) => {
    const measured = answers.map(({ response }) => response.meta.timings[stage]).filter((value): value is number => value !== undefined);
    return measured.length > 0 ? { ...stages, [stage]: sum(measured) } : stages;
  }, {});
}

export function composeSeason(answers: SeasonAnswer[]): ForecastResponse {
  const routes = [...new Set(answers.flatMap(({ response }) => response.series.map((series) => series.route)))].sort((first, second) => first - second);
  return {
    start: `${seasonMonths[0]}-01T00:00`,
    step: "1mo",
    bundle: answers[0].response.bundle,
    series: routes.map((route) => {
      const base = seasonMonths.map(() => 0);
      const value = seasonMonths.map(() => 0);
      const usual = seasonMonths.map(() => 0);
      answers.forEach(({ response }) => {
        const series = response.series.find((item) => item.route === route);
        series?.value.forEach((_, index) => {
          const slot = seasonMonths.indexOf(dayjs(response.start).add(index, "day").format("YYYY-MM"));
          if (slot < 0) return;
          base[slot] += series.base[index];
          value[slot] += series.value[index];
          usual[slot] += series.usual[index];
        });
      });
      return { route, base, value, usual, base_total: sum(base), total: sum(value), peak_index: value.indexOf(Math.max(...value)) };
    }),
    conditions: [],
    warnings: answers.flatMap(({ response }) => response.warnings).filter((warning, index, all) =>
      all.findIndex((item) => item.code === warning.code && (item.condition_id ?? "") === (warning.condition_id ?? "") && item.message === warning.message) === index),
    meta: {
      rows: sum(answers.map(({ response }) => response.meta.rows)),
      model: answers[0].response.meta.model,
      timings: mergedTimings(answers),
    },
  };
}
