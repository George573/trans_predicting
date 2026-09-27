import dayjs from "dayjs";
import type { components } from "@/shared/api/schema/generated";

type ApiResponse = components["schemas"]["ForecastResponse"];

export type ForecastResponse = Omit<ApiResponse, "step"> & { step: ApiResponse["step"] | "1mo" };

export const seasonRange: [string, string] = ["2025-11-01", "2026-04-30"];

export const seasonMonths = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04"];

export const seasonDays = dayjs(seasonRange[1]).diff(seasonRange[0], "day") + 1;

export function composeSeason(response: ApiResponse): ForecastResponse {
  const slots = response.series[0]?.value.map((_, index) => seasonMonths.indexOf(dayjs(response.start).add(index, "day").format("YYYY-MM"))) ?? [];
  const monthly = (values: number[]) => values.reduce((sums, value, index) => {
    if (slots[index] >= 0) sums[slots[index]] += value;
    return sums;
  }, seasonMonths.map(() => 0));
  return {
    ...response,
    start: `${seasonMonths[0]}-01T00:00`,
    step: "1mo",
    series: response.series.map(({ route, base, value, usual, base_total, total }) => {
      const sums = monthly(value);
      return { route, base: monthly(base), value: sums, usual: monthly(usual), base_total, total, peak_index: sums.indexOf(Math.max(...sums)) };
    }),
  };
}
