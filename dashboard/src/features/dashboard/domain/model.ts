import dayjs from "dayjs";
import type { components } from "@/shared/api/schema/generated";
import type { ForecastResponse } from "./season";

export type ModelName = components["schemas"]["ModelName"];
export type ModelInfo = components["schemas"]["ModelInfo"];

export const modelTitles: Record<ModelName, string> = { catboost: "Резервная (ML)", cnn: "Основная (CNN)" };

export function recursiveFrom(info?: ModelInfo): string | null {
  return typeof info?.recursive_from === "string" ? info.recursive_from : null;
}

export function recursiveIndex(data: ForecastResponse, from: string | null): number | null {
  if (!from) return null;
  const points = data.series[0]?.value.length ?? 0;
  const unit = data.step === "1h" ? "hour" : data.step === "1d" ? "day" : "month";
  const index = dayjs(from).diff(dayjs(data.start).startOf(unit === "month" ? "month" : "day"), unit);
  return index > 0 && index < points ? index : null;
}
