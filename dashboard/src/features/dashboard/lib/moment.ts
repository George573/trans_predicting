import dayjs from "dayjs";
import type { components } from "@/shared/api/schema/generated";

type ForecastResponse = components["schemas"]["ForecastResponse"];

export function momentAt(data: ForecastResponse, index: number) {
  return dayjs(data.start).add(index, data.step === "1h" ? "hour" : data.step === "1d" ? "day" : "month");
}

export function momentLabel(data: ForecastResponse, index: number) {
  return momentAt(data, index).format(data.step === "1h" ? "DD.MM HH:mm" : data.step === "1d" ? "DD.MM.YYYY" : "MM.YYYY");
}

export function quantityLabel(step: ForecastResponse["step"]) {
  return step === "1h" ? "посадки в час" : step === "1d" ? "посадки в сутки" : "посадки за месяц";
}
