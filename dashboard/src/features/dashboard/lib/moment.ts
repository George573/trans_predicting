import dayjs from "dayjs";

type Timeline = { start: string; step: "1h" | "1d" | "1mo" };

export function momentAt(data: Timeline, index: number) {
  return dayjs(data.start).add(index, data.step === "1h" ? "hour" : data.step === "1d" ? "day" : "month");
}

export function momentLabel(data: Timeline, index: number) {
  return momentAt(data, index).format(data.step === "1h" ? "DD.MM HH:mm" : data.step === "1d" ? "DD.MM.YYYY" : "MM.YYYY");
}

export function quantityLabel(step: Timeline["step"]) {
  return step === "1h" ? "посадки в час" : step === "1d" ? "посадки в сутки" : "посадки за месяц";
}
