import { Stack, Text } from "@mantine/core";
import { BarChart } from "@mantine/charts";
import type { ForecastResponse } from "../../model/forecast-state";
import { momentAt, momentLabel } from "../../lib/moment";
import { Panel } from "../panel";

type Props = { data: ForecastResponse; selectIndex: (index: number) => void };

const sumAt = (data: ForecastResponse, key: "value" | "usual", index: number) => data.series.reduce((sum, series) => sum + (series[key][index] ?? 0), 0);
const signed = (value: number) => `${value > 0 ? "+" : ""}${value}%`;
const scope = (data: ForecastResponse) => data.series.length === 1 ? `маршрут ${data.series[0].route}` : "сумма по сети";

export function UsualDeviation({ data, selectIndex }: Props) {
  const length = data.series[0]?.value.length ?? 0;
  const rows = Array.from({ length }, (_, index) => {
    const usual = sumAt(data, "usual", index);
    return { index, label: data.step === "1h" ? momentAt(data, index).format("HH") : momentLabel(data, index), deviation: usual > 0 ? Math.round(100 * sumAt(data, "value", index) / usual) - 100 : 0 };
  });
  return <Panel title="Отличие от обычного уровня" note={scope(data)}>
    <Stack gap={6}>
      <BarChart
        h={160}
        data={rows}
        dataKey="label"
        valueFormatter={signed}
        getBarColor={(value) => value > 0 ? "#f2b544" : "#4f6d8c"}
        referenceLines={[{ y: 0, color: "gray.6" }]}
        barChartProps={{ onClick: (state) => { const row = rows.find((item) => item.label === state?.activeLabel); if (row) selectIndex(row.index); } }}
        series={[{ name: "deviation", label: "К обычному", color: "gray.6" }]}
      />
      <Text fz={11} c="dimmed">Насколько прогноз выше или ниже обычного уровня в каждой точке. Клик по столбцу выбирает момент.</Text>
    </Stack>
  </Panel>;
}

const periods = [
  { label: "Утро", hours: [5, 6, 7, 8, 9, 10] },
  { label: "День", hours: [11, 12, 13, 14, 15] },
  { label: "Вечер", hours: [16, 17, 18, 19, 20] },
  { label: "Поздно", hours: [21, 22, 23, 0, 1, 2, 3, 4] },
];

export function DayPeriods({ data }: { data: ForecastResponse }) {
  if (data.step !== "1h") return null;
  const length = data.series[0]?.value.length ?? 0;
  const rows = periods.map((period) => {
    let value = 0, usual = 0;
    for (let index = 0; index < length; index += 1) {
      if (!period.hours.includes(momentAt(data, index).hour())) continue;
      value += sumAt(data, "value", index);
      usual += sumAt(data, "usual", index);
    }
    const difference = usual > 0 ? Math.round(100 * value / usual) - 100 : null;
    return { period: difference === null ? period.label : `${period.label} ${signed(difference)}`, value: Math.round(value), usual: Math.round(usual) };
  });
  return <Panel title="Прогноз против обычного дня" note={scope(data)}>
    <Stack gap={6}>
      <BarChart
        h={170}
        data={rows}
        dataKey="period"
        withLegend
        valueFormatter={(value) => value.toLocaleString("ru-RU")}
        series={[{ name: "value", label: "Прогноз", color: "#4c9aff" }, { name: "usual", label: "Обычный день", color: "gray.6" }]}
      />
      <Text fz={11} c="dimmed">Четыре части суток парами столбиков, рядом с названием - разница с обычным уровнем.</Text>
    </Stack>
  </Panel>;
}
