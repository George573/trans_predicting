import { Paper, Stack, Text } from "@mantine/core";
import { CompositeChart } from "@mantine/charts";
import type { ForecastResponse } from "../../model/forecast-state";
import { momentLabel, quantityLabel } from "../../lib/moment";

type Props = { data: ForecastResponse; selectedIndex: number; selectIndex: (index: number) => void };
type Point = { index: number; value: number; base: number; usual: number; band?: [number, number] };

export function ForecastChart({ data, selectedIndex, selectIndex }: Props) {
  const single = data.series.length === 1 ? data.series[0] : null;
  const bounds = single?.lo && single.hi ? { lo: single.lo, hi: single.hi } : null;
  const points: Point[] = Array.from({ length: data.series[0]?.value.length ?? 0 }, (_, index) => ({
    index,
    value: single ? single.value[index] : data.series.reduce((sum, series) => sum + (series.value[index] ?? 0), 0),
    base: single ? single.base[index] : data.series.reduce((sum, series) => sum + (series.base[index] ?? 0), 0),
    usual: single ? single.usual[index] : data.series.reduce((sum, series) => sum + (series.usual[index] ?? 0), 0),
    ...(bounds && { band: [bounds.lo[index], bounds.hi[index]] as [number, number] }),
  }));

  return <Paper p="md" withBorder>
    <Stack gap="xs">
      <Text fw={600}>Прогноз посадок: {single ? `маршрут ${single.route}` : `сумма маршрутов ${data.series.map((series) => series.route).join(", ")}`}</Text>
      <Text size="xs" c="dimmed">Величина: {quantityLabel(data.step)}. Выбранный момент: {points.length > 0 ? momentLabel(data, Math.min(selectedIndex, points.length - 1)) : "нет точек"}</Text>
      <CompositeChart
        h={260}
        data={points}
        dataKey="index"
        withDots={false}
        curveType="linear"
        withLegend
        referenceLines={selectedIndex < points.length ? [{ x: selectedIndex, color: "red.6" }] : []}
        xAxisProps={{ tickFormatter: (value: number) => momentLabel(data, Number(value)), minTickGap: 32 }}
        composedChartProps={{ onClick: (state) => { const index = Number(state?.activeLabel); if (Number.isInteger(index) && index >= 0 && index < points.length) selectIndex(index); } }}
        tooltipProps={{ content: ({ active, payload }) => {
          const point = payload?.[0]?.payload as Point | undefined;
          if (!active || !point) return null;
          return <Paper px="md" py="sm" withBorder shadow="md" radius="md">
            <Text fw={600} size="sm">{single ? `Маршрут ${single.route}` : "Сумма маршрутов"} · {momentLabel(data, point.index)}</Text>
            <Text size="xs">С условиями: {Math.round(point.value)}</Text>
            <Text size="xs">Без условий: {Math.round(point.base)}</Text>
            <Text size="xs">Обычный уровень: {Math.round(point.usual)}</Text>
            <Text size="xs">{point.band ? `Коридор: ${Math.round(point.band[0])} - ${Math.round(point.band[1])}` : "Коридор не показан"}</Text>
          </Paper>;
        } }}
        series={[
          ...(bounds ? [{ name: "band", label: "Коридор", color: "indigo.2", type: "area" as const }] : []),
          { name: "usual", label: "Обычный уровень", color: "gray.5", type: "line" as const, strokeDasharray: "4 4" },
          { name: "base", label: "Без условий", color: "teal.6", type: "line" as const, strokeDasharray: "6 3" },
          { name: "value", label: "С условиями", color: "indigo.6", type: "line" as const },
        ]}
      />
      {!bounds && <Text size="xs" c="dimmed">{single ? "Границы коридора не запрошены: в ответе нет lo и hi." : "Коридор не показан: общих границ для суммы маршрутов сервис не даёт, а складывать коридоры отдельных маршрутов нельзя."}</Text>}
    </Stack>
  </Paper>;
}
