import { Paper, SimpleGrid, Stack, Text, Title } from "@mantine/core";
import { BarChart, DonutChart } from "@mantine/charts";
import type { ForecastResponse } from "../model/forecast-state";
import { routeColor, routeLevel } from "../lib/level";

type Props = { data: ForecastResponse; comparison: { title: string; data: ForecastResponse } | null; routes: { route: number }[]; onRouteSelect: (route: number) => void };

const sum = (values: number[]) => values.reduce((total, value) => total + value, 0);

export function RouteCharts({ data, comparison, routes, onRouteSelect }: Props) {
  const rows = data.series.map((series) => {
    const usual = sum(series.usual);
    const other = comparison?.data.series.find((item) => item.route === series.route);
    return { route: `№${series.route}`, number: series.route, value: Math.round(series.total), base: Math.round(series.base_total), usual: Math.round(usual), deviation: usual > 0 ? Math.round(100 * series.total / usual) - 100 : null, ...(other && { other: Math.round(other.total) }) };
  });
  const total = sum(rows.map((row) => row.value));
  const select = (label?: string | number) => { const row = rows.find((item) => item.route === label); if (row) onRouteSelect(row.number); };

  return <SimpleGrid cols={{ base: 1, lg: rows.length > 1 ? 3 : 2 }}>
    <Paper p="md" withBorder>
      <Stack gap="xs">
        <Title order={6}>Посадки за период по маршрутам</Title>
        <BarChart
          h={220}
          data={rows}
          dataKey="route"
          withLegend
          valueFormatter={(value) => value.toLocaleString("ru-RU")}
          barChartProps={{ onClick: (state) => select(state?.activeLabel) }}
          series={[
            { name: "value", label: "С условиями", color: "indigo.6" },
            { name: "base", label: "Без условий", color: "teal.6" },
            { name: "usual", label: "Обычный уровень", color: "gray.6" },
            ...(comparison ? [{ name: "other", label: comparison.title, color: "orange.5" }] : []),
          ]}
        />
      </Stack>
    </Paper>
    <Paper p="md" withBorder>
      <Stack gap="xs">
        <Title order={6}>Отклонение от обычного уровня</Title>
        <BarChart
          h={220}
          data={rows}
          dataKey="route"
          valueFormatter={(value) => `${value > 0 ? "+" : ""}${value}%`}
          getBarColor={(value) => routeLevel(value + 100, 100).color}
          referenceLines={[{ y: 0, color: "gray.5" }]}
          barChartProps={{ onClick: (state) => select(state?.activeLabel) }}
          series={[{ name: "deviation", label: "Отклонение", color: "gray.6" }]}
        />
        <Text size="xs" c="dimmed">Итог прогноза за период против суммы обычного уровня, в процентах. Клик по столбцу выбирает маршрут.</Text>
      </Stack>
    </Paper>
    {rows.length > 1 && <Paper p="md" withBorder>
      <Stack gap="xs" align="center">
        <Title order={6} style={{ alignSelf: "flex-start" }}>Доля маршрутов в сумме</Title>
        <DonutChart
          size={180}
          thickness={28}
          withLabels
          withLabelsLine
          labelsType="percent"
          chartLabel={total.toLocaleString("ru-RU")}
          valueFormatter={(value) => value.toLocaleString("ru-RU")}
          data={rows.map((row) => ({ name: `Маршрут ${row.number}`, value: row.value, color: routeColor(routes, row.number) }))}
        />
      </Stack>
    </Paper>}
  </SimpleGrid>;
}
