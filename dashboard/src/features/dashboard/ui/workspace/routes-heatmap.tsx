import { Fragment } from "react";
import { Group, Paper, ScrollArea, Stack, Text } from "@mantine/core";
import type { ForecastResponse, Route } from "../../model/forecast-state";
import { routeLevel } from "../../lib/level";
import { momentAt, momentLabel, quantityLabel } from "../../lib/moment";

type Props = { data: ForecastResponse; routes: Route[]; selectedIndex: number; selectIndex: (index: number) => void; route: number | null; onRoute: (route: number | null) => void };

export function RoutesHeatmap({ data, routes, selectedIndex, selectIndex, route, onRoute }: Props) {
  const points = data.series[0]?.value.length ?? 0;
  const labelEvery = Math.max(1, Math.ceil(points / 24));
  const shortLabel = (index: number) => momentAt(data, index).format(data.step === "1h" ? "HH" : data.step === "1d" ? "DD" : "MM");

  return <Paper p="md" withBorder>
    <Stack gap="xs">
      <Text fw={600}>Маршруты и время</Text>
      <Text size="xs" c="dimmed">Цвет - % от обычного уровня, та же шкала, что на карте. Число в подсказке - {quantityLabel(data.step)}.</Text>
      <Group gap="md">
        {[{ color: "#3bb8a3", label: "ниже 80% от обычного уровня" }, { color: "#e3b350", label: "80-120% от обычного уровня" }, { color: "#ef6b73", label: "выше 120% от обычного уровня" }, { color: "#778899", label: "нет сравнения" }].map((item) =>
          <Group key={item.label} gap={4}><span style={{ width: 12, height: 12, background: item.color, display: "inline-block", borderRadius: 2 }} /><Text size="xs">{item.label}</Text></Group>)}
      </Group>
      <ScrollArea type="auto">
        <div style={{ display: "grid", gridTemplateColumns: `max-content repeat(${points}, 16px)`, gap: 1, alignItems: "center" }}>
          <span />
          {Array.from({ length: points }, (_, index) => <Text key={index} size="10px" ta="center">{index % labelEvery === 0 ? shortLabel(index) : ""}</Text>)}
          {data.series.map((series) => <Fragment key={series.route}>
            <Text size="xs" pr="xs" fw={route === series.route ? 700 : 400} style={{ whiteSpace: "nowrap", position: "sticky", left: 0, background: "var(--mantine-color-body)" }}>
              Маршрут {series.route}{routes.find((item) => item.route === series.route)?.has_history === false ? " · оценка" : ""}
            </Text>
            {Array.from({ length: points }, (_, index) => {
              const value = series.value[index];
              const level = routeLevel(value, series.usual[index]);
              const above = series.hi !== undefined && value !== undefined && value > series.hi[index];
              const chosen = index === selectedIndex && route === series.route;
              return <button
                key={`${series.route}-${index}`}
                type="button"
                title={value === undefined
                  ? `Маршрут ${series.route} · ${momentLabel(data, index)} · нет данных`
                  : `Маршрут ${series.route} · ${momentLabel(data, index)} · ${Math.round(value).toLocaleString("ru-RU")} (${quantityLabel(data.step)}) · ${level.label}${above ? " · выше верхней границы" : ""}`}
                onClick={() => { selectIndex(index); onRoute(series.route); }}
                style={{
                  height: 18,
                  border: chosen ? "2px solid var(--mantine-color-red-6)" : index === selectedIndex ? "1px solid var(--mantine-color-dark-2)" : "none",
                  padding: 0,
                  cursor: "pointer",
                  borderRadius: 2,
                  background: value === undefined ? "repeating-linear-gradient(45deg, #444, #444 2px, transparent 2px, transparent 4px)" : level.color,
                  outlineOffset: 1,
                  boxShadow: above ? "inset 0 0 0 2px #2b0b0e" : undefined,
                }}
              />;
            })}
          </Fragment>)}
        </div>
      </ScrollArea>
      <Text size="xs" c="dimmed">Клетка выбирает маршрут и момент для разбора и карты, нового прогноза не запрашивает. Тёмная рамка внутри клетки - превышение верхней границы коридора; при выключенном коридоре её нет. Штриховка - данных нет, это не ноль.</Text>
    </Stack>
  </Paper>;
}
