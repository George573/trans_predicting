import { Box, Group, Paper, SegmentedControl, Select, Skeleton, Stack, Text, Title, UnstyledButton } from "@mantine/core";
import type { ForecastResponse, Route } from "../model/forecast-state";
import { momentLabel } from "../lib/moment";
import { routeLevel } from "../lib/level";

type Props = { routes: Route[]; selected: number | null; onSelect: (route: number | null) => void; data: ForecastResponse | null; stale: boolean };

const mono = "var(--mantine-font-family-monospace)";

function RouteBadge({ route }: { route: number }) {
  return <Box component="span" miw={22} h={22} px={5} fz={12} fw={600} style={{ display: "inline-flex", alignItems: "center", justifyContent: "center", borderRadius: 4, background: "#e6eaee", color: "#0e1217", fontFamily: mono, flexShrink: 0 }}>{route}</Box>;
}

function worstPoint(data: ForecastResponse | null, route: number) {
  const series = data?.series.find((item) => item.route === route);
  if (!series || !data) return null;
  let worst: { index: number; percent: number } | null = null;
  series.value.forEach((value, index) => {
    const usual = series.usual[index];
    if (!(usual > 0)) return;
    const percent = 100 * value / usual;
    if (Number.isFinite(percent) && (!worst || percent > worst.percent)) worst = { index, percent };
  });
  return worst as { index: number; percent: number } | null;
}

export function ObjectPicker({ routes, selected, onSelect, data, stale }: Props) {
  const rows = routes.map((route) => ({ route, worst: worstPoint(data, route.route) }));
  const current = routes.find((route) => route.route === selected);
  return <Stack gap={8}>
    <Group justify="space-between" align="baseline">
      <Title order={6}>Объект</Title>
      <Text fz={11} c="dimmed">уровень: {selected === null ? "сеть" : "маршрут"}</Text>
    </Group>
    <SegmentedControl fullWidth size="xs" value={selected === null ? "network" : "route"} onChange={(value) => onSelect(value === "network" ? null : [...rows].sort((first, second) => (second.worst?.percent ?? -1) - (first.worst?.percent ?? -1))[0]?.route.route ?? null)} data={[{ label: "Сеть", value: "network" }, { label: "Маршрут", value: "route" }]} />
    <Text fz={11} c="dark.3" lh={1.35}>Прогноз - по маршруту целиком. Остановки, направления и пересадки вне модели.</Text>
    <Select
      searchable
      size="sm"
      placeholder="Маршрут: номер или название"
      nothingFoundMessage="Нет такого маршрута"
      value={selected === null ? null : String(selected)}
      onChange={(value) => onSelect(value === null ? null : Number(value))}
      data={routes.map((route) => ({ value: String(route.route), label: route.name }))}
      leftSection={current ? <RouteBadge route={current.route} /> : undefined}
      leftSectionWidth={current ? 36 : undefined}
    />
    {current && <Text fz={11} c="dimmed">{!current.has_history && "оценка без истории · "}{!current.has_geometry ? "нет линии на карте" : current.geometry_source === "spravochnik" ? "линия из справочника" : current.geometry_source === "osm" ? "линия из OSM" : ""}</Text>}
    <Text fz={12} c="dimmed">{selected === null ? "Маршруты сети · процент - максимум к обычному уровню" : "Маршруты сети · клик переключает маршрут"}</Text>
    {routes.length === 0 && <Skeleton height={120} />}
    <Stack gap={2}>
      {rows.map(({ route, worst }) => {
        const level = worst ? routeLevel(worst.percent, 100) : null;
        const active = route.route === selected;
        return <UnstyledButton key={route.route} onClick={() => onSelect(route.route)} mih={40} py={4} px={8} style={{ borderRadius: 6, display: "flex", alignItems: "center", gap: 8, background: active ? "#1e3350" : undefined, outline: active ? "1px solid #4c9aff" : undefined }}>
          <RouteBadge route={route.route} />
          <Box flex={1} miw={0}>
            <Text fz={12} truncate>{route.name}</Text>
            {worst && data && <Text fz={11} c="dimmed" style={{ fontFamily: mono }}>{momentLabel(data, worst.index)}</Text>}
          </Box>
          {level && <Paper component="span" px={6} py={2} radius={4} fz={12} fw={600} bg={`${level.color}26`} c={level.color} style={{ fontFamily: mono, opacity: stale ? 0.5 : 1 }}>{Math.round(worst!.percent)}%</Paper>}
        </UnstyledButton>;
      })}
    </Stack>
  </Stack>;
}
