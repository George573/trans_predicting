import { useEffect, useState } from "react";
import dayjs from "dayjs";
import { Alert, Button, Group, SegmentedControl, Stack, Text } from "@mantine/core";
import { fetchClient } from "@/shared/api/instance";
import type { ForecastRequest, ForecastResponse } from "../model/forecast-state";
import { seasonRange } from "../domain/season";

type Props = { request: ForecastRequest; data: ForecastResponse | null; bounds?: string[]; change: (from: string, to: string, horizon: ForecastRequest["horizon"], granularity: ForecastRequest["granularity"]) => void; selectedIndex: number; selectIndex: (index: number) => void };
const date = (value: string) => dayjs(value).format("YYYY-MM-DD");
const within = (from: string, to: string, bounds?: string[]) => !bounds || (from >= bounds[0] && to <= bounds[1]);

export function PeriodControls({ request, data, bounds, change, selectedIndex, selectIndex }: Props) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const [hourly, setHourly] = useState<ForecastResponse | null>(null);
  const [hourError, setHourError] = useState("");
  const week = request.horizon === "week";
  const month = request.horizon === "month";
  const season = request.horizon === "season";

  useEffect(() => {
    if (!expanded || !week) return;
    const controller = new AbortController();
    queueMicrotask(async () => {
      const body: ForecastRequest = { ...request, from: expanded, to: expanded, granularity: "hour" };
      try {
        const result = await fetchClient.POST("/api/v1/forecast", { body, signal: controller.signal });
        if (!controller.signal.aborted) {
          setHourly(result.data ?? null);
          setHourError(result.error?.error.message ?? "");
        }
      } catch { if (!controller.signal.aborted) setHourError("Почасовые данные недоступны"); }
    });
    return () => controller.abort();
  }, [expanded, request, week]);

  function changeHorizon(horizon: string) {
    if (horizon === "season") {
      if (within(seasonRange[0], seasonRange[1], bounds)) change(seasonRange[0], seasonRange[1], "season", "day");
      selectIndex(0);
      setExpanded(null);
      return;
    }
    const day = dayjs(request.from);
    const from = horizon === "month" ? date(day.startOf("month").toString()) : horizon === "week" ? date(day.subtract((day.day() + 6) % 7, "day").toString()) : request.from;
    const to = horizon === "month" ? date(dayjs(from).endOf("month").toString()) : horizon === "week" ? date(dayjs(from).add(6, "day").toString()) : from;
    if (within(from, to, bounds)) change(from, to, horizon as ForecastRequest["horizon"], horizon === "day" ? "hour" : "day");
    selectIndex(0);
    setExpanded(null);
  }

  function navigate(direction: number) {
    const from = month ? date(dayjs(request.from).add(direction, "month").startOf("month").toString()) : date(dayjs(request.from).add(direction * (week ? 7 : 1), "day").toString());
    const to = month ? date(dayjs(from).endOf("month").toString()) : week ? date(dayjs(from).add(6, "day").toString()) : from;
    if (within(from, to, bounds)) { change(from, to, request.horizon, request.granularity); selectIndex(0); setExpanded(null); }
  }

  const previous = month ? date(dayjs(request.from).subtract(1, "month").startOf("month").toString()) : date(dayjs(request.from).subtract(week ? 7 : 1, "day").toString());
  const next = date(dayjs(request.to).add(1, "day").toString());
  const today = date(dayjs().toString());
  const targetEnd = (from: string) => month ? date(dayjs(from).endOf("month").toString()) : week ? date(dayjs(from).add(6, "day").toString()) : from;
  const todayFrom = month ? date(dayjs(today).startOf("month").toString()) : week ? date(dayjs(today).subtract((dayjs(today).day() + 6) % 7, "day").toString()) : today;
  const offset = (dayjs(request.from).day() + 6) % 7;
  const daysInMonth = dayjs(request.from).daysInMonth();
  return <Stack gap="xs">
    <Text fw={600}>Период</Text>
    <SegmentedControl fullWidth size="xs" value={request.horizon} onChange={changeHorizon} data={[{ label: "День", value: "day" }, { label: "Неделя", value: "week" }, { label: "Месяц", value: "month" }, { label: "Сезон", value: "season" }]} />
    {!season && <Group justify="space-between"><Button size="xs" disabled={!within(previous, targetEnd(previous), bounds)} onClick={() => navigate(-1)}>Назад</Button><Text size="sm">{request.from}{(week || month) && ` - ${request.to}`}</Text><Button size="xs" disabled={!within(next, targetEnd(next), bounds)} onClick={() => navigate(1)}>Вперёд</Button></Group>}
    {!season && <Button variant="subtle" size="xs" disabled={!within(todayFrom, targetEnd(todayFrom), bounds)} title={!within(todayFrom, targetEnd(todayFrom), bounds) ? "Сегодня вне периода модели" : undefined} onClick={() => change(todayFrom, targetEnd(todayFrom), request.horizon, request.granularity)}>Сегодня</Button>}
    {month && data && <div style={{ display: "grid", gridTemplateColumns: "repeat(7, 1fr)", gap: 3 }}>
      {["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"].map((name) => <Text key={name} size="xs" ta="center">{name}</Text>)}
      {Array.from({ length: offset }, (_, index) => <span key={`empty-${index}`} />)}
      {Array.from({ length: daysInMonth }, (_, index) => {
        const value = data.series.reduce((sum, series) => sum + (series.value[index] ?? 0), 0);
        return <Button key={index} size="compact-xs" variant={selectedIndex === index ? "filled" : "light"} onClick={() => selectIndex(index)} title={`${date(dayjs(request.from).add(index, "day").toString())}: ${Math.round(value)} посадок`}>{index + 1}</Button>;
      })}
    </div>}
    {season && <Text size="xs" c="dimmed">Сезон запрашивается тремя частями по суткам и показывается шестью месячными суммами. Момент выбирается на графике или тепловой карте.</Text>}
    {!month && !season && data && <Group gap={4}>{Array.from({ length: week ? 7 : 24 }, (_, index) => {
      const label = week ? date(dayjs(request.from).add(index, "day").toString()) : `${String(index).padStart(2, "0")}:00`;
      return <Button key={index} size="compact-xs" variant={selectedIndex === index ? "filled" : "light"} onClick={() => { selectIndex(index); if (week) { setHourly(null); setExpanded(label); } }}>{label}</Button>;
    })}</Group>}
    {expanded && week && <Stack gap={4}><Text size="sm">Почасовые данные: {expanded}</Text>{hourError && <Alert color="red">{hourError}</Alert>}{hourly && <Group gap={4}>{hourly.series[0]?.value.map((value, index) => <Text key={index} size="xs">{index}:00 - {Math.round(value)}</Text>)}</Group>}</Stack>}
  </Stack>;
}
