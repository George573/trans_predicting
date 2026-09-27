import { useEffect, useState } from "react";
import dayjs from "dayjs";
import { ActionIcon, Alert, Button, Group, SegmentedControl, Stack, Text, Title, UnstyledButton } from "@mantine/core";
import { fetchClient } from "@/shared/api/instance";
import { formatDate } from "@/features/scenario";
import type { ForecastRequest, ForecastResponse } from "../model/forecast-state";
import { seasonDays, seasonRange } from "../domain/season";

type Props = { request: ForecastRequest; data: ForecastResponse | null; bounds?: string[]; change: (from: string, to: string, horizon: ForecastRequest["horizon"], granularity: ForecastRequest["granularity"]) => void; selectedIndex: number; selectIndex: (index: number) => void };
const date = (value: string) => dayjs(value).format("YYYY-MM-DD");
const within = (from: string, to: string, bounds?: string[]) => !bounds || (from >= bounds[0] && to <= bounds[1]);

export function PeriodControls({ request, data, bounds, change, selectedIndex, selectIndex }: Props) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const [hourly, setHourly] = useState<ForecastResponse | null>(null);
  const [hourError, setHourError] = useState("");
  const [playing, setPlaying] = useState(false);
  const [night, setNight] = useState(false);
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
    selectIndex(horizon === "day" ? 8 : 0);
    setExpanded(null);
  }

  function navigate(direction: number) {
    const from = month ? date(dayjs(request.from).add(direction, "month").startOf("month").toString()) : date(dayjs(request.from).add(direction * (week ? 7 : 1), "day").toString());
    const to = month ? date(dayjs(from).endOf("month").toString()) : week ? date(dayjs(from).add(6, "day").toString()) : from;
    if (within(from, to, bounds)) { change(from, to, request.horizon, request.granularity); if (week || month) selectIndex(0); setExpanded(null); }
  }

  const previous = month ? date(dayjs(request.from).subtract(1, "month").startOf("month").toString()) : date(dayjs(request.from).subtract(week ? 7 : 1, "day").toString());
  const next = date(dayjs(request.to).add(1, "day").toString());
  const today = date(dayjs().toString());
  const targetEnd = (from: string) => month ? date(dayjs(from).endOf("month").toString()) : week ? date(dayjs(from).add(6, "day").toString()) : from;
  const todayFrom = month ? date(dayjs(today).startOf("month").toString()) : week ? date(dayjs(today).subtract((dayjs(today).day() + 6) % 7, "day").toString()) : today;
  const offset = (dayjs(request.from).day() + 6) % 7;
  const daysInMonth = dayjs(request.from).daysInMonth();
  const total = (response: ForecastResponse, index: number) => response.series.reduce((sum, series) => sum + (series.value[index] ?? 0), 0);
  const hours = Array.from({ length: 24 }, (_, hour) => hour).filter((hour) => night || hour >= 5 || selectedIndex === hour);
  const hourlyDay = !week && !month && !season && data?.step === "1h" ? data : null;
  const dailyWeek = week && data?.step === "1d" ? data : null;
  const dayOf = (index: number) => date(dayjs(request.from).add(index, "day").toString());
  const title = week || month ? `${formatDate(request.from)} - ${formatDate(request.to)}` : `${weekdayNames[dayjs(request.from).day()]}, ${formatDate(request.from)}`;
  const hourRange = (hour: number) => `${String(hour).padStart(2, "0")}:00-${String(hour + 1).padStart(2, "0")}:00`;

  useEffect(() => {
    if (!playing || !hourlyDay) return;
    const timer = window.setTimeout(() => {
      const next = hours.find((hour) => hour > selectedIndex);
      if (next === undefined) setPlaying(false);
      else selectIndex(next);
    }, 700);
    return () => window.clearTimeout(timer);
  }, [playing, hourlyDay, hours, selectedIndex, selectIndex]);

  return <Stack gap="xs">
    <Title order={6}>Период</Title>
    <SegmentedControl fullWidth size="xs" value={request.horizon} onChange={changeHorizon} data={[{ label: "День", value: "day" }, { label: "Неделя", value: "week" }, { label: "Месяц", value: "month" }, { label: "Сезон", value: "season" }]} />
    {!season && <Group gap="xs" wrap="nowrap">
      <ActionIcon variant="default" size="lg" aria-label="Назад" disabled={!within(previous, targetEnd(previous), bounds)} onClick={() => navigate(-1)}>‹</ActionIcon>
      <Text fw={700} size="sm" ta="center" flex={1}>{title}</Text>
      <ActionIcon variant="default" size="lg" aria-label="Вперёд" disabled={!within(next, targetEnd(next), bounds)} onClick={() => navigate(1)}>›</ActionIcon>
      <Button variant="default" size="sm" px="xs" disabled={!within(todayFrom, targetEnd(todayFrom), bounds)} title={!within(todayFrom, targetEnd(todayFrom), bounds) ? "Сегодня вне периода модели" : undefined} onClick={() => change(todayFrom, targetEnd(todayFrom), request.horizon, request.granularity)}>Сегодня</Button>
    </Group>}
    {hourlyDay && <>
      <Group justify="space-between"><Text size="sm" c="dimmed">Час</Text><Text fw={700} ff="monospace">{hourRange(selectedIndex)}</Text></Group>
      <Group gap="xs" wrap="nowrap" align="flex-end">
        <ActionIcon size={40} radius="xl" aria-label={playing ? "Пауза" : "Проиграть сутки по часам"} onClick={() => { if (!playing && selectedIndex >= hours[hours.length - 1]) selectIndex(hours[0]); setPlaying(!playing); }}>{playing ? "❚❚" : "▶"}</ActionIcon>
        <Bars values={hours.map((hour) => total(hourlyDay, hour))} labels={hours.map((hour) => String(hour).padStart(2, "0"))} titles={hours.map((hour) => `${hourRange(hour)}: ${Math.round(total(hourlyDay, hour)).toLocaleString("ru-RU")} посадок`)} selected={hours.indexOf(selectedIndex)} onSelect={(index) => { setPlaying(false); selectIndex(hours[index]); }} />
      </Group>
      <Button variant="subtle" size="compact-xs" onClick={() => setNight(!night)}>{night ? "Свернуть ночные часы" : "Показать ночные часы 00-04"}</Button>
    </>}
    {dailyWeek && <>
      <Group justify="space-between"><Text size="sm" c="dimmed">День</Text><Text fw={700} ff="monospace">{formatDate(dayOf(selectedIndex))}</Text></Group>
      <Bars values={dailyWeek.series[0]?.value.map((_, index) => total(dailyWeek, index)) ?? []} labels={dailyWeek.series[0]?.value.map((_, index) => weekdayNames[dayjs(dayOf(index)).day()]) ?? []} titles={dailyWeek.series[0]?.value.map((_, index) => `${formatDate(dayOf(index))}: ${Math.round(total(dailyWeek, index)).toLocaleString("ru-RU")} посадок`) ?? []} selected={selectedIndex} onSelect={(index) => { selectIndex(index); setHourly(null); setExpanded(dayOf(index)); }} />
    </>}
    {month && data && <div style={{ display: "grid", gridTemplateColumns: "repeat(7, 1fr)", gap: 3 }}>
      {["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"].map((name) => <Text key={name} size="xs" ta="center">{name}</Text>)}
      {Array.from({ length: offset }, (_, index) => <span key={`empty-${index}`} />)}
      {Array.from({ length: daysInMonth }, (_, index) => <Button key={index} size="compact-xs" variant={selectedIndex === index ? "filled" : "light"} onClick={() => selectIndex(index)} title={`${formatDate(dayOf(index))}: ${Math.round(total(data, index)).toLocaleString("ru-RU")} посадок`}>{index + 1}</Button>)}
    </div>}
    {season && <Text size="xs" c="dimmed">Ноябрь 2025 - апрель 2026, {seasonDays} суток: один запрос по суткам ({(seasonDays * 24 * (request.routes?.length || 10)).toLocaleString("ru-RU")} строк модели), на экране шесть месячных сумм. Месячного коридора нет: суточные границы в месячную ширину не складываются. Момент выбирается на графике или тепловой карте.</Text>}
    {expanded && week && <Stack gap={4}>
      <Text size="sm" c="dimmed">По часам: {formatDate(expanded)}</Text>
      {hourError && <Alert color="red">{hourError}</Alert>}
      {hourly && <Bars values={Array.from({ length: 24 }, (_, hour) => total(hourly, hour))} labels={Array.from({ length: 24 }, (_, hour) => hour % 3 === 0 ? String(hour).padStart(2, "0") : "")} titles={Array.from({ length: 24 }, (_, hour) => `${hourRange(hour)}: ${Math.round(total(hourly, hour)).toLocaleString("ru-RU")} посадок`)} selected={null} />}
    </Stack>}
  </Stack>;
}

const weekdayNames = ["Вс", "Пн", "Вт", "Ср", "Чт", "Пт", "Сб"];

function Bars({ values, labels, titles, selected, onSelect }: { values: number[]; labels: string[]; titles: string[]; selected: number | null; onSelect?: (index: number) => void }) {
  const max = Math.max(1, ...values);
  return <Group gap={3} wrap="nowrap" align="flex-end" flex={1} miw={0}>
    {values.map((value, index) => <UnstyledButton key={index} title={titles[index]} onClick={onSelect && (() => onSelect(index))} style={{ flex: 1, minWidth: 0, display: "flex", flexDirection: "column", alignItems: "center", gap: 4, cursor: onSelect ? "pointer" : "default" }}>
      <div style={{ width: "100%", height: 56, display: "flex", alignItems: "flex-end" }}>
        <div style={{ width: "100%", height: `${Math.max(4, 100 * value / max)}%`, borderRadius: 3, background: index === selected ? "var(--mantine-primary-color-filled)" : "var(--mantine-color-dark-5)" }} />
      </div>
      <Text fz={10} ff="monospace" lh={1} fw={index === selected ? 700 : 400} c={index === selected ? "bright" : "dimmed"}>{labels[index]}</Text>
    </UnstyledButton>)}
  </Group>;
}
