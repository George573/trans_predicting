import { useEffect, useRef, useState } from "react";
import { Alert, Button, Group, Paper, Slider, Stack, Text } from "@mantine/core";
import { apiBaseUrl, fetchClient } from "@/shared/api/instance";
import type { components } from "@/shared/api/schema/generated";
import { session } from "@/shared/model/session";
import type { ForecastRequest } from "../../model/forecast-state";

type Stats = components["schemas"]["Stats"];
type Props = { request: ForecastRequest | null; ready: boolean };

export function ProcessLoad({ request, ready }: Props) {
  const [opened, setOpened] = useState(false);
  const [threads, setThreads] = useState(1);
  const [running, setRunning] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [successes, setSuccesses] = useState(0);
  const [failures, setFailures] = useState(0);
  const [canceled, setCanceled] = useState(0);
  const [stats, setStats] = useState<Stats | null>(null);
  const [statsError, setStatsError] = useState("");
  const controller = useRef<AbortController | null>(null);
  const timer = useRef<number | null>(null);
  const ticker = useRef<number | null>(null);
  const started = useRef(0);

  function stop() {
    if (!controller.current) return;
    controller.current.abort();
    controller.current = null;
    if (timer.current !== null) window.clearTimeout(timer.current);
    if (ticker.current !== null) window.clearInterval(ticker.current);
    timer.current = null;
    ticker.current = null;
    setElapsed(Math.min(15, (performance.now() - started.current) / 1000));
    setRunning(false);
  }

  useEffect(() => {
    if (!opened) return;
    const pollController = new AbortController();
    const poll = async () => {
      try {
        const result = await fetchClient.GET("/api/v1/stats", { signal: pollController.signal });
        if (pollController.signal.aborted) return;
        if (result.response.status === 401) stop();
        if (result.data) { setStats(result.data); setStatsError(""); }
        else { setStats(null); setStatsError(result.error?.error.message ?? "Метрики недоступны"); }
      } catch { if (!pollController.signal.aborted) { setStats(null); setStatsError("Метрики недоступны"); } }
    };
    const first = window.setTimeout(() => void poll(), 0);
    const interval = window.setInterval(() => void poll(), 1000);
    return () => { pollController.abort(); window.clearTimeout(first); window.clearInterval(interval); };
  }, [opened]);

  useEffect(() => () => {
    controller.current?.abort();
    if (timer.current !== null) window.clearTimeout(timer.current);
    if (ticker.current !== null) window.clearInterval(ticker.current);
  }, []);

  function start() {
    if (!request || !ready || running) return;
    const snapshot = structuredClone(request);
    const body = JSON.stringify(snapshot);
    const current = new AbortController();
    controller.current = current;
    started.current = performance.now();
    setSuccesses(0);
    setFailures(0);
    setCanceled(0);
    setElapsed(0);
    setRunning(true);
    const deadline = started.current + 15000;
    const finish = () => { if (controller.current === current) stop(); };
    timer.current = window.setTimeout(finish, 15000);
    ticker.current = window.setInterval(() => setElapsed(Math.min(15, (performance.now() - started.current) / 1000)), 100);
    const worker = async () => {
      while (!current.signal.aborted && performance.now() < deadline) {
        try {
          const headers = new Headers({ "Content-Type": "application/json" });
          const authorization = session.authorization();
          if (authorization) headers.set("Authorization", authorization);
          const response = await fetch(`${apiBaseUrl}/api/v1/forecast`, { method: "POST", headers, body, signal: current.signal });
          if (response.status === 401) { setFailures((value) => value + 1); session.requireLogin(); finish(); return; }
          if (response.ok) setSuccesses((value) => value + 1);
          else setFailures((value) => value + 1);
        } catch {
          if (current.signal.aborted) { setCanceled((value) => value + 1); return; }
          setFailures((value) => value + 1);
        }
      }
    };
    void Promise.all(Array.from({ length: threads }, () => worker())).then(finish);
  }

  const display = (value?: number) => value === undefined ? "нет данных" : String(Math.round(value * 10) / 10);
  return <Paper p="md" withBorder><Stack gap="xs">
    <Group justify="space-between"><Text fw={600}>Метрики и нагрузка</Text><Button size="compact-xs" variant="subtle" onClick={() => { if (opened) stop(); setOpened(!opened); }}>{opened ? "Свернуть" : "Развернуть"}</Button></Group>
    {opened && <>
      <Text size="xs">Параллельных запросов: {threads}</Text><Slider min={1} max={12} step={1} value={threads} onChange={setThreads} disabled={running} marks={[{ value: 1, label: "1" }, { value: 6, label: "6" }, { value: 12, label: "12" }]} mb="lg" />
      <Group><Button size="xs" onClick={start} disabled={!ready || running}>Старт на 15 секунд</Button><Button size="xs" variant="light" onClick={stop} disabled={!running}>Стоп</Button><Text size="xs">{elapsed.toFixed(1)} / 15 с</Text></Group>
      {!ready && <Text size="xs" c="dimmed">Для прогона нужен успешный текущий прогноз.</Text>}
      <Text size="xs">Успехов: {successes} · ошибок: {failures} · отменено: {canceled}</Text>
      <Text size="xs">RPS этого браузера: {elapsed > 0 ? display(successes / elapsed) : "нет данных"} · RPS сервиса за минуту: {display(stats?.rps_1m)}</Text>
      <Text size="xs">CPU: {display(stats?.cpu_pct)}% одного ядра · память: {display(stats?.rss_mb)} МБ · p50: {display(stats?.p50_ms)} мс · p95: {display(stats?.p95_ms)} мс</Text>
      <Text size="xs">С запуска сервиса: запросов {stats ? stats.requests.toLocaleString("ru-RU") : "нет данных"} · ошибок {stats ? stats.errors.toLocaleString("ru-RU") : "нет данных"} · строк модели {stats ? stats.rows.toLocaleString("ru-RU") : "нет данных"} · работает {stats ? `${Math.floor(stats.uptime_s / 3600)} ч ${Math.floor(stats.uptime_s % 3600 / 60)} мин` : "нет данных"}</Text>
      <Text size="xs" c="dimmed">Браузер может ограничить число соединений примерно шестью: его RPS - нижняя оценка. Для точного замера используйте нагрузочный клиент сервиса.</Text>
      {statsError && <Alert color="yellow">{statsError}. Ресурсы неизвестны.</Alert>}
    </>}
  </Stack></Paper>;
}
