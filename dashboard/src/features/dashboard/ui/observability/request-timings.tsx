import { Paper, Progress, Stack, Text } from "@mantine/core";
import type { RequestTelemetry } from "../../model/forecast-state";

const stages = [{ key: "parse", label: "Разбор" }, { key: "features", label: "Признаки" }, { key: "model", label: "Модель" }, { key: "corrections", label: "Условия" }, { key: "json", label: "JSON" }] as const;

export function RequestTimings({ telemetry }: { telemetry: RequestTelemetry }) {
  const server = stages.reduce((sum, stage) => sum + (telemetry.stages[stage.key] ?? 0), 0);
  const overhead = telemetry.browserMs === null ? null : Math.max(0, telemetry.browserMs - server);
  const total = telemetry.browserMs ?? 0;
  return <Paper p="md" withBorder><Stack gap="xs">
    <Text fw={600}>Запрос прогноза</Text>
    <Text size="xs">POST /api/v1/forecast · статус: {telemetry.status} · строк модели: {telemetry.rows ?? "нет данных"}</Text>
    <Text size="xs">Время браузера: {telemetry.browserMs === null ? "нет данных" : `${telemetry.browserMs.toFixed(1)} мс`}</Text>
    {stages.map((stage) => <div key={stage.key}><Text size="xs">{stage.label}: {telemetry.stages[stage.key] === undefined ? "нет данных" : `${telemetry.stages[stage.key]?.toFixed(1)} мс`}</Text><Progress value={total > 0 ? 100 * (telemetry.stages[stage.key] ?? 0) / total : 0} size="xs" /></div>)}
    <Text size="xs">Сеть и накладные расходы: {overhead === null ? "нет данных" : `${overhead.toFixed(1)} мс`}</Text>
  </Stack></Paper>;
}
