import { useEffect, useState } from "react";
import { Alert, Badge, Button, Group, Paper, ScrollArea, Select, Skeleton, Stack, Table, Text, Title } from "@mantine/core";
import { fetchClient } from "@/shared/api/instance";
import type { components } from "@/shared/api/schema/generated";
import type { ForecastRequest, ForecastResponse } from "../../model/forecast-state";
import { formatDate } from "@/features/scenario";
import { momentAt } from "../../lib/moment";

type ExplainResponse = components["schemas"]["ExplainResponse"];
type Props = { data: ForecastResponse; request: ForecastRequest | null; selectedIndex: number; route: number | null; onRoute: (route: number | null) => void };

const amount = (value: number) => Math.round(value).toLocaleString("ru-RU");

export function PointBreakdown({ data, request, selectedIndex, route, onRoute }: Props) {
  const [hour, setHour] = useState<string | null>(null);
  const [day, setDay] = useState<string | null>(null);
  const [result, setResult] = useState<ExplainResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [revision, retry] = useState(0);

  const moment = momentAt(data, Math.min(selectedIndex, Math.max(0, (data.series[0]?.value.length ?? 1) - 1)));
  const routes = data.series.map((series) => series.route);
  const askedRoute = routes.length === 1 ? routes[0] : routes.find((item) => item === route) ?? null;
  const askedDate = data.step === "1mo" ? day : moment.format("YYYY-MM-DD");
  const askedHour = data.step === "1h" ? moment.hour() : hour === null ? null : Number(hour);
  const context = JSON.stringify({ model: request?.model, horizon: request?.horizon ?? "day", conditions: request?.conditions ?? [] });

  useEffect(() => {
    if (askedRoute === null || !askedDate || askedHour === null) {
      setResult(null);
      setError("");
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    setError("");
    queueMicrotask(async () => {
      if (controller.signal.aborted) return;
      try {
        const body: components["schemas"]["ExplainRequest"] = { ...JSON.parse(context), route: askedRoute as components["schemas"]["RouteNumber"], date: askedDate, hour: askedHour };
        const answer = await fetchClient.POST("/api/v1/explain", { body, signal: controller.signal });
        if (controller.signal.aborted) return;
        if (answer.error) { setError(answer.error.error.message); setResult(null); }
        else setResult(answer.data);
      } catch {
        if (!controller.signal.aborted) { setError("Нет связи с сервисом: разбор точки не получен"); setResult(null); }
      } finally { if (!controller.signal.aborted) setLoading(false); }
    });
    return () => controller.abort();
  }, [askedRoute, askedDate, askedHour, context, revision]);

  return <Paper p="md" withBorder>
    <Stack gap="xs">
      <Title order={6}>Разбор точки</Title>
      <Group gap="xs" align="end">
        {routes.length > 1 && <Select size="xs" w={140} label="Маршрут" placeholder="выберите" value={askedRoute === null ? null : String(askedRoute)} onChange={(value) => onRoute(value === null ? null : Number(value))} data={routes.map((item) => String(item))} />}
        {data.step === "1mo" && <Select size="xs" w={140} label="Дата" placeholder="выберите" value={day} onChange={setDay} data={Array.from({ length: moment.daysInMonth() }, (_, index) => { const value = moment.date(index + 1).format("YYYY-MM-DD"); return { value, label: formatDate(value) }; })} />}
        {data.step !== "1h" && <Select size="xs" w={110} label="Час" placeholder="выберите" value={hour} onChange={setHour} data={Array.from({ length: 24 }, (_, index) => String(index))} />}
      </Group>
      <Text size="xs" c="dimmed">
        {askedRoute === null ? "Выберите маршрут: сумма сети не разбирается." : askedDate === null ? "Выберите дату внутри месяца." : askedHour === null ? "Выберите час: агрегат за сутки не подписывается значением разбора." : `Маршрут ${askedRoute}, ${formatDate(askedDate)}, ${String(askedHour).padStart(2, "0")}:00`}
      </Text>
      {loading && <Skeleton height={120} />}
      {error && <Alert color="red" p="xs">{error}<Button size="compact-xs" ml="sm" onClick={() => retry((value) => value + 1)}>Повторить</Button></Alert>}
      {result && !loading && <Stack gap="xs">
        <Table withTableBorder={false} verticalSpacing={2} fz="xs">
          <Table.Thead><Table.Tr><Table.Th>Шаг</Table.Th><Table.Th>Подробность</Table.Th><Table.Th>Множитель</Table.Th><Table.Th>Значение</Table.Th></Table.Tr></Table.Thead>
          <Table.Tbody>{result.steps.map((step, index) => <Table.Tr key={index}>
            <Table.Td>{step.step}</Table.Td>
            <Table.Td>{step.detail ?? ""}</Table.Td>
            <Table.Td>{step.factor === null || step.factor === undefined ? "нет" : step.factor}</Table.Td>
            <Table.Td>{amount(step.value)}</Table.Td>
          </Table.Tr>)}</Table.Tbody>
        </Table>
        <Text size="sm">Итог: {amount(result.value)} · модель {result.model}</Text>
        {result.conditions.length > 0 && <Table withTableBorder={false} verticalSpacing={2} fz="xs">
          <Table.Thead><Table.Tr><Table.Th>Условие за сутки</Table.Th><Table.Th>Состояние</Table.Th><Table.Th>Множитель</Table.Th><Table.Th>Вклад</Table.Th><Table.Th>Часов</Table.Th></Table.Tr></Table.Thead>
          <Table.Tbody>{result.conditions.map((condition) => <Table.Tr key={condition.id}>
            <Table.Td>{condition.id}</Table.Td>
            <Table.Td>{condition.applied ? <Badge color="teal" size="xs">применено</Badge> : <Badge color="gray" size="xs">не применено</Badge>}</Table.Td>
            <Table.Td>{condition.factor}</Table.Td>
            <Table.Td>{condition.applied ? `${condition.contribution_pct}%` : "-"}</Table.Td>
            <Table.Td>{condition.points}</Table.Td>
          </Table.Tr>)}</Table.Tbody>
        </Table>}
        {result.warnings.map((warning, index) => <Alert key={index} color="yellow" p="xs">{warning.message}</Alert>)}
        <Text size="xs" c="dimmed">Календарь: {Object.entries(result.calendar).map(([key, value]) => `${key} = ${key === "date" ? formatDate(String(value)) : String(value)}`).join(", ")}</Text>
        <ScrollArea.Autosize mah={140}>
          <Table withTableBorder={false} verticalSpacing={1} fz="xs">
            <Table.Thead><Table.Tr><Table.Th>Признак</Table.Th><Table.Th>Значение</Table.Th><Table.Th>Вид</Table.Th><Table.Th>Хеш</Table.Th></Table.Tr></Table.Thead>
            <Table.Tbody>{result.features.map((feature) => <Table.Tr key={feature.name}>
              <Table.Td>{feature.name}</Table.Td>
              <Table.Td>{String(feature.value)}</Table.Td>
              <Table.Td>{feature.kind ?? ""}</Table.Td>
              <Table.Td>{feature.hash ?? ""}</Table.Td>
            </Table.Tr>)}</Table.Tbody>
          </Table>
        </ScrollArea.Autosize>
      </Stack>}
    </Stack>
  </Paper>;
}
