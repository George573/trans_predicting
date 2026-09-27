import { Alert, Anchor, Badge, Button, Group, Paper, Stack, Table, Text } from "@mantine/core";
import type { ForecastResponse } from "../../model/forecast-state";
import { momentLabel, quantityLabel } from "../../lib/moment";
import type { Comparison } from "./workspace";

type Props = { data: ForecastResponse; comparison: Comparison; selectIndex: (index: number) => void; onRouteSelect: (route: number) => void };

const amount = (value: number) => Math.round(value).toLocaleString("ru-RU");

function peakOfNetwork(data: ForecastResponse) {
  const sums = Array.from({ length: data.series[0]?.value.length ?? 0 }, (_, index) => data.series.reduce((sum, series) => sum + (series.value[index] ?? 0), 0));
  const index = sums.reduce((best, value, current) => value > sums[best] ? current : best, 0);
  return { index, value: sums[index] ?? 0 };
}

function exceedances(data: ForecastResponse) {
  return data.series.flatMap((series) => {
    const hi = series.hi;
    return hi ? series.value.flatMap((value, index) => value > hi[index] ? [{ route: series.route, index, value, hi: hi[index] }] : []) : [];
  });
}

export function Summary({ data, comparison, selectIndex, onRouteSelect }: Props) {
  if (data.series.length === 0) return <Paper p="md" withBorder><Text fw={600}>Цифры и тревоги</Text><Text size="sm" c="dimmed">Нет данных: сервис вернул прогноз без рядов.</Text></Paper>;
  const total = data.series.reduce((sum, series) => sum + series.total, 0);
  const baseTotal = data.series.reduce((sum, series) => sum + series.base_total, 0);
  const difference = total - baseTotal;
  const peak = peakOfNetwork(data);
  const withCorridor = data.series.filter((series) => series.hi);
  const alerts = exceedances(data);
  const shown = alerts.slice(0, 12);
  const otherTotal = comparison?.data.series.filter((series) => data.series.some((item) => item.route === series.route)).reduce((sum, series) => sum + series.total, 0);

  return <Paper p="md" withBorder>
    <Stack gap="sm">
      <Text fw={600}>Цифры и тревоги</Text>

      <Stack gap={2}>
        <Text>Посадки за период: {amount(total)}</Text>
        <Text size="sm" c="dimmed">Без условий: {amount(baseTotal)}</Text>
        <Text size="sm">Изменение: {difference >= 0 ? "+" : ""}{amount(difference)}{baseTotal > 0 ? ` (${difference >= 0 ? "+" : ""}${(100 * difference / baseTotal).toFixed(1)}%)` : ""}</Text>
        {baseTotal === 0 && <Text size="xs" c="dimmed">Базовый прогноз равен нулю, доля изменения не считается: показана абсолютная разница.</Text>}
        {comparison && otherTotal !== undefined && <Text size="sm">{comparison.title}: {amount(otherTotal)}{otherTotal > 0 ? `, показанная модель ${total >= otherTotal ? "выше" : "ниже"} на ${(100 * Math.abs(total - otherTotal) / otherTotal).toFixed(1)}%` : ""}</Text>}
      </Stack>

      <Stack gap={2}>
        <Text fw={500}>Пик{data.series.length > 1 ? " по выбранным маршрутам" : ""}</Text>
        <Group gap="xs">
          <Text size="sm">{momentLabel(data, peak.index)} · {amount(peak.value)} ({quantityLabel(data.step)})</Text>
          <Anchor component="button" type="button" size="sm" onClick={() => selectIndex(peak.index)}>показать</Anchor>
        </Group>
        {data.series.length > 1 && <Text size="xs" c="dimmed">Считается по суммам точек, поэтому может не совпадать с пиками отдельных маршрутов: {data.series.map((series) => `${series.route} - ${momentLabel(data, series.peak_index)}`).join("; ")}</Text>}
      </Stack>

      <Stack gap={4}>
        <Group gap="xs"><Text fw={500}>Превышения коридора</Text>{withCorridor.length > 0 && <Badge color={alerts.length > 0 ? "red" : "teal"}>{alerts.length}</Badge>}</Group>
        {withCorridor.length === 0
          ? <Text size="sm" c="dimmed">{data.step === "1mo" ? "Месячного коридора нет: суточные границы в месячную ширину не складываются, поэтому тревог по месяцам нет." : "Коридор не запрошен: границ hi в ответе нет, поэтому тревоги недоступны."} Это не значит, что превышений нет.</Text>
          : alerts.length === 0
            ? <Text size="sm" c="dimmed">Превышений нет: ни одна точка не вышла за верхнюю границу.</Text>
            : <Table withTableBorder={false} verticalSpacing={2} fz="xs">
                <Table.Thead><Table.Tr><Table.Th>Маршрут</Table.Th><Table.Th>Момент</Table.Th><Table.Th>Прогноз</Table.Th><Table.Th>Верх</Table.Th><Table.Th /></Table.Tr></Table.Thead>
                <Table.Tbody>{shown.map((alert) => <Table.Tr key={`${alert.route}-${alert.index}`}>
                  <Table.Td>{alert.route}</Table.Td>
                  <Table.Td>{momentLabel(data, alert.index)}</Table.Td>
                  <Table.Td>{amount(alert.value)}</Table.Td>
                  <Table.Td>{amount(alert.hi)}</Table.Td>
                  <Table.Td><Group gap={4} wrap="nowrap"><Button size="compact-xs" variant="subtle" onClick={() => selectIndex(alert.index)}>момент</Button>{data.series.length > 1 && <Button size="compact-xs" variant="subtle" onClick={() => onRouteSelect(alert.route)}>маршрут</Button>}</Group></Table.Td>
                </Table.Tr>)}</Table.Tbody>
              </Table>}
        {alerts.length > shown.length && <Text size="xs" c="dimmed">Показаны первые {shown.length} из {alerts.length}.</Text>}
        {withCorridor.length > 0 && withCorridor.length < data.series.length && <Text size="xs" c="dimmed">Границы пришли не по всем маршрутам: тревоги посчитаны по {withCorridor.length} из {data.series.length}.</Text>}
      </Stack>

      {data.warnings.length > 0 && <Stack gap={4}>
        <Text fw={500}>Предупреждения сервиса</Text>
        {data.warnings.map((warning, index) => <Alert key={index} color="yellow" p="xs">{warning.message}</Alert>)}
      </Stack>}
    </Stack>
  </Paper>;
}
