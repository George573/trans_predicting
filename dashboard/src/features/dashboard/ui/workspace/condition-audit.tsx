import { Badge, Stack, Table, Text } from "@mantine/core";
import { ConditionPassport, conditionTypeTitles, formatSignedPercent } from "@/features/scenario";
import { Panel } from "../panel";
import type { ForecastResponse } from "../../model/forecast-state";

const amount = (value: number) => Math.round(value).toLocaleString("ru-RU");

export function ConditionAudit({ data, title = "Аудит условий" }: { data: ForecastResponse; title?: string }) {
  const total = data.series.reduce((sum, series) => sum + series.total, 0);
  const baseTotal = data.series.reduce((sum, series) => sum + series.base_total, 0);
  const difference = total - baseTotal;

  return <Panel title={title} note="вклад сценария">
    <Stack gap="xs">
      <Text size="sm">Базовый прогноз {amount(baseTotal)} -&gt; итог {amount(total)}, разница {difference >= 0 ? "+" : ""}{amount(difference)}{baseTotal > 0 ? ` (${formatSignedPercent(100 * difference / baseTotal)})` : ""}</Text>
      {baseTotal === 0 && <Text size="xs" c="dimmed">Базовая сумма равна нулю, доля изменения не считается.</Text>}
      {data.conditions.length === 0
        ? <Text size="sm" c="dimmed">Условий в сценарии нет: итог равен базовому прогнозу.</Text>
        : <>
          <Table.ScrollContainer minWidth={420} type="native"><Table withTableBorder={false} verticalSpacing={2} fz="xs">
            <Table.Thead><Table.Tr><Table.Th>Условие</Table.Th><Table.Th>Состояние</Table.Th><Table.Th>Множитель</Table.Th><Table.Th>Вклад</Table.Th><Table.Th>Точек</Table.Th><Table.Th /></Table.Tr></Table.Thead>
            <Table.Tbody>{data.conditions.map((condition) => {
              const reason = data.warnings.find((warning) => warning.condition_id === condition.id);
              return <Table.Tr key={condition.id}>
                <Table.Td>{conditionTypeTitles[condition.type]} · {condition.id}</Table.Td>
                <Table.Td>{condition.applied ? <Badge color="teal" size="xs">применено</Badge> : <Badge color="gray" size="xs">не применено</Badge>}{!condition.applied && <Text size="xs" c="dimmed">{reason?.message ?? "Причина в ответе не указана"}</Text>}</Table.Td>
                <Table.Td>{condition.factor}</Table.Td>
                <Table.Td>{condition.applied ? formatSignedPercent(condition.contribution_pct) : "-"}</Table.Td>
                <Table.Td>{condition.points}</Table.Td>
                <Table.Td><ConditionPassport passport={condition.passport} title={conditionTypeTitles[condition.type]} /></Table.Td>
              </Table.Tr>;
            })}</Table.Tbody>
          </Table></Table.ScrollContainer>
          <Text size="xs" c="dimmed">Вклады не складываются в разницу сумм: условия перемножаются, их области могут пересекаться, а произведение ограничивается коридором корректировки. Проценты взяты из ответа сервиса как есть.</Text>
        </>}
    </Stack>
  </Panel>;
}
