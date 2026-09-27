import { Alert, Card, Group, Stack, Switch, Text } from "@mantine/core";
import { clampedWarning, conditionOverlaps, appliedCondition, notAppliedIds, pluralConditions, warningsFor } from "../domain/applicability";
import { formatNumber, formatSignedPercent, horizonTitles, type ConditionInput, type ForecastResponse, type Horizon } from "../domain/conditions";
import { describeHours } from "../domain/scope";

export function ApplicabilityBanner({ response, horizon }: { response: ForecastResponse | null; horizon: Horizon }) {
  const ids = notAppliedIds(response);
  if (ids.length === 0) return null;
  const reasons = [...new Set(ids.flatMap((id) => warningsFor(response, id).map((warning) => warning.message)))];
  return (
    <Alert color="yellow" p="xs" title={`${pluralConditions(ids.length)} на горизонте ${horizonTitles[horizon]}`}>
      <Stack gap={2}>
        <Text size="xs">Условия сохранены и вернутся в работу на сутках и неделе: {ids.join(", ")}.</Text>
        {reasons.map((reason) => <Text key={reason} size="xs">{reason}</Text>)}
      </Stack>
    </Alert>
  );
}

export function ConditionAppliedNote({ response, id, stale }: { response: ForecastResponse | null; id: string; stale: boolean }) {
  const applied = appliedCondition(response, id);
  const warnings = warningsFor(response, id);
  if (!applied) return warnings.length > 0 ? <Text size="xs" c="yellow">{warnings.map((warning) => warning.message).join(" ")}</Text> : null;
  return (
    <Stack gap={2}>
      <Text size="xs" c={applied.applied ? "dimmed" : "yellow"}>
        {applied.applied
          ? `Применено: точек в области ${applied.points}, вклад ${formatSignedPercent(applied.contribution_pct)}, коэффициент ${formatNumber(applied.factor)}`
          : "Сохранено, но не применено на этом горизонте. Прогноз не изменён, поправка не обнулена."}
        {stale && " (ответ обновляется)"}
      </Text>
      {!applied.applied && warnings.map((warning) => <Text key={warning.code} size="xs" c="yellow">{warning.message}</Text>)}
    </Stack>
  );
}

export function OperationalExplanation({ horizon }: { horizon: Horizon }) {
  return (
    <Card withBorder padding="xs">
      <Stack gap={2}>
        <Text size="xs" fw={500}>Оперативные условия на горизонте {horizonTitles[horizon]} не применяются</Text>
        <Text size="xs" c="dimmed">Измеренный эффект дождя, температуры, события и задержки меньше ширины коридора на этом кванте. Условия остаются в сценарии и снова работают на сутках и неделе.</Text>
        <Text size="xs" c="dimmed">Структурные условия действуют на всех горизонтах: их можно добавлять и править здесь же.</Text>
      </Stack>
    </Card>
  );
}

export function SeasonModelNote({ horizon }: { horizon: Horizon }) {
  return (
    <Card withBorder padding="xs">
      <Stack gap={4}>
        <Group justify="space-between" wrap="nowrap">
          <Text size="xs" fw={500}>Сезонность и праздничный блок: учтены моделью</Text>
          <Switch size="xs" checked disabled aria-label="Сезонность учтена моделью, ручная поправка недоступна" />
        </Group>
        <Text size="xs" c="dimmed">Контрол заблокирован сознательно: множитель поверх модельного признака считал бы эффект дважды.</Text>
        <Text size="xs" fw={500}>Погода на горизонте {horizonTitles[horizon]}</Text>
        <Text size="xs" c="dimmed">В сервисном режиме погоды в признаках модели нет: она приходит условием, а оперативные условия на этом горизонте не применяются.</Text>
        <Text size="xs" c="dimmed">Климатическая норма периода в контракте не предоставлена, поэтому числа здесь нет.</Text>
      </Stack>
    </Card>
  );
}

export function OverlapList({ conditions, response, from, to }: { conditions: ConditionInput[]; response: ForecastResponse | null; from: string; to: string }) {
  const overlaps = conditionOverlaps(conditions, response, from, to);
  const clamped = clampedWarning(response);
  if (overlaps.length === 0) return null;
  return (
    <Card withBorder padding="xs">
      <Stack gap={4}>
        <Text size="xs" fw={500}>Пересечения условий</Text>
        {overlaps.map((overlap) => (
          <Stack key={overlap.ids.join("-")} gap={0}>
            <Text size="xs">{overlap.ids.join(" и ")}: дат {overlap.dates.length}, {describeHours(overlap.hours)}</Text>
            <Text size="xs" c="dimmed">
              {overlap.factor === null
                ? "Композиция недоступна: на этом горизонте условия не применены"
                : `Композиция коэффициентов справочно: ${formatNumber(overlap.factor)}`}
            </Text>
          </Stack>
        ))}
        {clamped && <Text size="xs" c="yellow">{clamped.message} Итоговые посадки берутся из ответа сервиса, произведение коэффициентов их не заменяет.</Text>}
      </Stack>
    </Card>
  );
}
