import { useState } from "react";
import { ActionIcon, Badge, Card, Group, NumberInput, Slider, Stack, Text } from "@mantine/core";
import { ConditionPassport } from "./condition-passport";
import { clampValue, conditionClassOf, conditionClassTitles, conditionStep, conditionTitleOf, formatNumber, formatSignedPercent, passportFor, type CatalogEntry, type ConditionInput } from "../domain/conditions";

function ConditionValue({ entry, value, onChange }: { entry: CatalogEntry; value: number; onChange: (value: number) => void }) {
  const [draft, setDraft] = useState<number | null>(null);
  const current = draft ?? value;
  const step = conditionStep(entry);
  const commit = (next: number) => {
    const clamped = clampValue(entry, Number.isFinite(next) ? next : value);
    setDraft(null);
    if (clamped !== value) onChange(clamped);
  };
  return (
    <Stack gap={4}>
      <Group gap="xs" wrap="nowrap">
        <NumberInput
          size="xs"
          w={110}
          value={current}
          min={entry.range.min}
          max={entry.range.max}
          step={step ?? undefined}
          decimalScale={step === null ? undefined : String(step).split(".")[1]?.length}
          onChange={(next) => setDraft(typeof next === "number" ? next : Number(next))}
          onBlur={() => commit(current)}
          onKeyDown={(event) => { if (event.key === "Enter") commit(current); }}
          aria-label={`Значение условия, ${entry.passport.unit}`}
        />
        <Text size="xs" c="dimmed">{entry.passport.unit}</Text>
      </Group>
      {step !== null && (
        <Slider
          size="sm"
          value={current}
          min={entry.range.min}
          max={entry.range.max}
          step={step}
          onChange={setDraft}
          onChangeEnd={commit}
          label={(shown) => `${formatNumber(shown)} ${entry.passport.unit}`}
        />
      )}
      <Text size="xs" c="dimmed">
        Пределы: от {formatNumber(entry.range.min)} до {formatNumber(entry.range.max)} {entry.passport.unit}
        {step === null ? ", шаг не задан: ввод непрерывный" : `, шаг ${formatNumber(step)}`}
      </Text>
    </Stack>
  );
}

export function ConditionCard({ condition, entry, routes, onChange, onRemove, scopeEditor, muted, note }: {
  condition: ConditionInput;
  entry?: CatalogEntry;
  routes: number[];
  onChange: (condition: ConditionInput) => void;
  onRemove: () => void;
  scopeEditor?: React.ReactNode;
  muted?: boolean;
  note?: React.ReactNode;
}) {
  const title = conditionTitleOf(condition.type, entry);
  const passport = entry && passportFor(entry, routes);
  return (
    <Card withBorder padding="xs" opacity={muted ? 0.65 : 1}>
      <Stack gap={6}>
        <Group justify="space-between" wrap="nowrap" gap={4}>
          <Group gap={6} wrap="nowrap" miw={0}>
            <Text size="sm" fw={600} truncate>{title}</Text>
            <Badge size="xs" variant="light" color={conditionClassOf(condition.type, entry) === "operational" ? "cyan" : "orange"}>
              {conditionClassTitles[conditionClassOf(condition.type, entry)]}
            </Badge>
          </Group>
          <Group gap={2} wrap="nowrap">
            {passport && <ConditionPassport passport={passport} title={title} />}
            <ActionIcon variant="subtle" color="red" size="sm" aria-label={`Удалить условие: ${title}`} onClick={onRemove}>x</ActionIcon>
          </Group>
        </Group>
        {entry
          ? <ConditionValue entry={entry} value={condition.value} onChange={(value) => onChange({ ...condition, value })} />
          : <Text size="xs" c="dimmed">Значение: {formatNumber(condition.value)}. Паспорт не получен, пределы ввода неизвестны.</Text>}
        {passport && <Text size="xs" c="dimmed">Эффект по измерениям: {formatSignedPercent(passport.effect_pct)}, интервал от {formatSignedPercent(passport.ci_pct[0])} до {formatSignedPercent(passport.ci_pct[1])}</Text>}
        {condition.type === "route_change" && <Text size="xs" c="dimmed">Применяется к маршруту целиком: участок в контракте не задан.</Text>}
        {note}
        {scopeEditor}
      </Stack>
    </Card>
  );
}
