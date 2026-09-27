import { useState } from "react";
import { Anchor, Button, Popover, Stack, Text } from "@mantine/core";
import { formatNumber, formatSignedPercent, horizonTitles, type Passport } from "../domain/conditions";

export function ConditionPassport({ passport, title, curve, auto }: { passport: Passport; title: string; curve?: number[][]; auto?: boolean }) {
  const [pinned, setPinned] = useState(false);
  const [hovered, setHovered] = useState(false);
  return (
    <Popover opened={pinned || hovered} width={300} position="right-start" withArrow shadow="md">
      <Popover.Target>
        <Button
          variant="subtle"
          size="compact-xs"
          aria-label={`Паспорт условия: ${title}`}
          onClick={() => setPinned((value) => !value)}
          onMouseEnter={() => setHovered(true)}
          onMouseLeave={() => setHovered(false)}
          onFocus={() => setHovered(true)}
          onBlur={() => setHovered(false)}
        >
          Паспорт
        </Button>
      </Popover.Target>
      <Popover.Dropdown>
        <Stack gap={4}>
          <Text size="sm" fw={600}>{title}</Text>
          <Text size="xs">Единица: {passport.unit}</Text>
          <Text size="xs">{passport.step === undefined ? "Шаг измерения не задан, значение вводится непрерывно" : `Шаг: ${formatNumber(passport.step)} ${passport.unit}`}</Text>
          <Text size="xs">
            Источник: {passport.source_url ? <Anchor href={passport.source_url} target="_blank" rel="noreferrer" inherit>{passport.source}</Anchor> : passport.source}
          </Text>
          <Text size="xs">Измеренный эффект: {formatSignedPercent(passport.effect_pct)}</Text>
          <Text size="xs">Интервал: от {formatSignedPercent(passport.ci_pct[0])} до {formatSignedPercent(passport.ci_pct[1])}</Text>
          {passport.sample !== undefined && <Text size="xs">Выборка: {formatNumber(passport.sample)}</Text>}
          <Text size="xs">Горизонты: {passport.horizons.map((horizon) => horizonTitles[horizon]).join(", ")}</Text>
          {curve && curve.length > 0 && <Text size="xs">Кривая эффекта: {curve.map(([value, effect]) => `${formatNumber(value)} ${passport.unit} - ${formatSignedPercent(effect)}`).join("; ")}. Между точками эффект интерполируется, повтор значения - ступенька.</Text>}
          {auto === false && <Text size="xs" c="dimmed">Из данных автоматически не применяется: условие задаёт диспетчер.</Text>}
        </Stack>
      </Popover.Dropdown>
    </Popover>
  );
}
