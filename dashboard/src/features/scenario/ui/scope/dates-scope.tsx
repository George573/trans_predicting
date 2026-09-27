import { useState } from "react";
import { Button, Group, SegmentedControl, Stack, Text } from "@mantine/core";
import { DatePicker } from "@mantine/dates";
import type { Scope } from "../../domain/conditions";
import { datesLimit, datesOfWeekdays, datesOutsidePeriod, expandDateRange, fitsDatesLimit, formatDate, monthDates, monthTitle, seasonMonths, weekdayTemplates, withDates } from "../../domain/scope";

function SeasonStrip({ selected, onToggle }: { selected: string[]; onToggle: (month: string) => void }) {
  return (
    <Stack gap={2}>
      <Text size="xs" fw={500}>Полоса сезона: ноябрь 2025 - апрель 2026</Text>
      <Group gap={2}>
        {seasonMonths.map((month) => (
          <Button
            key={month}
            size="compact-xs"
            px={6}
            variant={selected.includes(month) ? "filled" : "light"}
            aria-pressed={selected.includes(month)}
            onClick={() => onToggle(month)}
          >
            {monthTitle(month).slice(0, 3)} {month.slice(2, 4)}
          </Button>
        ))}
      </Group>
    </Stack>
  );
}

export function DatesScope({ scope, from, to, season, onChange }: {
  scope?: Scope;
  from: string;
  to: string;
  season: boolean;
  onChange: (scope: Scope) => void;
}) {
  const [mode, setMode] = useState("dates");
  const [range, setRange] = useState<[string | null, string | null]>([null, null]);
  const explicit = scope?.dates ?? null;
  const chosen = explicit ?? [];
  const outside = datesOutsidePeriod(scope, from, to);
  const monthsSelected = seasonMonths.filter((month) => monthDates(month).every((date) => chosen.includes(date)));

  const apply = (dates: string[]) => { if (fitsDatesLimit(dates)) onChange(withDates(scope, dates)); };

  return (
    <Stack gap={4}>
      <Text size="xs" fw={500}>Когда: явные даты</Text>
      <Group gap={2}>
        {weekdayTemplates.map((template) => (
          <Button key={template.label} size="compact-xs" variant="light" onClick={() => apply(datesOfWeekdays(from, to, template.weekdays))}>
            {template.label}
          </Button>
        ))}
        <Button size="compact-xs" variant="light" onClick={() => apply(expandDateRange(from, to))}>Весь период</Button>
        <Button size="compact-xs" variant="subtle" onClick={() => onChange(withDates(scope, []))}>Сбросить</Button>
      </Group>
      <Text size="xs" c="dimmed">Будни и выходные это дни недели, а не производственный календарь: переносы и праздники сюда не входят.</Text>
      {season && (
        <SeasonStrip
          selected={monthsSelected}
          onToggle={(month) => {
            const dates = monthDates(month);
            apply(monthsSelected.includes(month) ? chosen.filter((date) => !dates.includes(date)) : [...chosen, ...dates]);
          }}
        />
      )}
      <SegmentedControl
        size="xs"
        fullWidth
        value={mode}
        onChange={setMode}
        data={[{ value: "dates", label: "Свои даты" }, { value: "range", label: "Диапазон" }]}
      />
      {mode === "dates"
        ? <DatePicker type="multiple" size="xs" value={chosen} minDate={from} maxDate={to} onChange={apply} />
        : <DatePicker
            type="range"
            size="xs"
            value={range}
            minDate={from}
            maxDate={to}
            onChange={(next) => {
              setRange(next);
              if (next[0] && next[1]) apply(expandDateRange(next[0], next[1]));
            }}
          />}
      <Text size="xs" c={explicit === null ? "dimmed" : undefined}>
        {explicit === null ? "dates не задан: все даты периода" : `Отправляется dates: ${explicit.length} дат, от ${formatDate(explicit[0])} до ${formatDate(explicit[explicit.length - 1])}`}
      </Text>
      {!fitsDatesLimit(chosen) && <Text size="xs" c="yellow">Контракт принимает не больше {datesLimit} дат.</Text>}
      {outside.length > 0 && <Text size="xs" c="yellow">Вне текущего периода сохранено дат: {outside.length}. Область не изменена.</Text>}
    </Stack>
  );
}
