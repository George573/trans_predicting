import { Button, Group, Stack, Text } from "@mantine/core";
import type { Scope } from "../../domain/conditions";
import { datesOutsidePeriod, effectiveDates, expandDateRange, operationalScenarioDays, weekdayNumbers, weekdayOf, weekdayTitles, withDates, withWeekdays } from "../../domain/scope";

export function WeekScope({ scope, from, to, operational, emptyWarning, onChange }: {
  scope?: Scope;
  from: string;
  to: string;
  operational: boolean;
  emptyWarning?: string;
  onChange: (scope: Scope) => void;
}) {
  const period = expandDateRange(from, to);
  const explicit = scope?.dates ?? null;
  const weekdays = scope?.weekdays ?? null;
  const intersection = effectiveDates(scope, from, to);
  const outside = datesOutsidePeriod(scope, from, to);

  const toggleDate = (date: string) => {
    const base = explicit ?? period;
    onChange(withDates(scope, base.includes(date) ? base.filter((item) => item !== date) : [...base, date]));
  };

  return (
    <Stack gap={4}>
      <Text size="xs" fw={500}>Дни недели: понедельник 1, воскресенье 7</Text>
      <Group gap={2}>
        {weekdayNumbers.map((weekday) => (
          <Button
            key={weekday}
            size="compact-xs"
            px={6}
            variant={weekdays?.includes(weekday) ? "filled" : "light"}
            aria-pressed={weekdays?.includes(weekday) ?? false}
            onClick={() => onChange(withWeekdays(scope, weekdays?.includes(weekday) ? weekdays.filter((item) => item !== weekday) : [...(weekdays ?? []), weekday]))}
          >
            {weekdayTitles[weekday - 1]}
          </Button>
        ))}
      </Group>
      <Text size="xs" c="dimmed">{weekdays ? `Отправляется weekdays: ${weekdays.join(", ")}` : "weekdays не задан: все дни недели"}</Text>
      <Text size="xs" fw={500}>Даты периода</Text>
      <Group gap={2}>
        {period.map((date) => {
          const selected = explicit === null || explicit.includes(date);
          const counted = intersection.includes(date);
          return (
            <Button
              key={date}
              size="compact-xs"
              px={6}
              variant={counted ? "filled" : selected ? "light" : "default"}
              aria-pressed={selected}
              onClick={() => toggleDate(date)}
            >
              {date.slice(8)}.{date.slice(5, 7)} {weekdayTitles[weekdayOf(date) - 1]}
            </Button>
          );
        })}
      </Group>
      {explicit === null && <Text size="xs" c="dimmed">dates не задан: все даты периода</Text>}
      {explicit !== null && <Text size="xs" c="dimmed">Отправляется dates: {explicit.length} дат</Text>}
      <Text size="xs" c={intersection.length === 0 ? "yellow" : "dimmed"}>
        {intersection.length === 0 ? "Пересечение дат и дней недели пусто" : `Пересечение: ${intersection.length} дат, общий диапазон часов применяется к каждой`}
      </Text>
      {emptyWarning && <Text size="xs" c="yellow">{emptyWarning}</Text>}
      {outside.length > 0 && <Text size="xs" c="yellow">Вне текущего периода сохранено дат: {outside.length} ({outside[0]} и далее). Область не изменена.</Text>}
      {operational && intersection.length > operationalScenarioDays && (
        <Text size="xs" c="yellow">Оперативный сценарий рассчитан не больше чем на {operationalScenarioDays} суток, выбрано {intersection.length}.</Text>
      )}
    </Stack>
  );
}
