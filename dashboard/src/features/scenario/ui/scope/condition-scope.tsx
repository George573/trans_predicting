import { Button, Stack, Text } from "@mantine/core";
import { conditionClassOf, conditionLimit, nextConditionId, type CatalogEntry, type ConditionInput, type ForecastResponse, type Horizon, type RouteNumber, type Scope } from "../../domain/conditions";
import { warningOf } from "../../domain/applicability";
import { describeHours, isOvernight, scopeHours, splitOvernight, withHours } from "../../domain/scope";
import { HoursScope } from "./hours-scope";
import { ScopeRoutes } from "./scope-routes";
import { WeekScope } from "./week-scope";
import { DatesScope } from "./dates-scope";

export type ConditionScopeProps = {
  condition: ConditionInput;
  conditions: ConditionInput[];
  onConditionsChange: (conditions: ConditionInput[]) => void;
  horizon: Horizon;
  from: string;
  to: string;
  availableRoutes: RouteNumber[];
  requestRoutes: RouteNumber[];
  entry?: CatalogEntry;
  response: ForecastResponse | null;
};

export function ConditionScope({ condition, conditions, onConditionsChange, horizon, from, to, availableRoutes, requestRoutes, entry, response }: ConditionScopeProps) {
  const hours = scopeHours(condition.scope);
  const update = (scope: Scope) => onConditionsChange(conditions.map((item) => (item.id === condition.id ? { ...item, scope } : item)));
  const full = conditions.length >= conditionLimit;
  const parts = hours && isOvernight(hours[0], hours[1]) ? splitOvernight(condition, hours, from, to, conditions) : null;
  const splitAllowed = parts !== null && !full;
  const duplicate = () => onConditionsChange([...conditions, { ...condition, id: nextConditionId(conditions), scope: structuredClone(condition.scope ?? {}) }]);
  return (
    <Stack gap={6}>
      <ScopeRoutes scope={condition.scope} availableRoutes={availableRoutes} requestRoutes={requestRoutes} onChange={update} />
      {(horizon === "month" || horizon === "season") && (
        <DatesScope scope={condition.scope} from={from} to={to} season={horizon === "season"} onChange={update} />
      )}
      {horizon === "week" && (
        <WeekScope
          scope={condition.scope}
          from={from}
          to={to}
          operational={conditionClassOf(condition.type, entry) === "operational"}
          emptyWarning={warningOf(response, condition.id, "condition_scope_empty")?.message}
          onChange={update}
        />
      )}
      {(horizon === "month" || horizon === "season") && (
        <Text size="xs" c="dimmed">Часовая область сохранена: {describeHours(hours)}. Она применится при возврате на сутки или неделю.</Text>
      )}
      {(horizon === "day" || horizon === "week") && <HoursScope
        value={hours}
        onChange={(next) => update(withHours(condition.scope, next))}
        onSplit={splitAllowed ? () => onConditionsChange(conditions.flatMap((item) => (item.id === condition.id ? parts : [item]))) : undefined}
        splitReason={parts === null ? "Вторые сутки выходят за период прогноза, разбить нельзя" : `Лимит ${conditionLimit} условий исчерпан, разбить нельзя`}
      />}
      {(horizon === "day" || horizon === "week") && (full
        ? <Text size="xs" c="dimmed">Для других часов нужно отдельное условие, но лимит {conditionLimit} исчерпан.</Text>
        : <Button size="compact-xs" variant="subtle" onClick={duplicate}>Отдельное условие с другими часами</Button>)}
    </Stack>
  );
}
