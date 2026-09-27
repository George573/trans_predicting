import { MultiSelect, Stack, Text } from "@mantine/core";
import type { RouteNumber, Scope } from "../../domain/conditions";
import { describeRoutes, withRoutes } from "../../domain/scope";

export function ScopeRoutes({ scope, availableRoutes, requestRoutes, onChange }: {
  scope?: Scope;
  availableRoutes: RouteNumber[];
  requestRoutes: RouteNumber[];
  onChange: (scope: Scope) => void;
}) {
  const selected = scope?.routes ?? [];
  const outside = selected.filter((route) => requestRoutes.length > 0 && !requestRoutes.includes(route));
  return (
    <Stack gap={2}>
      <MultiSelect
        size="xs"
        label="Где"
        placeholder={selected.length === 0 ? "Все маршруты запроса" : undefined}
        data={availableRoutes.map((route) => ({ value: String(route), label: `Маршрут ${route}` }))}
        value={selected.map(String)}
        onChange={(values) => onChange(withRoutes(scope, values.map((value) => Number(value) as RouteNumber)))}
        maxValues={10}
        clearable
        searchable
      />
      <Text size="xs" c="dimmed">Область: {describeRoutes(scope)}</Text>
      {outside.length > 0 && <Text size="xs" c="yellow">Вне текущего выбора объекта: {outside.join(", ")}. Область условия сохранена как задана.</Text>}
    </Stack>
  );
}
