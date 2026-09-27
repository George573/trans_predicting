import { Alert, Button, Group, Menu, Skeleton, Stack, Text, Title } from "@mantine/core";
import { useConditionsCatalog } from "../api/catalog";
import { ConditionCard } from "./condition-card";
import { conditionClassOf, conditionClassTitles, conditionLimit, conditionTitleOf, createCondition, type CatalogEntry, type ConditionClass, type ConditionInput, type ForecastResponse, type Horizon, type RouteNumber } from "../domain/conditions";
import { ConditionScope } from "./scope";
import { ApplicabilityBanner, ConditionAppliedNote, OperationalExplanation, OverlapList, SeasonModelNote } from "./applicability";
import { appliedCondition } from "../domain/applicability";

function AddConditionMenu({ entries, disabled, onAdd }: { entries: CatalogEntry[]; disabled: boolean; onAdd: (entry: CatalogEntry) => void }) {
  if (entries.length === 0) return null;
  return (
    <Menu position="bottom-start" withinPortal>
      <Menu.Target>
        <Button size="compact-xs" variant="light" disabled={disabled}>Добавить</Button>
      </Menu.Target>
      <Menu.Dropdown>
        {entries.map((entry) => (
          <Menu.Item key={entry.type} onClick={() => onAdd(entry)}>{conditionTitleOf(entry.type, entry)}</Menu.Item>
        ))}
      </Menu.Dropdown>
    </Menu>
  );
}

export type ScenarioPanelProps = {
  conditions: ConditionInput[];
  requestRoutes: RouteNumber[];
  availableRoutes: RouteNumber[];
  horizon: Horizon;
  from: string;
  to: string;
  response: ForecastResponse | null;
  stale: boolean;
  onChange: (conditions: ConditionInput[]) => void;
};

export function ScenarioPanel({ conditions, requestRoutes, availableRoutes, horizon, from, to, response, stale, onChange }: ScenarioPanelProps) {
  const catalog = useConditionsCatalog();
  const entryOf = (type: ConditionInput["type"]) => catalog.entries.find((entry) => entry.type === type);
  const full = conditions.length >= conditionLimit;
  const classes: ConditionClass[] = ["operational", "structural"];

  return (
    <Stack gap="sm">
      <Group justify="space-between">
        <Title order={6}>Условия</Title>
        <Text size="xs" c="dimmed">{conditions.length} из {conditionLimit}</Text>
      </Group>
      <ApplicabilityBanner response={response} horizon={horizon} />
      {(horizon === "month" || horizon === "season") && <OperationalExplanation horizon={horizon} />}
      {(horizon === "month" || horizon === "season") && <SeasonModelNote horizon={horizon} />}
      {catalog.loading && <Skeleton height={80} />}
      {catalog.error && <Alert color="red" p="xs">{catalog.error}<Button size="compact-xs" ml="sm" onClick={catalog.retry}>Повторить</Button></Alert>}
      {!catalog.loading && !catalog.error && catalog.entries.length === 0 && (
        <Text size="xs" c="dimmed">Каталог условий пуст. Прогноз считается без условий.</Text>
      )}
      {full && <Text size="xs" c="yellow">Контракт принимает не больше {conditionLimit} условий. Удалите ненужное, чтобы добавить новое.</Text>}
      {classes.map((conditionClass) => (
        <Stack key={conditionClass} gap={6}>
          <Group justify="space-between">
            <Text size="sm" fw={500}>{conditionClassTitles[conditionClass]}</Text>
            <AddConditionMenu
              entries={catalog.entries.filter((entry) => conditionClassOf(entry.type, entry) === conditionClass)}
              disabled={full}
              onAdd={(entry) => onChange([...conditions, createCondition(entry, conditions)])}
            />
          </Group>
          {conditions.filter((condition) => conditionClassOf(condition.type, entryOf(condition.type)) === conditionClass).map((condition) => (
            <ConditionCard
              key={condition.id}
              condition={condition}
              entry={entryOf(condition.type)}
              onChange={(next) => onChange(conditions.map((item) => (item.id === condition.id ? next : item)))}
              onRemove={() => onChange(conditions.filter((item) => item.id !== condition.id))}
              muted={appliedCondition(response, condition.id)?.applied === false}
              note={<ConditionAppliedNote response={response} id={condition.id} stale={stale} />}
              scopeEditor={
                <ConditionScope
                  condition={condition}
                  conditions={conditions}
                  onConditionsChange={onChange}
                  horizon={horizon}
                  from={from}
                  to={to}
                  availableRoutes={availableRoutes}
                  requestRoutes={requestRoutes}
                  entry={entryOf(condition.type)}
                  response={response}
                />
              }
            />
          ))}
        </Stack>
      ))}
      <OverlapList conditions={conditions} response={response} from={from} to={to} />
    </Stack>
  );
}
