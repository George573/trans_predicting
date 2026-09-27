import { Button, Group, Paper, Stack, Text } from "@mantine/core";
import type { ForecastRequest, HistoryEntry } from "../../model/forecast-state";

export function RequestHistory({ entries, repeat }: { entries: HistoryEntry[]; repeat: (request: ForecastRequest) => void }) {
  return <Paper p="md" withBorder><Stack gap="xs"><Text fw={600}>Последние запросы</Text>
    {entries.length === 0 && <Text size="xs" c="dimmed">Запросов пока нет</Text>}
    {entries.map((entry) => <Group key={entry.id} justify="space-between" wrap="nowrap">
      <div><Text size="xs">{new Date(entry.at).toLocaleTimeString("ru-RU")} · {entry.request.routes?.length ? `маршруты ${entry.request.routes.join(", ")}` : "вся сеть"} · {entry.request.from} - {entry.request.to}</Text><Text size="xs" c="dimmed">{entry.status}{entry.durationMs !== null && ` · ${entry.durationMs.toFixed(1)} мс`}</Text></div>
      <Button size="compact-xs" variant="light" onClick={() => repeat(entry.request)}>Повторить</Button>
    </Group>)}
  </Stack></Paper>;
}
