import { Badge, Group, Stack, Text } from "@mantine/core";
import dayjs from "dayjs";
import { splitSeasonRequest, type SeasonRequest } from "../domain/season";

export function SeasonRequestPanel({ request, active }: { request: SeasonRequest; active: boolean }) {
  const parts = splitSeasonRequest(request);
  const seasonDays = parts.reduce((total, part) => total + part.days, 0);

  return (
    <Stack gap="xs">
      <Group justify="space-between">
        <Text fw={600}>Сезон</Text>
        <Badge variant="light" color={active ? "teal" : "gray"}>{active ? "показан" : "горизонт не выбран"}</Badge>
      </Group>

      <Text size="sm" c="dimmed">
        Ноябрь 2025 - апрель 2026, {seasonDays} суток. Горизонт «Сезон» запрашивает {parts.length} части по суткам,
        каждая под консервативным пределом 20 000 маршруто-часов, и показывает результат шестью месячными суммами.
      </Text>

      <Stack gap={6}>
        {parts.map((part) => (
          <Stack key={part.label} gap={0}>
            <Text size="sm">{part.label}</Text>
            <Text size="xs" c="dimmed">
              {dayjs(part.request.from).format("DD.MM.YYYY")} - {dayjs(part.request.to).format("DD.MM.YYYY")},{" "}
              {part.days} суток, {part.routeHours.toLocaleString("ru-RU")} маршруто-часов
            </Text>
          </Stack>
        ))}
      </Stack>

      <Text size="xs" c="dimmed">
        Части публикуются только вместе и только при одинаковой версии модели: ошибка любой части не превращается в
        нулевой месяц. Месячный коридор не строится - суточные границы нельзя сложить в месячную ширину, поэтому
        месячных тревог тоже нет. Выгрузка CSV и XLSX для сезона всей сети не предусмотрена.
      </Text>
    </Stack>
  );
}
