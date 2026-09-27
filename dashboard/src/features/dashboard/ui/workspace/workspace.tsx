import { useState } from "react";
import { Alert, Badge, Box, Button, Flex, Group, SimpleGrid, Skeleton, Stack, Text } from "@mantine/core";
import { Panel } from "../panel";
import type { ForecastResponse } from "../../model/forecast-state";
import type { Route } from "../../model/forecast-state";
import { ForecastChart } from "./forecast-chart";
import { Summary } from "./summary";
import { PointBreakdown } from "./point-breakdown";
import { ConditionAudit } from "./condition-audit";
import { RoutesHeatmap } from "./routes-heatmap";
import { DayPeriods, UsualDeviation } from "./simple-charts";
import { RouteMap } from "../route-map";
import { RouteCharts } from "../route-charts";
import type { ForecastRequest } from "../../model/forecast-state";
import type { ForecastFailure } from "../../model/forecast-state";
import { sameRequest } from "../../forecast-request";
import { formatDate } from "@/features/scenario";

export type Comparison = { title: string; data: ForecastResponse } | null;
type Props = { data: ForecastResponse | null; comparison: Comparison; recursive: string | null; loading: boolean; updatedAt: number | null; settling: boolean; failure: ForecastFailure | null; retry: () => void; layout: "map" | "mosaic" | "heat"; routes: Route[]; selected: number[]; selectedIndex: number; selectIndex: (index: number) => void; onRouteSelect: (route: number) => void; request: ForecastRequest; acceptedRequest: ForecastRequest | null };

function failureTitle(source: ForecastFailure["source"]) {
  if (source === "запрос") return "Запрос не отправлен";
  if (source === "сеть") return "Нет связи с сервисом";
  return "Сервис отклонил запрос";
}

export function Workspace({ data, comparison, recursive, loading, updatedAt, settling, failure, retry, layout, routes, selected, selectedIndex, selectIndex, onRouteSelect, request, acceptedRequest }: Props) {
  const [pointRoute, setPointRoute] = useState<number | null>(null);
  const matched = !!acceptedRequest && sameRequest(request, acceptedRequest);
  const empty = !!data && data.series.length === 0;
  const zeros = !!data && data.series.length > 0 && data.series.every((series) => series.total === 0);
  const head = <>
    <Group gap="xs">
      <Badge color={matched && !settling ? "teal" : "yellow"} variant="light">{matched && !settling ? "параметры совпадают с показанным" : "параметры изменились"}</Badge>
      <Badge color={loading ? "blue" : "gray"} variant="light">{loading ? "обновляется" : settling ? "ждёт ввода" : "запрос не идёт"}</Badge>
      <Badge color="gray" variant="light">{updatedAt === null ? "успешного ответа ещё не было" : `последний успешный ответ в ${new Date(updatedAt).toLocaleTimeString("ru-RU")}`}</Badge>
    </Group>
    {failure && <Alert color={failure.source === "запрос" ? "yellow" : "red"} title={failureTitle(failure.source)}>
      <Stack gap="xs">
        <Text size="sm">{failure.message}</Text>
        {failure.field && <Text size="xs" c="dimmed">Поле запроса: {failure.field}</Text>}
        {acceptedRequest && data && <Text size="xs" c="dimmed">Ниже прошлый успешный прогноз за {acceptedRequest.from === acceptedRequest.to ? formatDate(acceptedRequest.from) : `${formatDate(acceptedRequest.from)} - ${formatDate(acceptedRequest.to)}`}, он не отвечает текущим параметрам.</Text>}
        {failure.source !== "запрос" && <Group><Button size="xs" onClick={retry}>Повторить</Button></Group>}
      </Stack>
    </Alert>}
    {loading && !data && <Skeleton height={150} />}
    {empty && <Alert color="gray" title="Пустой ответ">Сервис вернул прогноз без рядов: для выбранных маршрутов и периода данных нет.</Alert>}
    {zeros && <Alert color="yellow" title="Нулевой прогноз">Ряды получены, но все значения равны нулю: посадок на выбранном отрезке модель не ожидает.</Alert>}
  </>;
  if (layout === "map") return <Flex direction={{ base: "column", md: "row" }} h={{ base: "auto", md: "100%" }} gap={8} p={8}>
    <Stack gap={8} w={{ base: "100%", md: "clamp(400px, 38%, 640px)" }} h={{ base: "auto", md: "100%" }} style={{ flexShrink: 0, overflowY: "auto" }}>
      {head}
      {data && !empty && <>
        <ForecastChart data={data} comparison={comparison} recursive={recursive} selectedIndex={selectedIndex} selectIndex={selectIndex} />
        <UsualDeviation data={data} selectIndex={selectIndex} />
        <DayPeriods data={data} />
        <Summary data={data} comparison={comparison} selectIndex={selectIndex} onRouteSelect={onRouteSelect} />
        {data.series.length > 1 && <RouteCharts data={data} comparison={comparison} routes={routes} onRouteSelect={onRouteSelect} stacked />}
        <PointBreakdown data={data} request={acceptedRequest} selectedIndex={selectedIndex} route={pointRoute} onRoute={setPointRoute} />
        <ConditionAudit data={data} />
      </>}
      {!loading && !failure && !data && <Text c="dimmed">Нет данных прогноза</Text>}
    </Stack>
    {data && !empty && <Box flex={1} miw={0} h={{ base: "auto", md: "100%" }}>
      <RouteMap routes={routes} selected={selected} onSelect={onRouteSelect} data={data} selectedIndex={selectedIndex} height="calc(100dvh - 104px)" />
    </Box>}
  </Flex>;
  return <Stack p={8} gap={8}>
    {head}
    {data && !empty && (layout === "heat" ? <>
      <RoutesHeatmap data={data} routes={routes} selectedIndex={selectedIndex} selectIndex={selectIndex} route={pointRoute} onRoute={setPointRoute} />
      <SimpleGrid cols={{ base: 1, lg: 2 }}><ForecastChart data={data} comparison={comparison} recursive={recursive} selectedIndex={selectedIndex} selectIndex={selectIndex} /><Summary data={data} comparison={comparison} selectIndex={selectIndex} onRouteSelect={onRouteSelect} /></SimpleGrid>
      <PointBreakdown data={data} request={acceptedRequest} selectedIndex={selectedIndex} route={pointRoute} onRoute={setPointRoute} />
    </> : <SimpleGrid cols={{ base: 1, lg: 2 }}>
      <ForecastChart data={data} comparison={comparison} recursive={recursive} selectedIndex={selectedIndex} selectIndex={selectIndex} /><Summary data={data} comparison={comparison} selectIndex={selectIndex} onRouteSelect={onRouteSelect} />
      <PointBreakdown data={data} request={acceptedRequest} selectedIndex={selectedIndex} route={pointRoute} onRoute={setPointRoute} />
      <ConditionAudit data={data} />
      <UsualDeviation data={data} selectIndex={selectIndex} />
      <DayPeriods data={data} />
      <Panel title="Сведения о запросе"><Text size="sm">Модель: {data.meta.model}</Text><Text size="sm">Версия бандла: {data.bundle}</Text><Text size="sm">Строк модели: {data.meta.rows}</Text></Panel>
    </SimpleGrid>)}
    {data && !empty && <RouteCharts data={data} comparison={comparison} routes={routes} onRouteSelect={onRouteSelect} />}
    {!loading && !failure && !data && <Text c="dimmed">Нет данных прогноза</Text>}
  </Stack>;
}
