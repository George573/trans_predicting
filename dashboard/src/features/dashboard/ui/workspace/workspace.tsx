import { useState } from "react";
import { Alert, Badge, Button, Group, Paper, SimpleGrid, Skeleton, Stack, Text } from "@mantine/core";
import type { ForecastResponse } from "../../model/forecast-state";
import type { Route } from "../../model/forecast-state";
import { ForecastChart } from "./forecast-chart";
import { Summary } from "./summary";
import { PointBreakdown } from "./point-breakdown";
import { ConditionAudit } from "./condition-audit";
import { RoutesHeatmap } from "./routes-heatmap";
import { RouteMap } from "../route-map";
import { ExportControls } from "../export";
import type { ForecastRequest } from "../../model/forecast-state";
import type { RequestTelemetry, ForecastFailure } from "../../model/forecast-state";
import { sameRequest } from "../../forecast-request";
import { ProcessLoad, RequestHistory, RequestTimings } from "../observability";
import type { HistoryEntry } from "../../model/forecast-state";

export type Comparison = { title: string; data: ForecastResponse } | null;
type Props = { data: ForecastResponse | null; comparison: Comparison; recursive: string | null; loading: boolean; updatedAt: number | null; settling: boolean; failure: ForecastFailure | null; retry: () => void; layout: "map" | "mosaic" | "heat"; routes: Route[]; selected: number[]; selectedIndex: number; selectIndex: (index: number) => void; onRouteSelect: (route: number) => void; request: ForecastRequest; acceptedRequest: ForecastRequest | null; telemetry: RequestTelemetry; history: HistoryEntry[]; repeat: (request: ForecastRequest) => void };

function failureTitle(source: ForecastFailure["source"]) {
  if (source === "запрос") return "Запрос не отправлен";
  if (source === "сеть") return "Нет связи с сервисом";
  return "Сервис отклонил запрос";
}

export function Workspace({ data, comparison, recursive, loading, updatedAt, settling, failure, retry, layout, routes, selected, selectedIndex, selectIndex, onRouteSelect, request, acceptedRequest, telemetry, history, repeat }: Props) {
  const [pointRoute, setPointRoute] = useState<number | null>(null);
  const matched = !!acceptedRequest && sameRequest(request, acceptedRequest);
  const empty = !!data && data.series.length === 0;
  const zeros = !!data && data.series.length > 0 && data.series.every((series) => series.total === 0);
  return <Stack p="md" gap="md">
    <Group gap="xs">
      <Badge color={matched && !settling ? "teal" : "yellow"} variant="light">{matched && !settling ? "параметры совпадают с показанным" : "параметры изменились"}</Badge>
      <Badge color={loading ? "blue" : "gray"} variant="light">{loading ? "обновляется" : settling ? "ждёт ввода" : "запрос не идёт"}</Badge>
      <Badge color="gray" variant="light">{updatedAt === null ? "успешного ответа ещё не было" : `последний успешный ответ в ${new Date(updatedAt).toLocaleTimeString("ru-RU")}`}</Badge>
    </Group>
    {failure && <Alert color={failure.source === "запрос" ? "yellow" : "red"} title={failureTitle(failure.source)}>
      <Stack gap="xs">
        <Text size="sm">{failure.message}</Text>
        {failure.field && <Text size="xs" c="dimmed">Поле запроса: {failure.field}</Text>}
        {acceptedRequest && data && <Text size="xs" c="dimmed">Ниже прошлый успешный прогноз за {acceptedRequest.from === acceptedRequest.to ? acceptedRequest.from : `${acceptedRequest.from} - ${acceptedRequest.to}`}, он не отвечает текущим параметрам.</Text>}
        {failure.source !== "запрос" && <Group><Button size="xs" onClick={retry}>Повторить</Button></Group>}
      </Stack>
    </Alert>}
    {loading && !data && <Skeleton height={150} />}
    {empty && <Alert color="gray" title="Пустой ответ">Сервис вернул прогноз без рядов: для выбранных маршрутов и периода данных нет.</Alert>}
    {zeros && <Alert color="yellow" title="Нулевой прогноз">Ряды получены, но все значения равны нулю: посадок на выбранном отрезке модель не ожидает.</Alert>}
    {data && !empty && (layout === "heat" ? <>
      <RoutesHeatmap data={data} routes={routes} selectedIndex={selectedIndex} selectIndex={selectIndex} route={pointRoute} onRoute={setPointRoute} />
      <SimpleGrid cols={{ base: 1, lg: 2 }}><ForecastChart data={data} comparison={comparison} recursive={recursive} selectedIndex={selectedIndex} selectIndex={selectIndex} /><Summary data={data} comparison={comparison} selectIndex={selectIndex} onRouteSelect={onRouteSelect} /></SimpleGrid>
      <PointBreakdown data={data} request={acceptedRequest} selectedIndex={selectedIndex} route={pointRoute} onRoute={setPointRoute} />
    </> : layout === "map" ? <>
      <RouteMap routes={routes} selected={selected} onSelect={onRouteSelect} data={data} selectedIndex={selectedIndex} />
      <SimpleGrid cols={{ base: 1, lg: 2 }}><ForecastChart data={data} comparison={comparison} recursive={recursive} selectedIndex={selectedIndex} selectIndex={selectIndex} /><Summary data={data} comparison={comparison} selectIndex={selectIndex} onRouteSelect={onRouteSelect} /></SimpleGrid>
      <SimpleGrid cols={{ base: 1, lg: 2 }}><PointBreakdown data={data} request={acceptedRequest} selectedIndex={selectedIndex} route={pointRoute} onRoute={setPointRoute} /><ConditionAudit data={data} /></SimpleGrid>
    </> : <SimpleGrid cols={{ base: 1, lg: 2 }}>
      <ForecastChart data={data} comparison={comparison} recursive={recursive} selectedIndex={selectedIndex} selectIndex={selectIndex} /><Summary data={data} comparison={comparison} selectIndex={selectIndex} onRouteSelect={onRouteSelect} />
      <PointBreakdown data={data} request={acceptedRequest} selectedIndex={selectedIndex} route={pointRoute} onRoute={setPointRoute} />
      <ConditionAudit data={data} />
      <Paper p="md" withBorder><Text fw={600}>Сведения о запросе</Text><Text size="sm">Модель: {data.meta.model}</Text><Text size="sm">Версия бандла: {data.bundle}</Text><Text size="sm">Строк модели: {data.meta.rows}</Text></Paper>
    </SimpleGrid>)}
    {!loading && !failure && !data && <Text c="dimmed">Нет данных прогноза</Text>}
    <RequestTimings telemetry={telemetry} />
    <RequestHistory entries={history} repeat={repeat} />
    <ProcessLoad request={acceptedRequest} ready={!!data && !loading && !failure && matched} />
    <ExportControls request={request} acceptedRequest={acceptedRequest} ready={!!data && !empty && !loading && !failure && matched} />
  </Stack>;
}
