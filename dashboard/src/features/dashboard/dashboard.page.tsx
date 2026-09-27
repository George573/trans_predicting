import { useCallback, useEffect, useState } from "react";
import { Affix, Alert, Badge, Box, Button, Divider, Drawer, Flex, Group, ScrollArea, Stack, Title } from "@mantine/core";
import { apiBaseUrl, fetchClient } from "@/shared/api/instance";
import { useComparison, useForecast, type Route } from "./model/forecast-state";
import { fitPeriod, forecastDomain, sameRequest } from "./forecast-request";
import { PeriodControls } from "./ui/period";
import { ScenarioPanel } from "@/features/scenario";
import { ModelPanel } from "./ui/model-panel";
import { Workspace } from "./ui/workspace";
import { modelTitles, recursiveFrom, type ModelInfo, type ModelName } from "./domain/model";
import { ObjectPicker } from "./ui/object-picker";
import { ExportControls } from "./ui/export";
import { ViewPicker, type Layout } from "./ui/view-picker";

function useServiceModels() {
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [defaultModel, setDefaultModel] = useState<ModelName>("cnn");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const reload = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      const health = await fetch(`${apiBaseUrl}/healthz`);
      if (!health.ok) {
        const body = await health.json().catch(() => null);
        throw new Error(body?.error?.message ?? "Сервис пока не готов");
      }
      const result = await fetchClient.GET("/api/v1/model");
      if (result.error || !result.data) throw new Error(result.error?.error.message ?? "Не удалось получить сведения о моделях");
      setModels(result.data.models);
      setDefaultModel(result.data.models.find((model) => model.name === "cnn")?.name ?? result.data.models.find((model) => model.name === result.data.default)?.name ?? result.data.models[0]?.name ?? "catboost");
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Сервис недоступен");
    } finally { setLoading(false); }
  }, []);
  useEffect(() => { queueMicrotask(() => void reload()); }, [reload]);
  return { models, defaultModel, loading, error, reload };
}

function DashboardPage() {
  const { models, defaultModel, loading, error, reload } = useServiceModels();
  const forecast = useForecast((name) => models.find((model) => model.name === (name ?? defaultModel))?.horizon);
  const [compare, setCompare] = useState(false);
  const active = forecast.request.model ?? defaultModel;
  const bounds = models.find((model) => model.name === active)?.horizon ?? forecastDomain;
  const shown = forecast.acceptedRequest?.model ?? defaultModel;
  const other = compare ? models.find((model) => model.name !== shown)?.name ?? null : null;
  const comparison = useComparison(forecast.acceptedRequest, other);
  const [selectedIndex, setSelectedIndex] = useState(8);
  const [routes, setRoutes] = useState<Route[]>([]);
  const [layout, setLayout] = useState<Layout>("map");
  const [controlsOpen, setControlsOpen] = useState(false);
  useEffect(() => { queueMicrotask(async () => { const result = await fetchClient.GET("/api/v1/routes"); if (result.data) setRoutes(result.data.routes); }); }, []);
  const rejectedField = (fields: string[]) => forecast.failure?.field && fields.includes(forecast.failure.field) ? <Alert color={forecast.failure.source === "запрос" ? "yellow" : "red"} p="xs">{forecast.failure.message}</Alert> : null;
  const selectModel = (model: ModelName) => {
    const { from, to } = fitPeriod(forecast.request, models.find((item) => item.name === model)?.horizon ?? forecastDomain);
    if (from !== forecast.request.from) setSelectedIndex(0);
    forecast.dispatch({ type: "model", model, from, to });
  };
  const controls = <Stack p="md"><Group justify="space-between"><Title order={6}>Прогноз посадок трамваев</Title><Badge size="sm" color={error ? "red" : loading ? "gray" : "teal"}>{error ? "Сервис недоступен" : loading ? "Подключение" : "Сервис готов"}</Badge></Group><ModelPanel models={models} active={active} onSelect={selectModel} compare={compare} onCompare={setCompare} to={forecast.request.to} comparisonError={comparison.model === other ? comparison.error : ""} /><Divider /><ObjectPicker routes={routes} selected={forecast.request.routes?.[0] ?? null} onSelect={(route) => forecast.dispatch({ type: "routes", routes: route === null ? [] : [route] })} data={forecast.data} stale={forecast.loading || !!forecast.failure || !forecast.acceptedRequest || !sameRequest(forecast.request, forecast.acceptedRequest)} />{rejectedField(["routes"])}<Divider /><PeriodControls request={forecast.request} data={forecast.data} bounds={bounds} change={(from, to, horizon, granularity) => forecast.dispatch({ type: "period", from, to, horizon, granularity })} selectedIndex={selectedIndex} selectIndex={setSelectedIndex} />{rejectedField(["from", "to", "granularity", "horizon"])}<Divider /><ScenarioPanel conditions={forecast.request.conditions} requestRoutes={forecast.request.routes ?? []} availableRoutes={routes.map((route) => route.route)} horizon={forecast.request.horizon} from={forecast.request.from} to={forecast.request.to} response={forecast.data} stale={forecast.loading} onChange={(conditions) => forecast.dispatch({ type: "conditions", conditions })} />{rejectedField(["conditions"])}<Divider /><ViewPicker value={layout} onChange={setLayout} /><Divider /><ExportControls request={forecast.request} acceptedRequest={forecast.acceptedRequest} ready={!!forecast.data && forecast.data.series.length > 0 && !forecast.loading && !forecast.failure && !!forecast.acceptedRequest && sameRequest(forecast.request, forecast.acceptedRequest)} /></Stack>;
  return (
    <Stack h="100dvh" gap={0}>
      {error && <Alert color="red" title="Нет связи с сервисом" m="md">{error}<Button size="xs" ml="md" onClick={() => void reload()}>Повторить</Button></Alert>}
      <Flex flex={1} mih={0} style={{ overflow: "hidden" }}>
      <ScrollArea h="100%" flex="0 0 clamp(320px, 24vw, 420px)" visibleFrom="md">
        {controls}
      </ScrollArea>

      <Divider orientation="vertical" visibleFrom="md" />

      <Box h="100%" flex={1} miw={0} style={{ overflowY: "auto" }}>
        <Workspace data={forecast.data} comparison={other && comparison.model === other && comparison.data ? { title: modelTitles[other], data: comparison.data } : null} recursive={recursiveFrom(models.find((model) => model.name === shown))} loading={forecast.loading} updatedAt={forecast.updatedAt} settling={forecast.settling} failure={forecast.failure} retry={forecast.retry} layout={layout} routes={routes} selected={forecast.request.routes ?? []} selectedIndex={selectedIndex} selectIndex={setSelectedIndex} onRouteSelect={(route) => forecast.dispatch({ type: "routes", routes: [route] })} request={forecast.request} acceptedRequest={forecast.acceptedRequest} />
      </Box>
    </Flex>
    <Affix position={{ bottom: 16, right: 16 }} hiddenFrom="md"><Button onClick={() => setControlsOpen(true)}>Управление</Button></Affix>
    <Drawer opened={controlsOpen} onClose={() => setControlsOpen(false)} title="Управление прогнозом" hiddenFrom="md" size="min(92vw, 420px)"><ScrollArea h="calc(100dvh - 80px)">{controls}</ScrollArea></Drawer>
    </Stack>
  );
}

export const Component = DashboardPage;
