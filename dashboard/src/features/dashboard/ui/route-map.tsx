import { useEffect, useRef, useState } from "react";
import L from "leaflet";
import { Alert, Group, Paper, Text } from "@mantine/core";
import type { ForecastResponse, Route } from "../model/forecast-state";
import { apiBaseUrl, authorizedFetch } from "@/shared/api/instance";
import { routeLevel } from "../lib/level";
import { momentLabel, quantityLabel } from "../lib/moment";
import "leaflet/dist/leaflet.css";

type Props = { routes: Route[]; selected: number[]; onSelect: (route: number) => void; data: ForecastResponse; selectedIndex: number };
type Feature = { type: "Feature"; properties: { kind: "line" | "stop"; route: number; name?: string; source?: string }; geometry: { type: "MultiLineString" | "Point"; coordinates: unknown } };
type FeatureCollection = { type: "FeatureCollection"; features: Feature[] };

export function RouteMap({ routes, selected, onSelect, data, selectedIndex }: Props) {
  const element = useRef<HTMLDivElement>(null);
  const map = useRef<L.Map | null>(null);
  const lines = useRef<L.GeoJSON | null>(null);
  const rings = useRef<L.LayerGroup | null>(null);
  const stopLayer = useRef<L.GeoJSON | null>(null);
  const fitted = useRef("");
  const [geometry, setGeometry] = useState<FeatureCollection | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!element.current) return;
    const instance = L.map(element.current, { zoomControl: true, attributionControl: true }).setView([55.75, 37.62], 10);
    instance.attributionControl.setPrefix(false);
    map.current = instance;
    if (navigator.onLine) L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { attribution: "© OpenStreetMap", maxZoom: 18 }).addTo(instance);
    return () => { instance.remove(); map.current = null; };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    authorizedFetch(`${apiBaseUrl}/geo/routes.geojson`, { signal: controller.signal })
      .then((response) => { if (!response.ok) throw new Error("Локальная геометрия недоступна"); return response.json(); })
      .then((value: FeatureCollection) => {
        if (value.type !== "FeatureCollection" || !Array.isArray(value.features)) throw new Error("Файл маршрутов повреждён");
        setGeometry(value);
      })
      .catch((cause) => { if (!controller.signal.aborted) setError(cause instanceof Error ? cause.message : "Нет геометрии маршрутов"); });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!map.current || !geometry) return;
    lines.current?.remove();
    rings.current?.remove();
    const allowed = new Set(routes.filter((route) => route.has_geometry).map((route) => route.route));
    const filtered: FeatureCollection = { type: "FeatureCollection", features: geometry.features.filter((feature) => feature.properties.kind === "line" && allowed.has(Number(feature.properties.route) as Route["route"])) };
    const layer = L.geoJSON(filtered as unknown as Parameters<typeof L.geoJSON>[0], {
      style: (feature) => {
        const route = Number(feature?.properties?.route);
        const series = data.series.find((item) => item.route === route);
        const level = routeLevel(series?.value[selectedIndex], series?.usual[selectedIndex]);
        return { color: level.color, weight: selected.includes(route) ? 6 : 4, opacity: selected.length && !selected.includes(route) ? 0.45 : 0.95 };
      },
      onEachFeature: (feature, item) => {
        const route = Number(feature.properties?.route);
        const series = data.series.find((entry) => entry.route === route);
        const value = series?.value[selectedIndex];
        const level = routeLevel(value, series?.usual[selectedIndex]);
        item.bindTooltip(`Маршрут ${route} · ${momentLabel(data, selectedIndex)} · ${value === undefined ? "нет прогноза" : `${Math.round(value).toLocaleString("ru-RU")} (${quantityLabel(data.step)})`} · ${level.label}`);
        item.on("click", () => onSelect(route));
      },
    }).addTo(map.current);
    lines.current = layer;
    const ringLayer = L.layerGroup().addTo(map.current);
    layer.eachLayer((item) => {
      const route = Number((item as L.Polyline & { feature?: { properties?: { route?: number } } }).feature?.properties?.route);
      const series = data.series.find((entry) => entry.route === route);
      if (!series?.hi || series.value[selectedIndex] <= series.hi[selectedIndex]) return;
      const bounds = (item as L.Polyline).getBounds();
      if (bounds.isValid()) L.circleMarker(bounds.getCenter(), { radius: 10, color: "#ef6b73", fillOpacity: 0, weight: 3 }).bindTooltip(`Маршрут ${route}: выше верхней границы коридора. Кольцо отмечает маршрут в целом.`).addTo(ringLayer);
    });
    rings.current = ringLayer;
    const shown = `${[...allowed].sort().join(",")}|${[...selected].sort().join(",")}`;
    if (fitted.current !== shown && layer.getBounds().isValid()) {
      map.current.fitBounds(layer.getBounds(), { padding: [16, 16], maxZoom: 12 });
      fitted.current = shown;
    }
    return () => { layer.remove(); ringLayer.remove(); };
  }, [geometry, routes, selected, onSelect, data, selectedIndex]);

  useEffect(() => {
    if (!map.current) return;
    stopLayer.current?.remove();
    if (!geometry) return;
    const available = new Set(routes.map((route) => route.route));
    const visible: FeatureCollection = { type: "FeatureCollection", features: geometry.features.filter((feature) => feature.properties.kind === "stop" && available.has(feature.properties.route as Route["route"]) && (selected.length === 0 || selected.includes(feature.properties.route))) };
    const layer = L.geoJSON(visible as unknown as Parameters<typeof L.geoJSON>[0], {
      pointToLayer: (_, latlng) => L.circleMarker(latlng, { radius: 3, color: "#d9e2eb", weight: 1, fillColor: "#d9e2eb", fillOpacity: 0.8 }),
      onEachFeature: (feature, item) => { if (feature.properties?.name) item.bindTooltip(String(feature.properties.name)); },
    }).addTo(map.current);
    stopLayer.current = layer;
    return () => { layer.remove(); };
  }, [geometry, routes, selected]);

  const missing = routes.filter((route) => route.has_geometry && geometry && !geometry.features.some((feature) => feature.properties.kind === "line" && feature.properties.route === route.route));
  return <Paper p="md" withBorder>
    <Text fw={600} mb="sm">Карта маршрутов</Text>
    {error && <Alert color="yellow" mb="sm">{error}. Маршруты доступны в списке и графиках.</Alert>}
    {geometry && !geometry.features.some((feature) => feature.properties.kind === "stop" && (selected.length === 0 || selected.includes(feature.properties.route))) && <Text size="xs" c="dimmed">Для выбранных маршрутов нет остановок в геометрии.</Text>}
    {missing.length > 0 && <Text size="xs" c="dimmed" mb="xs">Нет линии для маршрутов: {missing.map((route) => route.route).join(", ")}</Text>}
    <div ref={element} style={{ height: 300, background: "#17232c", borderRadius: 8 }} aria-label="Карта трамвайных маршрутов" />
    <Text size="xs" mt="xs">% от обычного уровня</Text>
    <Group gap="md"><Text size="xs" c="#3bb8a3">До 80%</Text><Text size="xs" c="#e3b350">80-120%</Text><Text size="xs" c="#ef6b73">Выше 120%</Text><Text size="xs" c="dimmed">Серый: нет сравнения</Text></Group>
    <Text size="xs" c="dimmed">Кольцо - превышение верхней границы коридора по маршруту в целом.</Text>
    <Text size="xs" c="dimmed">Остановки показаны только как геометрия, без данных о посадках.</Text>
    <Text size="xs" c="dimmed">Геометрия маршрутов и остановок: справочник и © OpenStreetMap, локальная копия из сервиса.</Text>
  </Paper>;
}
