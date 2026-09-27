import type { components } from "@/shared/api/schema/generated";

export type ConditionInput = components["schemas"]["ConditionInput"];
export type ConditionType = components["schemas"]["ConditionType"];
export type ConditionClass = components["schemas"]["ConditionClass"];
export type CatalogEntry = components["schemas"]["CatalogEntry"];
export type Passport = components["schemas"]["Passport"];
export type Scope = components["schemas"]["Scope"];
export type RouteNumber = components["schemas"]["RouteNumber"];
export type AppliedCondition = components["schemas"]["AppliedCondition"];
export type ForecastWarning = components["schemas"]["Warning"];
export type ForecastResponse = components["schemas"]["ForecastResponse"];
export type Horizon = components["schemas"]["Horizon"];

export const conditionLimit = 16;

export const conditionTypeTitles: Record<ConditionType, string> = {
  rain: "Дождь",
  temperature: "Температура",
  event: "Событие рядом",
  delay: "Задержка трамваев",
  closure: "Отмена или перекрытие",
  route_change: "Изменение маршрута",
};

export const conditionTypeClasses: Record<ConditionType, ConditionClass> = {
  rain: "operational",
  temperature: "operational",
  event: "operational",
  delay: "operational",
  closure: "structural",
  route_change: "structural",
};

export const conditionClassTitles: Record<ConditionClass, string> = {
  operational: "Оперативные",
  structural: "Структурные",
};

export const horizonTitles: Record<Horizon, string> = {
  day: "сутки",
  week: "неделя",
  month: "месяц",
  season: "сезон",
};

export function conditionClassOf(type: ConditionType, entry?: CatalogEntry): ConditionClass {
  return entry?.class ?? conditionTypeClasses[type];
}

export function conditionTitleOf(type: ConditionType, entry?: CatalogEntry): string {
  return entry?.title ?? conditionTypeTitles[type];
}

export function nextConditionId(conditions: ConditionInput[]): string {
  const used = new Set(conditions.map((condition) => condition.id));
  let index = 1;
  while (used.has(`c${index}`)) index += 1;
  return `c${index}`;
}

export function conditionStep(entry: CatalogEntry): number | null {
  return entry.range.step ?? entry.passport.step ?? null;
}

export function clampValue(entry: CatalogEntry, value: number): number {
  return Math.min(Math.max(value, entry.range.min), entry.range.max);
}

export function createCondition(entry: CatalogEntry, conditions: ConditionInput[]): ConditionInput {
  return { id: nextConditionId(conditions), type: entry.type, value: clampValue(entry, 0), scope: {} };
}

export function passportFor(entry: CatalogEntry, routes: number[]): Passport {
  if (routes.length !== 1) return entry.passport;
  return entry.by_route?.[String(routes[0])] ?? entry.passport;
}

export function formatNumber(value: number): string {
  return value.toLocaleString("ru-RU", { maximumFractionDigits: 3 });
}

export function formatSignedPercent(value: number): string {
  return `${value > 0 ? "+" : ""}${formatNumber(value)}%`;
}
