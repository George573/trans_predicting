import type { AppliedCondition, ConditionInput, ForecastResponse, ForecastWarning } from "./conditions";
import { effectiveDates, scopeHours } from "./scope";

export function appliedCondition(response: ForecastResponse | null, id: string): AppliedCondition | undefined {
  return response?.conditions.find((condition) => condition.id === id);
}

export function warningsFor(response: ForecastResponse | null, id: string): ForecastWarning[] {
  if (!id) return [];
  return response?.warnings.filter((warning) => warning.condition_id === id) ?? [];
}

export function warningOf(response: ForecastResponse | null, id: string, code: ForecastWarning["code"]): ForecastWarning | undefined {
  return warningsFor(response, id).find((warning) => warning.code === code);
}

export function notAppliedIds(response: ForecastResponse | null): string[] {
  return [...new Set(response?.conditions.filter((condition) => !condition.applied).map((condition) => condition.id) ?? [])];
}

export function clampedWarning(response: ForecastResponse | null): ForecastWarning | undefined {
  return response?.warnings.find((warning) => warning.code === "correction_clamped");
}

function routesIntersect(left: ConditionInput, right: ConditionInput): boolean {
  const first = left.scope?.routes;
  const second = right.scope?.routes;
  if (!first || !second || first.length === 0 || second.length === 0) return true;
  return first.some((route) => second.includes(route));
}

function hoursIntersection(left: ConditionInput, right: ConditionInput): [number, number] | null {
  const first = scopeHours(left.scope) ?? [0, 23];
  const second = scopeHours(right.scope) ?? [0, 23];
  const from = Math.max(first[0], second[0]);
  const to = Math.min(first[1], second[1]);
  return from <= to ? [from, to] : null;
}

export type ConditionOverlap = {
  ids: [string, string];
  dates: string[];
  hours: [number, number];
  factor: number | null;
};

export function conditionOverlaps(conditions: ConditionInput[], response: ForecastResponse | null, from: string, to: string): ConditionOverlap[] {
  const overlaps: ConditionOverlap[] = [];
  for (let left = 0; left < conditions.length; left += 1) {
    for (let right = left + 1; right < conditions.length; right += 1) {
      const first = conditions[left];
      const second = conditions[right];
      if (!routesIntersect(first, second)) continue;
      const hours = hoursIntersection(first, second);
      if (!hours) continue;
      const dates = effectiveDates(first.scope, from, to).filter((date) => effectiveDates(second.scope, from, to).includes(date));
      if (dates.length === 0) continue;
      const applied = [first.id, second.id].map((id) => appliedCondition(response, id));
      const factor = applied.every((condition) => condition?.applied) ? applied.reduce((product, condition) => product * (condition?.factor ?? 1), 1) : null;
      overlaps.push({ ids: [first.id, second.id], dates, hours, factor });
    }
  }
  return overlaps;
}

export function pluralConditions(count: number): string {
  const tail = count % 100;
  if (tail >= 11 && tail <= 14) return `${count} условий не применимы`;
  const last = count % 10;
  if (last === 1) return `${count} условие не применимо`;
  if (last >= 2 && last <= 4) return `${count} условия не применимы`;
  return `${count} условий не применимы`;
}
