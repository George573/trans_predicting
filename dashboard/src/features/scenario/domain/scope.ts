import dayjs from "dayjs";
import { nextConditionId, type ConditionInput, type RouteNumber, type Scope } from "./conditions";

export const datesLimit = 366;
export const forecastDomain = { from: "2025-01-01", to: "2026-04-30" };
export const operationalScenarioDays = 7;

export const weekdayTitles = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"];
export const weekdayNumbers = [1, 2, 3, 4, 5, 6, 7];

export const monthTitles = ["Январь", "Февраль", "Март", "Апрель", "Май", "Июнь", "Июль", "Август", "Сентябрь", "Октябрь", "Ноябрь", "Декабрь"];

export const seasonMonths = ["2025-11", "2025-12", "2026-01", "2026-02", "2026-03", "2026-04"];

export const weekdayTemplates: { label: string; weekdays: number[] }[] = [
  { label: "Все будни", weekdays: [1, 2, 3, 4, 5] },
  { label: "Выходные", weekdays: [6, 7] },
];

export const hoursPresets: { label: string; hours: [number, number] | null }[] = [
  { label: "Утренний пик 07-09", hours: [7, 9] },
  { label: "Вечерний пик 16-19", hours: [16, 19] },
  { label: "Весь день", hours: null },
];

export function scopeHours(scope?: Scope): [number, number] | null {
  const hours = scope?.hours;
  if (!hours || hours.length !== 2) return null;
  return [hours[0], hours[1]];
}

export function withHours(scope: Scope | undefined, hours: [number, number] | null): Scope {
  return { ...scope, hours };
}

export function withRoutes(scope: Scope | undefined, routes: RouteNumber[]): Scope {
  return { ...scope, routes: routes.length === 0 ? null : [...routes].sort((left, right) => left - right) };
}

export function withDates(scope: Scope | undefined, dates: string[]): Scope {
  const unique = sortUniqueDates(dates);
  return { ...scope, dates: unique.length === 0 ? null : unique };
}

export function withWeekdays(scope: Scope | undefined, weekdays: number[]): Scope {
  const unique = [...new Set(weekdays)].sort((left, right) => left - right);
  return { ...scope, weekdays: unique.length === 0 ? null : unique };
}

export function hoursSpan(hours: [number, number] | null): number {
  return hours ? hours[1] - hours[0] + 1 : 24;
}

export function describeHours(hours: [number, number] | null): string {
  return hours ? `часы ${hours[0]}-${hours[1]} включительно` : "все 24 часа";
}

export function describeRoutes(scope?: Scope): string {
  const routes = scope?.routes;
  return routes && routes.length > 0 ? `маршруты ${routes.join(", ")}` : "все маршруты запроса";
}

export function isOvernight(from: number, to: number): boolean {
  return from > to;
}

export function sortUniqueDates(dates: string[]): string[] {
  return [...new Set(dates)].sort();
}

export function expandDateRange(from: string, to: string): string[] {
  const start = dayjs(from);
  const end = dayjs(to);
  if (!start.isValid() || !end.isValid() || end.isBefore(start)) return [];
  const dates: string[] = [];
  for (let current = start; !current.isAfter(end); current = current.add(1, "day")) dates.push(current.format("YYYY-MM-DD"));
  return dates;
}

export function weekdayOf(date: string): number {
  const day = dayjs(date).day();
  return day === 0 ? 7 : day;
}

export function shiftDate(date: string, days: number): string {
  return dayjs(date).add(days, "day").format("YYYY-MM-DD");
}

export function withinDomain(dates: string[]): string[] {
  return dates.filter((date) => date >= forecastDomain.from && date <= forecastDomain.to);
}

export function datesOfWeekdays(from: string, to: string, weekdays: number[]): string[] {
  return expandDateRange(from, to).filter((date) => weekdays.includes(weekdayOf(date)));
}

export function effectiveDates(scope: Scope | undefined, from: string, to: string): string[] {
  const period = expandDateRange(from, to);
  const chosen = scope?.dates && scope.dates.length > 0 ? sortUniqueDates(scope.dates).filter((date) => period.includes(date)) : period;
  const weekdays = scope?.weekdays;
  return weekdays && weekdays.length > 0 ? chosen.filter((date) => weekdays.includes(weekdayOf(date))) : chosen;
}

export function datesOutsidePeriod(scope: Scope | undefined, from: string, to: string): string[] {
  if (!scope?.dates) return [];
  return sortUniqueDates(scope.dates).filter((date) => date < from || date > to);
}

export function isScopeEmptyOnPeriod(scope: Scope | undefined, from: string, to: string): boolean {
  return effectiveDates(scope, from, to).length === 0;
}

export function splitOvernight(condition: ConditionInput, hours: [number, number], from: string, to: string, conditions: ConditionInput[]): ConditionInput[] | null {
  const evening = withinDomain(effectiveDates(condition.scope, from, to));
  const morning = withinDomain(evening.map((date) => shiftDate(date, 1)));
  if (evening.length === 0 || morning.length === 0) return null;
  const first: ConditionInput = { ...condition, scope: { ...withDates(condition.scope, evening), hours: [hours[0], 23] } };
  const second: ConditionInput = { ...condition, id: nextConditionId(conditions), scope: { ...withDates(condition.scope, morning), hours: [0, hours[1]] } };
  return [first, second];
}

export function monthTitle(month: string): string {
  const [year, index] = month.split("-");
  return `${monthTitles[Number(index) - 1]} ${year}`;
}

export function monthDates(month: string): string[] {
  const start = dayjs(`${month}-01`);
  return withinDomain(expandDateRange(start.format("YYYY-MM-DD"), start.endOf("month").format("YYYY-MM-DD")));
}

export function monthsDates(months: string[]): string[] {
  return sortUniqueDates(months.flatMap(monthDates));
}

export function fitsDatesLimit(dates: string[]): boolean {
  return dates.length <= datesLimit;
}
