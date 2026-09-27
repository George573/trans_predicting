export function routeLevel(value?: number, usual?: number) {
  if (value === undefined || usual === undefined || usual <= 0) return { percent: null, color: "#778899", label: "Нет сравнения" };
  const percent = 100 * value / usual;
  if (!Number.isFinite(percent)) return { percent: null, color: "#778899", label: "Нет сравнения" };
  return { percent, color: percent < 80 ? "#3bb8a3" : percent <= 120 ? "#e3b350" : "#ef6b73", label: `${Math.round(percent)}% от обычного уровня` };
}
