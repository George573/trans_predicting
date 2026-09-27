export function routeLevel(value?: number, usual?: number) {
  if (value === undefined || usual === undefined || usual <= 0) return { percent: null, color: "#778899", label: "Нет сравнения" };
  const percent = 100 * value / usual;
  if (!Number.isFinite(percent)) return { percent: null, color: "#778899", label: "Нет сравнения" };
  return { percent, color: percent < 80 ? "#3bb8a3" : percent <= 120 ? "#e3b350" : "#ef6b73", label: `${Math.round(percent)}% от обычного уровня` };
}

const routePalette = ["#4c9aff", "#f2b544", "#3bb8a3", "#ef6b73", "#b57bff", "#ff8f40", "#5ad1e6", "#9bd35a", "#ff6fb5", "#e8e1c4"];

export function routeColor(routes: { route: number }[], route: number) {
  const index = routes.findIndex((item) => item.route === route);
  return routePalette[(index < 0 ? route : index) % routePalette.length];
}
