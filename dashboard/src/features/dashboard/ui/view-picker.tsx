import type { ReactNode } from "react";
import { Group, Stack, Text, Title, Tooltip, UnstyledButton } from "@mantine/core";

export type Layout = "map" | "mosaic" | "heat";

const frame = <rect x="3" y="4" width="18" height="16" rx="2" />;
const views: { value: Layout; label: string; icon: ReactNode }[] = [
  { value: "map", label: "Графики и карта рядом", icon: <>{frame}<path d="M11 4 V20" /></> },
  { value: "mosaic", label: "Мозаика 2x2", icon: <>{frame}<path d="M12 4 V20" /><path d="M3 12 H21" /></> },
  { value: "heat", label: "Маршруты и время", icon: <>{frame}<path d="M3 9.3 H21" /><path d="M3 14.6 H21" /><path d="M9 4 V20" /><path d="M15 4 V20" /></> },
];

export function ViewPicker({ value, onChange }: { value: Layout; onChange: (layout: Layout) => void }) {
  return <Stack gap={8}>
    <Group justify="space-between" align="baseline"><Title order={6}>Вид</Title><Text fz={11} c="dimmed">раскладка экрана</Text></Group>
    <Group gap={4} grow>
      {views.map((view) => {
        const active = view.value === value;
        return <Tooltip key={view.value} label={view.label} withArrow>
          <UnstyledButton aria-label={view.label} aria-pressed={active} onClick={() => onChange(view.value)} h={34} style={{ display: "flex", alignItems: "center", justifyContent: "center", borderRadius: 6, border: `1px solid ${active ? "#4c9aff" : "#3a4655"}`, background: active ? "#1e3350" : "transparent", color: active ? "#e6eaee" : "#9aa5b1" }}>
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">{view.icon}</svg>
          </UnstyledButton>
        </Tooltip>;
      })}
    </Group>
    <Text fz={12} c="dimmed">{views.find((view) => view.value === value)?.label}</Text>
  </Stack>;
}
