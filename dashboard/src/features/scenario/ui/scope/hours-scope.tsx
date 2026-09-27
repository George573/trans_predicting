import { useEffect, useState } from "react";
import { Button, Group, NumberInput, Stack, Text, UnstyledButton } from "@mantine/core";
import { describeHours, hoursPresets, hoursSpan, isOvernight } from "../../domain/scope";

const hours = Array.from({ length: 24 }, (_, hour) => hour);

function HoursBand({ value, onChange }: { value: [number, number] | null; onChange: (value: [number, number]) => void }) {
  const [drag, setDrag] = useState<{ anchor: number; hover: number } | null>(null);
  const [keyAnchor, setKeyAnchor] = useState<number | null>(null);
  const shown: [number, number] | null = drag ? [Math.min(drag.anchor, drag.hover), Math.max(drag.anchor, drag.hover)] : value;

  useEffect(() => {
    if (!drag) return;
    const finish = () => {
      setDrag(null);
      onChange([Math.min(drag.anchor, drag.hover), Math.max(drag.anchor, drag.hover)]);
    };
    window.addEventListener("pointerup", finish);
    return () => window.removeEventListener("pointerup", finish);
  }, [drag, onChange]);

  const press = (hour: number) => {
    if (keyAnchor === null) { setKeyAnchor(hour); return; }
    setKeyAnchor(null);
    onChange([Math.min(keyAnchor, hour), Math.max(keyAnchor, hour)]);
  };

  return (
    <Stack gap={2}>
      <Group gap={1} wrap="nowrap" style={{ touchAction: "none" }}>
        {hours.map((hour) => {
          const inside = shown !== null && hour >= shown[0] && hour <= shown[1];
          const anchored = keyAnchor === hour;
          return (
            <UnstyledButton
              key={hour}
              h={26}
              flex={1}
              bg={inside ? "var(--mantine-color-indigo-6)" : "var(--mantine-color-dark-5)"}
              style={{ borderRadius: 2, outline: anchored ? "1px solid var(--mantine-color-indigo-3)" : undefined }}
              aria-label={`Час ${hour}`}
              aria-pressed={inside}
              onPointerDown={() => setDrag({ anchor: hour, hover: hour })}
              onPointerEnter={() => setDrag((current) => (current ? { ...current, hover: hour } : null))}
              onKeyDown={(event) => { if (event.key === "Enter" || event.key === " ") { event.preventDefault(); press(hour); } }}
            />
          );
        })}
      </Group>
      <Group justify="space-between">
        {[0, 6, 12, 18, 23].map((hour) => <Text key={hour} size="xs" c="dimmed">{hour}</Text>)}
      </Group>
      <Text size="xs" c="dimmed">{keyAnchor === null ? "Протяните мышью или выберите начало и конец с клавиатуры" : `Начало ${keyAnchor}: выберите конец диапазона`}</Text>
    </Stack>
  );
}

export function HoursScope({ value, onChange, onSplit, splitReason }: {
  value: [number, number] | null;
  onChange: (value: [number, number] | null) => void;
  onSplit?: () => void;
  splitReason?: string;
}) {
  const from = value?.[0] ?? 0;
  const to = value?.[1] ?? 23;
  const overnight = isOvernight(from, to);
  return (
    <Stack gap={4}>
      <Text size="xs" fw={500}>Когда: {describeHours(value)} ({hoursSpan(value)} ч)</Text>
      <HoursBand value={value} onChange={onChange} />
      <Group gap={4} wrap="nowrap">
        <NumberInput size="xs" w={78} label="с" min={0} max={23} clampBehavior="strict" value={from} onChange={(next) => onChange([Number(next), to])} />
        <NumberInput size="xs" w={78} label="по" min={0} max={23} clampBehavior="strict" value={to} onChange={(next) => onChange([from, Number(next)])} />
      </Group>
      <Group gap={4}>
        {hoursPresets.map((preset) => (
          <Button
            key={preset.label}
            size="compact-xs"
            variant={JSON.stringify(preset.hours) === JSON.stringify(value) ? "filled" : "light"}
            onClick={() => onChange(preset.hours)}
          >
            {preset.label}
          </Button>
        ))}
      </Group>
      {overnight && (
        <Stack gap={2}>
          <Text size="xs" c="yellow">Диапазон через полночь контракт не принимает: пара часов читается как {from}-{to} в пределах суток.</Text>
          {onSplit
            ? <Button size="compact-xs" variant="light" onClick={onSplit}>Разбить на два условия с явными датами</Button>
            : <Text size="xs" c="dimmed">{splitReason}</Text>}
        </Stack>
      )}
    </Stack>
  );
}
