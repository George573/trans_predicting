import type { ReactNode } from "react";
import { Box, Group, Paper, Text } from "@mantine/core";

type Props = { title: ReactNode; note?: ReactNode; extra?: ReactNode; children: ReactNode };

export function Panel({ title, note, extra, children }: Props) {
  return <Paper withBorder radius={6} style={{ overflow: "hidden", display: "flex", flexDirection: "column", flexShrink: 0, minWidth: 0, maxWidth: "100%" }}>
    <Group h={32} px={12} gap={10} wrap="nowrap" style={{ flexShrink: 0, borderBottom: "1px solid var(--mantine-color-dark-4)" }}>
      <Text fz={13} fw={600} style={{ whiteSpace: "nowrap" }}>{title}</Text>
      {note && <Text fz={12} c="dimmed" truncate>{note}</Text>}
      <Box flex={1} />
      {extra}
    </Group>
    <Box p={12} miw={0}>{children}</Box>
  </Paper>;
}
