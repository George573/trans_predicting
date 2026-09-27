import type { PropsWithChildren } from "react";

import "@mantine/core/styles.layer.css";
import "@mantine/charts/styles.layer.css";
import "@mantine/dates/styles.layer.css";
import "dayjs/locale/ru";

import { MantineProvider } from "@mantine/core";
import { DatesProvider } from "@mantine/dates";

import { createTheme } from "@mantine/core";
import { QueryClientProvider } from "@tanstack/react-query";
import { queryClient } from "@/shared/api/query-client";

const theme = createTheme({
  colors: {
    dark: ["#e6eaee", "#c4ccd4", "#9aa5b1", "#7f8a96", "#27303a", "#3a4655", "#1b222b", "#0e1217", "#0a0d11", "#06090c"],
    accent: ["#e5f0ff", "#cce0ff", "#99c2ff", "#80b3ff", "#66a6ff", "#4c9aff", "#3d85e6", "#2f6fcc", "#2459a6", "#1a4280"],
  },
  primaryColor: "accent",
  primaryShade: { light: 6, dark: 5 },
  fontFamily: "IBM Plex Sans, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
  fontFamilyMonospace: "IBM Plex Mono, ui-monospace, SFMono-Regular, Menlo, monospace",
  defaultRadius: 6,
  cursorType: "pointer",
  autoContrast: true,
  headings: {
    fontFamily: "IBM Plex Sans, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
    fontWeight: "600",
  },
  components: {
    Paper: { defaultProps: { bg: "#141a21" } },
    SegmentedControl: { defaultProps: { color: "accent" } },
    Title: { styles: (_: unknown, props: { order?: number }) => props.order === 6 ? { root: { textTransform: "uppercase", letterSpacing: "0.08em", color: "var(--mantine-color-dark-2)", fontWeight: 600, fontSize: 11 } } : {} },
  },
});

export function Providers({ children }: PropsWithChildren) {
  return (
    <MantineProvider theme={theme} defaultColorScheme="dark">
      <DatesProvider settings={{ locale: "ru", firstDayOfWeek: 1 }}>
        <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
      </DatesProvider>
    </MantineProvider>
  );
}
