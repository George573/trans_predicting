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
  primaryColor: "indigo",
  defaultRadius: "md",
  cursorType: "pointer",
  autoContrast: true,
  headings: {
    fontWeight: "600",
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
