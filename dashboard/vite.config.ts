import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

import { forecastMock } from "./vite/forecast-mock.ts";

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), ...(process.env.VITE_FORECAST_MOCK ? [forecastMock()] : [])],
  server: { proxy: { "/api": "http://localhost:8080", "/geo": "http://localhost:8080", "/healthz": "http://localhost:8080" } },
  resolve: {
    tsconfigPaths: true,
  },
});
