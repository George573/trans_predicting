import { createBrowserRouter } from "react-router";
import { RouterProvider } from "react-router/dom";

import { Routes } from "@/shared/model/routes";

import { App } from "./app";
import { Providers } from "./providers";

const router = createBrowserRouter([
  {
    element: (
      <Providers>
        <App />
      </Providers>
    ),
    children: [
      {
        path: Routes.DASHBOARD,
        lazy: () => import("@/features/dashboard/dashboard.page"),
      },
    ],
  },
]);

export function Router() {
  return <RouterProvider router={router} />;
}
