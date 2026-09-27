import { AppShell } from "@mantine/core";
import { Outlet } from "react-router";
import { LoginDialog } from "@/features/auth";

export function App() {
  return (
    <AppShell>
      <AppShell.Main>
        <Outlet />
      </AppShell.Main>
      <LoginDialog />
    </AppShell>
  );
}
