import { useState, useSyncExternalStore, type FormEvent } from "react";
import { Alert, Button, Modal, PasswordInput, Stack, TextInput } from "@mantine/core";
import { apiBaseUrl } from "@/shared/api/instance";
import { session } from "@/shared/model/session";

export function LoginDialog() {
  const opened = useSyncExternalStore(session.subscribe, session.getSnapshot);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      const authorization = `Basic ${btoa(unescape(encodeURIComponent(`${username}:${password}`)))}`;
      const response = await fetch(`${apiBaseUrl}/api/v1/model`, { headers: { Authorization: authorization } });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        setError(body?.error?.message ?? `Сервис ответил: ${response.status}`);
        return;
      }
      session.accept(username, password);
      setPassword("");
    } catch {
      setError("Не удалось связаться с сервисом. Попробуйте ещё раз.");
    } finally { setLoading(false); }
  }

  return <Modal opened={opened} onClose={() => {}} withCloseButton={false} closeOnEscape={false} closeOnClickOutside={false} title="Вход в дашборд" centered>
    <form onSubmit={submit}><Stack>
      <TextInput label="Логин" autoComplete="username" required autoFocus value={username} onChange={(event) => setUsername(event.currentTarget.value)} />
      <PasswordInput label="Пароль" autoComplete="current-password" required value={password} onChange={(event) => setPassword(event.currentTarget.value)} />
      {error && <Alert color="red">{error}</Alert>}
      <Button type="submit" loading={loading}>Войти</Button>
    </Stack></form>
  </Modal>;
}
