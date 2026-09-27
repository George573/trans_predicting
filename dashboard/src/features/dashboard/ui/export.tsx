import { useState } from "react";
import { Alert, Button, Group, Paper, Text } from "@mantine/core";
import { apiBaseUrl, authorizedFetch } from "@/shared/api/instance";
import type { ForecastRequest } from "../model/forecast-state";

type Format = "csv" | "xlsx";
type Props = { request: ForecastRequest; acceptedRequest: ForecastRequest | null; ready: boolean };

function downloadName(disposition: string | null, format: Format, request: ForecastRequest) {
  const encoded = disposition?.match(/filename\*\s*=\s*UTF-8''([^;]+)/i)?.[1];
  const plain = disposition?.match(/filename\s*=\s*"?([^";]+)"?/i)?.[1];
  let name = plain;
  if (encoded) {
    try { name = decodeURIComponent(encoded.replace(/^"|"$/g, "")); } catch { name = plain; }
  }
  name = name?.split(/[\\/]/).pop()?.replace(/\p{Cc}/gu, "").trim();
  return name || `forecast_${request.from}_${request.to}.${format}`;
}

export function ExportControls({ request, acceptedRequest, ready }: Props) {
  const [pending, setPending] = useState<Format | null>(null);
  const [error, setError] = useState("");
  const season = request.horizon === "season";
  const available = !season && ready && !!acceptedRequest && JSON.stringify(acceptedRequest) === JSON.stringify(request);

  async function download(format: Format) {
    if (!available || !acceptedRequest) return;
    setPending(format);
    setError("");
    try {
      const response = await authorizedFetch(`${apiBaseUrl}/api/v1/forecast/export?format=${format}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(acceptedRequest),
      });
      if (!response.ok) {
        const body = await response.json().catch(() => null);
        throw new Error(body?.error?.message ?? `Сервис ответил: ${response.status}`);
      }
      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = downloadName(response.headers.get("Content-Disposition"), format, acceptedRequest);
      document.body.append(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "Не удалось скачать прогноз");
    } finally { setPending(null); }
  }

  return <Paper p="md" withBorder>
    <Text fw={600} mb="xs">Выгрузка текущего прогноза</Text>
    <Group><Button disabled={!available || !!pending} loading={pending === "csv"} onClick={() => void download("csv")}>CSV</Button><Button disabled={!available || !!pending} loading={pending === "xlsx"} onClick={() => void download("xlsx")}>XLSX</Button></Group>
    {season ? <Alert color="yellow" mt="xs">Сезон показан суммами за месяцы. Сервис выгружает один запрос с суточными данными и не формирует файл текущего месячного вида. Выгрузка сезона всей сети в этой поставке недоступна.</Alert> : !available && <Text size="xs" c="dimmed" mt="xs">Дождитесь успешного обновления прогноза перед выгрузкой.</Text>}
    {error && <Alert color="red" mt="xs">{error}</Alert>}
  </Paper>;
}
