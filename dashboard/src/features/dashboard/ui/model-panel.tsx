import { Alert, Checkbox, SegmentedControl, Stack, Text, Title } from "@mantine/core";
import { formatDate } from "@/features/scenario";
import { modelTitles, recursiveFrom, type ModelInfo, type ModelName } from "../domain/model";

type Props = { models: ModelInfo[]; active: ModelName; onSelect: (model: ModelName) => void; compare: boolean; onCompare: (compare: boolean) => void; to: string; comparisonError: string };

export function ModelPanel({ models, active, onSelect, compare, onCompare, to, comparisonError }: Props) {
  const info = models.find((model) => model.name === active);
  const other = models.find((model) => model.name !== active);
  const recursive = recursiveFrom(info);
  return <Stack gap="xs">
    <Title order={6}>Модель</Title>
    {models.length > 1 && <SegmentedControl fullWidth size="xs" value={active} onChange={(value) => onSelect(value as ModelName)} data={models.map((model) => ({ label: modelTitles[model.name], value: model.name }))} />}
    {info && <Text size="xs" c="dimmed">{info.describe}{info.wape_score !== undefined && ` · WAPE-score ${info.wape_score}`} · прогноз {info.horizon.map(formatDate).join(" - ")}</Text>}
    {recursive && to >= recursive && <Alert color="yellow" p="xs">С {formatDate(recursive)} {modelTitles[active]} считает авторегрессией: каждый следующий день опирается на свой же прогноз. Точность на отрезках длиннее трёх месяцев не проверялась, коридор на этих днях не измерялся.</Alert>}
    {other && <Checkbox size="xs" label={`Сравнить с ${modelTitles[other.name]} на графике и в цифрах`} checked={compare} onChange={(event) => onCompare(event.currentTarget.checked)} />}
    {compare && comparisonError && <Text size="xs" c="yellow">Сравнение недоступно: {comparisonError}</Text>}
  </Stack>;
}
