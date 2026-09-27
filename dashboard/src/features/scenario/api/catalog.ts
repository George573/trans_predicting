import { rqClient } from "@/shared/api/instance";
import type { CatalogEntry } from "../domain/conditions";

export function useConditionsCatalog() {
  const query = rqClient.useQuery("get", "/api/v1/conditions", {}, { staleTime: Infinity, retry: false });
  const entries: CatalogEntry[] = query.data?.conditions ?? [];
  return {
    entries,
    loading: query.isPending,
    error: query.isError ? "Каталог условий недоступен, условия недоступны для добавления" : "",
    retry: () => void query.refetch(),
  };
}
