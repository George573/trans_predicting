import createFetchClient from "openapi-fetch";
import createReactQueryClient from "openapi-react-query";
import { session } from "@/shared/model/session";
import type { ApiPaths } from "./schema";

export const apiBaseUrl = (import.meta.env.VITE_API_URL ?? "").replace(/\/$/, "");

export async function authorizedFetch(input: RequestInfo | URL, init?: RequestInit) {
  const request = new Request(input, init);
  const send = () => {
    const attempt = request.clone();
    const authorization = session.authorization();
    if (authorization) attempt.headers.set("Authorization", authorization);
    return fetch(attempt);
  };
  let response = await send();
  if (response.status !== 401) return response;
  session.requireLogin();
  if (!(await session.waitForLogin(request.signal))) return response;
  response = await send();
  if (response.status === 401) session.requireLogin();
  return response;
}

export const fetchClient = createFetchClient<ApiPaths>({
  baseUrl: apiBaseUrl,
  fetch: authorizedFetch,
});

export const rqClient = createReactQueryClient(fetchClient);
