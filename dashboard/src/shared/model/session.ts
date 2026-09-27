type Credentials = { username: string; password: string };
type Listener = () => void;

let credentials: Credentials | null = null;
let loginRequired = false;
const listeners = new Set<Listener>();
const waiters = new Set<(ready: boolean) => void>();

function emit() { listeners.forEach((listener) => listener()); }

export const session = {
  subscribe(listener: Listener) { listeners.add(listener); return () => listeners.delete(listener); },
  getSnapshot: () => loginRequired,
  authorization() {
    if (!credentials) return null;
    return `Basic ${btoa(unescape(encodeURIComponent(`${credentials.username}:${credentials.password}`)))}`;
  },
  requireLogin() { credentials = null; loginRequired = true; emit(); },
  accept(username: string, password: string) {
    credentials = { username, password };
    loginRequired = false;
    emit();
    waiters.forEach((resolve) => resolve(true));
    waiters.clear();
  },
  waitForLogin(signal?: AbortSignal) {
    if (signal?.aborted) return Promise.resolve(false);
    return new Promise<boolean>((resolve) => {
      const done = (ready: boolean) => {
        waiters.delete(done);
        signal?.removeEventListener("abort", abort);
        resolve(ready);
      };
      const abort = () => done(false);
      waiters.add(done);
      signal?.addEventListener("abort", abort, { once: true });
    });
  },
};
