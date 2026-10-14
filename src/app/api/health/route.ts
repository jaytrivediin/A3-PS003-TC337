import { BACKEND_URL } from "@/lib/backend-url";
import type { BackendHealth } from "@/types/api";

export const dynamic = "force-dynamic";

/** Web-server health plus a reachability probe of the Python backend. Always 200. */
export async function GET() {
  const backend: { reachable: boolean; url: string; detail: BackendHealth | null; error?: string } = {
    reachable: false,
    url: BACKEND_URL,
    detail: null,
  };
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 2500);
    const res = await fetch(`${BACKEND_URL}/api/health`, { cache: "no-store", signal: controller.signal });
    clearTimeout(timer);
    if (res.ok) {
      backend.reachable = true;
      backend.detail = (await res.json()) as BackendHealth;
    } else {
      backend.error = `HTTP ${res.status}`;
    }
  } catch (error) {
    backend.error = error instanceof Error ? error.message : String(error);
  }
  return Response.json({ status: "ok", ok: true, app: "fedtraffic-web", time: new Date().toISOString(), backend });
}
