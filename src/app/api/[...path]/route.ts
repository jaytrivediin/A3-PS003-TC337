import type { NextRequest } from "next/server";
import { BACKEND_URL } from "@/lib/backend-url";

export const dynamic = "force-dynamic";

/**
 * Same-origin proxy: /api/<path> -> <BACKEND_URL>/api/<path>.
 * Keeps the browser on one origin (no CORS issues) and lets BACKEND_URL stay server-side.
 */
async function proxy(req: NextRequest, context: { params: Promise<{ path: string[] }> }) {
  const { path } = await context.params;
  const target = `${BACKEND_URL}/api/${path.join("/")}${req.nextUrl.search}`;
  const headers = new Headers();
  const contentType = req.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);
  headers.set("accept", req.headers.get("accept") ?? "application/json");

  const init: RequestInit = { method: req.method, headers, cache: "no-store" };
  if (req.method !== "GET" && req.method !== "HEAD") {
    init.body = await req.text();
  }
  try {
    const upstream = await fetch(target, init);
    const resHeaders = new Headers();
    for (const name of ["content-type", "content-disposition", "cache-control"]) {
      const value = upstream.headers.get(name);
      if (value) resHeaders.set(name, value);
    }
    return new Response(upstream.body, { status: upstream.status, headers: resHeaders });
  } catch (error) {
    return Response.json(
      {
        error: "backend_unreachable",
        message: `The FedTraffic Python backend at ${BACKEND_URL} is not reachable. Start it with: cd backend && uvicorn app.main:app --port 8000`,
        backend_url: BACKEND_URL,
        cause: error instanceof Error ? error.message : String(error),
      },
      { status: 503 },
    );
  }
}

export { proxy as GET, proxy as POST, proxy as PUT, proxy as PATCH, proxy as DELETE };
