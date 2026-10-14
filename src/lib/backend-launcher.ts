/**
 * Optional convenience: start the Python backend together with the Next.js server.
 * Enabled only when FEDTRAFFIC_AUTOSTART_BACKEND=1 (see .env.example). If a backend is
 * already listening at BACKEND_URL nothing is spawned.
 */
import { spawn, type ChildProcess } from "node:child_process";
import { existsSync } from "node:fs";
import path from "node:path";
import { BACKEND_URL } from "./backend-url";

let child: ChildProcess | null = null;

async function reachable(url: string): Promise<boolean> {
  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 1500);
    const res = await fetch(`${url}/api/health`, { cache: "no-store", signal: controller.signal });
    clearTimeout(timer);
    return res.ok;
  } catch {
    return false;
  }
}

export async function startBackend(): Promise<void> {
  if (child) return;
  if (await reachable(BACKEND_URL)) {
    console.log(`[fedtraffic] backend already running at ${BACKEND_URL}`);
    return;
  }
  const backendDir = path.resolve(/*turbopackIgnore: true*/ process.cwd(), process.env.FEDTRAFFIC_BACKEND_DIR ?? "backend");
  if (!existsSync(path.join(/*turbopackIgnore: true*/ backendDir, "app", "main.py"))) {
    console.warn(`[fedtraffic] backend directory not found at ${backendDir}; skipping auto-start`);
    return;
  }
  const venvPython = path.join(/*turbopackIgnore: true*/ backendDir, ".venv", "bin", "python");
  const python = process.env.FEDTRAFFIC_PYTHON ?? (existsSync(venvPython) ? venvPython : "python3");
  const url = new URL(BACKEND_URL);
  const args = ["-m", "uvicorn", "app.main:app", "--host", url.hostname, "--port", url.port || "8000"];
  console.log(`[fedtraffic] starting backend: ${python} ${args.join(" ")} (cwd ${backendDir})`);
  child = spawn(python, args, {
    cwd: backendDir,
    stdio: ["ignore", "inherit", "inherit"],
    env: { ...process.env, PYTHONUNBUFFERED: "1" },
  });
  child.on("error", (err) => {
    console.warn(`[fedtraffic] could not start backend (${err.message}). Start it manually: cd backend && uvicorn app.main:app --port 8000`);
    child = null;
  });
  child.on("exit", (code) => {
    console.log(`[fedtraffic] backend process exited (code ${code})`);
    child = null;
  });
  const cleanup = () => {
    if (child && !child.killed) child.kill("SIGTERM");
  };
  process.once("exit", cleanup);
  process.once("SIGINT", cleanup);
  process.once("SIGTERM", cleanup);
}
