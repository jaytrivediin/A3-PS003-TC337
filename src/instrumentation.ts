export async function register() {
  if (process.env.NEXT_RUNTIME !== "nodejs") return;
  if (process.env.NEXT_PHASE === "phase-production-build") return;
  if (process.env.FEDTRAFFIC_AUTOSTART_BACKEND !== "1") return;
  const { startBackend } = await import("./lib/backend-launcher");
  await startBackend();
}
