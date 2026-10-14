/** Server-side only: where the FastAPI backend lives. */
export const BACKEND_URL = (process.env.BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");
