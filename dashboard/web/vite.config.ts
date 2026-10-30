import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The API (dashboard/server) runs on 4317; in dev, Vite forwards API and image requests to it.
const api = process.env.API_URL ?? "http://127.0.0.1:4317";

export default defineConfig({
  plugins: [react()],
  server: { port: 5317, proxy: { "/api": api, "/images": api } },
});
