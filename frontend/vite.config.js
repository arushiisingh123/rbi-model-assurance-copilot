import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    // Dev-only convenience: lets the app call /api/... on the same origin so
    // there is no CORS setup on the FastAPI side. The API layer uses
    // VITE_API_BASE_URL when set, so production builds bypass this entirely.
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
    },
  },
  build: { outDir: "dist", sourcemap: false },
});
