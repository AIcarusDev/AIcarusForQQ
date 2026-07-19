import { fileURLToPath, URL } from "node:url";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

export default defineConfig({
  base: "/static/new/",
  plugins: [react()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  build: {
    outDir: "../src/static/new",
    emptyOutDir: true,
  },
  server: {
    host: "0.0.0.0",
    allowedHosts: ["localhost", "127.0.0.1", "terminal.local"],
    proxy: {
      "/api": "http://127.0.0.1:5000",
      "/login": "http://127.0.0.1:5000",
    },
  },
});
