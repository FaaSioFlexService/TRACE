/**
 * Configuración de Vite para la aplicación de benchmarking.
 *
 * COOP/COEP se preparan desde el principio porque serán necesarios cuando el
 * backend incorpore SharedArrayBuffer y WebAssembly threads.
 */

import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  preview: {
    headers: {
      "Cross-Origin-Opener-Policy": "same-origin",
      "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
});
