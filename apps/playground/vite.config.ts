import { defineConfig } from 'vite';
import { fileURLToPath, URL } from 'node:url';

export default defineConfig(({ command }) => ({
  base: command === 'build' ? '/sparkplusplus/' : '/',
  server: {
    // just allow all hosts
    allowedHosts: true,
  },
  resolve: {
    alias: {
      '@sparkplusplus/spark': fileURLToPath(
        new URL('../../packages/spark/src/index.ts', import.meta.url),
      ),
    },
  },
}));
