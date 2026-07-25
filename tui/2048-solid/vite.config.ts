import { univiewSolid } from "@uniview/tui-solid/vite";
import { defineConfig } from "vitest/config";

export default defineConfig({
  // @uniview/tui-solid@0.0.1's runtime is compatible with Vite 5, but its
  // published hook type is narrower than Vite's Plugin type. Keep the adapter
  // at this app boundary until the package declaration is widened upstream.
  plugins: [univiewSolid()],
  test: {
    environment: "node",
    include: ["tests/**/*.test.{ts,tsx}"],
  },
});
