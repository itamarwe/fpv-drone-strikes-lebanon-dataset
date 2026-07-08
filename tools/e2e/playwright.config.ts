import { defineConfig } from "@playwright/test";

const baseURL = process.env.FPV_TEST_BASE_URL ?? "http://127.0.0.1:8766";

export default defineConfig({
  testDir: ".",
  timeout: 60_000,
  retries: 0,
  use: {
    baseURL,
    headless: true,
  },
  reporter: [["list"]],
});
