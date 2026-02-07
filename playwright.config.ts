import { defineConfig, devices } from "@playwright/test";

export default defineConfig({
  testDir: "./tests",
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 2 : 0,
  workers: process.env.CI ? 1 : undefined,
  reporter: "html",
  use: {
    baseURL: "http://localhost:3000",
    trace: "on-first-retry",
  },
  projects: [
    {
      name: "Mobile 360",
      use: { ...devices["iPhone SE"], viewport: { width: 360, height: 640 } },
    },
    {
      name: "Mobile 390",
      use: { ...devices["iPhone 14"], viewport: { width: 390, height: 844 } },
    },
    {
      name: "Tablet 768",
      use: { viewport: { width: 768, height: 1024 } },
    },
    {
      name: "Desktop 1024",
      use: { viewport: { width: 1024, height: 768 } },
    },
    {
      name: "Desktop 1280",
      use: { viewport: { width: 1280, height: 800 } },
    },
    {
      name: "Desktop 1440",
      use: { viewport: { width: 1440, height: 900 } },
    },
  ],
  webServer: {
    command: "npm run dev",
    url: "http://localhost:3000",
    reuseExistingServer: !process.env.CI,
  },
});
