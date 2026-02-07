import { test, expect } from "@playwright/test";

test.describe("Responsive — Zéro scroll horizontal", () => {
  const routes = ["/", "/catalogue"];

  for (const route of routes) {
    test(`${route} — no horizontal scroll`, async ({ page }) => {
      await page.goto(route, { waitUntil: "networkidle" });
      const scrollWidth = await page.evaluate(() => document.documentElement.scrollWidth);
      const clientWidth = await page.evaluate(() => document.documentElement.clientWidth);
      expect(scrollWidth).toBeLessThanOrEqual(clientWidth);
    });
  }
});

test.describe("Navigation", () => {
  test("home page loads and has hero", async ({ page }) => {
    await page.goto("/", { waitUntil: "networkidle" });
    await expect(page.locator("h1")).toContainText("Agents IA");
  });

  test("catalogue page loads", async ({ page }) => {
    await page.goto("/catalogue", { waitUntil: "networkidle" });
    await expect(page.locator("h1")).toContainText("Catalogue");
  });

  test("use case card links to detail page", async ({ page }) => {
    await page.goto("/catalogue", { waitUntil: "networkidle" });
    const firstCard = page.locator("a[href^='/use-case/']").first();
    await firstCard.click();
    await page.waitForURL("**/use-case/**");
    await expect(page.locator("h1")).toBeVisible();
  });
});

test.describe("Responsive screenshots", () => {
  test("home page screenshot", async ({ page }) => {
    await page.goto("/", { waitUntil: "networkidle" });
    await expect(page.locator("h1")).toBeVisible();
    await page.screenshot({
      path: `tests/screenshots/home-${test.info().project.name}.png`,
      fullPage: true,
    });
  });
});
