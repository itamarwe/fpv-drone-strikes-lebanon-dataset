import { expect, test } from "@playwright/test";

const SCENE_VIEWER =
  "/scenes/2026-06-05_merkava_tank_beaufort_castle_mmirleb_17447/2026-06-05_merkava_tank_beaufort_castle_mmirleb_17447_seg01/viewer/index.html";

const OPTIONAL_404 = /\/tools\/scene_viewer\/assets\/fpv_kamikaze_drone\.glb$/;

test("scene viewer loads without 404s", async ({ page }) => {
  const failed: string[] = [];
  page.on("response", (response) => {
    if (response.status() !== 404) return;
    if (OPTIONAL_404.test(response.url())) return;
    failed.push(`${response.status()} ${response.url()}`);
  });

  await page.goto(SCENE_VIEWER, { waitUntil: "load" });

  const html = await page.content();
  expect(html).not.toContain("__APP_BASE__");
  expect(html).not.toContain("__API_BASE__");

  await expect(page.locator("#title")).not.toHaveText("VGGT Scene", { timeout: 15_000 });
  await expect(page.locator("#meta")).not.toBeEmpty();

  expect(failed, `unexpected 404 responses:\n${failed.join("\n")}`).toEqual([]);
});

test("scene browser lists scenes", async ({ page }) => {
  await page.goto("/scenes/");
  await expect(page.locator("body")).toContainText(/scene/i);
});
