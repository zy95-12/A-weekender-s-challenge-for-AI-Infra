/* Read-only verification of chapter 05 against its actual published evidence. */
const fs = require("fs");
const path = require("path");
const assert = require("assert");
const { chromium } = require("playwright");

(async () => {
  const base = process.env.DEMO_URL || "http://127.0.0.1:8088";
  const out = process.env.DEMO_EVIDENCE_DIR || path.join(__dirname, "evidence");
  const browser = await chromium.launch({
    headless: true,
    args: ["--no-sandbox"],
    ...(process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH
      ? { executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH }
      : {}),
  });
  try {
    const page = await browser.newPage({
      viewport: { width: 1440, height: 1000 },
    });
    const errors = [];
    page.on("pageerror", (e) => errors.push(e.message));
    await page.goto(base, { waitUntil: "networkidle" });
    const response = await page.request.get(base + "/api/evidence");
    assert.equal(response.status(), 200);
    const evidence = (await response.json())["open-loop-sweep"];
    assert(evidence && evidence.points.length);
    await page.waitForFunction(() =>
      document.querySelector("#open-loop-ttft svg"),
    );
    assert.equal(
      await page.locator("#open-loop-table tbody tr").count(),
      evidence.points.length,
    );
    assert.equal(
      await page.locator("#open-loop-status").textContent(),
      evidence.conclusion,
    );
    const cells = await page
      .locator("#open-loop-table tbody tr")
      .allTextContents();
    evidence.points.forEach((r, i) => {
      assert(cells[i].includes(r.completed_qps.toFixed(3)));
      assert(cells[i].includes(r.target_arrival_rate.toFixed(3)));
      assert(cells[i].includes((r.arrival_slo * 100).toFixed(2)));
    });
    for (const metric of ["ttft", "tpot"]) {
      const plot = page.locator(`#open-loop-${metric} svg`);
      assert.equal(
        await plot.locator("circle").count(),
        evidence.points.length * 2,
      );
      assert((await plot.textContent()).includes("Baseline"));
      assert((await plot.textContent()).includes("优化系统"));
    }
    const downloads = {};
    for (const file of [
      "open-loop-sweep.csv",
      "open-loop-ttft-qps.png",
      "open-loop-tpot-qps.png",
      "open-loop-raw.zip",
      "open-loop-report.md",
    ]) {
      const r = await page.request.get(base + "/evidence/" + file);
      assert.equal(r.status(), 200, file);
      const size = (await r.body()).length;
      assert(size > 0);
      downloads[file] = size;
    }
    await page.locator("#open-loop-status").scrollIntoViewIfNeeded();
    await page
      .locator("#open-loop-ttft")
      .screenshot({ path: path.join(out, "open-loop-browser-ttft.png") });
    await page
      .locator("#open-loop-tpot")
      .screenshot({ path: path.join(out, "open-loop-browser-tpot.png") });
    await page.setViewportSize({ width: 390, height: 844 });
    await page.locator("#open-loop-ttft").scrollIntoViewIfNeeded();
    assert(await page.locator("#open-loop-ttft svg").isVisible());
    assert.deepEqual(errors, []);
    const report = {
      url: base,
      points: evidence.points.length,
      conclusion: evidence.conclusion,
      errors,
      downloads,
      desktop_and_mobile: true,
    };
    fs.writeFileSync(
      path.join(out, "open-loop-browser-validation.json"),
      JSON.stringify(report, null, 2),
    );
    console.log(JSON.stringify(report));
  } finally {
    await browser.close();
  }
})().catch((e) => {
  console.error(e);
  process.exit(1);
});
