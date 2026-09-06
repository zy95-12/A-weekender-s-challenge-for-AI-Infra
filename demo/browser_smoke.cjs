// Optional real four-GPU browser smoke. Requires Playwright; begin with ./poc down.
const { chromium } = require("playwright");
const fs = require("fs");
const path = require("path");
const output = process.env.DEMO_EVIDENCE_DIR || "/tmp/live-demo-smoke";
fs.mkdirSync(output, { recursive: true });
(async () => {
  const b = await chromium.launch({
    headless: true,
    executablePath: process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH,
    args: ["--no-sandbox"],
  });
  const page = await b.newPage({ viewport: { width: 1440, height: 1100 } });
  page.setDefaultTimeout(600000);
  const errors = [],
    jobs = [];
  page.on("pageerror", (e) => errors.push(e.message));
  page.on("response", async (r) => {
    if (r.url().includes("/api/jobs/")) {
      const j = await r.json();
      if (j.status !== "running") jobs.push(j);
    }
  });
  await page.goto(process.env.DEMO_URL || "http://127.0.0.1:8088");
  await page.waitForFunction(
    () => document.querySelector("#service-state").textContent === "服务未启动",
  );
  await page.click("#service-toggle");
  await page.waitForFunction(
    () => !document.querySelector("#service-toggle").disabled,
  );
  if (
    !(await page.locator("#service-state").innerText()).includes("服务已启动")
  )
    throw Error(await page.locator("#service-log").innerText());
  const chats = [];
  for (const prompt of [
    "请用一句话介绍你自己。",
    "What is 17 plus 26? Answer briefly.",
  ]) {
    await page.fill("#chat-text", prompt);
    await page.click("#chat-form button");
    await page.waitForFunction(
      () => !document.querySelector("#chat-form button").disabled,
    );
    const text = await page
      .locator("#messages .assistant p")
      .last()
      .innerText();
    if (!text || text.includes("[失败]")) throw Error(text);
    chats.push({
      prompt,
      text,
      ttft: await page.locator("#chat-ttft").innerText(),
      tpot: await page.locator("#chat-tpot").innerText(),
    });
  }
  await page.fill(
    "#attack-text",
    "企业数据仅用于本次演示。The secret code is 7391.",
  );
  await page.click("#encode-run");
  await page.waitForFunction(
    () => !document.querySelector("#encode-run").disabled,
  );
  if (await page.locator("#attack-run").isDisabled())
    throw Error(await page.locator("#attack-output").innerText());
  await page.click("#attack-run");
  await page.waitForFunction(
    () => !document.querySelector("#attack-run").disabled,
  );
  const recovered = await page.locator("#recovered-output").innerText();
  if (!recovered.includes("100.00%")) throw Error(recovered);
  for (const variant of ["optimized", "baseline"]) {
    await page.selectOption("#variant", variant);
    await page.fill("#sim-levels", "1");
    await page.click("#sim-run");
    await page.waitForFunction(
      () => !document.querySelector("#sim-run").disabled,
    );
    if (!(await page.locator("#sim-status").innerText()).includes("仿真完成"))
      throw Error(await page.locator("#sim-status").innerText());
  }
  const report = { errors, chats, recovered, jobs };
  fs.writeFileSync(
    path.join(output, "demo-browser-e2e.json"),
    JSON.stringify(report, null, 2),
  );
  await page
    .locator("#security")
    .screenshot({ path: path.join(output, "demo-security-live.png") });
  await page
    .locator("#system")
    .screenshot({ path: path.join(output, "demo-system-live.png") });
  await page
    .locator("#optimization")
    .screenshot({ path: path.join(output, "demo-optimization-live.png") });
  await b.close();
  console.log(
    JSON.stringify({
      errors,
      chats,
      recovered,
      jobs: jobs.map((j) => ({ id: j.id, kind: j.kind, status: j.status })),
    }),
  );
  if (errors.length) process.exit(1);
})();
