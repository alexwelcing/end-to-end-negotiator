// usage: node shoot.mjs <dir> <width> <height> <scale>   — screenshots dir/*.html to dir/*.png
import { createRequire } from "node:module";
const { chromium } = createRequire(import.meta.url)("playwright");   // the pre-installed Playwright is on NODE_PATH
import { readdirSync } from "node:fs";
import { join, resolve } from "node:path";
const [dir, w, h, scale] = process.argv.slice(2);
const b = await chromium.launch();
const p = await b.newPage({ viewport: { width: +w, height: +h }, deviceScaleFactor: +scale });
for (const f of readdirSync(dir).filter((x) => x.endsWith(".html")).sort()) {
  await p.goto("file://" + resolve(dir, f));
  await p.screenshot({ path: join(dir, f.replace(/\.html$/, ".png")) });
}
await b.close();
