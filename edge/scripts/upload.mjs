// Upload an edge export (and any PULSE patches) to R2 with wrangler.
// usage: node scripts/upload.mjs <spec> <edge.json> [patch_dir]
import { execSync } from "node:child_process";
import { readdirSync } from "node:fs";
const [spec, file, patches] = process.argv.slice(2);
if (!spec || !file) { console.error("usage: node scripts/upload.mjs <spec> <edge.json> [patch_dir]"); process.exit(1); }
execSync(`wrangler r2 object put nerve-weights/${spec}/edge.json --file ${file}`, { stdio: "inherit" });
if (patches) for (const p of readdirSync(patches).filter((f) => f.endsWith(".nrvp")))
  execSync(`wrangler r2 object put nerve-weights/${spec}/patches/${p} --file ${patches}/${p}`, { stdio: "inherit" });
