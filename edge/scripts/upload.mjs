// Upload export/<spec>/ (edge.json + patches/*.nrvp) to R2 with wrangler. Same layout workerd serves from disk.
// usage: node scripts/upload.mjs <spec> [export_dir=export]
import { execSync } from "node:child_process";
import { existsSync, readdirSync } from "node:fs";
const [spec, root = "export"] = process.argv.slice(2);
if (!spec) { console.error("usage: node scripts/upload.mjs <spec> [export_dir]"); process.exit(1); }
const put = (key, file) => execSync(`wrangler r2 object put nerve-weights/${key} --file ${file}`, { stdio: "inherit" });
put(`${spec}/edge.json`, `${root}/${spec}/edge.json`);
const pd = `${root}/${spec}/patches`;
if (existsSync(pd)) for (const p of readdirSync(pd).filter((f) => f.endsWith(".nrvp"))) put(`${spec}/patches/${p}`, `${pd}/${p}`);
