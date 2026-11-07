/* node tests/hosts.mjs <motion.json> <character.glb> <blender-baked.glb>: the two hosts' bone places, compared (tests/hosts.html) */
import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path'; import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
let pw; try { pw = require('playwright'); } catch (e) { pw = require(path.join(process.env.HOME, 'Documents/node_modules/playwright')); }
const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..'); const files = process.argv.slice(2).filter((f) => !f.startsWith('--')).map((f) => path.resolve(f));
const T = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.glb': 'model/gltf-binary' };
const srv = http.createServer((q, r) => { const u = decodeURIComponent(new URL(q.url, 'http://x').pathname); const f = u.startsWith('/f/') ? files[+u.split('/')[2]] : path.join(ROOT, u);
  fs.readFile(f, (e, b) => { if (e) { r.writeHead(404); return r.end(); } r.writeHead(200, { 'content-type': T[path.extname(f)] || 'application/octet-stream' }); r.end(b); }); }).listen(0, '127.0.0.1');
await new Promise((ok) => srv.on('listening', ok)); const port = srv.address().port;
const b = await pw.chromium.launch(); const page = await (await b.newContext()).newPage(); const errs = []; page.on('pageerror', (e) => errs.push(String(e)));
await page.goto(`http://127.0.0.1:${port}/tests/hosts.html?motion=/f/0/m.json&character=/f/1/c.glb&baked=/f/2/b.glb` + (process.argv.includes('--plant-late') ? '&plant=late' : ''));
const r = await page.waitForFunction(() => window.__hosts, null, { timeout: 120000 }).then((h) => h.jsonValue()).catch(() => null);
console.log(JSON.stringify(r || { errors: errs }));
await b.close(); srv.close(); process.exit(r && r.worst_cm < 2.0 && r.bones >= 15 ? 0 : 1);
