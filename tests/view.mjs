/* THE VIEWER, FILMED: the three.js example wearing a motion on a character, stilled at chosen times (the clock held: __rmo.set).
 *   node tests/view.mjs <motion dir> <out dir> [--character examples/models/street-skater.glb] [--times 0,0.1,...] [--cam x,y,z]
 * Serves the kit and the motion over http on 127.0.0.1 (ES modules do not load from file://); allows only itself and jsdelivr
 * (three.js). Writes <out>/view_<i>.png and <out>/view.json (the times). */
import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path'; import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
let pw; try { pw = require('playwright'); } catch (e) { pw = require(path.join(process.env.HOME, 'Documents/node_modules/playwright')); }
const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const [motionDir, outDir] = process.argv.slice(2); const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
fs.mkdirSync(outDir, { recursive: true });
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.glb': 'model/gltf-binary', '.bvh': 'text/plain' };
const srv = http.createServer((q, r) => {
  const u = decodeURIComponent(new URL(q.url, 'http://x').pathname); const f = u.startsWith('/data/') ? path.join(path.resolve(motionDir), u.slice(6)) : path.join(ROOT, u);
  if (!f.startsWith(ROOT) && !f.startsWith(path.resolve(motionDir))) { r.writeHead(403); return r.end(); }
  fs.readFile(f, (e, b) => { if (e) { r.writeHead(404); return r.end(); } r.writeHead(200, { 'content-type': TYPES[path.extname(f)] || 'application/octet-stream' }); r.end(b); });
}).listen(0, '127.0.0.1');
await new Promise((ok) => srv.on('listening', ok)); const port = srv.address().port;
const b = await pw.chromium.launch({ args: ['--ignore-gpu-blocklist', '--enable-gpu'] });
const ctx = await b.newContext({ viewport: { width: 960, height: 1080 } });
await ctx.route('**/*', (r) => { const h = new URL(r.request().url()).hostname; return (h === '127.0.0.1' || h === 'cdn.jsdelivr.net') ? r.continue() : r.abort(); });
const page = await ctx.newPage(); const errs = []; page.on('pageerror', (e) => errs.push(String(e))); page.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
const char = arg('character', 'examples/models/street-skater.glb');
await page.goto(`http://127.0.0.1:${port}/examples/threejs/index.html?motion=/data/motion.json&character=/${char}`);
await page.waitForFunction(() => window.__rmo && window.__rmo.ready, null, { timeout: 60000 });
const info = await page.evaluate(() => ({ duration: window.__rmo.duration, bones: window.__rmo.bones }));
const cam = arg('cam', null);
if (cam) await page.evaluate((c) => { const [x, y, z] = c.split(',').map(Number); window.__rmo.camera.position.set(x, y, z); window.__rmo.controls.update(); }, cam);
const times = (arg('times', null) || Array.from({ length: 8 }, (_, i) => (i * info.duration / 8).toFixed(3)).join(',')).split(',').map(Number);
const probes = [];
for (let i = 0; i < times.length; i++) { probes.push(await page.evaluate((t) => window.__rmo.probe(t), times[i])); await page.screenshot({ path: path.join(outDir, `view_${i}.png`) }); }
fs.writeFileSync(path.join(outDir, 'probe.json'), JSON.stringify(probes));
fs.writeFileSync(path.join(outDir, 'view.json'), JSON.stringify({ times, duration: info.duration, bones_moved: info.bones, errors: errs }));
console.log(`viewer: ${times.length} stills, ${info.bones.length} of the character's bones moved, ${errs.length} page errors${errs.length ? ': ' + errs.slice(0, 3).join(' | ') : ''}`);
await b.close(); srv.close();
