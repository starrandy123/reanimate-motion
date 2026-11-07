/* rmo film <work> [--character examples/models/street-skater.glb] [--video <the source video>] [--size 1080] [--bg #f4f2ee]
 *                 [--hide attach-righthand-mesh] [--still] [--frames N]
 * The character wearing <work>/solve/motion.json, drawn at the moment of EVERY video frame it was solved from, so it plays beside the
 * video frame for frame (a slow-motion clip stays slow, at the clock the clean step measured):
 *   <work>/film/character.mp4   the character alone, square, the camera following the hips (--still: it holds where it starts)
 *   <work>/film/tracked.mp4     the same frames of track_overlay.mp4 (the video with its mapping drawn)
 *   <work>/film/source.mp4      the same frames of the video itself, when --video is given
 *   <work>/film/side.mp4        tracked | character, to judge by eye
 * Video frame j (counted from the first tracked frame) is motion time (j - clip_from) / video_per_real_second. */
import http from 'node:http'; import fs from 'node:fs'; import path from 'node:path'; import { spawnSync } from 'node:child_process'; import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
let pw; try { pw = require('playwright'); } catch (e) { pw = require(path.join(process.env.HOME, 'Documents/node_modules/playwright')); }
const ROOT = path.resolve(path.dirname(new URL(import.meta.url).pathname), '..');
const work = path.resolve(process.argv[2] || ''); const arg = (k, d) => { const i = process.argv.indexOf('--' + k); return i > 0 ? process.argv[i + 1] : d; };
const solveDir = path.join(work, 'solve'), out = path.join(work, 'film'), frames = path.join(out, 'frames');
const readJSON = (f) => JSON.parse(fs.readFileSync(f, 'utf8').replace(/\bNaN\b/g, 'null'));   // clean.json keeps unseen points as NaN
const motion = readJSON(path.join(solveDir, 'motion.json')), clean = readJSON(path.join(work, 'clean.json'));
const vfps = clean.fps, real = motion.video_per_real_second, a = motion.clip_from || 0;
const b = motion.clip_to == null ? clean.X.length : motion.clip_to, first = clean.first_frame || 0;
const size = +arg('size', 1080), char = arg('character', 'examples/models/street-skater.glb');
fs.rmSync(frames, { recursive: true, force: true }); fs.mkdirSync(frames, { recursive: true });
const TYPES = { '.html': 'text/html', '.js': 'text/javascript', '.json': 'application/json', '.glb': 'model/gltf-binary' };
const srv = http.createServer((q, r) => {
  const u = decodeURIComponent(new URL(q.url, 'http://x').pathname); const f = u.startsWith('/data/') ? path.join(solveDir, u.slice(6)) : path.join(ROOT, u);
  if (!f.startsWith(ROOT) && !f.startsWith(solveDir)) { r.writeHead(403); return r.end(); }
  fs.readFile(f, (e, buf) => { if (e) { r.writeHead(404); return r.end(); } r.writeHead(200, { 'content-type': TYPES[path.extname(f)] || 'application/octet-stream' }); r.end(buf); });
}).listen(0, '127.0.0.1');
await new Promise((ok) => srv.on('listening', ok));
const browser = await pw.chromium.launch({ args: ['--ignore-gpu-blocklist', '--enable-gpu'] });
const ctx = await browser.newContext({ viewport: { width: size, height: size }, deviceScaleFactor: 1 });
await ctx.route('**/*', (r) => { const h = new URL(r.request().url()).hostname; return (h === '127.0.0.1' || h === 'cdn.jsdelivr.net') ? r.continue() : r.abort(); });
const page = await ctx.newPage(); const errs = []; page.on('pageerror', (e) => errs.push(String(e)));
const q = new URLSearchParams({ motion: '/data/motion.json', character: '/' + char, only: 'character', film: '1', ui: '0', follow: process.argv.includes('--still') ? '0' : '1' });
if (arg('bg', null)) q.set('bg', arg('bg')); if (arg('hide', null)) q.set('hide', arg('hide'));
await page.goto(`http://127.0.0.1:${srv.address().port}/examples/threejs/index.html?${q}`);
await page.waitForFunction(() => window.__rmo && window.__rmo.ready, null, { timeout: 60000 });
const n = Math.min(b - a, +arg('frames', 1e9));   // --frames N: only the first N (a quick look)
for (let j = 0; j < n; j++) {
  const t = Math.min(j / real, window_dur(motion));
  await page.evaluate((tt) => window.__rmo.set(tt), t);
  await page.locator('canvas').screenshot({ path: path.join(frames, String(j).padStart(5, '0') + '.jpg'), type: 'jpeg', quality: 92 });
}
function window_dur(m) { return (m.frames - 1) / m.fps; }
await browser.close(); srv.close();
const ff = (args) => { const r = spawnSync('ffmpeg', ['-v', 'error', '-y', ...args], { stdio: 'inherit' }); if (r.status) throw new Error('ffmpeg failed: ' + args.join(' ')); };
const enc = ['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-r', String(vfps)];
ff(['-framerate', String(vfps), '-i', path.join(frames, '%05d.jpg'), ...enc, path.join(out, 'character.mp4')]);
const cut = (src, dst, from) => ff(['-i', src, '-vf', `select='between(n\\,${from}\\,${from + n - 1})',setpts=N/(${vfps}*TB),scale=-2:${size}`, '-an', ...enc, dst]);
cut(path.join(work, 'track_overlay.mp4'), path.join(out, 'tracked.mp4'), a);
if (arg('video', null)) cut(path.resolve(arg('video')), path.join(out, 'source.mp4'), first + a);
ff(['-i', path.join(out, 'tracked.mp4'), '-i', path.join(out, 'character.mp4'), '-filter_complex', 'hstack=inputs=2', ...enc, path.join(out, 'side.mp4')]);
console.log(`film: ${n} frames at ${vfps} fps (${(n / vfps).toFixed(2)} s of video = ${(n / real).toFixed(2)} s of motion), ${errs.length} page errors${errs.length ? ': ' + errs.slice(0, 3).join(' | ') : ''} -> ${out}`);
