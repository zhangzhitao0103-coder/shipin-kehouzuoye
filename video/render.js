// 逐帧渲染 scene.html → 合成 mp4
// 用法: node render.js                     完整渲染到 ../output/
//       node render.js --cover             出封面 ../output/封面.jpg
//       node render.js --preview 3.2,8,15  只截几张静帧到 build/preview/
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

const HERE = __dirname;
const tl = JSON.parse(fs.readFileSync(path.join(HERE, 'build/timeline.json'), 'utf8'));
const args = process.argv.slice(2);
const previewArg = args.includes('--preview') ? args[args.indexOf('--preview') + 1] : null;
const outArg = args.includes('--out') ? args[args.indexOf('--out') + 1] : null;
const OUT = outArg || path.join(HERE, '..', 'output', '胡源_课前测答案速览_1080x1920.mp4');

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: tl.size[0], height: tl.size[1] }, deviceScaleFactor: 1 });
  page.on('pageerror', e => { console.error('page error:', e.message); process.exit(1); });
  await page.addInitScript(`window.TIMELINE = ${JSON.stringify(tl)};`);
  await page.goto('file://' + path.join(HERE, 'scene.html'));
  await page.waitForFunction(() => window.READY === true);
  await page.waitForTimeout(300);

  if (args.includes('--cover')) {           // 封面：开场画面定格，不带字幕
    await page.evaluate(t => window.render(t), tl.lines.L02.end - 0.05);
    await page.evaluate(() => { document.getElementById('cap').style.display = 'none'; });
    const cover = path.join(HERE, '..', 'output', '封面.jpg');
    await page.screenshot({ path: cover, type: 'jpeg', quality: 92 });
    console.log('cover →', cover);
    await browser.close();
    return;
  }

  if (previewArg) {
    const dir = path.join(HERE, 'build/preview');
    fs.mkdirSync(dir, { recursive: true });
    for (const t of previewArg.split(',').map(Number)) {
      await page.evaluate(t => window.render(t), t);
      await page.screenshot({ path: path.join(dir, `t${t.toFixed(2)}.png`) });
      console.log('preview', t);
    }
    await browser.close();
    return;
  }

  const frames = Math.ceil(tl.duration * tl.fps);
  fs.mkdirSync(path.dirname(OUT), { recursive: true });
  const ff = spawn('ffmpeg', ['-y', '-loglevel', 'error',
    '-f', 'image2pipe', '-framerate', String(tl.fps), '-c:v', 'png', '-i', '-',
    '-i', path.join(HERE, 'build/mix.wav'),
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '20', '-pix_fmt', 'yuv420p', '-profile:v', 'high', '-level', '4.1',
    '-af', 'loudnorm=I=-16:TP=-1.5:LRA=11', '-c:a', 'aac', '-b:a', '192k', '-ar', '48000', '-ac', '2',
    '-movflags', '+faststart', '-shortest', OUT], { stdio: ['pipe', 'inherit', 'inherit'] });
  const t0 = Date.now();
  for (let f = 0; f < frames; f++) {
    await page.evaluate(t => window.render(t), f / tl.fps);
    const buf = await page.screenshot({ type: 'png' });
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    if (f % 150 === 0) console.log(`frame ${f}/${frames}  ${((Date.now() - t0) / 1000).toFixed(0)}s`);
  }
  ff.stdin.end();
  await new Promise(r => ff.on('close', r));
  await browser.close();
  console.log('done →', OUT);
})();
