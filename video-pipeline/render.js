// usage: node render.js timeline.json out.mp4
const { chromium } = require('playwright');
const { spawn } = require('child_process');
const fs = require('fs'), path = require('path');
(async () => {
  const [tlPath, out] = process.argv.slice(2);
  const tl = JSON.parse(fs.readFileSync(tlPath, 'utf8'));
  const fps = tl.fps, n = Math.ceil(tl.duration * fps);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 } });
  await page.goto('file://' + path.join(__dirname, 'player.html'));
  await page.evaluate(t => window.setup(t), tl);
  await page.evaluate(() => document.fonts.ready);
  await page.waitForTimeout(500);
  const ff = spawn('ffmpeg', ['-v', 'error', '-y', '-f', 'image2pipe', '-framerate', String(fps), '-c:v', 'mjpeg', '-i', '-',
    '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', out], { stdio: ['pipe', 'inherit', 'inherit'] });
  for (let i = 0; i < n; i++) {
    await page.evaluate(t => window.render(t), i / fps);
    const buf = await page.screenshot({ type: 'jpeg', quality: 92 });
    if (!ff.stdin.write(buf)) await new Promise(r => ff.stdin.once('drain', r));
    if (i % 250 === 0) console.log(`frame ${i}/${n}`);
  }
  ff.stdin.end();
  await new Promise(r => ff.on('close', r));
  await browser.close();
})();
