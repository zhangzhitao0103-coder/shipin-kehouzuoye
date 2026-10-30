// usage: node render.js preview out_dir t1 t2 ...   |  node render.js frames out_dir fps start end
const { chromium } = require('playwright');
const path = require('path'); const fs = require('fs');
(async () => {
  const [mode, out, ...rest] = process.argv.slice(2);
  fs.mkdirSync(out, { recursive: true });
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1080, height: 1920 }, deviceScaleFactor: 1 });
  const errs = [];
  page.on('pageerror', e => errs.push(String(e))); page.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  await page.goto('file://' + path.resolve(__dirname, 'page.html'));
  await page.evaluate(() => window.ready);
  await page.waitForTimeout(300);
  if (mode === 'preview') {
    for (const t of rest.map(Number)) {
      await page.evaluate(t => window.seek(t), t);
      await page.screenshot({ path: path.join(out, `t${t.toFixed(2)}.png`) });
    }
  } else {
    const [fps, a, b] = rest.map(Number);
    for (let f = a; f < b; f++) {
      await page.evaluate(t => window.seek(t), f / fps);
      await page.screenshot({ path: path.join(out, `f${String(f).padStart(5, '0')}.jpg`), type: 'jpeg', quality: 93 });
    }
  }
  if (errs.length) console.log('ERRORS:', errs.join('\n'));
  await browser.close();
})();
