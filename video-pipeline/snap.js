// usage: node snap.js timeline.json outprefix t1 t2 ...
const { chromium } = require('playwright'); const fs=require('fs'), path=require('path');
(async()=>{const [tlp,pre,...ts]=process.argv.slice(2); const tl=JSON.parse(fs.readFileSync(tlp,'utf8'));
const [w,h]=tl.size||[1080,1920]; const b=await chromium.launch(); const p=await b.newPage({viewport:{width:w,height:h}});
p.on('pageerror',e=>console.log('ERR',e.message)); p.on('console',m=>console.log('LOG',m.text()));
await p.goto('file://'+path.join(__dirname,tl.page||'player.html')); await p.evaluate(t=>window.setup(t),tl); await p.evaluate(()=>document.fonts.ready); await p.waitForTimeout(400);
for(const t of ts){ // walk time forward so scroll easing state is consistent
  for(let x=0;x<=+t;x+=0.2) await p.evaluate(v=>window.render(v),x);
  await p.evaluate(v=>window.render(v),+t); await p.screenshot({path:`${pre}_${t}.png`});}
await b.close();})();
