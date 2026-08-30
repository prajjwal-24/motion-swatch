/* Same anchor/tip measurement as scarf_anchor.js, but on the built-in scenery flag —
   asking whether the extracted cloth swatch picks the right END to anchor on an object
   it was not tuned against. wave.js compares two moments 250ms apart, which measures
   velocity at an arbitrary phase; this measures displacement from rest over a whole loop. */
const puppeteer = require('puppeteer-core');
const { installSwatch } = require('./installSwatch');
const { pathPoints } = require('./pathabs');

(async () => {
  const b = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', args: ['--no-sandbox'] });
  const p = await b.newPage();
  const errs = []; p.on('pageerror', e => errs.push(String(e)));
  await p.goto('http://localhost:8000/index.html', { waitUntil: 'networkidle0' });
  const swatch = await installSwatch(p);
  if (!swatch.ok) { console.log('SETUP FAILED: ' + swatch.why); process.exit(1); }
  const out = await p.evaluate(async () => {
    const { sel, library, animator, loadScene } = window.__ms;
    await loadScene('scenery'); await new Promise(r => setTimeout(r, 400));
    const svg = document.querySelector('#artwork-container svg');
    sel._createSVGSelection(svg.querySelector('[data-ms-name="flag"]'));
    const s = sel.selections[sel.activeIdx];
    s.motionId = 'flutter-flag-autumn'; s.intensity = 1.0;
    if (sel.setHighlightsHidden) sel.setHighlightsHidden(true);
    animator._applyAll(0);
    const ax = s._ribbon.axis, rows = [];
    for (let i = 0; i < 48; i++) {
      const t = i * 4 / 48;
      animator._applyAll(t);
      rows.push(s._warp.els.map(o => ({ d0: o.d0, d: o.el.getAttribute('d') })));
    }
    return { pole: (() => {   // where the pole is, so "anchored end" can be checked
      const f = svg.querySelector('[data-ms-name="flag"]').getBBox();
      const p = svg.querySelector('[data-ms-name="pole"]');
      return p ? { x: p.getBBox().x, flagX: f.x, flagW: f.width } : null;
    })(), ax: { cos: ax.cos, sin: ax.sin, mx: ax.mx, my: ax.my, sMin: ax.sMin, len: ax.len, half: ax.half }, rows };
  });
  await b.close();
  const { cos, sin, mx, my, sMin, len, half } = out.ax;
  console.log('pole vs flag bbox:', JSON.stringify(out.pole));
  console.log(`axis direction: cos=${cos.toFixed(3)} sin=${sin.toFixed(3)} (u=0 end is at the ${cos>=0?'LEFT':'RIGHT'})`);
  console.log(`axis len=${len.toFixed(1)}px  half-width=${half.toFixed(1)}px (thickness ${(2*half).toFixed(1)}px)`);
  // anchor band = nearest 10% of the rest length; tip band = farthest 10%
  let aMax = 0, tipMax = 0, pullMax = 0;
  for (const frame of out.rows) {
    for (const { d0, d } of frame) {
      const A = pathPoints(d0), B = pathPoints(d);
      if (A.length !== B.length) continue;
      for (let i = 0; i + 1 < A.length; i += 2) {
        const u = ((A[i] - mx) * cos + (A[i + 1] - my) * sin - sMin) / len;
        const disp = Math.hypot(B[i] - A[i], B[i + 1] - A[i + 1]);
        if (u <= 0.10) aMax = Math.max(aMax, disp);
        if (u >= 0.90) {
          tipMax = Math.max(tipMax, disp);
          const sr = (B[i] - mx) * cos + (B[i + 1] - my) * sin - sMin;   // along-axis reach
          pullMax = Math.max(pullMax, len - sr);
        }
      }
    }
  }
  console.log(`anchor band (u<=0.10): max displacement over the loop  ${aMax.toFixed(2)}px  (${(100*aMax/len).toFixed(2)}% of length)`);
  console.log(`free end   (u>=0.90): max displacement over the loop  ${tipMax.toFixed(1)}px  (${(100*tipMax/len).toFixed(0)}% of length)`);
  console.log(`free end pulls back toward the anchor by up to        ${pullMax.toFixed(1)}px  (${(100*pullMax/len).toFixed(0)}% of length)`);
  console.log(errs.length ? `PAGE ERRORS: ${errs.join(' | ')}` : 'no page errors');
})();
