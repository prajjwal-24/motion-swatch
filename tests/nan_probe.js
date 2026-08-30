/* Version-agnostic: does applying MOTION to #Scarf produce NaN in any path `d`?
   Touches no method that only exists on one revision, so it can bisect animate.js. */
const puppeteer = require('puppeteer-core');
(async () => {
  const b = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', args: ['--no-sandbox'] });
  const p = await b.newPage();
  const errs = []; p.on('pageerror', e => errs.push(String(e)));
  p.on('console', m => { if (m.type() === 'error') errs.push(m.text()); });
  await p.goto('http://localhost:8000/index.html', { waitUntil: 'networkidle0' });
  const svgText = require('fs').readFileSync('/Users/prajjwas/Downloads/Scene3.svg', 'utf8');
  const r = await p.evaluate(async (svgText, id, wave) => {
    const { sel, library, animator, loadUploadedSVG } = window.__ms;
    await loadUploadedSVG(svgText, 'Scene3.svg'); await new Promise(r => setTimeout(r, 400));
    if (!library.getById(id)) return { error: `no motion ${id}` };
    const svg = document.querySelector('#artwork-container svg');
    sel._createSVGSelection(sel._wrapOne(svg.querySelector('#Scarf')));
    const s = sel.selections[sel.activeIdx];
    s.motionId = id; s.intensity = 1.0; if (wave) s.waveMode = true;
    if (sel.setHighlightsHidden) sel.setHighlightsHidden(true);
    let nan = 0, seen = 0;
    for (let i = 0; i < 12; i++) {
      animator._applyAll(i * 4 / 12);
      for (const el of svg.querySelectorAll('#Scarf path')) {
        seen++; if (/NaN/.test(el.getAttribute('d') || '')) nan++;
      }
    }
    return { nan, seen, warp: !!s._warp };
  }, svgText, process.argv[2] || 'flag-flutter', process.env.MS_WAVE === '1');
  await b.close();
  console.log(JSON.stringify(r), errs.length ? `errors=${errs.length}` : 'errors=0');
})();
