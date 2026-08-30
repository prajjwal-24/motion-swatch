/* A route committed while the scene is already playing must still begin its journey at the
   START of the route, not wherever the global clock happens to be in the loop. */
const puppeteer = require('puppeteer-core');
const { installSwatch } = require('./installSwatch');
(async () => {
  const b = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', args: ['--no-sandbox'] });
  const p = await b.newPage();
  await p.setViewport({ width: 1600, height: 1000 });
  const errs = []; p.on('pageerror', e => errs.push(String(e)));
  await p.goto('http://localhost:8000/index.html', { waitUntil: 'networkidle0' });
  const sw = await installSwatch(p);
  if (!sw.ok) { console.log('SETUP FAILED: ' + sw.why); process.exit(1); }
  const svgText = require('fs').readFileSync('/Users/prajjwas/Downloads/Scene3.svg', 'utf8');
  const out = await p.evaluate(async (svgText) => {
    const { sel, library, animator, loadUploadedSVG } = window.__ms;
    await loadUploadedSVG(svgText, 'Scene3.svg'); await new Promise(r => setTimeout(r, 400));
    const svg = document.querySelector('#artwork-container svg');
    sel._createSVGSelection(sel._wrapOne(svg.querySelector('#Scarf')));
    const s = sel.selections[sel.activeIdx];
    s.motionId = 'flutter-flag-autumn';
    if (sel.setHighlightsHidden) sel.setHighlightsHidden(true);
    const at = (vx, vy) => { const m = svg.getScreenCTM(), q = svg.createSVGPoint();
      q.x = vx; q.y = vy; const c = q.matrixTransform(m);
      const o = { clientX: c.x, clientY: c.y, bubbles: true, cancelable: true };
      svg.dispatchEvent(new MouseEvent('mousemove', o)); svg.dispatchEvent(new MouseEvent('click', o)); };
    sel.beginRoute(); at(470, 135); at(667, 67);
    svg.dispatchEvent(new MouseEvent('dblclick', { bubbles: true, cancelable: true }));

    const off = (t) => {
      animator._applyAll(t);
      const m = /translate\(([-\d.]+) ([-\d.]+)\)/.exec(s.wrap.getAttribute('transform') || '');
      return m ? +Math.hypot(+m[1], +m[2]).toFixed(1) : 0;
    };
    // clocks a user could realistically be at when they finish drawing the route
    const res = {};
    for (const start of [0, 2.9, 6.1, 13.7]) {
      s._routeT0 = null; s._routeT0Rev = null;        // as if the route were just committed
      res['t0=' + start] = [0, 0.25, 0.5, 1, 2].map(d => off(start + d));
    }
    return res;
  }, svgText);
  await b.close();
  console.log('offset (px) at +0, +0.25, +0.5, +1, +2s after the route is committed:');
  for (const k of Object.keys(out)) console.log(`  clock ${k.padEnd(9)} -> ${out[k].join(', ')}`);
  console.log(errs.length ? 'PAGE ERRORS: ' + errs.join(' | ') : 'no page errors');
})();
