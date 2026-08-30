/* Does the authored route guide leak into the exported artwork, and does travel survive? */
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
    const at = (vx, vy) => { const m = svg.getScreenCTM(), q = svg.createSVGPoint();
      q.x = vx; q.y = vy; const c = q.matrixTransform(m);
      const o = { clientX: c.x, clientY: c.y, bubbles: true, cancelable: true };
      svg.dispatchEvent(new MouseEvent('mousemove', o)); svg.dispatchEvent(new MouseEvent('click', o)); };
    sel.beginRoute(); at(470, 135); at(667, 67);
    svg.dispatchEvent(new MouseEvent('dblclick', { bubbles: true, cancelable: true }));
    const withMotion = buildExportSVG(sel, library) || '';
    s.motionId = null;                       // route only
    const routeOnly = buildExportSVG(sel, library);
    return {
      guideInDom: !!svg.querySelector('#ms-routes'),
      guideInExport: withMotion.includes('ms-routes'),
      dashesInExport: (withMotion.match(/stroke-dasharray/g) || []).length,
      hasTravel: /animateTransform|translate\(/.test(withMotion),
      routeOnlyExport: routeOnly === null ? 'null (object skipped entirely)' : `${routeOnly.length} bytes`,
    };
  }, svgText);
  await b.close();
  console.log(out);
  console.log(errs.length ? 'PAGE ERRORS: ' + errs.join(' | ') : 'no page errors');
})();
