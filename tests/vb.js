const puppeteer = require('puppeteer-core');
(async () => {
  const b = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', args: ['--no-sandbox'] });
  const p = await b.newPage();
  await p.goto('http://localhost:8000/index.html', { waitUntil: 'networkidle0' });
  const t = require('fs').readFileSync('/Users/prajjwas/Downloads/Scene3.svg', 'utf8');
  console.log(JSON.stringify(await p.evaluate(async (t) => {
    const { sel, loadUploadedSVG } = window.__ms;
    await loadUploadedSVG(t, 'Scene3.svg'); await new Promise(r => setTimeout(r, 400));
    const svg = document.querySelector('#artwork-container svg');
    const r = svg.getBoundingClientRect(), vb = svg.viewBox.baseVal;
    sel._createSVGSelection(sel._wrapOne(svg.querySelector('#Scarf')));
    const probe = f => {
      const e = { clientX: r.left + r.width * f, clientY: r.top + r.height * 0.5 };
      const q = sel._toViewBox(e);
      return q ? [+q[0].toFixed(1), +q[1].toFixed(1)] : null;
    };
    return {
      rect: [r.left, r.top, r.width, r.height].map(v => +v.toFixed(1)),
      viewBox: [vb.x, vb.y, vb.width, vb.height],
      svgAttrs: { w: svg.getAttribute('width'), h: svg.getAttribute('height'),
                  par: svg.getAttribute('preserveAspectRatio') },
      at0: probe(0), at05: probe(0.5), at1: probe(1.0),
      container: (() => { const c = document.getElementById('artwork-container').getBoundingClientRect();
                          return [c.left, c.top, c.width, c.height].map(v => +v.toFixed(1)); })(),
    };
  }, t), null, 1));
  await b.close();
})();
