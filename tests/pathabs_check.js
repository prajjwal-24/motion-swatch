/* Validate pathPoints() against the browser's own path geometry.
   A parser that gets relative commands wrong is worse than no metric at all, so it is
   checked against getBBox() (which the renderer computes) on real artwork. */
const puppeteer = require('puppeteer-core');
const { pathPoints } = require('./pathabs');

(async () => {
  const b = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', args: ['--no-sandbox'] });
  const p = await b.newPage();
  await p.goto('http://localhost:8000/index.html', { waitUntil: 'networkidle0' });
  const t = require('fs').readFileSync('/Users/prajjwas/Downloads/Scene3.svg', 'utf8');
  const data = await p.evaluate(async (t) => {
    const { loadUploadedSVG } = window.__ms;
    await loadUploadedSVG(t, 'Scene3.svg'); await new Promise(r => setTimeout(r, 300));
    const svg = document.querySelector('#artwork-container svg');
    return [...svg.querySelectorAll('#Scarf path')].map(el => {
      const bb = el.getBBox(), L = el.getTotalLength();
      const e = el.getPointAtLength(L), s = el.getPointAtLength(0);
      return { d: el.getAttribute('d'), bb: [bb.x, bb.y, bb.width, bb.height],
               start: [s.x, s.y], end: [e.x, e.y] };
    });
  }, t);
  await b.close();

  // Control-point hulls bound the curve, so my bbox is >= the true one; the START point
  // must match exactly, and the max coordinate error is what actually matters.
  let worstStart = 0, worstSlack = 0, n = 0;
  for (const o of data) {
    const pts = pathPoints(o.d);
    if (pts.length < 2) continue;
    n++;
    worstStart = Math.max(worstStart, Math.hypot(pts[0] - o.start[0], pts[1] - o.start[1]));
    let xs = [], ys = [];
    for (let i = 0; i < pts.length; i += 2) { xs.push(pts[i]); ys.push(pts[i + 1]); }
    const mine = [Math.min(...xs), Math.min(...ys), Math.max(...xs) - Math.min(...xs), Math.max(...ys) - Math.min(...ys)];
    // slack = how far my hull bbox exceeds the rendered bbox (must be >= 0 and small)
    const slack = Math.max(o.bb[0] - mine[0], o.bb[1] - mine[1],
                           (mine[0] + mine[2]) - (o.bb[0] + o.bb[2]), (mine[1] + mine[3]) - (o.bb[1] + o.bb[3]));
    worstSlack = Math.max(worstSlack, -Math.min(0, slack));   // NEGATIVE slack = parser is wrong
  }
  console.log(`checked ${n} paths`);
  console.log(`worst start-point error vs getPointAtLength(0): ${worstStart.toFixed(6)} px`);
  console.log(`worst bbox UNDER-shoot (must be 0 — my hull must contain the curve): ${worstSlack.toFixed(6)} px`);
  console.log(worstStart < 1e-3 && worstSlack < 1e-3 ? 'PARSER OK' : 'PARSER WRONG');
})();
