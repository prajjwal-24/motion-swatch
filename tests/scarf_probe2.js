const puppeteer = require('puppeteer-core');
(async () => {
  const b = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', args: ['--no-sandbox'] });
  const p = await b.newPage();
  p.on('pageerror', e => console.log('ERR', String(e)));
  await p.goto('http://localhost:8000/index.html', { waitUntil: 'networkidle0' });
  const t = require('fs').readFileSync('/Users/prajjwas/Downloads/Scene3.svg', 'utf8');
  console.log(JSON.stringify(await p.evaluate(async (t) => {
    const { sel, library, animator, loadUploadedSVG } = window.__ms;
    await loadUploadedSVG(t, 'Scene3.svg'); await new Promise(r => setTimeout(r, 400));
    const m = library.getById('flutter-flag-autumn');
    const svg = document.querySelector('#artwork-container svg');
    sel._createSVGSelection(sel._wrapOne(svg.querySelector('#Scarf')));
    const s = sel.selections[sel.activeIdx];
    s.motionId = m.id; s.intensity = 1.0;
    sel.setHighlightsHidden && sel.setHighlightsHidden(true);
    animator._applyAll(0);
    const cl = animator._centrelineFor(m);
    // what does the sampler actually return?
    const samples = [];
    for (const u of [0, 0.25, 0.5, 0.75, 1]) samples.push(+animator._sampleCentreline(cl, u, 0.35).toFixed(4));
    // raw swatch range
    let lo = 1e9, hi = -1e9;
    for (const row of cl.frames) for (const v of row) { if (v < lo) lo = v; if (v > hi) hi = v; }
    const bb0 = svg.querySelector('#Scarf').getBBox();
    animator._applyAll(0.35);
    const bb1 = svg.querySelector('#Scarf').getBBox();
    // check whether the pristine d strings are relative
    const ds = s._warp.els.slice(0, 3).map(o => o.d0.slice(0, 70));
    const rel = s._warp.els.filter(o => /[mlhvcsqtaz]/.test(o.d0)).length;
    return {
      axis: { deg: +(Math.atan2(s._ribbon.axis.sin, s._ribbon.axis.cos) * 180 / Math.PI).toFixed(2), len: +s._ribbon.axis.len.toFixed(1) },
      samplesAtT035: samples, swatchRange: [+lo.toFixed(4), +hi.toFixed(4)],
      fps: cl.fps, nframes: cl.frames.length, nstations: cl.u.length,
      bboxRest: [bb0.x, bb0.y, bb0.width, bb0.height].map(v => +v.toFixed(1)),
      bboxAt035: [bb1.x, bb1.y, bb1.width, bb1.height].map(v => +v.toFixed(1)),
      pathsWithLowercaseCmds: rel, totalPaths: s._warp.els.length, sampleD: ds,
    };
  }, t), null, 1));
  await b.close();
})();
