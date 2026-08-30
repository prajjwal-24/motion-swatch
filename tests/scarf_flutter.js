/*
 * Measure the extracted flag flutter on Scene3's #Scarf.
 *
 * The metric is LOCAL SHAPE DISTORTION with exact point correspondence: warpPathD
 * preserves command structure, so the k-th coordinate pair of the warped `d` is the
 * same point as the k-th pair of the pristine data-ms-d0. Comparing the distance
 * between consecutive pairs before and after the warp measures how much the artwork's
 * own geometry was stretched, independent of how far the whole thing moved.
 *
 * Pairs are kept only when the rest distance is 0.4..6 px — shorter is numerical noise
 * in the source art, longer spans a whole limb of the path and averages the local
 * stretch away.
 */
const puppeteer = require('puppeteer-core');
const { installSwatch } = require('./installSwatch');
const { pathPoints } = require('./pathabs');

// Default probes five arbitrary phases; MS_T=a,b,c (or MS_T=sweep:N) sweeps the whole
// loop, which is how the worst-case stretch is found rather than sampled at.
const T = (() => {
  const e = process.env.MS_T;
  if (!e) return [0.15, 0.35, 0.6, 0.9, 1.2];
  const m = /^sweep:(\d+):([\d.]+)$/.exec(e);
  if (m) return Array.from({ length: +m[1] }, (_, i) => +(i * (+m[2]) / +m[1]).toFixed(4));
  return e.split(',').map(Number);
})();
const MOTION = process.argv[2] || 'flutter-flag-autumn';
const INTENSITY = parseFloat(process.argv[3] || '1.0');

// #Scarf is drawn with RELATIVE commands, so the raw numbers in `d` are deltas —
// positions must be walked out of the path. See pathabs.js.
const nums = pathPoints;

(async () => {
  const browser = await puppeteer.launch({
    executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    args: ['--no-sandbox'],
  });
  const page = await browser.newPage();
  const errs = [];
  page.on('pageerror', (e) => errs.push(String(e)));
  page.on('console', (m) => { if (m.type() === 'error') errs.push('console: ' + m.text()); });
  await page.goto('http://localhost:8000/index.html', { waitUntil: 'networkidle0' });
  const swatch = await installSwatch(page);
  if (!swatch.ok) { console.log('SETUP FAILED: ' + swatch.why); process.exit(1); }

  const svgText = require('fs').readFileSync('/Users/prajjwas/Downloads/Scene3.svg', 'utf8');
  const setup = await page.evaluate(async (svgText, motionId, intensity, process_env_wave) => {
    const { sel, library, animator, loadUploadedSVG } = window.__ms;
    await loadUploadedSVG(svgText, 'Scene3.svg');
    await new Promise(r => setTimeout(r, 400));
    const m = library.getById(motionId);
    if (!m) return { error: `motion ${motionId} not in library` };
    const svg = document.querySelector('#artwork-container svg');
    const g = svg.querySelector('#Scarf');
    if (!g) return { error: 'no #Scarf' };
    sel._createSVGSelection(sel._wrapOne(g));
    const s = sel.selections[sel.activeIdx];
    s.motionId = m.id;
    s.intensity = intensity;
    if (process_env_wave) s.waveMode = true;   // presets only deform the geometry when asked
    if (sel.setHighlightsHidden) sel.setHighlightsHidden(true);
    animator._applyAll(0);        // builds the deform cache + the axis
    const ax = s._ribbon && s._ribbon.axis;
    return {
      name: s.name, paths: s._warp ? s._warp.els.length : 0,
      waveMode: !!s.waveMode,
      applicator: animator._applicatorFor(m),
      centreline: !!animator._centrelineFor(m),
      axisDeg: ax ? +(Math.atan2(ax.sin, ax.cos) * 180 / Math.PI).toFixed(2) : null,
      axisLen: ax ? +ax.len.toFixed(1) : null,
    };
  }, svgText, MOTION, INTENSITY, process.env.MS_WAVE === '1');

  if (setup.error) { console.log('SETUP FAILED:', setup.error); await browser.close(); process.exit(1); }
  console.log(`selection "${setup.name}"  paths=${setup.paths}  waveMode=${setup.waveMode}`);
  console.log(`applicator=${setup.applicator}  centreline=${setup.centreline}` +
              `  axis=${setup.axisDeg}deg  len=${setup.axisLen}px  intensity=${INTENSITY}`);

  const rows = [];
  for (const t of T) {
    const frame = await page.evaluate((t) => {
      const { sel, animator } = window.__ms;
      animator._applyAll(t);
      const s = sel.selections[sel.activeIdx];
      // A motion that only transforms the wrapper never builds a warp — that is a real
      // answer about that motion, not a crash.
      return s._warp ? s._warp.els.map(o => ({ d0: o.d0, d: o.el.getAttribute('d') })) : null;
    }, t);
    if (!frame) { console.log(`t=${t}: this motion does not deform the geometry (no warp built)`); continue; }

    const ratios = [];
    let moved = 0, npts = 0;
    for (const { d0, d } of frame) {
      const A = nums(d0), B = nums(d);
      if (A.length !== B.length) continue;
      for (let i = 0; i + 3 < A.length; i += 2) {
        const r0 = Math.hypot(A[i + 2] - A[i], A[i + 3] - A[i + 1]);
        if (r0 <= 0.4 || r0 >= 6) continue;
        const r1 = Math.hypot(B[i + 2] - B[i], B[i + 3] - B[i + 1]);
        ratios.push(Math.abs(r1 - r0) / r0);
      }
      for (let i = 0; i + 1 < A.length; i += 2) {
        moved += Math.hypot(B[i] - A[i], B[i + 1] - A[i + 1]); npts++;
      }
    }
    ratios.sort((a, b) => a - b);
    const q = (p) => ratios.length ? ratios[Math.min(ratios.length - 1, Math.floor(p * ratios.length))] : NaN;
    rows.push({ t, n: ratios.length, med: q(0.5), p90: q(0.9), worst: ratios[ratios.length - 1],
                travel: moved / Math.max(1, npts) });
  }

  console.log('\n   t   pairs   median     p90    worst   mean travel');
  for (const r of rows) {
    console.log(`${r.t.toFixed(2).padStart(5)} ${String(r.n).padStart(6)} ` +
      `${(r.med * 100).toFixed(1).padStart(7)}% ${(r.p90 * 100).toFixed(1).padStart(6)}% ` +
      `${(r.worst * 100).toFixed(0).padStart(7)}% ${r.travel.toFixed(1).padStart(9)}px`);
  }
  const meds = rows.map(r => r.med).sort((a, b) => a - b);
  console.log(`\nmedian across frames: ${(meds[Math.floor(meds.length / 2)] * 100).toFixed(1)}%` +
              `   worst overall: ${(Math.max(...rows.map(r => r.worst)) * 100).toFixed(0)}%`);
  console.log(errs.length ? `\nPAGE ERRORS:\n  ${errs.join('\n  ')}` : '\nno page errors');
  await browser.close();
})();
