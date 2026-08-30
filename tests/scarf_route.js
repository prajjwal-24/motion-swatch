/*
 * Flutter + authored travel route on #Scarf, composed.
 *
 * The load-bearing invariant: travel is a RIGID translation, so it cannot change local
 * shape distortion at all. If composing the two changes the distortion numbers even
 * slightly, the route is leaking into the deformation and the composition is wrong.
 *
 * Also checks the route is driven through the real UI path (beginRoute + synthetic clicks
 * + dblclick), not by poking the data structure, so _toViewBox and the click interception
 * are exercised too.
 */
const puppeteer = require('puppeteer-core');
const { installSwatch } = require('./installSwatch');
const { pathPoints } = require('./pathabs');

const T = Array.from({ length: 48 }, (_, i) => +(i * 4 / 48).toFixed(4));

(async () => {
  const b = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', args: ['--no-sandbox'] });
  const p = await b.newPage();
  await p.setViewport({ width: +(process.env.MS_VW || 1600), height: +(process.env.MS_VH || 1000) });
  const errs = [];
  p.on('pageerror', e => errs.push(String(e)));
  p.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text()); });
  await p.goto('http://localhost:8000/index.html', { waitUntil: 'networkidle0' });
  const swatch = await installSwatch(p);
  if (!swatch.ok) { console.log('SETUP FAILED: ' + swatch.why); process.exit(1); }
  const svgText = require('fs').readFileSync('/Users/prajjwas/Downloads/Scene3.svg', 'utf8');

  const out = await p.evaluate(async (svgText, T) => {
    const { sel, library, animator, loadUploadedSVG } = window.__ms;
    await loadUploadedSVG(svgText, 'Scene3.svg');
    await new Promise(r => setTimeout(r, 400));
    const svg = document.querySelector('#artwork-container svg');
    const g = svg.querySelector('#Scarf');
    sel._createSVGSelection(sel._wrapOne(g));
    const s = sel.selections[sel.activeIdx];
    s.motionId = 'flutter-flag-autumn'; s.intensity = 1.0;
    if (sel.setHighlightsHidden) sel.setHighlightsHidden(true);

    const sample = () => T.map(t => {
      animator._applyAll(t);
      return {
        tf: s.wrap.getAttribute('transform') || '',
        paths: s._warp.els.map(o => ({ d0: o.d0, d: o.el.getAttribute('d') })),
      };
    });

    // ---- 1. flutter alone (baseline) ----
    animator._applyAll(0);
    const before = sample();

    // ---- 2. draw a route through the real UI: begin, click, click, double-click ----
    const started = sel.beginRoute();
    // click in VIEWBOX coords, mapped forward through the same CTM _toViewBox inverts, so
    // the test does not care how the artwork is letterboxed into the window
    const clickAt = (vx, vy) => {
      const m = svg.getScreenCTM();
      const q = svg.createSVGPoint(); q.x = vx; q.y = vy;
      const c = q.matrixTransform(m);
      const o = { clientX: c.x, clientY: c.y, bubbles: true, cancelable: true };
      svg.dispatchEvent(new MouseEvent('mousemove', o));   // rubber band
      svg.dispatchEvent(new MouseEvent('click', o));
    };
    const vb = svg.viewBox.baseVal;
    clickAt(vb.width * 0.62, vb.height * 0.32);   // out over the lake
    clickAt(vb.width * 0.88, vb.height * 0.16);   // and up toward the far bank

    // A click in the letterbox margin is inside the svg element but outside the artwork.
    // It must be pulled onto the canvas, or the guide is drawn where nothing is visible.
    const r = svg.getBoundingClientRect();
    const far = { clientX: r.left + 2, clientY: r.top + r.height - 2, bubbles: true, cancelable: true };
    svg.dispatchEvent(new MouseEvent('mousemove', far));
    const hover = sel.routing && sel.routing.hover;
    svg.dispatchEvent(new MouseEvent('click', far));
    const clampRaw = sel._toViewBox(far).map(v => +v.toFixed(1));
    const clamped = sel.routing.pts[sel.routing.pts.length - 1].map(v => +v.toFixed(1));
    sel.routing.pts.pop();                       // drop the probe point; keep the real route
    svg.dispatchEvent(new MouseEvent('dblclick', { bubbles: true, cancelable: true }));
    const route = s.route ? { pts: s.route.pts, authored: s.route.authored, dur: s.route.duration } : null;
    const guide = !!svg.querySelector('#ms-routes');

    // The guide is authoring chrome: it belongs to the SELECTED object only.
    // NB the harness hid highlights during setup, so unhide first — the committed guide
    // is suppressed during playback for the same reason the selection outline is.
    sel.setHighlightsHidden(false);
    const vis = { selected: !!svg.querySelector('#ms-routes') };
    sel.deselect();
    vis.deselected = !!svg.querySelector('#ms-routes');
    sel.selectByIndex(sel.selections.indexOf(s));
    vis.reselected = !!svg.querySelector('#ms-routes');
    sel.setHighlightsHidden(true);
    vis.playing = !!svg.querySelector('#ms-routes');
    sel.setHighlightsHidden(false);
    vis.legs = svg.querySelectorAll('#ms-routes polyline').length;
    vis.live = (() => {                      // a route being drawn always shows
      sel.setHighlightsHidden(true);
      sel.beginRoute();
      const on = !!svg.querySelector('#ms-routes');
      sel.endRoute(false);                   // abandon: keeps the committed route intact
      sel.setHighlightsHidden(true);
      return on;
    })();

    // ---- 3. flutter + travel ----
    const after = sample();

    // ---- 4. route with NO motion: must not accumulate frame over frame ----
    s.motionId = null;
    const solo = [];
    for (let k = 0; k < 3; k++) for (const t of T) {
      animator._applyAll(t);
      const m = /translate\(([-\d.]+) ([-\d.]+)\)/.exec(s.wrap.getAttribute('transform') || '');
      if (m) solo.push([+m[1], +m[2]]);
    }
    return { started, route, guide, vis, clampRaw, clamped, hover: hover && hover.map(v => +v.toFixed(1)), before, after, vb: [svg.viewBox.baseVal.width, svg.viewBox.baseVal.height],
             soloMax: Math.max(...solo.map(v => Math.hypot(v[0], v[1]))) };
  }, svgText, T);
  await b.close();

  if (!out.route) { console.log('FAILED: no route was created through the UI path'); process.exit(1); }
  console.log(`beginRoute ok=${out.started}  route guide drawn=${out.guide}`);
  console.log(`guide visible: selected=${out.vis.selected} deselected=${out.vis.deselected}` +
              ` reselected=${out.vis.reselected} during playback=${out.vis.playing} while drawing=${out.vis.live}` +
              `  (${out.vis.legs} polylines = 1 route)`);
  console.log(`clamp: click mapping to (${out.clampRaw}) stored as (${out.clamped})` +
              `  rubber band (${out.hover})`);
  console.log(`route: ${out.route.pts.length} pts, authored=${out.route.authored}, ${out.route.dur}s`);
  console.log(`  ${out.route.pts.map(q => `(${q[0].toFixed(0)},${q[1].toFixed(0)})`).join(' -> ')}`);

  // distortion, computed identically for both runs
  const distortion = (frames) => {
    const meds = [], worsts = [];
    for (const fr of frames) {
      const rs = [];
      for (const { d0, d } of fr.paths) {
        const A = pathPoints(d0), B = pathPoints(d);
        if (A.length !== B.length) continue;
        for (let i = 0; i + 3 < A.length; i += 2) {
          const r0 = Math.hypot(A[i + 2] - A[i], A[i + 3] - A[i + 1]);
          if (r0 <= 0.4 || r0 >= 6) continue;
          rs.push(Math.abs(Math.hypot(B[i + 2] - B[i], B[i + 3] - B[i + 1]) - r0) / r0);
        }
      }
      rs.sort((x, y) => x - y);
      meds.push(rs[Math.floor(rs.length / 2)]); worsts.push(rs[rs.length - 1]);
    }
    meds.sort((x, y) => x - y);
    return { med: meds[Math.floor(meds.length / 2)], worst: Math.max(...worsts) };
  };
  const A = distortion(out.before), B = distortion(out.after);
  console.log(`\nflutter alone       median ${(A.med*100).toFixed(2)}%  worst ${(A.worst*100).toFixed(1)}%`);
  console.log(`flutter + travel    median ${(B.med*100).toFixed(2)}%  worst ${(B.worst*100).toFixed(1)}%`);
  const drift = Math.max(Math.abs(A.med - B.med), Math.abs(A.worst - B.worst));
  console.log(drift < 1e-9
    ? 'INVARIANT HELD: travel is rigid — identical distortion, so it does not leak into the flutter'
    : `INVARIANT BROKEN: distortion changed by ${drift}`);

  // travel actually happened, and the geometry still flutters underneath it
  let tmax = 0, shapeChanges = 0;
  for (let i = 0; i < out.after.length; i++) {
    const m = /translate\(([-\d.]+) ([-\d.]+)\)/.exec(out.after[i].tf);
    if (m) tmax = Math.max(tmax, Math.hypot(+m[1], +m[2]));
    if (out.after[i].paths[0].d !== out.before[0].paths[0].d) shapeChanges++;
  }
  console.log(`\nmax travel offset over the loop: ${tmax.toFixed(1)} viewBox units` +
              `  (viewBox is ${out.vb[0]} x ${out.vb[1]})`);
  console.log(`frames whose geometry differs from rest: ${shapeChanges}/${out.after.length} (flutter still running)`);
  console.log(`route-only, 3 loops: max offset ${out.soloMax.toFixed(1)} (bounded => no per-frame accumulation)`);
  const nan = out.after.some(f => f.paths.some(q => /NaN/.test(q.d))) || /NaN/.test(out.after.map(f => f.tf).join());
  console.log(nan ? 'NaN PRESENT' : 'no NaN in geometry or transforms');
  console.log(errs.length ? `\nPAGE ERRORS:\n  ${errs.join('\n  ')}` : '\nno page errors');
})();
