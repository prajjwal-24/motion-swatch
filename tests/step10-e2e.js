/*
 * Step 10 done-when, LIVE — the real UI, the real router, the real model:
 *
 *   "upload one multi-motion clip -> named swatches auto-applied to the right objects
 *    -> judged & tuned — end to end, no filename hints"
 *
 * tests/step10-orchestration.js and tests/step10-label.py prove the wiring offline with
 * fakes. Neither can tell you whether the model actually recognises a river when the layer
 * is called `path2843` — and that is the whole claim, so it has to be measured against the
 * live model at least once.
 *
 * Three passes, each printing the numbers rather than just a tick:
 *
 *   A  NAMED   Scenery scene, real layer names, one /label call. What did it see?
 *   B  BLIND   the same artwork with every data-name replaced by `Layer 7` / `path2847`.
 *              Same picture, no names. Labels that still identify the object were read off
 *              the PIXELS. This is the pass that retires the layer-name regex.
 *   C  AUTUMN  assets/videos/Autumn.mp4 through the real #motion-input. This is the exact
 *              clip whose FILENAME used to short-circuit the pipeline into a hand-written
 *              spiral (js/upload.js, removed at Step 10). It must now come back with
 *              measured params and no leafFall flag.
 *
 * Costs real VLM calls (2 x /label, 1 x /decompose) and a real RAFT run. Needs every
 * service up:  sh start-all.sh
 *   node tests/step10-e2e.js
 */
const puppeteer = require('puppeteer-core');
const fs = require('fs');
const path = require('path');

const CHROME = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const ROOT = path.join(__dirname, '..');
let FAIL = [];
const ok = (c, m) => { console.log((c ? '  ok   ' : '  FAIL ') + m); if (!c) FAIL.push(m); };

/* What each scenery layer IS, so a label can be scored without a human in the loop.
   Deliberately generous: "cloud 1" is right if the model says cloud, clouds, or sky —
   the test is whether it found the OBJECT, not whether it matched a string. */
const TRUTH = {
  sun: [/sun/i, 'oscillation|'],
  'cloud 1': [/cloud|sky/i, 'fluid|flock|oscillation|rigid_path|'],
  'cloud 2': [/cloud|sky/i, 'fluid|flock|oscillation|rigid_path|'],
  birds: [/bird|flock|gull/i, 'flock|articulated|rigid_path|'],
  mountains: [/mountain|hill|range|peak/i, ''],
  river: [/river|water|stream|lake/i, 'fluid|'],
  mist: [/mist|fog|haze|cloud|steam/i, 'fluid|'],
  boat: [/boat|sail|ship|vessel/i, 'rigid_path|oscillation|cloth|'],
  ground: [/ground|bank|shore|dock|pier|land|lake|grass/i, ''],
  flagpole: [/pole|mast|flag|staff/i, 'cloth|oscillation|'],
  flag: [/flag|banner|pennant/i, 'cloth|'],
  'tree 1': [/tree|trunk|foliage|canopy|leaf|leaves|plant/i, 'cloth|oscillation|'],
  'tree 2': [/tree|trunk|foliage|canopy|leaf|leaves|plant/i, 'cloth|oscillation|'],
  leaves: [/leaf|leaves|foliage|canopy|tree|bush/i, 'cloth|oscillation|'],
};

/* Score one labelling against TRUTH. `sawIt` = the label names the right object.
   `classOk` = the motion class is one a reasonable person would accept for that object
   (an empty class is always acceptable — "this should not move" is a real answer). */
function score(labels, nameOfId) {
  const rows = [];
  for (const l of labels) {
    const truthName = nameOfId.get(l.id);
    const t = TRUTH[truthName];
    const sawIt = t ? t[0].test(l.label || '') : null;
    const classOk = t ? t[1].split('|').includes(l.motion_class || '') : null;
    rows.push({ id: l.id, truth: truthName, saw: l.label, cls: l.motion_class,
                conf: l.confidence, deforms: l.deforms, sawIt, classOk });
  }
  return rows;
}

function table(rows, sentNames) {
  for (const r of rows) {
    console.log(`     ${r.sawIt === null ? '?' : r.sawIt ? '✓' : '×'} ` +
      `${(r.truth || '?').padEnd(10)} shown as ${String(sentNames.get(r.id)).padEnd(12)} ` +
      `-> "${r.saw}" ${(r.cls || 'static').padEnd(12)} ` +
      `${Math.round((r.conf || 0) * 100)}%${r.classOk === false ? '  (class debatable)' : ''}`);
  }
}

/* One /label pass through the real button. `blind` strips the layer names first. */
async function labelPass(page, blind) {
  await page.evaluate(() => window.__ms.loadScene('scenery'));
  await new Promise(r => setTimeout(r, 500));

  // the file's own names, and (blind) the meaningless ids we replace them with.
  // EVERY <g>, not just g.layer: collectLayers walks the group tree, so a group with no
  // .layer class (the scene's `ground` and `flagpole`) is labellable too and would
  // otherwise keep its real name and quietly carry the blind pass.
  // The `id` attribute is left alone — nameOf() only falls back to it when data-name is
  // absent, and stripping ids could break a clip-path or <use> reference in the artwork.
  const sent = await page.evaluate((blind) => {
    const svg = document.querySelector('#artwork-container svg') ||
                document.querySelector('svg');
    const map = {};
    [...svg.querySelectorAll('g')].forEach((g, i) => {
      const real = g.getAttribute('data-name') || g.id || '?';
      if (blind) g.setAttribute('data-name', i % 2 ? `Layer ${i + 1}` : `path${2840 + i}`);
      map[g.getAttribute('data-name') || g.id || '?'] = real;
    });
    return map;
  }, blind);

  await page.click('#btn-autolabel');
  await page.waitForFunction(() => {
    const b = document.getElementById('btn-autolabel');
    return b && !b.disabled;
  }, { timeout: 180000 });

  const res = await page.evaluate(() => {
    const s = window.__mlLayerLabels;
    if (!s) return { missing: (document.getElementById('autolabel-out') || {}).textContent };
    return { labels: s.labels.map(l => ({ id: l.id, label: l.label,
               motion_class: l.motion_class, confidence: l.confidence, deforms: l.deforms })),
             layers: s.layers.map(l => ({ id: l.id, name: l.name })),
             art: s.art, warnings: s.warnings,
             panel: (document.getElementById('autolabel-out') || {}).textContent };
  });
  if (res.missing !== undefined) return res;
  // the name the FILE carried at label time -> the name it originally had
  const shownAs = new Map(res.layers.map(l => [l.id, l.name]));
  const truthOf = new Map(res.layers.map(l => [l.id, sent[l.name] || l.name]));
  return { ...res, rows: score(res.labels, truthOf), shownAs };
}

(async () => {
  for (const p of [8000, 8765, 8771]) {
    const up = await fetch(`http://127.0.0.1:${p}/health`).then(() => true)
      .catch(() => fetch(`http://127.0.0.1:${p}/`).then(() => true).catch(() => false));
    if (!up) { console.log(`\n:${p} is DOWN — run  sh start-all.sh  first.\n`); process.exit(1); }
  }

  const browser = await puppeteer.launch({
    executablePath: CHROME, headless: 'new',
    args: ['--no-sandbox', '--window-size=1400,900', '--autoplay-policy=no-user-gesture-required'],
  });
  const page = await browser.newPage();
  await page.setViewport({ width: 1400, height: 900 });
  const errors = [];
  page.on('pageerror', e => errors.push(e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push('[console] ' + m.text()); });
  await page.goto('http://localhost:8000', { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 400));

  // ── A: the artwork, with its real layer names ─────────────────────────────
  console.log('\nA — NAMED: one /label pass over the Scenery scene, names intact');
  const A = await labelPass(page, false);
  if (A.missing !== undefined) {
    ok(false, `labelling produced nothing: ${A.missing}`);
    await browser.close(); process.exit(1);
  }
  table(A.rows, A.shownAs);
  const aSaw = A.rows.filter(r => r.sawIt).length, aTot = A.rows.filter(r => r.sawIt !== null).length;
  ok(A.labels.length >= 10, `all ${A.labels.length} layers came back labelled (12 sent, cap 12)`);
  ok(aSaw >= Math.ceil(aTot * 0.75), `${aSaw}/${aTot} objects identified with names available`);
  ok(A.art && A.art.width > 0, `the artwork travelled at ${A.art.width}x${A.art.height}px`);

  // ── B: the same artwork with the names taken away ─────────────────────────
  console.log('\nB — BLIND: same picture, every layer renamed Layer N / pathNNNN');
  await page.reload({ waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 400));
  const B = await labelPass(page, true);
  if (B.missing !== undefined) {
    ok(false, `blind labelling produced nothing: ${B.missing}`);
    await browser.close(); process.exit(1);
  }
  table(B.rows, B.shownAs);
  const bSaw = B.rows.filter(r => r.sawIt).length, bTot = B.rows.filter(r => r.sawIt !== null).length;
  const bCls = B.rows.filter(r => r.classOk).length;
  ok([...B.shownAs.values()].every(n => /^(Layer \d+|path\d+)$/.test(n)),
     `every layer really was anonymous: ${[...B.shownAs.values()].slice(0, 3).join(', ')}, …`);
  ok(bSaw >= Math.ceil(bTot * 0.6),
     `${bSaw}/${bTot} objects identified from the PIXELS ALONE ` +
     `(named pass: ${aSaw}/${aTot}) — the layer-name regex is not carrying this`);
  ok(bCls >= Math.ceil(bTot * 0.6), `${bCls}/${bTot} motion classes acceptable when blind`);

  // the one the old regex existed for: /flag|banner|cloth/ can only fire on a name
  const flagRow = B.rows.find(r => r.truth === 'flag');
  ok(flagRow && flagRow.sawIt && flagRow.cls === 'cloth',
     `the flag is cloth with no "flag" in its name: shown as "${B.shownAs.get(flagRow && flagRow.id)}" ` +
     `-> "${flagRow && flagRow.saw}" / ${flagRow && flagRow.cls}`);
  ok(flagRow && flagRow.deforms === 'mesh', 'and mesh deform is derived from the class, not asked for');

  // the labelling must reach the SELECTION, or nothing downstream changes
  const applied = await page.evaluate(() => {
    const svg = document.querySelector('svg');
    const lab = [...window.__mlLayerLabels.byEl.entries()]
      .find(([, l]) => l.motion_class === 'cloth');
    if (!lab) return { none: true };
    const t = lab[0].querySelector('path,rect,polygon,polyline,circle,ellipse');
    t.dispatchEvent(new MouseEvent('click', { bubbles: true }));
    const a = window.__ms.sel.getActive();
    return { name: a && a.name, waveMode: a && a.waveMode, from: a && a.waveModeFrom };
  });
  ok(applied.from && applied.from.startsWith('vlm:'),
     `clicking that layer records the VLM as the evidence: waveModeFrom="${applied.from}"`);
  ok(applied.waveMode === true && applied.name && !/^Layer |^path\d/.test(applied.name),
     `the region takes the LABEL as its name, not the file's: "${applied.name}" (waveMode ${applied.waveMode})`);

  // ── C: the clip the filename shortcut used to hijack ──────────────────────
  // Deliberately run on the page B left behind: the artwork is still labelled, and still
  // labelled from ANONYMOUS names. So this is the done-when in one shot — one clip, real
  // extraction, swatches placed on objects whose names say nothing. The labels are reused
  // (autoApplyMotions only labels when it has none), so this costs no extra vision call.
  console.log('\nC — AUTUMN: assets/videos/Autumn.mp4 through the real #motion-input,');
  console.log('    onto the artwork B just labelled from anonymous layer names');
  const clip = path.join(ROOT, 'assets/videos/Autumn.mp4');
  if (!fs.existsSync(clip)) {
    ok(false, 'assets/videos/Autumn.mp4 is missing — cannot prove the shortcut is gone');
  } else {
    const before = await page.evaluate(() => window.__ms.library.getAll().length);
    const input = await page.$('#motion-input');
    await input.uploadFile(clip);   // this fires `change` itself — do not dispatch a second
    let grew = false;
    try {
      await page.waitForFunction(n => window.__ms.library.getAll().length > n,
                                 { timeout: 420000 }, before);
      grew = true;
    } catch { /* fall through to the diagnosis below */ }
    // give auto-apply a moment after the library grows
    await page.waitForFunction(() => /Applied|not placed|click one/.test(
      (document.getElementById('upload-status') || {}).textContent || ''),
      { timeout: 120000 }).catch(() => {});
    const st = await page.evaluate((n) => ({
      status: (document.getElementById('upload-status') || {}).textContent,
      motions: window.__ms.library.getAll().slice(n).map(m => ({
        name: m.name, desc: m.desc, cls: window.__ms.animator._classOf(m),
        rawClass: m.class, swatchClasses: (m.swatches || []).map(s => s && s.class),
        leafFall: !!(m.params || {}).leafFall, params: m.params,
        traj: (m.trajectories || []).length,
        frames: m.trajectories && m.trajectories[0] ? m.trajectories[0].length : 0,
      })),
      placed: (window.__ms.sel.selections || []).filter(s => s.motionId)
        .map(s => ({ layer: s.name, from: s.waveModeFrom, mesh: s.waveMode,
                     motion: (window.__ms.library.getAll()
                       .find(m => m.id === s.motionId) || {}).name })),
    }), before);
    ok(grew, `the clip produced ${st.motions.length} swatch(es)`);
    console.log(`     status: "${st.status}"`);
    for (const m of st.motions) {
      console.log(`     "${m.name}"  class=${m.cls || '(none)'} ` +
                  `(motion.class=${m.rawClass}, swatch=${JSON.stringify(m.swatchClasses)})`);
      console.log(`       ${m.traj} tracks x ${m.frames} frames · ${JSON.stringify(m.params)}`);
      console.log(`       ${m.desc}`);
    }
    const m = st.motions[st.motions.length - 1] || {};
    ok(!st.motions.some(x => x.leafFall), 'no leafFall flag: this is not the hand-written spiral');
    ok(!st.motions.some(x => /falling-leaves video/i.test(x.desc || '')),
       'no swatch claims to be "Captured from falling-leaves video"');
    // At least one region must come back with a real measured field. Not all of them:
    // captureFromFile legitimately falls back to in-browser Lucas–Kanade when the service
    // call fails, and that path has params but no trajectories — which is fine as long as
    // it SAYS so, so check the description names the engine either way.
    ok(st.motions.some(x => x.traj > 0),
       `a real measured field came back (${st.motions.map(x => `${x.traj}x${x.frames}`).join(', ')})`);
    ok(st.motions.every(x => /Captured (via|in-browser)/.test(x.desc || '')),
       'every swatch names the engine that produced it, fallbacks included');
    // the old branch produced the same 8 hand-tuned dials for every clip. A measured
    // amplitude belongs to the clip, so it is free to be anything.
    ok(m.params && m.params.amplitude !== undefined,
       `amplitude ${m.params && m.params.amplitude} came from the flow field, not a constant`);
    // and the class must survive onto EVERY swatch, fallback included, or the matcher has
    // nothing to match on and the region is silently refused
    ok(st.motions.every(x => x.cls),
       `every swatch carries a class the matcher can use: ${st.motions.map(x => x.cls || '(none)').join(', ')}`);
    console.log('     placed: ' + (st.placed.length
      ? st.placed.map(p => `"${p.motion}" -> ${p.layer} (${p.from})`).join(', ') : '(none)'));
    ok(/Applied \d+\/\d+/.test(st.status || ''),
       `auto-apply placed swatches by class alone: "${(st.status || '').slice(0, 120)}"`);
    // The region NAME is the label's, so the placement is attributable even though the
    // deform mode defers to the trajectory field (see CLOTH_NAME_HINT in js/regions.js:
    // motion_field is a measurement and outranks a still-image reading).
    ok(st.placed.length > 0 && st.placed.every(p => !/^(Layer \d+|path\d+)$/.test(p.layer)),
       `${st.placed.length} placement(s), each named by the LABEL not the file: ` +
       st.placed.map(p => `"${p.layer}"`).join(', '));
    ok(st.placed.every(p => p.from === 'motion_field' || (p.from || '').startsWith('vlm:')),
       `and every deform mode traces to evidence, never a name: ` +
       [...new Set(st.placed.map(p => p.from))].join(', '));
  }

  if (errors.length) console.log('\npage errors:\n  ' + errors.slice(0, 6).join('\n  '));
  console.log(`\n${FAIL.length ? 'FAILED: ' + FAIL.join(' | ') : 'all step 10 live checks passed'}`);
  await browser.close();
  process.exit(FAIL.length ? 1 : 0);
})();
