/*
 * Step 10 selftest — the whole chain, and the honesty of it.
 *
 * The done-when is one sentence from docs/BUILD_PLAN.md: "upload one multi-motion clip ->
 * named swatches auto-applied to the right objects -> judged & tuned — end to end, no
 * filename hints, minimal hardcoding." Four of those clauses are checkable without a
 * browser, a video or an API key, and this file checks them:
 *
 *   1. THE MATCH IS ONE RULE, NOT TWO. js/autolabel.js has to decide swatch->layer in the
 *      browser and service/contracts.py has to be able to decide it without a DOM. Two
 *      implementations of one rule is a drift bug waiting to happen, so the SAME fixtures
 *      are run through BOTH (python via a subprocess) and the answers must be identical.
 *   2. NO FILENAME HINTS. Asserted against the sources, because "we deleted it" is the
 *      kind of claim that quietly stops being true.
 *   3. THE SURVIVING HARDCODING IS GATED. regions.js still has a cloth name regex. It
 *      must lose to a label every time, and every selection must say which one decided it.
 *   4. HONEST WHEN THE ROUTER IS DOWN. No labels must mean nothing applied — never a
 *      fallback to name matching.
 *
 * It also carries the flock heading fix that shipped with this step (js/animate.js), whose
 * numbers were measured on the pre-fix code and are quoted in the check names.
 *
 * Dependency-free: node + python3, no browser and no service.
 * Run: node tests/step10-orchestration.js
 */
const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const ROOT = path.resolve(__dirname, '..');
const src = f => fs.readFileSync(path.join(ROOT, 'js', f), 'utf8');

let fail = 0;
const ck = (name, ok, extra = '') => {
  console.log((ok ? '  ok   ' : '  FAIL ') + name + (extra ? '  ' + extra : ''));
  if (!ok) fail++;
};
const section = t => console.log('\n' + t);

// ---------------------------------------------------------------- load the app's JS
// Classic scripts sharing one global scope, so they go into ONE eval scope (animate.js
// reads motionfields.js's consts, and regions.js's deformDefault is a bare function
// declaration — it lands in this module's scope, callable by name below).
global.window = global;
global.document = {
  createElementNS: () => ({
    _attrs: {}, children: [],
    setAttribute(k, v) { this._attrs[k] = v; },
    getAttribute(k) { return this._attrs[k] === undefined ? null : this._attrs[k]; },
    appendChild(c) { this.children.push(c); return c; },
    remove() {},
    querySelector: () => null,
    querySelectorAll: () => [],
  }),
};
// autolabel.js is an IIFE that publishes window.MotionAutoLabel; the rest export by name.
const { Animator, SelectionManager } =
  eval(src('motionfields.js') + '\n' + src('animate.js') + '\n' + src('regions.js')
       + '\n' + src('autolabel.js') + '\n;({ Animator, SelectionManager })');
const AL = window.MotionAutoLabel;
ck('js/autolabel.js loads and publishes MotionAutoLabel', !!AL && typeof AL.label === 'function');

// ================================================================ 1. one rule, two impls
section('The swatch->layer rule is ONE rule: js/autolabel.js === contracts.py');

/* Fixtures chosen so a plausible wrong implementation fails at least one of them:
   a straight match, a contested class, a classless swatch, a below-floor layer, an
   order flip, an exactly-at-the-floor layer, and no labels at all. */
const L = (id, cls, conf) => ({ id, motion_class: cls, confidence: conf });
const S = cls => ({ kind: 'texture', class: cls });
const CASES = [
  { why: 'one swatch per class', sw: [S('cloth'), S('flock')],
    labs: [L('L1', 'cloth', 0.91), L('L2', 'flock', 0.7), L('L3', '', 0.95)] },
  { why: 'two swatches contest one layer', sw: [S('cloth'), S('cloth')],
    labs: [L('L1', 'cloth', 0.91), L('L2', 'flock', 0.7)] },
  { why: 'a classless swatch', sw: [S(''), S('cloth')],
    labs: [L('L1', 'cloth', 0.5)] },
  { why: 'a class nothing depicts', sw: [S('fluid')],
    labs: [L('L1', 'cloth', 0.9), L('L2', 'flock', 0.9)] },
  { why: 'best confidence wins', sw: [S('cloth')],
    labs: [L('Lo', 'cloth', 0.5), L('Hi', 'cloth', 0.88)] },
  { why: '...and input order does not matter', sw: [S('cloth')],
    labs: [L('Hi', 'cloth', 0.88), L('Lo', 'cloth', 0.5)] },
  { why: 'below the confidence floor', sw: [S('cloth')], labs: [L('L1', 'cloth', 0.2)] },
  { why: 'exactly at the floor is IN', sw: [S('cloth')], labs: [L('L1', 'cloth', 0.35)] },
  { why: 'just under the floor is OUT', sw: [S('cloth')], labs: [L('L1', 'cloth', 0.34)] },
  { why: 'no labels at all', sw: [S('cloth'), S('flock')], labs: [] },
  { why: 'no swatches at all', sw: [], labs: [L('L1', 'cloth', 0.9)] },
  { why: 'three swatches, three layers, mixed classes',
    sw: [S('flock'), S('cloth'), S('flock')],
    labs: [L('L1', 'flock', 0.4), L('L2', 'cloth', 0.8), L('L3', 'flock', 0.9)] },
];

const DRIVER = `
import json, os, sys
sys.path.insert(0, os.path.join(${JSON.stringify(ROOT)}, "service"))
import contracts as C
out = []
for c in json.load(sys.stdin):
    pairs, unm = C.match_swatches_to_layers(c["sw"], c["labs"])
    out.append({"pairs": [list(p) for p in pairs], "unmatched": unm})
print(json.dumps(out))
`;
let pyOut = null, pyErr = '';
try {
  pyOut = JSON.parse(execFileSync('python3', ['-c', DRIVER],
    { input: JSON.stringify(CASES), encoding: 'utf8' }));
} catch (e) { pyErr = (e.stderr || e.message || '').toString().trim().split('\n').pop(); }
ck('contracts.py answers for the same fixtures (python3 reachable)', !!pyOut, pyErr);

if (pyOut) {
  CASES.forEach((c, i) => {
    const js = AL.matchSwatchesToLayers(c.sw, c.labs);
    const jsKey = JSON.stringify({ pairs: js.pairs, unmatched: js.unmatched });
    const pyKey = JSON.stringify(pyOut[i]);
    ck(`js and python agree — ${c.why}`, jsKey === pyKey, jsKey === pyKey ? jsKey
       : `js ${jsKey} vs py ${pyKey}`);
  });
}
// the floor itself has to be the same number in both files, or the two agree only by luck
const pyFloor = /def match_swatches_to_layers\(swatches, labels, conf_min=([\d.]+)\)/
  .exec(fs.readFileSync(path.join(ROOT, 'service', 'contracts.py'), 'utf8'));
ck('and the confidence floor is literally the same number in both',
   pyFloor && parseFloat(pyFloor[1]) === AL.LABEL_CONF_MIN,
   `js ${AL.LABEL_CONF_MIN} vs py ${pyFloor && pyFloor[1]}`);

// ================================================================ 2. no filename hints
section('No filename hints survive in the upload path');

const UPLOAD = fs.readFileSync(path.join(ROOT, 'js', 'upload.js'), 'utf8');
const MAIN = fs.readFileSync(path.join(ROOT, 'js', 'main.js'), 'utf8');
/* Strip comments before scanning: the deletions are DOCUMENTED in comments on purpose
   (a reader should be able to find out what used to happen and why it went), so a naive
   grep would flag the documentation as the offence. */
const code = s => s.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
const upCode = code(UPLOAD), mainCode = code(MAIN);
ck('js/upload.js never tests file.name against a motion vocabulary',
   !/file\.name\s*\)|test\(\s*file\.name/.test(upCode) || !/leaf|leaves|autumn|falling/i.test(upCode));
ck('...and the leaf/autumn regex is gone from the code entirely',
   !/leaf|leaves|autumn/i.test(upCode));
ck('synthFallTrajectories() no longer exists in js/main.js', !/synthFallTrajectories/.test(mainCode));
ck('...and nothing still calls it', !/synthFallTrajectories/.test(upCode));
ck('file.name is still used for the SWATCH NAME (that is naming, not routing)',
   /file\.name\.replace/.test(upCode));
// the hand-tuned leaf behaviour is allowed to live on, but only as a declared preset
const MOTIONS = fs.readFileSync(path.join(ROOT, 'js', 'motions.js'), 'utf8');
ck('the leaf look survives as a PRESET in js/motions.js, where it is labelled as one',
   /Autumn Fall/.test(MOTIONS) && /leafFall/.test(MOTIONS));
ck('...and no upload path claims a preset was "captured from video"',
   !/Captured from falling-leaves video/.test(upCode));

// ================================================================ 3. gated hardcoding
section('The cloth name regex is gated behind the label, and always says which won');

const wrapStub = () => ({
  _attrs: {}, querySelector: () => null, querySelectorAll: () => [],
  setAttribute(k, v) { this._attrs[k] = v; }, getAttribute(k) { return this._attrs[k]; },
  getBBox: () => ({ x: 10, y: 20, width: 100, height: 60 }),
  contains: () => false,
  ownerSVGElement: null,
});

const noLabels = () => { delete window.__mlLayerLabels; };
noLabels();
let d = deformDefault(wrapStub(), 'Flag');
ck('router offline + a layer called "Flag" -> the regex answers, and OWNS it',
   d.waveMode === true && d.waveModeFrom === 'name_hint');
d = deformDefault(wrapStub(), 'Layer 7');
ck('router offline + an unrecognised name -> rigid, marked as a default',
   d.waveMode === false && d.waveModeFrom === 'default');

// now the VLM has looked at the artwork
const el = wrapStub();
const withLabel = (lab) => { window.__mlLayerLabels = { byEl: new Map([[el, lab]]) }; };
withLabel({ id: 'L1', label: 'pennant', motion_class: 'cloth', deforms: 'mesh', confidence: 0.9 });
d = deformDefault(el, 'Layer 7');
ck('a cloth label beats the name: an unnamed layer ripples anyway',
   d.waveMode === true && d.waveModeFrom === 'vlm:cloth');
ck('...and the label renames it, because the file name is what we distrust',
   d.name === 'pennant');
withLabel({ id: 'L1', label: 'flag pole', motion_class: '', deforms: 'rigid', confidence: 0.8 });
d = deformDefault(el, 'Flag');
ck('a label saying "this does not move" beats the regex that says flag -> cloth',
   d.waveMode === false && d.waveModeFrom === 'vlm:static');
ck('the losing regex leaves no trace in the provenance', !/name_hint/.test(d.waveModeFrom));
noLabels();

/* The provenance must reach the SELECTION, not just the helper: _createSVGSelection
   spreads deformDefault LAST so the label's name overrides the wrap's own. */
function mgr() {
  const overlay = { getContext: () => ({ clearRect() {} }), addEventListener() {},
                    style: {}, classList: { add() {} }, width: 800, height: 500 };
  const m = new SelectionManager(overlay, { querySelector: () => null });
  m._renderSVGHighlights = () => {};
  return m;
}
const m1 = mgr();
const w1 = wrapStub(); w1._attrs['data-ms-name'] = 'Banner';
m1._createSVGSelection(w1);
ck('a new selection records waveModeFrom, so the UI can never present a guess as a reading',
   m1.selections[0].waveModeFrom === 'name_hint' && m1.selections[0].waveMode === true);
const m2 = mgr();
const w2 = wrapStub(); w2._attrs['data-ms-name'] = 'Banner';
window.__mlLayerLabels = { byEl: new Map([[w2, { id: 'L2', label: 'ground', motion_class: '',
                                                 deforms: 'rigid', confidence: 0.77 }]]) };
m2._createSVGSelection(w2);
ck('...and with a label the selection takes the LABEL name, not the file name',
   m2.selections[0].name === 'ground' && m2.selections[0].waveModeFrom === 'vlm:static',
   `${m2.selections[0].name} / ${m2.selections[0].waveModeFrom}`);
ck('the whole label is kept on the selection, so the panel can show its confidence',
   m2.selections[0].layerLabel && m2.selections[0].layerLabel.confidence === 0.77);
noLabels();

/* The precedence LIST in regions.js is documentation, and documentation drifts. The order
   that actually happens is decided by two writers in main.js, so pin the doc to them.
   (Found by tests/step10-e2e.js: the list used to mark vlm:<class> "preferred" while
   applyMotionToActive overwrote it with motion_field on every real placement.) */
const regionsSrc = src('regions.js'), mainSrc = src('main.js');
// (?=\s) not \b: "vlm:<class>" ends in a non-word char, so \b never fires after it.
const doc = [...regionsSrc.matchAll(/^ \* {3}(preset_leaffall|artwork_rigid|motion_field|vlm:<class>|name_hint|default)(?=\s)/gm)]
  .map(m => m[1]);
ck('regions.js documents the precedence strongest-first, matching the writers in main.js',
   JSON.stringify(doc) === JSON.stringify(['preset_leaffall', 'artwork_rigid', 'motion_field',
                                           'vlm:<class>', 'name_hint', 'default']),
   JSON.stringify(doc));
ck('...and runAutoLabel still refuses to demote the two sources ranked above it',
   /waveModeFrom !== 'motion_field' && s\.waveModeFrom !== 'artwork_rigid'/.test(mainSrc));
ck('...while applyMotionToActive writes motion_field only for a REAL trajectory field',
   /m\.trajectories && m\.trajectories\.length[^\n]*\n?[^\n]*waveModeFrom = 'motion_field'/.test(mainSrc)
   || /trajectories\.length.*\{[\s\S]{0,120}?'motion_field'/.test(mainSrc));

// ================================================================ 4. honest when down
section('Auto-apply is honest when the router is down');

const applyStub = () => {
  const calls = { selected: [], applied: 0 };
  return {
    sel: { selections: [], selectByIndex() {}, getActive: () => null },
    library: { select(id) { calls.selected.push(id); } },
    animator: new Animator({ selections: [], syncHighlights() {} }, { getById: () => null }),
    applyMotionToActive() { calls.applied++; },
    calls,
  };
};
const MOT = [{ id: 'm1', class: 'cloth', swatches: [] }, { id: 'm2', class: 'flock', swatches: [] }];

let st;
(async () => {
  st = applyStub();
  let r = await AL.apply({ motions: MOT, labelling: null, sel: st.sel, library: st.library,
                           animator: st.animator, applyMotionToActive: st.applyMotionToActive });
  ck('no labelling at all -> NOTHING is applied', r.applied.length === 0 && st.calls.applied === 0);
  ck('...and every motion is reported skipped, with the reason',
     r.skipped.length === 2 && r.skipped.every(s => s.why === 'no layer labels'));

  st = applyStub();
  r = await AL.apply({ motions: MOT, labelling: { labels: [] }, sel: st.sel, library: st.library,
                       animator: st.animator, applyMotionToActive: st.applyMotionToActive });
  ck('an EMPTY labelling is the same as none — not an excuse to guess',
     r.applied.length === 0 && st.calls.applied === 0 && r.skipped.length === 2);

  /* And with labels it really does apply — through the DOM path, so this needs a
     clickable element. selectLayer dispatches a real click; here the stub answers it by
     handing back an existing region, which is the branch a re-labelled artwork takes. */
  st = applyStub();
  const region = { id: 'r1', name: 'Layer 4', wrap: el, waveMode: false, waveModeFrom: 'default' };
  st.sel.selections = [region];
  st.sel.getActive = () => region;
  const labelling = {
    labels: [{ id: 'L1', label: 'flag', motion_class: 'cloth', deforms: 'mesh',
               confidence: 0.9, el }],
    byId: new Map([['L1', { id: 'L1', label: 'flag', motion_class: 'cloth', deforms: 'mesh',
                            confidence: 0.9, el }]]),
  };
  r = await AL.apply({ motions: [MOT[0]], labelling, sel: st.sel, library: st.library,
                       animator: st.animator, applyMotionToActive: st.applyMotionToActive });
  ck('a cloth motion lands on the layer labelled cloth', r.applied.length === 1
     && r.applied[0].layerId === 'L1' && st.calls.applied === 1, JSON.stringify(r.skipped));
  ck('...and the region it landed on is renamed and re-moded from the LABEL',
     region.name === 'flag' && region.waveMode === true && region.waveModeFrom === 'vlm:cloth');
  ck('...and the swatch that got applied is the one the library was told to select',
     st.calls.selected.length === 1 && st.calls.selected[0] === 'm1');

  // the swatch's class outranks the motion's — _classOf is the single definition of that
  st = applyStub();
  st.sel.selections = [region]; st.sel.getActive = () => region;
  const mixed = [{ id: 'm3', class: 'flock',
                   swatches: [{ kind: 'texture', class: 'cloth', applicator: 'wave' }] }];
  r = await AL.apply({ motions: mixed, labelling, sel: st.sel, library: st.library,
                       animator: st.animator, applyMotionToActive: st.applyMotionToActive });
  ck('the SWATCH class decides the match, not a stale motion.class',
     r.applied.length === 1 && r.applied[0].layerId === 'L1', JSON.stringify(r.skipped));

  /* A classless swatch is unmatchable, so the class must survive the FALLBACK extraction
     too. captureFromFile drops to in-browser Lucas–Kanade when the service call fails, and
     that result has params but no Contract-B swatch — hence no class. Found live in
     tests/step10-e2e.js: one of Autumn.mp4's two regions lost its service call and was
     then refused here with "motion has no class", so the region never reached an object. */
  st = applyStub();
  st.sel.selections = [region]; st.sel.getActive = () => region;
  r = await AL.apply({ motions: [{ id: 'm4', swatches: [] }], labelling, sel: st.sel,
                       library: st.library, animator: st.animator,
                       applyMotionToActive: st.applyMotionToActive });
  ck('a classless swatch is refused rather than dropped on the nearest layer',
     r.applied.length === 0 && r.skipped.length === 1
     && r.skipped[0].why === 'motion has no class', JSON.stringify(r.skipped));
  ck('...so js/upload.js gives a fallback swatch the ROUTER class it already knew',
     /if \(!sw\.class\) sw\.class = m\.class \|\| '';/.test(upCode));
  ck('...and the single-motion path does the same before it adds to the library',
     /if \(routed && routed\.class && !motion\.class\) motion\.class = routed\.class;/.test(upCode));

  // ============================================================== 5. layer collection
  section('collectLayers: the artwork root is descended into, never labelled');

  const IDENT = { a: 1, b: 0, c: 0, d: 1, e: 0, f: 0, inverse() { return this; },
                  multiply(m) { return m; } };
  const svgStub = {
    viewBox: { baseVal: { x: 0, y: 0, width: 1000, height: 800 } },
    getScreenCTM: () => IDENT,
  };
  const node = (name, box, kids = []) => ({
    _attrs: { 'data-name': name }, _kids: kids,
    getAttribute(k) { return this._attrs[k] === undefined ? null : this._attrs[k]; },
    getBBox: () => ({ x: box[0], y: box[1], width: box[2], height: box[3] }),
    getScreenCTM: () => IDENT,
  });
  // one giant wrapper holding three real objects
  const flagN = node('Flag_x20_1', [100, 100, 200, 150]);
  const birdsN = node('birds', [500, 60, 220, 90]);
  const skyN = node('sky', [0, 0, 990, 300]);
  const rootN = node('Layer_1', [0, 0, 1000, 800], [flagN, birdsN, skyN]);
  const groupsOf = p => (p === svgStub ? [rootN] : (p._kids || []));
  const got = AL.collectLayers(svgStub, groupsOf);
  const names = got.layers.map(l => l.name);
  ck('the canvas-covering wrapper is not offered as a layer', !names.includes('Layer_1'));
  ck('its children are, so the real objects are what get labelled',
     names.length === 3 && names.includes('birds') && names.includes('sky'), names.join(','));
  ck('Illustrator _xHH_ escapes are decoded before the model sees the name',
     names.includes('Flag 1'), names.join(','));
  ck('every layer gets a stable id the router can echo back',
     got.layers.every((l, i) => l.id === 'L' + (i + 1)));
  const fb = got.layers[names.indexOf('Flag 1')].bbox;
  ck('bboxes are normalized 0-1 against the viewBox, not raw user units',
     JSON.stringify(fb) === JSON.stringify([0.1, 0.125, 0.2, 0.1875]), JSON.stringify(fb));
  const capped = AL.collectLayers(svgStub, groupsOf, { cap: 2 });
  ck('a cap is REPORTED, never a silent truncation',
     capped.layers.length === 2 && capped.dropped === 1, `dropped ${capped.dropped}`);
  const zero = AL.collectLayers(svgStub, p => (p === svgStub ? [node('empty', [0, 0, 0, 0])] : []));
  ck('a zero-size layer is skipped, and the skip is named',
     zero.layers.length === 0 && zero.skipped.length === 1, zero.skipped.join(','));
  ck('the cap defaults to the router\'s LABEL_MAX_LAYERS (12)',
     /LABEL_MAX_LAYERS.*"12"/.test(fs.readFileSync(path.join(ROOT, 'service', 'vlm_router.py'), 'utf8'))
     && /cap != null \? opts\.cap : 12/.test(src('autolabel.js')));

  // ============================================================== 6. flock sign
  section('Flock heading: an unsigned axis resolved against the measured drift');

  /* distill.py emits `direction` as `% 180.0` — a dominant AXIS, not a heading — so a
     falling flock (270) and a rising one (90) arrive identically as 90. The drift is the
     only signed evidence. Pre-fix, the axis was averaged with the drift before the sign
     was resolved; the numbers in these check names were measured on that code. */
  const PARAMS = { frequency: 1.2, amplitude: 0.4, direction: 30, turbulence: 0,
                   damping: 0.1, phaseSpread: 0.3, driftX: 0, driftY: 0 };
  function flockRun(over, n = 6) {
    const kids = Array.from({ length: n }, (_, i) => ({
      _t: null,
      getBBox: () => ({ x: 400 + i * 10, y: 350, width: 8, height: 6 }),
      setAttribute(k, v) { if (k === 'transform') this._t = v; },
      removeAttribute() { this._t = null; },
    }));
    const s = { kind: 'svg', name: 'Layer 4', motionId: 'f1', wrap: wrapStub(),
                center: [50, 30], speed: 1, intensity: 1 };
    s.wrap.querySelectorAll = () => kids;
    s.wrap.ownerSVGElement = { viewBox: { baseVal: { width: 1000, height: 800 } } };
    const anim = new Animator({ selections: [s], syncHighlights() {} }, { getById: () => null });
    const mo = { id: 'f1', class: 'flock', params: Object.assign({}, PARAMS, over),
                 swatches: [], trajectories: null, path: null, pose: null };
    const pts = [];
    for (let t = 0.5; t < 12; t += 0.5) {
      anim._applyFlock(s, mo, t, 1);
      kids.forEach(k => {
        const g = (k._t || '').match(/translate\(([-\d.]+) ([-\d.]+)\)/);
        if (g) pts.push([+g[1], +g[2]]);
      });
    }
    const xs = pts.map(p => p[0]), ys = pts.map(p => p[1]);
    return { minX: Math.min(...xs), maxX: Math.max(...xs),
             minY: Math.min(...ys), maxY: Math.max(...ys) };
  }
  const fallen = flockRun({ direction: 90, driftY: 0.9 });
  ck('a FALLING flock (axis 90, driftY +0.9) now drifts DOWN — it drifted 52.08px sideways before',
     fallen.maxY > 40 && Math.abs(fallen.maxX) < 0.6 && Math.abs(fallen.minX) < 0.6,
     `dy ${fallen.maxY.toFixed(2)} dx [${fallen.minX.toFixed(2)}, ${fallen.maxX.toFixed(2)}]`);
  const left = flockRun({ direction: 0, driftX: -0.9 });
  ck('a LEFTWARD flock (axis 0, driftX -0.9) now drifts LEFT — it froze at 0.00px before',
     left.minX < -40 && Math.abs(left.maxY) < 0.6,
     `dx ${left.minX.toFixed(2)} dy ${left.maxY.toFixed(2)}`);
  const rose = flockRun({ direction: 90, driftY: -0.9 });
  ck('a RISING flock still rises (the two cases that already worked are unchanged)',
     rose.minY < -40 && Math.abs(rose.maxX) < 0.6, `dy ${rose.minY.toFixed(2)}`);
  const right = flockRun({ direction: 0, driftX: 0.9 });
  ck('...and a RIGHTWARD flock still goes right', right.maxX > 40 && Math.abs(right.maxY) < 0.6,
     `dx ${right.maxX.toFixed(2)}`);
  const perp = flockRun({ direction: 90, driftX: 0.9 });
  ck('drift perpendicular to the axis blends instead of cancelling to a standstill',
     Math.hypot(perp.maxX - perp.minX, perp.maxY - perp.minY) > 10,
     `dx ${perp.maxX.toFixed(1)} dy ${perp.minY.toFixed(1)}`);
  const still = flockRun({ direction: 90, driftY: 0 });
  ck('with NO measured drift the axis is used as-is (no evidence to resolve the sign)',
     Math.abs(still.minY) > 40, `dy ${still.minY.toFixed(2)}`);

  // ============================================================== 7. the /label contract
  section('The browser and the router agree on Contract D');

  const ROUTER = fs.readFileSync(path.join(ROOT, 'service', 'vlm_router.py'), 'utf8');
  ck('the router exposes /label', /"\/label"/.test(ROUTER));
  ck('...and js/capture.js posts to it', /\/label'/.test(src('capture.js')));
  ck('the router only offers the 6 real classes plus "" (it cannot invent a class)',
     /\[""\] \+ list\(contracts\.MOTION_CLASSES\.keys\(\)\)/.test(ROUTER));

  /* Driven, not grepped: labelLayers is called for real against a fetch that behaves the
     way a stopped service behaves. It must resolve to null so the caller can say "the
     labels are missing" — an exception here would abort a multi-motion upload that had
     already extracted its swatches. */
  const { MotionCapture } = eval(src('capture.js') + '\n;({ MotionCapture })');
  const noRouter = { fetch: () => Promise.reject(new Error('Failed to fetch')) };
  global.fetch = noRouter.fetch;
  global.console.warn = () => {};                        // the warning is expected; hush it
  const down = await MotionCapture.prototype.labelLayers.call({}, 'data:image/jpeg;base64,x', []);
  ck('capture.labelLayers resolves to null when the router is down (never throws upward)',
     down === null, String(down));
  global.fetch = () => Promise.resolve({ json: () => Promise.resolve({ error: 'no api key' }) });
  const unauthed = await MotionCapture.prototype.labelLayers.call({}, 'data:', []);
  ck('...and an {error} body is a failure too, not a labelling of zero layers',
     unauthed === null, JSON.stringify(unauthed));
  global.fetch = () => Promise.resolve({
    json: () => Promise.resolve({ kind: 'layer_labels', labels: [{ id: 'L1' }] }) });
  const up = await MotionCapture.prototype.labelLayers.call({}, 'data:', []);
  ck('...and a real answer is passed through untouched', up && up.labels.length === 1);
  delete global.fetch;

  ck('index.html loads js/autolabel.js',
     /js\/autolabel\.js\?v=/.test(fs.readFileSync(path.join(ROOT, 'index.html'), 'utf8')));

  console.log(fail ? `\n${fail} CHECK(S) FAILED` : '\nall step 10 checks passed');
  process.exit(fail ? 1 : 0);
})();
