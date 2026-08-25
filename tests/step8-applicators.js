/*
 * Step 8 selftest — class-keyed application.
 *
 * The claim Step 8 has to make good on is one sentence from docs/BUILD_PLAN.md:
 * "the correct behavior fires from the class even if the layer is renamed". That is
 * exactly what this file drives — the REAL Animator._applyAll, with the applicator
 * methods spied, so a test passes only if the dispatch genuinely routes on the
 * motion's class and genuinely stops consulting the layer's name.
 *
 * Dependency-free: node only, no browser and no service. The DOM is stubbed just
 * deeply enough for the guards in _applyAll (querySelector for the rig/canopy
 * checks, setAttribute for the transform write) — the applicator bodies themselves
 * are replaced by spies, because what is under test is WHICH one runs.
 *
 * Pass an /analyze?swatch=1 response as argv[2] to add the real-field mesh-warp
 * measurements (see docs, or just run:
 *   curl -s -X POST 'localhost:8765/analyze?swatch=1&cls=cloth' \
 *        -F file=@assets/videos/flag.mp4 -o /tmp/flag.json
 *   node tests/step8-applicators.js /tmp/flag.json
 * ). Without it the mesh checks run on a synthetic field of realistic amplitude.
 *
 * Run: node tests/step8-applicators.js
 */
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const src = f => fs.readFileSync(path.join(ROOT, 'js', f), 'utf8');

// The app's JS are classic scripts sharing one global scope, so load them into ONE
// eval scope — `const`/`class` are lexical and would otherwise not escape it (animate.js
// reads motionfields.js's WAVE_AMP_PX, and this file needs its Animator).
global.window = global;
// Only the class + the const need exporting: `function` declarations inside a direct
// eval land in this module's own scope already, so buildMeshWarp/buildTrajField/... are
// callable below by name (and re-declaring them here would be a duplicate binding).
const { Animator, APPLICATOR_BY_CLASS } =
  eval(src('motionfields.js') + '\n' + src('animate.js')
       + '\n;({ Animator, APPLICATOR_BY_CLASS })');

let fail = 0;
const ck = (name, ok, extra = '') => {
  console.log((ok ? '  ok   ' : '  FAIL ') + name + (extra ? '  ' + extra : ''));
  if (!ok) fail++;
};
const section = t => console.log('\n' + t);

// ---------------------------------------------------------------- DOM stubs
function stubWrap() {
  return {
    _attrs: {},
    querySelector: () => null,            // no rig, no canopy, no <text>
    querySelectorAll: () => [],
    setAttribute(k, v) { this._attrs[k] = v; },
    getAttribute(k) { return this._attrs[k]; },
    getBBox: () => ({ x: 0, y: 0, width: 100, height: 60 }),
    ownerSVGElement: { viewBox: { baseVal: { width: 1000, height: 800 } } },
  };
}
function sel(name, motionId) {
  return { kind: 'svg', name, motionId, wrap: stubWrap(), center: [50, 30],
           speed: 1, intensity: 1 };
}

// ---------------------------------------------------------------- fixtures
const PARAMS = { frequency: 1.2, amplitude: 0.4, direction: 30, turbulence: 0.2,
                 damping: 0.1, phaseSpread: 0.3, driftX: 0.2, driftY: 0 };
/* A 4x4 grid of tracks — enough for buildTrajField (needs a square count >= 4).
   The per-column phase shift matters: buildTrajField deforms IN PLACE, subtracting
   the per-frame mean displacement, so a fixture where every track moves identically
   is pure bulk drift and correctly reduces to nothing. This one carries a travelling
   wave, i.e. real RELATIVE motion, at an amplitude in the range distill actually
   reports (a few % of the frame). */
const TRACKS = Array.from({ length: 16 }, (_, i) =>
  Array.from({ length: 20 }, (_, f) => [0.15 + 0.12 * (i % 4),
                                        0.15 + 0.12 * ((i / 4) | 0)
                                          + 0.05 * Math.sin(0.5 * f + 1.2 * (i % 4))]));
const swatch = (kind, cls, applicator, extra = {}) =>
  Object.assign({ schema_version: 1, kind, class: cls, applicator, engine: 'test',
                  fps: 15, frames: 20, confidence: 0.5, warnings: [] }, extra);
const texSw = cls => swatch('texture', cls, APPLICATOR_BY_CLASS[cls] === 'path_travel'
                              ? 'oscillate' : (APPLICATOR_BY_CLASS[cls] || 'oscillate'),
                            { params: PARAMS, tracks: TRACKS });
const PATH = { points: Array.from({ length: 20 }, (_, i) => [i, 0.01 * i, 0]),
               travel: { dist: 0.2 }, straightness: 0.9, label: 'boat' };

function motion(id, over = {}) {
  return Object.assign({ id, class: '', params: PARAMS, trajectories: null,
                         path: null, pose: null, swatches: [] }, over);
}

// ---------------------------------------------------------------- harness
const SPIED = ['_applyCloth', '_applyFluid', '_applyFlock', '_applyPathTravel',
               '_applyBirds', '_applyClouds', '_applyRiver', '_applyBoat',
               '_applyCharacter', '_applyLeafFall', '_applyTreeLeaves'];

/* Run the REAL _applyAll over one selection and report which applicator it chose.
   `handled` says whether the spy claimed the motion (the capability check) — a spy
   returning false is how "this artwork cannot take that applicator" is simulated. */
function run(s, m, opts = {}) {
  const anim = new Animator(
    { selections: [s], syncHighlights() {}, overlay: null, mode: 'svg' },
    { getById: id => (id === m.id ? m : null) });
  const calls = [];
  for (const k of SPIED) {
    anim[k] = function (sl, mo) {
      calls.push(k);
      return opts.cannotHandle === k ? false : true;
    };
  }
  // Only the parametric tail of _applyAll writes the wrapper transform (every
  // applicator is a spy here), so that attribute appearing IS the marker that the
  // dispatch fell through to the default.
  anim._applyAll(0.5);
  if (s.wrap._attrs.transform) calls.push('<parametric default>');
  return calls;
}

// ================================================================ 1. routing
section('Routing: the applicator comes from the CLASS, not the layer name');

for (const [cls, expect] of Object.entries(APPLICATOR_BY_CLASS)) {
  if (cls === 'articulated' || cls === 'rigid_path') continue;   // need a payload, below
  const m = motion('m-' + cls, { class: cls, swatches: [texSw(cls)] });
  const method = { wave: '_applyCloth', flow_field: '_applyFluid',
                   flock_drift: '_applyFlock', oscillate: '<parametric default>' }[expect];
  // the layer is named after a DIFFERENT object on purpose
  ck(`${cls} swatch on a layer named "Birds" -> ${expect}`,
     run(sel('Birds', m.id), m).includes(method));
}

const cloth = motion('m1', { class: 'cloth', swatches: [texSw('cloth')] });
ck('the SAME cloth swatch routes identically on "Flag", "Birds" and "Layer 7"',
   ['Flag', 'Birds', 'Layer 7', ''].every(n => run(sel(n, cloth.id), cloth).includes('_applyCloth')));

const flock = motion('m2', { class: 'flock', swatches: [texSw('flock')] });
ck('a flock swatch dropped on a layer named "Flag" drifts as a flock (not cloth)',
   (c => c.includes('_applyFlock') && !c.includes('_applyCloth'))(run(sel('Flag', flock.id), flock)));

ck('no curated name-keyed behaviour fires for a classified motion',
   ['Birds', 'Clouds', 'River', 'Boat'].every(n => {
     const c = run(sel(n, cloth.id), cloth);
     return !c.some(k => ['_applyBirds', '_applyClouds', '_applyRiver', '_applyBoat'].includes(k));
   }));

// ================================================================ 2. two axes
section('Routing takes (kind, class), not class alone');

const boatPath = motion('m3', { class: 'rigid_path', path: PATH,
                                swatches: [swatch('path', 'rigid_path', 'path_travel', { path: PATH }),
                                           texSw('rigid_path')] });
ck('a rigid_path motion WITH a tracked path travels', run(sel('Layer 3', boatPath.id), boatPath).includes('_applyPathTravel'));

const swOnlyPath = motion('m3b', { class: 'rigid_path',            // no legacy motion.path
                                   swatches: [swatch('path', 'rigid_path', 'path_travel', { path: PATH }),
                                              texSw('rigid_path')] });
ck('...and it travels from the path SWATCH alone, with no legacy motion.path',
   run(sel('Layer 3', swOnlyPath.id), swOnlyPath).includes('_applyPathTravel'));

const boatNoPath = motion('m4', { class: 'rigid_path', swatches: [texSw('rigid_path')] });
ck('a rigid_path TEXTURE swatch does NOT go to path_travel — it has no path',
   !run(sel('Boat', boatNoPath.id), boatNoPath).includes('_applyPathTravel'));
ck("...and it does not silently fall back to the curated boat bob either",
   !run(sel('Boat', boatNoPath.id), boatNoPath).includes('_applyBoat'));

const artTex = motion('m5', { class: 'articulated', swatches: [texSw('articulated')] });
ck('an articulated swatch on artwork with NO rig does not drive the rig',
   !run(sel('Duck', artTex.id), artTex).includes('_applyCharacter'));

/* The other half of that axis: a SKELETON swatch on artwork that DOES have a rig must
   drive it — and from the swatch alone, with no legacy top-level motion.pose, which is
   what lets the client stop sending the payload twice. */
const POSE = { joints: ['nose', 'l_hip', 'r_hip'], fps: 15,
               frames: Array.from({ length: 20 }, (_, f) =>
                 [[0.5, 0.2 + 0.01 * Math.sin(f), 1], [0.45, 0.5, 1], [0.55, 0.5, 1]]) };
function rigSel(name, motionId) {          // artwork with a character rig in it
  const s = sel(name, motionId);
  s.wrap.querySelector = q => (/character|data-role="body"/.test(q) ? { getAttribute: () => null } : null);
  return s;
}
const skel = motion('m6', { class: 'articulated',
                            swatches: [swatch('skeleton', 'articulated', 'skeletal', { pose: POSE })] });
ck('a skeleton swatch on rigged artwork drives the rig, from the swatch alone',
   run(rigSel('Layer 2', skel.id), skel).includes('_applyCharacter'));
ck('...and the same swatch on UNrigged artwork does not fake a walk',
   !run(sel('Layer 2', skel.id), skel).includes('_applyCharacter'));

// ================================================================ 3. fallback
section('The name-keyed curation survives as a PRESET-only fallback');

const preset = motion('p1');            // no class, no swatches, no field
for (const [name, method] of [['Birds', '_applyBirds'], ['Clouds', '_applyClouds'],
                              ['River', '_applyRiver'], ['Boat', '_applyBoat'],
                              ['Rowboat', '_applyBoat'], ['Ripples', '_applyRiver']]) {
  ck(`preset on "${name}" still gets ${method}`, run(sel(name, preset.id), preset).includes(method));
}
ck('a preset on an unrecognised name gets the parametric default',
   run(sel('Layer 9', preset.id), preset).includes('<parametric default>'));

// a swatch the service could not classify must NOT be mistaken for a classification
const unclassified = motion('p2', { swatches: [swatch('texture', '', 'oscillate',
                                                      { params: PARAMS, tracks: TRACKS })] });
ck('an UNCLASSIFIED swatch (empty class) leaves the curation in charge',
   run(sel('Birds', unclassified.id), unclassified).includes('_applyBirds'));

// ================================================================ 4. capability
section('An applicator that cannot run on this artwork falls through honestly');

ck('cloth on non-deformable artwork falls through to the parametric default',
   run(sel('Layer 1', cloth.id), cloth, { cannotHandle: '_applyCloth' })
     .includes('<parametric default>'));
ck('...and does NOT reach for a name-keyed behaviour to cover it',
   !run(sel('Birds', cloth.id), cloth, { cannotHandle: '_applyCloth' }).includes('_applyBirds'));

// ================================================================ 5. table sync
section('The JS table mirrors contracts.py');

const py = fs.readFileSync(path.join(ROOT, 'service', 'contracts.py'), 'utf8');
const pyClasses = [...py.matchAll(/^ {4}"(\w+)": \{$/gm)].map(m => m[1]);
const pyApps = [...py.matchAll(/^ {8}"applicator": "(\w+)"/gm)].map(m => m[1]);
ck('every class in MOTION_CLASSES has a JS applicator', pyClasses.length > 0
   && pyClasses.every(c => APPLICATOR_BY_CLASS[c]), pyClasses.join(','));
ck('and it is the SAME applicator contracts.py names',
   pyClasses.every((c, i) => APPLICATOR_BY_CLASS[c] === pyApps[i]),
   pyClasses.map((c, i) => `${c}=${pyApps[i]}/${APPLICATOR_BY_CLASS[c]}`).join(' '));
ck('the JS table adds nothing contracts.py does not know about',
   Object.keys(APPLICATOR_BY_CLASS).every(c => pyClasses.includes(c)));

// ================================================================ 6. mesh warp
section('Mesh warp (the cloth/fluid deformation) — properties');

const realJson = process.argv[2];
let field, label;
if (realJson && fs.existsSync(realJson)) {
  const j = JSON.parse(fs.readFileSync(realJson, 'utf8'));
  const sw = (j.swatches || []).find(s => s.kind === 'texture');
  field = sw && buildTrajField({ id: 'r', swatches: [sw] });
  label = `real captured field (${path.basename(realJson)})`;
}
if (!field) {
  // realistic amplitude: distill's normalized displacements are a few % of the frame
  field = (u, v, t) => ({ dx: 0.08 * Math.sin(u * 19 + t * 3) * Math.cos(v * 17 - t * 2),
                          dy: 0.08 * Math.sin(v * 23 + t * 5) * Math.cos(u * 13) });
  label = 'synthetic field at realistic amplitude';
}
console.log(`  (${label})`);

const box = { minX: 0, minY: 0, maxX: 180, maxY: 110 }, W = 180, H = 110;
const warp = buildMeshWarp(field, box, { anchor: 'x0' });
const free = buildMeshWarp(field, box, { anchor: 'none' });
ck('a warp is built and reports its lattice', !!warp && warp.controls === 20, `${warp && warp.controls} controls`);

// continuity: 1px-apart neighbours must stay ~1px apart (this is the guarantee)
let sep = 0, direct = 0;
for (let t = 0; t < 4; t += 0.25) for (let y = 2; y < H; y += 3) for (let x = 2; x < W; x += 3) {
  const a = warp(x, y, t, 1), b = warp(x + 1, y, t, 1);
  sep = Math.max(sep, Math.hypot(b.x - a.x, b.y - a.y));
  const fa = field(x / W, y / H, t), fb = field((x + 1) / W, y / H, t);
  direct = Math.max(direct, Math.hypot(1 + (fb.dx - fa.dx) * W * 0.5, (fb.dy - fa.dy) * H * 0.5));
}
ck('1px-apart neighbours stay within 2px (continuous: geometry cannot separate)', sep < 2.0, `${sep.toFixed(2)}px`);
ck('and less far apart than the per-point path it replaces', sep < direct,
   `mesh ${sep.toFixed(2)}px vs fieldD ${direct.toFixed(2)}px`);

// local SHAPE distortion — the number that justifies retiring the flag regex
function distortion(map) {
  const out = [], S = 4;
  for (let t = 0; t < 4; t += 0.25) for (let y = 2; y < H - S; y += 3) for (let x = 2; x < W - S; x += 3) {
    const p = [[x, y], [x + S, y], [x + S, y + S], [x, y + S]], q = p.map(c => map(c[0], c[1], t, 1));
    const pc = [x + S / 2, y + S / 2];
    const qc = [q.reduce((a, v) => a + v.x, 0) / 4, q.reduce((a, v) => a + v.y, 0) / 4];
    let sd = 0, sc = 0;
    for (let i = 0; i < 4; i++) {
      const ax = p[i][0] - pc[0], ay = p[i][1] - pc[1];
      const bx = q[i].x - qc[0], by = q[i].y - qc[1];
      sd += ax * bx + ay * by; sc += ax * by - ay * bx;
    }
    const th = Math.atan2(sc, sd), ct = Math.cos(th), st = Math.sin(th);
    let r = 0;
    for (let i = 0; i < 4; i++) {
      const ax = p[i][0] - pc[0], ay = p[i][1] - pc[1];
      r = Math.max(r, Math.hypot(q[i].x - qc[0] - (ax * ct - ay * st),
                                 q[i].y - qc[1] - (ax * st + ay * ct)));
    }
    out.push(r / S * 100);
  }
  out.sort((a, b) => a - b);
  return { p50: out[out.length >> 1], p95: out[Math.floor(out.length * 0.95)], max: out[out.length - 1] };
}
const perPoint = (x, y, t) => { const f = field(x / W, y / H, t); return { x: x + f.dx * W * 0.5, y: y + f.dy * H * 0.5 }; };
const dA = distortion(perPoint), dB = distortion(warp);
console.log(`    local shape distortion (% of a 4px patch):`);
console.log(`      per-point fieldD : p50 ${dA.p50.toFixed(1)}%  p95 ${dA.p95.toFixed(1)}%  max ${dA.max.toFixed(0)}%`);
console.log(`      rigid MLS warp   : p50 ${dB.p50.toFixed(1)}%  p95 ${dB.p95.toFixed(1)}%  max ${dB.max.toFixed(0)}%`);
ck('mesh warp cuts local shape distortion at least 2x at p95',
   dB.p95 * 2 <= dA.p95, `${(dA.p95 / dB.p95).toFixed(1)}x`);

// anchoring
let pin = 0, whip = 0, freeEdgeUnpinned = 0;
for (let t = 0; t < 4; t += 0.1) for (let y = 0; y <= H; y += 10) {
  const a = warp(0, y, t, 1), b = warp(W, y, t, 1), c = free(0, y, t, 1);
  pin = Math.max(pin, Math.hypot(a.x, a.y - y));
  whip = Math.max(whip, Math.hypot(b.x - W, b.y - y));
  freeEdgeUnpinned = Math.max(freeEdgeUnpinned, Math.hypot(c.x, c.y - y));
}
ck("anchor 'x0' holds the pinned edge", pin < 2.5, `${pin.toFixed(2)}px`);
ck("anchor 'x0' lets the free edge whip much further", whip > pin * 4, `${whip.toFixed(1)}px vs ${pin.toFixed(2)}px`);
ck('the whip is visible motion, not a twitch', whip > 4, `${whip.toFixed(1)}px`);
ck("anchor 'none' pins nothing (a river surface is held by nothing)", freeEdgeUnpinned > 2.5,
   `${freeEdgeUnpinned.toFixed(2)}px`);

// the field really drives it
const at = t => warp(W * 0.9, H * 0.5, t, 1);
let moved = 0;
for (let t = 0; t < 4; t += 0.05) moved = Math.max(moved, Math.hypot(at(t).x - at(0).x, at(t).y - at(0).y));
ck('the free edge moves over time (the captured field is driving it)', moved > 2, `${moved.toFixed(1)}px`);

let maxRot = 0;
for (let t = 0; t < 4; t += 0.05) for (let y = 5; y < H; y += 7) for (let x = 5; x < W; x += 9)
  maxRot = Math.max(maxRot, Math.abs(warp(x, y, t, 1).rot));
ck('a detail path gets a real rotation to ride, not just a slide', maxRot > 0.5, `${maxRot.toFixed(2)} deg`);

// no folding: a control column must never cross its neighbour
const NX = 5, NY = 4, sx = W / (NX - 1);
let overlap = 0;
for (let t = 0; t < 4; t += 0.1) {
  const P = [];
  for (let jj = 0; jj < NY; jj++) for (let ii = 0; ii < NX; ii++) {
    const u = ii / (NX - 1), v = jj / (NY - 1), s = field(u, v, t), r = Math.pow(u, 1.15);
    P.push([u * W + s.dx * W * 0.5 * r, v * H + s.dy * H * 0.5 * r]);
  }
  for (let jj = 0; jj < NY; jj++) for (let ii = 0; ii < NX - 1; ii++)
    overlap = Math.max(overlap, -(P[jj * NX + ii + 1][0] - P[jj * NX + ii][0]) / sx);
}
ck('the lattice does not fold (no column crosses its neighbour)', overlap <= 0,
   `worst ${(overlap * 100).toFixed(1)}% of a cell`);

// rest + determinism
const still = buildMeshWarp(() => ({ dx: 0, dy: 0 }), box, { anchor: 'x0' });
let maxStill = 0;
for (let y = 0; y <= H; y += 5) for (let x = 0; x <= W; x += 6) {
  const p = still(x, y, 1.5, 1);
  maxStill = Math.max(maxStill, Math.hypot(p.x - x, p.y - y), Math.abs(p.rot));
}
ck('a zero field leaves the artwork exactly where it was', maxStill < 1e-9, maxStill.toExponential(1));
const q1 = warp(37, 21, 2.5, 1), q2 = warp(37, 21, 2.5, 1), q3 = warp(37, 21, 2.5, 2);
ck('same (t, intensity) is deterministic', q1.x === q2.x && q1.y === q2.y);
ck('changing intensity re-evaluates the lattice (the cache is keyed on it)',
   Math.hypot(q3.x - q1.x, q3.y - q1.y) > 0.1);

// a swatch alone is a complete source — no raw `trajectories` needed
ck('buildTrajField works from a swatch alone (Step 7 payload is self-sufficient)',
   typeof buildTrajField({ id: 'x', swatches: [texSw('cloth')] }) === 'function');

// ================================================================ 7. flock
section('Flock drift is driven by the captured params, not by bird constants');

/* Drive the real _applyFlock over stub children and check the heading follows
   params.direction. Children are laid out with room in every direction. */
function flockRun(params, n = 6) {
  const kids = Array.from({ length: n }, (_, i) => ({
    _t: null,
    getBBox: () => ({ x: 400 + i * 10, y: 350, width: 8, height: 6 }),
    setAttribute(k, v) { if (k === 'transform') this._t = v; },
    removeAttribute() { this._t = null; },
  }));
  const s = sel('Layer 4', 'f1');
  s.wrap.querySelectorAll = () => kids;
  const anim = new Animator({ selections: [s], syncHighlights() {} }, { getById: () => null });
  const m = motion('f1', { class: 'flock', params, swatches: [texSw('flock')] });
  const out = [];
  for (let t = 0.5; t < 12; t += 0.5) {
    anim._applyFlock(s, m, t, 1);
    out.push(kids.map(k => (k._t.match(/translate\(([-\d.]+) ([-\d.]+)\)/) || [])
                             .slice(1).map(Number)));
  }
  return out;
}
const east = flockRun(Object.assign({}, PARAMS, { direction: 0, driftX: 0, turbulence: 0 }));
const north = flockRun(Object.assign({}, PARAMS, { direction: 90, driftX: 0, turbulence: 0 }));
const maxOf = (fr, ax) => Math.max(...fr.flat().map(p => p[ax]));
const minOf = (fr, ax) => Math.min(...fr.flat().map(p => p[ax]));
ck('direction 0 drifts the flock +x (screen right)', maxOf(east, 0) > 3 && Math.abs(maxOf(east, 1)) < 0.6,
   `dx ${maxOf(east, 0).toFixed(1)} dy ${maxOf(east, 1).toFixed(1)}`);
ck('direction 90 drifts it -y (screen up), so heading is really read',
   minOf(north, 1) < -3 && Math.abs(maxOf(north, 0)) < 0.6,
   `dx ${maxOf(north, 0).toFixed(1)} dy ${minOf(north, 1).toFixed(1)}`);
ck('members never reverse into the flock (drift only ever eases out and back)',
   minOf(east, 0) >= -1e-9, `min dx ${minOf(east, 0).toFixed(3)}`);
const calm = flockRun(Object.assign({}, PARAMS, { turbulence: 0, driftX: 0 }));
const rough = flockRun(Object.assign({}, PARAMS, { turbulence: 0.8, driftX: 0 }));
const spread = fr => Math.max(...fr.map(f => Math.max(...f.map(p => p[1])) - Math.min(...f.map(p => p[1]))));
ck('turbulence loosens the formation (captured turbulence is actually used)',
   spread(rough) > spread(calm) + 1, `${spread(calm).toFixed(1)} -> ${spread(rough).toFixed(1)}`);
/* One path cannot be a flock: _applyFlock must decline (so _applyByClass returns false
   and _applyAll's parametric tail sways the object as a whole) rather than invent a
   formation out of a single member. Asserted on the return value — nothing is written. */
ck('a single-path object is not a flock (declines instead of faking one)',
   (() => {
     const s = sel('Layer 4', 'f1'); s.wrap.querySelectorAll = () => [{}];
     const anim = new Animator({ selections: [s], syncHighlights() {} }, { getById: () => null });
     return anim._applyFlock(s, motion('f1', { class: 'flock', params: PARAMS }), 1, 1) === false
            && s.wrap._attrs.transform === undefined;
   })());

console.log(fail ? `\n${fail} CHECK(S) FAILED` : '\nall step 8 checks passed');
process.exit(fail ? 1 : 0);
