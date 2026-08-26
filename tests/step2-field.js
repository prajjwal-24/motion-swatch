/*
 * Step 2 selftest, renderer half — what a GATED trajectory field does downstream.
 *
 * tests/step2-preprocess.py proves the service excludes background cells by freezing
 * them. That is only half the done-when: a frozen cell is a claim ("nothing was
 * measured here"), and every average the renderer takes has to read it that way rather
 * than as a measured zero. This file drives the REAL buildTrajField over fields whose
 * correct answer is known by construction, so the checks are arithmetic, not eyeballing.
 *
 * The bug this pins down is concrete. buildTrajField subtracts the per-frame mean
 * displacement to keep a drifting object on its bed; the mean used to divide by all 144
 * tracks, so with 8 live cells the drift was under-subtracted 18x and the shape slid
 * clean off. That was already reachable before Step 2 (segment.py freezes out-of-region
 * cells the same way), which is why the fix is measured here rather than assumed.
 *
 * Dependency-free: node only, no browser, no service.
 *
 * Run: node tests/step2-field.js
 */
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const src = f => fs.readFileSync(path.join(ROOT, 'js', f), 'utf8');

// classic scripts share one global scope; load into ONE eval scope so lexical
// declarations are visible to each other (see tests/step8-applicators.js).
global.window = global;
eval(src('motionfields.js'));

const FAIL = [];
const check = (name, cond, detail = '') => {
  console.log((cond ? '  ok   ' : '  FAIL ') + name + (detail && !cond ? '  — ' + detail : ''));
  if (!cond) FAIL.push(name);
};

// ---------------------------------------------------------------- fixtures
const G = 12, T = 20;

/* A 12x12 field in service format ([track][frame][x, y], normalized, gy-major).
   `live(gx, gy)` decides which cells move; movers travel `(dx, dy)` per frame and
   additionally oscillate so the field has relative motion, not just bulk drift.
   Frozen cells repeat their seed point exactly, as grid_trajectories does. */
function field(live, dx, dy, osc = 0) {
  const tracks = [];
  for (let gy = 0; gy < G; gy++) {
    for (let gx = 0; gx < G; gx++) {
      const x = (gx + 0.5) / G, y = (gy + 0.5) / G;
      const pts = [[x, y]];
      const moving = live(gx, gy);
      for (let f = 1; f <= T; f++) {
        pts.push(moving
          ? [x + dx * f + osc * Math.sin(f * 0.7 + gx), y + dy * f]
          : [x, y]);
      }
      tracks.push(pts);
    }
  }
  return tracks;
}

const ALL = () => true;
const QUADRANT = (gx, gy) => gx < 6 && gy < 6;          // 36 of 144 cells
const EIGHT = (gx, gy) => gy >= 10 && gx >= 8;          // 8 of 144, like a real SAM 2 mask

// what the sampler says the field does at the centre of a cell, over the whole clip
function sampleRange(motion, u, v) {
  const f = buildTrajField(motion);
  if (!f) return null;
  let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
  for (let i = 0; i <= 60; i++) {
    const d = f(u, v, i / 20);
    lo = [Math.min(lo[0], d.dx), Math.min(lo[1], d.dy)];
    hi = [Math.max(hi[0], d.dx), Math.max(hi[1], d.dy)];
  }
  return { dx: hi[0] - lo[0], dy: hi[1] - lo[1], peak: Math.max(Math.abs(lo[1]), Math.abs(hi[1])) };
}

const mo = tracks => ({ trajectories: tracks, trajFps: 20 });

// ---------------------------------------------------------------- checks
console.log('ACTIVE WINDOW — one rule, shared by the chip and the replay:');
{
  const rel = t => t.map(tr => tr.map(p => [p[0] - tr[0][0], p[1] - tr[0][1]]));
  const full = activeCellWindow(rel(field(ALL, 0.002, 0.003, 0.01)), G);
  check('a field that moves everywhere spans the whole grid',
        JSON.stringify(full) === JSON.stringify([0, G - 1, 0, G - 1]), JSON.stringify(full));
  const q = activeCellWindow(rel(field(QUADRANT, 0.002, 0.003, 0.01)), G);
  check('a field live in one quadrant reports that quadrant plus a cell of margin',
        JSON.stringify(q) === JSON.stringify([0, 6, 0, 6]), JSON.stringify(q));
  const e = activeCellWindow(rel(field(EIGHT, 0.002, 0.003, 0.01)), G);
  check('an 8-cell mask reports an 8-cell corner, not the whole frame',
        e[0] >= 7 && e[2] >= 9 && e[1] === G - 1 && e[3] === G - 1, JSON.stringify(e));
  const dead = activeCellWindow(rel(field(() => false, 0, 0)), G);
  check('a field where nothing moved falls back to the whole grid, not an inverted one',
        JSON.stringify(dead) === JSON.stringify([0, G - 1, 0, G - 1]), JSON.stringify(dead));
}

console.log('\nFROZEN CELLS ARE NOT MEASURED ZEROS:');
{
  // The headline. Same object motion, same drift; only the number of frozen cells
  // differs, so a difference between the two IS the bug. The tolerance is calibrated on
  // the pre-fix code rather than picked: dividing by all 144 tracks left the 8-live field
  // with peak |dy| 0.41556 where the fix leaves 0.00000 (the residual is amplified past
  // the raw 0.080/clip drift because the sampler normalizes by the active window, which
  // is 2 cells tall here). A tolerance loose enough to pass 0.41556 would prove nothing.
  const drift = 0.004;                                   // per frame, downward
  const wide = sampleRange(mo(field(ALL, 0, drift, 0.012)), 0.5, 0.5);
  const narrow = sampleRange(mo(field(EIGHT, 0, drift, 0.012)), 0.5, 0.5);
  const BUDGET = 0.005;
  check('a fully live field replays with its bulk drift removed',
        wide && wide.peak < BUDGET, wide && String(wide.peak));
  check('an 8-live-cell field removes just as much — the mean is over LIVE tracks only',
        narrow && narrow.peak < BUDGET, narrow && String(narrow.peak));
  console.log(`       peak |dy| after drift removal: 144 live ${wide.peak.toFixed(5)}, `
            + `8 live ${narrow.peak.toFixed(5)} `
            + `(budget ${BUDGET}; pre-fix measured 0.41556 for 8 live)`);
  check('...so the number of frozen cells does not change how much drift survives',
        Math.abs(wide.peak - narrow.peak) < BUDGET,
        `${wide.peak.toFixed(5)} vs ${narrow.peak.toFixed(5)}`);

  // and the relative motion the object really had is still there
  check('the object\'s own oscillation survives the gate',
        narrow.dx > 0.004, String(narrow.dx));
}

console.log('\nPURE DRIFT — nothing to replay, said so rather than sliding:');
{
  // every live cell moves identically, so after mean removal there is no relative
  // motion at all. The field must report that, not emit a whole-object slide.
  const pure = sampleRange(mo(field(QUADRANT, 0, 0.004, 0)), 0.5, 0.5);
  check('a rigid drift with no internal motion replays as (near) nothing',
        pure === null || (pure.dx < 1e-6 && pure.dy < 1e-6),
        pure && `${pure.dx} / ${pure.dy}`);
}

console.log('\nDEGENERATE FIELDS — refused, not guessed at:');
{
  check('a field with too few live cells to interpolate is refused',
        buildTrajField(mo(field((gx, gy) => gx === 0 && gy === 0, 0.002, 0.003, 0.01))) === null);
  check('a field where nothing moved at all is refused',
        buildTrajField(mo(field(() => false, 0, 0))) === null);
  check('a non-square track count is refused rather than mis-indexed',
        buildTrajField(mo(field(ALL, 0.002, 0.003, 0.01).slice(0, 140))) === null);
  check('a motion with no field and no swatch is refused',
        buildTrajField({ trajectories: [] }) === null);
}

console.log('\nA SWATCH IS A COMPLETE SOURCE — a gated field survives Contract B:');
{
  const tracks = field(EIGHT, 0, 0.004, 0.012);
  const viaSwatch = buildTrajField({ swatches: [{ kind: 'texture', tracks, fps: 20 }] });
  check('the same gated field replays identically whether it arrives raw or in a swatch',
        viaSwatch !== null
        && JSON.stringify(viaSwatch(0.5, 0.5, 0.5))
           === JSON.stringify(buildTrajField(mo(tracks))(0.5, 0.5, 0.5)));
}

console.log('\n' + (FAIL.length ? 'FAILED: ' + FAIL.join(', ') : 'all checks passed'));
process.exit(FAIL.length ? 1 : 0);
