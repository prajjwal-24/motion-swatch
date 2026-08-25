/*
 * Step 9, client side: the frames we send have to actually contain the motion.
 *
 * Everything else in the judge loop is enforced by the server (see
 * tests/step9-judge-loop.py). What the server cannot check is WHEN the browser sampled
 * the animation — and getting that wrong fails silently in the worst possible way: eight
 * frames of a walk cycle sampled over 0.7s all show the same third of a stride, the judge
 * honestly reports "I see almost no motion", and the loop dutifully cranks the amplitude
 * on a motion that was fine. So cycleSeconds() gets its own checks.
 *
 *   node tests/step9-sampling.js
 *
 * Dependency-free: judge.js is a classic script, so it is eval'd with a window stub.
 */
const fs = require('fs'), path = require('path');
const ROOT = path.join(__dirname, '..');

global.window = global;
eval(fs.readFileSync(path.join(ROOT, 'js', 'judge.js'), 'utf8'));
const { cycleSeconds } = window.MotionJudge;

let fail = [];
const ck = (name, cond, detail = '') => {
  console.log((cond ? '  ok   ' : '  FAIL ') + name + (!cond && detail ? `  — ${detail}` : ''));
  if (!cond) fail.push(name);
};
const near = (a, b, eps = 1e-6) => Math.abs(a - b) < eps;

console.log('CAPTURED MOTION — the cycle is the clip, not 1/frequency:');

// a 4-second walk that distill happened to label 1.4 Hz. Sampling at 1/1.4s would show
// the same third of the stride eight times over.
const POSE = { joints: ['nose'], fps: 15, frames: Array.from({ length: 60 }, () => [[0.5, 0.5, 1]]) };
ck('a pose sequence spans the whole captured clip',
   near(cycleSeconds({ pose: POSE, params: { frequency: 1.4 } }, 1), 4),
   String(cycleSeconds({ pose: POSE, params: { frequency: 1.4 } }, 1)));
ck('...and frequency does not get a say when a pose exists',
   near(cycleSeconds({ pose: POSE, params: { frequency: 3.5 } }, 1),
        cycleSeconds({ pose: POSE, params: { frequency: 0.3 } }, 1)));
ck('a pose nested in a Step-7 skeleton swatch counts the same as a top-level one',
   near(cycleSeconds({ swatches: [{ kind: 'skeleton', pose: POSE }], params: { frequency: 1.4 } }, 1), 4));
ck('a pose with no frames is ignored rather than dividing by nothing',
   near(cycleSeconds({ pose: { joints: [], fps: 15, frames: [] }, params: { frequency: 1 } }, 1), 1));

const PATH = { fps: 30, points: Array.from({ length: 90 }, (_, i) => [i, 0.001 * i, 0]) };
ck('a travel path spans its own tracked duration',
   near(cycleSeconds({ path: PATH, params: { frequency: 2 } }, 1), 3),
   String(cycleSeconds({ path: PATH, params: { frequency: 2 } }, 1)));
ck('a path nested in a Step-7 path swatch counts the same',
   near(cycleSeconds({ swatches: [{ kind: 'path', path: PATH }], params: {} }, 1), 3));
ck('a one-point "path" is not a path and falls through to frequency',
   near(cycleSeconds({ path: { fps: 30, points: [[0, 0, 0]] }, params: { frequency: 2 } }, 1), 0.5));
ck('a pose outranks a path when a clip yielded both',
   near(cycleSeconds({ pose: POSE, path: PATH, params: {} }, 1), 4));

console.log('\nPARAMETRIC MOTION — one period, bounded at both ends:');
ck('the span is one period of the oscillation',
   near(cycleSeconds({ params: { frequency: 2 } }, 1), 0.5));
ck('a slow motion is capped so a run does not sample a 20-second window',
   near(cycleSeconds({ params: { frequency: 0.05001 } }, 1), 4),
   String(cycleSeconds({ params: { frequency: 0.05001 } }, 1)));
ck('a very fast motion still spans long enough to be visible',
   near(cycleSeconds({ params: { frequency: 100 } }, 1), 0.4));
ck('a zero frequency does not become an infinite span',
   near(cycleSeconds({ params: { frequency: 0 } }, 1), 2));
ck('missing params do not throw', near(cycleSeconds({}, 1), 2));
ck('a motion with neither params nor payload still yields a usable span',
   cycleSeconds({ swatches: [] }, 1) > 0);

console.log('\nSPEED — the layer dial changes what one cycle means in wall-clock time:');
// _applyAll multiplies t by the selection's speed, so at 2x a cycle takes half as long.
ck('at 2x speed one cycle takes half the time',
   near(cycleSeconds({ params: { frequency: 1 } }, 2), 0.5));
ck('at 0.5x speed it takes twice as long',
   near(cycleSeconds({ params: { frequency: 1 } }, 0.5), 2));
ck('a captured clip is rescaled by speed too',
   near(cycleSeconds({ pose: POSE }, 2), 2));
ck('an unset speed is treated as 1x, not as 0',
   near(cycleSeconds({ params: { frequency: 1 } }, undefined), 1)
   && near(cycleSeconds({ params: { frequency: 1 } }, 0), 1));
ck('every span is finite and positive for every case above',
   [{}, { params: {} }, { pose: POSE }, { path: PATH }, { params: { frequency: 0 } }]
     .every(m => [0, 0.5, 1, 3].every(sp => {
       const v = cycleSeconds(m, sp);
       return isFinite(v) && v > 0;
     })));

console.log('\nWHAT THE MODULE EXPOSES:');
for (const fn of ['tune', 'grabFrames', 'referenceFrames', 'cycleSeconds', 'render']) {
  ck(`MotionJudge.${fn} is callable`, typeof window.MotionJudge[fn] === 'function');
}
// the panel is built with innerHTML, so a critique is escaped rather than injected
const el = { innerHTML: '' };
window.MotionJudge.render(el, {
  verdict: { score: 0.5, verdict: 'tune', critique: '<img src=x onerror=alert(1)>',
             axes: {}, observations: ['<script>bad()</script>'], deltas: {} },
  iterations: 1, reason: 'ok',
});
ck('a critique from the model cannot inject markup into the panel',
   !el.innerHTML.includes('<img') && !el.innerHTML.includes('<script')
   && el.innerHTML.includes('&lt;img'), el.innerHTML.slice(0, 90));

console.log('\n' + (fail.length ? 'FAILED: ' + fail.join(', ') : 'all checks passed'));
process.exit(fail.length ? 1 : 0);
