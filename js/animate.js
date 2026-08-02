/*
 * animate.js — per-selection displacement engine.
 *
 * For each selection with a motion assigned, compute a parametric
 * displacement (dx, dy, rotation) each frame and apply it:
 *   - SVG selection   → set the wrapper <g>'s transform ATTRIBUTE
 *                       (verified: setting CSS transform on a group that
 *                        already has a transform="" attr conflicts; the
 *                        empty wrapper's attribute does not).
 *   - Raster selection → set the floating clone's CSS transform.
 */

const AMP_PX = 16;     // svg viewBox units at amplitude 1
const TURB_PX = 11;
const DRIFT_PX = 26;   // bounded travel range for steady drift (driftX/driftY)
const DRIFT_PERIOD = 10; // seconds per drift loop

function _hash(n) { const s = Math.sin(n) * 43758.5453; return s - Math.floor(s); }
function _noise(x) { const i = Math.floor(x), f = x - i; const u = f * f * (3 - 2 * f); return _hash(i) * (1 - u) + _hash(i + 1) * u; }
function _curl(x, y, t) {
  const s = 0.015;
  return {
    x: (_noise(x * s + t * 1.7 + 5.2) - 0.5) * 2 + (_noise(x * s * 3 + t * 3.1 + y * s * 2) - 0.5),
    y: (_noise(y * s + t * 1.9 + 91.7) - 0.5) * 2 + (_noise(y * s * 3 + t * 2.7 + x * s * 2) - 0.5),
  };
}
function _env(t, damping, seed) {
  if (damping <= 0.02) return 1;
  const rate = 0.6 + damping * 1.8;
  const n = _noise(t * rate + seed * 37.7);
  return (1 - damping) + damping * Math.pow(n, 1 + damping * 4) * 2.2;
}

/*
 * Pure displacement function — shared by the Animator and the motion-card
 * previews so a preview shows exactly what the motion will do to an object.
 * Returns {dx, dy, rot} where dx/dy are in viewBox units and rot is degrees.
 */
function computeMotion(params, seed, t, intensity = 1) {
  const p = params;
  const th = p.direction * Math.PI / 180;
  const dir = { x: Math.cos(th), y: -Math.sin(th) };
  const env = _env(t, p.damping, seed);
  const osc = Math.sin(2 * Math.PI * p.frequency * t + seed * p.phaseSpread * 3.0);
  const turb = _curl(seed * 40, seed * 70, t + seed);
  const A = p.amplitude * AMP_PX * intensity * env;
  // steady drift (falling leaves, flowing water): captured as driftX/driftY.
  // A poster object can't leave the frame, so drift becomes a slow sawtooth
  // loop — travels DRIFT_PX in the drift direction, eases back, repeats.
  let ddx = 0, ddy = 0;
  const dxv = p.driftX || 0, dyv = p.driftY || 0;
  if (dxv || dyv) {
    const phase = (t % DRIFT_PERIOD) / DRIFT_PERIOD;          // 0..1
    const loop = phase < 0.85 ? phase / 0.85 : (1 - phase) / 0.15; // ramp out, ease back
    ddx = dxv * DRIFT_PX * loop * intensity;
    ddy = dyv * DRIFT_PX * loop * intensity;
  }
  return {
    dx: A * osc * dir.x + p.turbulence * TURB_PX * turb.x * env * intensity + ddx,
    dy: A * osc * dir.y + p.turbulence * TURB_PX * turb.y * env * intensity + ddy,
    rot: (0.06 * p.amplitude * intensity * env
        * Math.cos(2 * Math.PI * p.frequency * t)) * (180 / Math.PI),
  };
}

/*
 * ---- Trajectory-field replay ----
 * Captured motions carry a 12x12 grid of REAL point trajectories from the
 * source video (normalized coords). Instead of a synthetic sine, we can
 * replay that field: each sampled path point bilinearly samples the
 * displacement of the corresponding region of the SOURCE VIDEO at time t.
 * The flag's free edge whips like the real free edge; water churns like the
 * real water; local stretching/compression (foreshortening) emerges because
 * neighboring points genuinely converge/diverge like they did on camera.
 *
 * buildTrajField(motion) → sampler(u, v, t) -> {dx, dy} in unit-square units
 */
function buildTrajField(motion) {
  const tracks = motion.trajectories;
  if (!tracks || tracks.length < 4) return null;
  const G = Math.round(Math.sqrt(tracks.length));          // 12
  if (G * G !== tracks.length) return null;
  const T = tracks[0].length;
  if (T < 4) return null;
  const fps = motion.trajFps || 15;

  // displacement-from-start per track
  const disp = tracks.map(tr => {
    const x0 = tr[0][0], y0 = tr[0][1];
    return tr.map(p => [p[0] - x0, p[1] - y0]);
  });

  // DEFORM-IN-PLACE: a flowing river / rippling silk has a strong bulk drift —
  // every point travels the same way, so replaying it raw would slide the
  // WHOLE object off its bed and leave the region empty. Subtract the
  // per-frame mean displacement so only the *relative* motion remains: the
  // surface ripples and flows in place, the shape stays where it belongs.
  for (let fi = 0; fi < T; fi++) {
    let mx = 0, my = 0;
    for (let c = 0; c < disp.length; c++) { mx += disp[c][fi][0]; my += disp[c][fi][1]; }
    mx /= disp.length; my /= disp.length;
    for (let c = 0; c < disp.length; c++) { disp[c][fi][0] -= mx; disp[c][fi][1] -= my; }
  }

  // ACTIVE REGION: the moving object usually fills only part of the video
  // frame (a flag against sky, waves in the lower half). Find the grid cells
  // with real motion and map the target object onto that window only —
  // otherwise parts of the artwork would sample static background and freeze.
  const energy = disp.map(tr => {
    let e = 0;
    for (const p of tr) { const m = Math.hypot(p[0], p[1]); if (m > e) e = m; }
    return e;
  });
  const eMax = Math.max(...energy);
  if (eMax < 0.002) return null;                           // nothing moved
  const thresh = Math.max(0.004, eMax * 0.25);
  let x0g = G - 1, x1g = 0, y0g = G - 1, y1g = 0;
  for (let gy = 0; gy < G; gy++)
    for (let gx = 0; gx < G; gx++)
      if (energy[gy * G + gx] >= thresh) {
        if (gx < x0g) x0g = gx; if (gx > x1g) x1g = gx;
        if (gy < y0g) y0g = gy; if (gy > y1g) y1g = gy;
      }
  x0g = Math.max(0, x0g - 1); x1g = Math.min(G - 1, x1g + 1);
  y0g = Math.max(0, y0g - 1); y1g = Math.min(G - 1, y1g + 1);
  const spanX = Math.max(1, x1g - x0g), spanY = Math.max(1, y1g - y0g);
  // active window size in frame units — displacements are normalized by this
  // so motion is proportional to the OBJECT, not the video frame
  const aW = Math.max(0.08, spanX / (G - 1));
  const aH = Math.max(0.08, spanY / (G - 1));

  // ping-pong the loop to avoid the end-snap
  const frameAt = (t) => {
    const f = (t * fps) % (2 * (T - 1));
    return f <= T - 1 ? f : 2 * (T - 1) - f;
  };
  const CAP = 0.55;    // max object-relative displacement

  const sample = function sample(u, v, t) {
    const f = frameAt(t);
    const f0 = Math.floor(f), f1 = Math.min(T - 1, f0 + 1), fw = f - f0;
    const gx = Math.min(G - 1.001, Math.max(0, (x0g + u * spanX)));
    const gy = Math.min(G - 1.001, Math.max(0, (y0g + v * spanY)));
    const xi = Math.floor(gx), yi = Math.floor(gy);
    const xw = gx - xi, yw = gy - yi;
    let dx = 0, dy = 0;
    for (const [ox, oy, w] of [[0, 0, (1 - xw) * (1 - yw)], [1, 0, xw * (1 - yw)],
                               [0, 1, (1 - xw) * yw], [1, 1, xw * yw]]) {
      const tr = disp[Math.min(G - 1, yi + oy) * G + Math.min(G - 1, xi + ox)];
      const a = tr[f0], b = tr[f1];
      dx += w * (a[0] * (1 - fw) + b[0] * fw);
      dy += w * (a[1] * (1 - fw) + b[1] * fw);
    }
    // object-relative units, capped
    return {
      dx: Math.max(-CAP, Math.min(CAP, dx / aW)),
      dy: Math.max(-CAP, Math.min(CAP, dy / aH)),
    };
  };
  sample.period = 2 * (T - 1) / fps;   // seconds per seamless ping-pong loop
  return sample;
}

/* One frame of trajectory-field deformation for a sampled path. */
function fieldD(pd, minX, minY, w, h, field, t, intensity) {
  let d = '';
  for (let i = 0; i < pd.pts.length; i++) {
    const [x0, y0] = pd.pts[i];
    const u = (x0 - minX) / w, v = (y0 - minY) / h;
    const s = field(u, v, t);
    const x = x0 + s.dx * w * 0.5 * intensity;
    const y = y0 + s.dy * h * 0.5 * intensity;
    d += (i ? 'L' : 'M') + x.toFixed(2) + ',' + y.toFixed(2);
  }
  return pd.closed ? d + 'Z' : d;
}

/*
 * ---- Per-glyph text animation ----
 * A <text> block moved rigidly reads as "the poster is shaking". Real
 * text-wearing-motion means each LETTER rides the motion at its own phase.
 * buildTextData splits every <text> in the selection into absolutely
 * positioned per-glyph <text> elements (visually identical — positions come
 * from getStartPositionOfChar, so kerning/letter-spacing/anchor are baked),
 * then the animator transforms each glyph independently: captured motions
 * sample the trajectory field at the glyph's (u,v); presets get a per-letter
 * phase offset. Letters also lean along the local wave slope.
 */
function buildTextData(wrap) {
  if (!wrap.getAttribute('data-ms-textsplit')) {
    const texts = [...wrap.querySelectorAll('text')];
    if (!texts.length) return null;
    for (const t of texts) {
      const content = t.textContent;
      const glyphs = [];
      for (let i = 0; i < content.length; i++) {
        const ch = content[i];
        if (!ch.trim()) continue;
        let pos, ext;
        try { pos = t.getStartPositionOfChar(i); ext = t.getExtentOfChar(i); }
        catch { continue; }
        const g = t.cloneNode(false);
        g.removeAttribute('text-anchor');
        g.removeAttribute('letter-spacing');
        g.setAttribute('x', pos.x.toFixed(2));
        g.setAttribute('y', pos.y.toFixed(2));
        g.setAttribute('data-ms-glyph', '1');
        g.setAttribute('data-ms-cx', (ext.x + ext.width / 2).toFixed(2));
        g.setAttribute('data-ms-cy', (ext.y + ext.height / 2).toFixed(2));
        g.textContent = ch;
        glyphs.push(g);
      }
      if (glyphs.length) {
        for (const g of glyphs) t.parentNode.insertBefore(g, t);
        t.remove();
      }
    }
    wrap.setAttribute('data-ms-textsplit', '1');
  }
  const els = [...wrap.querySelectorAll('text[data-ms-glyph]')];
  if (!els.length) return null;
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  const items = els.map(el => {
    const cx = parseFloat(el.getAttribute('data-ms-cx'));
    const cy = parseFloat(el.getAttribute('data-ms-cy'));
    if (cx < minX) minX = cx;
    if (cx > maxX) maxX = cx;
    if (cy < minY) minY = cy;
    if (cy > maxY) maxY = cy;
    return { el, cx, cy };
  });
  return { items, minX, maxX, minY, maxY };
}

/*
 * One glyph's transform under a motion at time t.
 *   field mode: sample the captured trajectory field at the glyph's (u,v),
 *               lean along the local slope (difference to the neighbor sample)
 *   preset mode: computeMotion with a per-letter phase seed (u across the line)
 */
function glyphTransform(td, it, field, params, t, intensity) {
  const tw = Math.max(1, td.maxX - td.minX);
  const th = Math.max(8, td.maxY - td.minY);
  const u = (it.cx - td.minX) / tw, v = (it.cy - td.minY) / th;
  let dx, dy, rot;
  if (field) {
    const c = field(u, v, t);
    dx = c.dx * tw * 0.5 * intensity;
    dy = c.dy * Math.max(th, tw * 0.2) * 0.5 * intensity;
    const cR = field(Math.min(1, u + 0.08), v, t);
    rot = (cR.dy - c.dy) * 60 * intensity;             // lean along wave slope
  } else {
    // per-letter phase: u spans 0..1 across the line; scale by phaseSpread so
    // high-spread motions (flutter/wave) travel visibly through the word
    const d = computeMotion(params, u * 2.2, t, intensity);
    dx = d.dx; dy = d.dy; rot = d.rot;
    // add an explicit vertical wave component riding through the letters —
    // pure-horizontal motions otherwise leave letters in near-lockstep
    const wavePhase = 2 * Math.PI * params.frequency * t - u * Math.PI * 2 * (0.5 + params.phaseSpread);
    const waveAmp = 4 * params.amplitude * intensity * (0.3 + params.phaseSpread);
    dy += waveAmp * Math.sin(wavePhase);
    rot += waveAmp * 0.55 * Math.cos(wavePhase);
  }
  return `translate(${dx.toFixed(2)} ${dy.toFixed(2)}) rotate(${rot.toFixed(2)} ${it.cx} ${it.cy})`;
}

/*
 * ---- Wave (cloth) deformation ----
 * Rigid transforms slide an object around; cloth needs the geometry itself
 * to ripple. Wave mode samples every <path> in the selection into WAVE_SAMPLES
 * points and, each frame, displaces each point with a traveling wave:
 *
 *   dy(x,t) = A · ramp(x) · sin(2πf·t − k·(x − xmin))
 *
 * ramp(x) grows 0→1 from the anchored edge (pole side) to the free tip, so
 * the cloth stays pinned at the pole and whips at the end — like a real flag.
 */
const WAVE_SAMPLES = 48;
const WAVE_AMP_PX = 9;          // amplitude at amplitude=1, intensity=1
const WAVE_CYCLES = 1.5;        // wavelengths across the object's width

function buildWaveData(wrap) {
  const paths = [];
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  // group bbox first, so we can tell big "cloth" paths (stripes) from small
  // high-detail paths (a flag's Ashoka chakra, an emblem) that must NOT be
  // point-resampled — resampling 48 points destroys their fine geometry.
  let gb; try { gb = wrap.getBBox(); } catch { gb = null; }
  const gw = gb ? gb.width : 0, gh = gb ? gb.height : 0;
  const DETAIL_FRAC = 0.34;   // path smaller than this fraction of the group → ride rigidly
  const els = [...wrap.querySelectorAll('path')];
  for (const el of els) {
    // remember the pristine geometry (survives re-application / re-build)
    let d0 = el.getAttribute('data-ms-d0');
    if (!d0) { d0 = el.getAttribute('d'); el.setAttribute('data-ms-d0', d0); }
    else el.setAttribute('d', d0);
    const len = el.getTotalLength();
    if (!len) continue;
    let pb; try { pb = el.getBBox(); } catch { pb = null; }
    // detail path: small vs. the whole group in BOTH dimensions → keep its exact
    // geometry and just translate it to follow the cloth (crisp chakra/emblem)
    const isDetail = pb && gw && gh &&
      (pb.width < gw * DETAIL_FRAC && pb.height < gh * DETAIL_FRAC);
    if (isDetail) {
      const cx = pb.x + pb.width / 2, cy = pb.y + pb.height / 2;
      paths.push({ el, detail: true, cx, cy });
      if (cx < minX) minX = cx; if (cx > maxX) maxX = cx;
      if (cy < minY) minY = cy; if (cy > maxY) maxY = cy;
      continue;
    }
    const pts = [];
    for (let i = 0; i <= WAVE_SAMPLES; i++) {
      const pt = el.getPointAtLength(len * i / WAVE_SAMPLES);
      pts.push([pt.x, pt.y]);
      if (pt.x < minX) minX = pt.x;
      if (pt.x > maxX) maxX = pt.x;
      if (pt.y < minY) minY = pt.y;
      if (pt.y > maxY) maxY = pt.y;
    }
    paths.push({ el, pts, closed: /z\s*$/i.test(d0) });
  }
  return paths.length ? { paths, minX, maxX, minY, maxY } : null;
}

/* Translate a detail (non-resampled) path so it rides the cloth's displacement
 * at its center. Keeps the path's crisp original geometry intact. */
function detailRide(pd, minX, minY, w, h, field, t, intensity) {
  const u = (pd.cx - minX) / w, v = (pd.cy - minY) / h;
  const s = field(u, v, t);
  const dx = s.dx * w * 0.5 * intensity;
  const dy = s.dy * h * 0.5 * intensity;
  pd.el.setAttribute('transform', `translate(${dx.toFixed(2)} ${dy.toFixed(2)})`);
}

/* One frame of the wave: returns the new `d` string for a sampled path. */
function waveD(pd, minX, width, A, k, phase, turb) {
  let d = '';
  for (let i = 0; i < pd.pts.length; i++) {
    const [x0, y0] = pd.pts[i];
    const ramp = Math.pow((x0 - minX) / width, 1.15);
    const arg = phase - k * (x0 - minX);
    const dy = A * ramp * Math.sin(arg) + turb * ramp * _noise(x0 * 0.11 + phase * 1.3);
    const dx = A * 0.22 * ramp * Math.cos(arg);
    d += (i ? 'L' : 'M') + (x0 + dx).toFixed(2) + ',' + (y0 + dy).toFixed(2);
  }
  return pd.closed ? d + 'Z' : d;
}

class Animator {
  constructor(selectionManager, motionLibrary) {
    this.sel = selectionManager;
    this.motions = motionLibrary;
    this.playing = false;
    this.t0 = performance.now() / 1000;
    this._raf = null;
  }

  play() { this.playing = true; this.t0 = performance.now() / 1000; this._tick(); }
  pause() { this.playing = false; if (this._raf) cancelAnimationFrame(this._raf); this._reset(); }
  toggle() { this.playing ? this.pause() : this.play(); return this.playing; }

  _tick() {
    if (!this.playing) return;
    this._raf = requestAnimationFrame(() => this._tick());
    const t = performance.now() / 1000 - this.t0;
    this._applyAll(t);
    this.sel.syncHighlights();
  }

  _applyAll(t) {
    for (const s of this.sel.selections) {
      if (!s.motionId) continue;
      const motion = this.motions.getById(s.motionId);
      if (!motion) continue;

      const speed = s.speed || 1, intensity = s.intensity || 1;
      const rt = t * speed;
      const seed = s.kind === 'svg'
        ? (s.center[0] * 0.01 + s.center[1] * 0.03)
        : (s.bounds.x * 0.01 + s.bounds.y * 0.03);

      // Keep dense canopy artwork intact; only sway its selectable overlay.
      if (s.kind === 'svg' && s.wrap.querySelector('[data-motion-role="tree-canopy"]')) {
        this._applyTreeLeaves(s, motion, rt, intensity);
        continue;
      }

      // ---- realistic falling leaves: each child leaf falls independently ----
      if (s.kind === 'svg' && motion.params && motion.params.leafFall) {
        this._applyLeafFall(s, motion, rt, intensity);
        continue;
      }

      // ---- hardcoded scenery behaviors, keyed on the object's name ----
      // (demo curation: whatever swatch is dropped on the "birds" / "clouds"
      //  group, it animates the way a viewer expects that object to move.)
      if (s.kind === 'svg' && /\bbirds?\b/i.test(s.name)) {
        this._applyBirds(s, motion, rt, intensity);
        continue;
      }
      if (s.kind === 'svg' && /\bclouds?\b/i.test(s.name)) {
        this._applyClouds(s, motion, rt, intensity);
        continue;
      }
      if (s.kind === 'svg' && /\briver|ripples?\b/i.test(s.name)) {
        this._applyRiver(s, motion, rt, intensity);
        continue;
      }
      if (s.kind === 'svg' && /\bboat|rowboat|canoe|ferry|ship\b/i.test(s.name)) {
        this._applyBoat(s, motion, rt, intensity);
        continue;
      }

      // ---- per-glyph text animation: letters ride the motion individually ----
      if (s.kind === 'svg' && s.wrap.querySelector('text')) {
        if (s._text === undefined) s._text = buildTextData(s.wrap);
        if (s._text) {
          if (s._field === undefined || s._fieldMotion !== motion.id) {
            s._field = buildTrajField(motion) || null;
            s._fieldMotion = motion.id;
          }
          for (const it of s._text.items) {
            it.el.setAttribute('transform',
              glyphTransform(s._text, it, s._field, motion.params, rt, intensity));
          }
          s.wrap.setAttribute('transform', '');
          continue;
        }
      }

      // ---- wave (cloth) mode: deform the geometry itself ----
      if (s.kind === 'svg' && s.waveMode) {
        if (!s._wave) s._wave = buildWaveData(s.wrap);
        if (s._wave) {
          const width = Math.max(1, s._wave.maxX - s._wave.minX);
          // captured motion with trajectories → replay the REAL motion field
          if (s._field === undefined) s._field = buildTrajField(motion) || null;
          if (s._field && s._fieldMotion !== motion.id) {
            s._field = buildTrajField(motion) || null;
            s._fieldMotion = motion.id;
          }
          if (s._field) {
            const height = Math.max(1, s._wave.maxY - s._wave.minY);
            for (const pd of s._wave.paths) {
              if (pd.detail) {
                // fine detail (chakra/emblem): ride the cloth, keep crisp geometry
                detailRide(pd, s._wave.minX, s._wave.minY, width, height, s._field, rt, intensity);
              } else {
                pd.el.setAttribute('d', fieldD(pd, s._wave.minX, s._wave.minY,
                  width, height, s._field, rt, intensity));
              }
            }
          } else {
            // preset → synthetic traveling sine
            const p = motion.params;
            const A = WAVE_AMP_PX * (0.35 + p.amplitude) * intensity;
            const k = 2 * Math.PI * WAVE_CYCLES * (0.5 + p.phaseSpread) / width;
            const phase = 2 * Math.PI * p.frequency * rt;
            const turb = p.turbulence * 4 * intensity;
            const height = Math.max(1, s._wave.maxY - s._wave.minY);
            for (const pd of s._wave.paths) {
              if (pd.detail) {
                // ride the synthetic wave at the detail's x, keeping crisp geometry
                const ramp = Math.pow((pd.cx - s._wave.minX) / width, 1.15);
                const arg = phase - k * (pd.cx - s._wave.minX);
                const dyv = A * ramp * Math.sin(arg) + turb * ramp * _noise(pd.cx * 0.11 + phase * 1.3);
                const dxv = A * 0.22 * ramp * Math.cos(arg);
                pd.el.setAttribute('transform', `translate(${dxv.toFixed(2)} ${dyv.toFixed(2)})`);
              } else {
                pd.el.setAttribute('d', waveD(pd, s._wave.minX, width, A, k, phase, turb));
              }
            }
          }
          s.wrap.setAttribute('transform', '');
          continue;
        }
      }

      const { dx, dy, rot } = computeMotion(motion.params, seed, rt, intensity);

      if (s.kind === 'svg') {
        // rotate around the element's own center for a natural sway
        const [cx, cy] = s.center;
        s.wrap.setAttribute('transform',
          `translate(${dx.toFixed(2)} ${dy.toFixed(2)}) rotate(${rot.toFixed(3)} ${cx.toFixed(1)} ${cy.toFixed(1)})`);
      } else if (s.floatEl) {
        // raster: dx/dy are in viewBox(=displayed px) units; convert to displayed px
        const rect = this.sel.overlay.getBoundingClientRect();
        const pxX = dx * rect.width / this.sel.overlay.width;
        const pxY = dy * rect.height / this.sel.overlay.height;
        s.floatEl.style.transform =
          `translate(${pxX.toFixed(2)}px, ${pxY.toFixed(2)}px) rotate(${rot.toFixed(3)}deg)`;
        s.floatEl.style.transformOrigin = 'center center';
      }
    }
  }

  // deterministic pseudo-random in [0,1) from a leaf index + channel
  _leafRnd(i, k) { const x = Math.sin(i * 12.9898 + k * 78.233) * 43758.5453; return x - Math.floor(x); }

  /*
   * Tree canopy: preserve the detailed silhouettes and sway the overlay around
   * a low pivot. The static source beneath it restores any area the wind opens.
   */
  _applyTreeLeaves(s, motion, t, intensity) {
    const wrap = s.wrap;
    const canopy = wrap.querySelector('[data-motion-role="tree-canopy"]');
    if (!canopy) return;
    if (!s._treeLeaves || s._treeLeaves.el !== canopy) {
      const b = canopy.getBBox();
      s._treeLeaves = {
        el: canopy,
        px: b.x + b.width * 0.78,
        py: b.y + b.height * 0.97,
      };
    }
    const p = motion.params || {};
    const frequency = Number.isFinite(p.frequency) ? p.frequency : 0.4;
    const amplitude = Number.isFinite(p.amplitude) ? p.amplitude : 0.45;
    const phase = 2 * Math.PI * Math.max(0.12, Math.min(0.65, frequency)) * t;
    const primary = Math.sin(phase);
    const harmonic = Math.sin(phase * 2);
    const reach = 2.5 + Math.max(0, Math.min(1, amplitude)) * 4;
    const dx = reach * intensity * (primary + harmonic * 0.16);
    const dy = 0.55 * intensity * harmonic;
    const angle = 0.42 * intensity * (primary + harmonic * 0.1);
    const { px, py } = s._treeLeaves;
    canopy.setAttribute('transform',
      `translate(${dx.toFixed(2)} ${dy.toFixed(2)}) ` +
      `rotate(${angle.toFixed(3)} ${px.toFixed(1)} ${py.toFixed(1)})`);
    wrap.setAttribute('transform', '');
  }

  /*
   * Realistic falling leaves: instead of moving the whole group rigidly, each
   * child leaf falls down a vertical corridor at its own speed, swaying and
   * tumbling, and wraps back to the top (fading in/out to hide the reset).
   * Reads as a continuous stream of leaves drifting to the ground.
   */
  _applyLeafFall(s, motion, t, intensity) {
    const wrap = s.wrap;
    if (!s._leaves || s._leavesMotion !== motion.id) {
      const kids = [...wrap.querySelectorAll('path')];
      const svg = wrap.ownerSVGElement;
      const H = (svg && svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.height) || 1377;
      s._corridor = { topY: H * 0.14, groundY: H * 0.80 };   // spawn line → ground line
      s._leaves = kids.map((el, i) => {
        const b = el.getBBox();
        return {
          el, cx: b.x + b.width / 2, cy: b.y + b.height / 2,
          vy: 60 + this._leafRnd(i, 1) * 95,               // fall speed (units/s)
          swayA: 12 + this._leafRnd(i, 2) * 30,            // horizontal sway amplitude
          swayF: 0.35 + this._leafRnd(i, 3) * 0.7,         // sway frequency (Hz)
          phase: this._leafRnd(i, 4) * Math.PI * 2,
          rot0: this._leafRnd(i, 5) * 360,
          rotV: (this._leafRnd(i, 6) - 0.5) * 170,         // tumble (deg/s)
        };
      });
      s._leavesMotion = motion.id;
    }
    const { topY, groundY } = s._corridor;
    const Hc = Math.max(1, groundY - topY);
    const spd = intensity;
    for (const lf of s._leaves) {
      const start = lf.cy - topY;
      const ph = (((start + lf.vy * t * spd) % Hc) + Hc) % Hc;   // 0..Hc, wraps
      const dy = (topY + ph) - lf.cy;
      const dx = lf.swayA * spd * Math.sin(lf.swayF * 2 * Math.PI * t + lf.phase);
      const ang = lf.rot0 + lf.rotV * t;
      const fin = Math.min(1, ph / (Hc * 0.07));                 // fade in near top
      const fout = Math.min(1, (Hc - ph) / (Hc * 0.14));         // fade out near ground
      lf.el.setAttribute('transform',
        `translate(${dx.toFixed(2)} ${dy.toFixed(2)}) rotate(${ang.toFixed(1)} ${lf.cx.toFixed(1)} ${lf.cy.toFixed(1)})`);
      lf.el.style.opacity = Math.max(0, Math.min(fin, fout)).toFixed(3);
    }
    wrap.setAttribute('transform', '');
  }

  /*
   * Birds: each child path is one bird, with a subtle WING FLAP — a small
   * vertical squash/stretch about the bird's own center (wings sweep up/down)
   * at a natural ~2-3 Hz, each bird on its own phase so the flock isn't in
   * lockstep.
   *
   * Travel: the flock is split into two groups by which half of the sky each
   * bird sits in. LEFT-half birds drift gently LEFT, RIGHT-half birds drift
   * gently RIGHT — so the flock fans outward instead of streaming off one edge.
   * The drift is a small, slow sine and each bird is clamped to its own
   * on-canvas room, so no bird ever leaves the artwork. Wing flap is untouched.
   */
  _applyBirds(s, motion, t, intensity) {
    const wrap = s.wrap;
    if (!s._birds || s._birdsMotion !== motion.id) {
      const kids = [...wrap.querySelectorAll('path')];
      const svg = wrap.ownerSVGElement;
      const vb = (svg && svg.viewBox && svg.viewBox.baseVal) || { width: 1121.71, height: 1121.73 };
      const vbW = vb.width, mid = vbW / 2, MARGIN = 8;
      const DRIFT = 55;   // max desired horizontal drift (viewBox units), room-clamped below
      s._birds = kids.map((el, i) => {
        const b = el.getBBox();
        // group by sky half: left-half → drift left (-1), right-half → right (+1)
        const goRight = (b.x + b.width / 2) >= mid;
        const roomLeft  = Math.max(0, b.x - MARGIN);
        const roomRight = Math.max(0, vbW - (b.x + b.width) - MARGIN);
        // amplitude bounded by the room on the side this bird moves toward
        const amp = Math.min(DRIFT, goRight ? roomRight : roomLeft);
        return {
          el, cx: b.x + b.width / 2, cy: b.y + b.height / 2,
          dir: goRight ? 1 : -1,
          amp: amp * (0.7 + this._leafRnd(i, 4) * 0.3),    // slight per-bird variety
          driftF: 0.05 + this._leafRnd(i, 7) * 0.05,       // slow drift (0.05–0.10 Hz)
          driftPh: this._leafRnd(i, 8) * Math.PI * 2,
          flapF: 2.1 + this._leafRnd(i, 1) * 1.3,          // 2.1–3.4 Hz wingbeat
          phase: this._leafRnd(i, 2) * Math.PI * 2,        // desync the flock
          flapAmp: 0.16 + this._leafRnd(i, 3) * 0.10,      // per-bird flap depth
          bobA: 4 + this._leafRnd(i, 5) * 6,               // gentle vertical waver
          bobF: 0.15 + this._leafRnd(i, 6) * 0.18,
          bobPh: this._leafRnd(i, 0) * Math.PI * 2,
        };
      });
      s._birdsMotion = motion.id;
    }
    const spd = intensity;
    for (const bd of s._birds) {
      // gentle bounded drift outward (left group left, right group right).
      // (1 - cos)/2 ramps 0→1→0 so it eases out and back without a hard turn,
      // and the sign never crosses zero → the bird only ever moves outward.
      const ramp = (1 - Math.cos(2 * Math.PI * bd.driftF * t + bd.driftPh)) / 2;
      const dx = bd.dir * bd.amp * spd * ramp;
      const dy = bd.bobA * spd * Math.sin(2 * Math.PI * bd.bobF * t + bd.bobPh);
      // wing flap: vertical scale oscillates about the bird's center (unchanged)
      const flap = 1 - bd.flapAmp * spd * (0.5 + 0.5 * Math.sin(2 * Math.PI * bd.flapF * t + bd.phase));
      const sy = Math.max(0.6, flap);
      bd.el.setAttribute('transform',
        `translate(${dx.toFixed(2)} ${dy.toFixed(2)}) ` +
        `translate(${bd.cx.toFixed(1)} ${bd.cy.toFixed(1)}) scale(1 ${sy.toFixed(3)}) ` +
        `translate(${(-bd.cx).toFixed(1)} ${(-bd.cy).toFixed(1)})`);
    }
    wrap.setAttribute('transform', '');
  }

  /*
   * Clouds: in the reference clip, clouds drift slowly and STEADILY in one
   * direction (wind) — they don't bob, pulse, or reverse. So each cloud gets a
   * gentle, uniform horizontal glide (all the same wind direction, slightly
   * different speeds), with a very slow, very shallow sine so the loop is
   * seamless without the drift ever reading as back-and-forth. No vertical bob,
   * no scale "breathing" (both looked unnatural). Amplitude is small and each
   * cloud is bounded to its own on-canvas room so none can wander off.
   */
  _applyClouds(s, motion, t, intensity) {
    const wrap = s.wrap;
    const CLOUD_SAMPLES = 64;
    if (!s._clouds || s._cloudsMotion !== motion.id) {
      const kids = [...wrap.querySelectorAll('path')];
      const svg = wrap.ownerSVGElement;
      const vbW = (svg && svg.viewBox && svg.viewBox.baseVal && svg.viewBox.baseVal.width) || 1121.71;
      const MARGIN = 8;
      s._clouds = kids.map((el, i) => {
        // remember pristine geometry so we can restore/rebuild each frame
        let d0 = el.getAttribute('data-ms-d0');
        if (!d0) { d0 = el.getAttribute('d'); el.setAttribute('data-ms-d0', d0); }
        else el.setAttribute('d', d0);
        const b = el.getBBox();
        // sample the cloud outline into points so we can gently billow the edge
        let pts = null;
        const len = el.getTotalLength ? el.getTotalLength() : 0;
        if (len) {
          pts = [];
          for (let k = 0; k <= CLOUD_SAMPLES; k++) {
            const pt = el.getPointAtLength(len * k / CLOUD_SAMPLES);
            pts.push([pt.x, pt.y]);
          }
        }
        const roomLeft  = Math.max(0, b.x - MARGIN);
        const roomRight = Math.max(0, vbW - (b.x + b.width) - MARGIN);
        const amp = Math.min(34, roomLeft, roomRight);
        return {
          el, pts, closed: /z\s*$/i.test(d0),
          cx: b.x + b.width / 2, cy: b.y + b.height / 2,
          w: Math.max(1, b.width), h: Math.max(1, b.height),
          amp: amp * (0.7 + this._leafRnd(i, 5) * 0.3),    // drift amplitude, per-cloud
          driftF: 0.010 + this._leafRnd(i, 1) * 0.010,     // extremely slow drift
          driftPh: this._leafRnd(i, 2) * Math.PI * 2,
          billowPh: this._leafRnd(i, 3) * Math.PI * 2,     // desync the billow
        };
      });
      s._cloudsMotion = motion.id;
    }
    // BILLOW: slow, shallow deformation of the outline so the cloud softly
    // morphs as it drifts (like the reference clip) instead of moving rigidly.
    const BILLOW = 2.4 * intensity;     // max edge displacement (viewBox units) — subtle
    const bf = 0.06;                    // billow frequency (very slow)
    for (const cd of s._clouds) {
      // steady wind drift (unchanged)
      const dx = cd.amp * intensity * Math.sin(2 * Math.PI * cd.driftF * t + cd.driftPh);
      if (cd.pts) {
        const ph = 2 * Math.PI * bf * t + cd.billowPh;
        let d = '';
        for (let k = 0; k < cd.pts.length; k++) {
          const [x0, y0] = cd.pts[k];
          // position-dependent phase so different parts of the outline swell at
          // different times → the silhouette breathes organically, not uniformly
          const u = (x0 - cd.cx) / cd.w, v = (y0 - cd.cy) / cd.h;
          const sx = BILLOW * Math.sin(ph + u * 4.0 + v * 2.3);
          // tops billow up a touch more than the flat base
          const sy = BILLOW * 0.7 * Math.cos(ph * 0.9 + v * 3.1 + u * 1.7);
          d += (k ? 'L' : 'M') + (x0 + sx).toFixed(2) + ',' + (y0 + sy).toFixed(2);
        }
        cd.el.setAttribute('d', cd.closed ? d + 'Z' : d);
      }
      cd.el.setAttribute('transform', `translate(${dx.toFixed(2)} 0)`);
    }
    wrap.setAttribute('transform', '');
  }

  /*
   * River ripples: the Water Ripple preset has no captured trajectory field, so
   * by default it would just rigidly shake the whole ripple group. Instead we
   * DEFORM the ripple geometry with a smooth LAMINAR traveling wave — glassy
   * downstream flow rather than choppy chop:
   *
   *   dy(x,y,t) = A · sin(k·x − 2πf·t + φ(y)) + small second harmonic
   *   dx(...)   = a gentle along-stream shear so ripple crests slide downstream
   *
   * Long wavelength + low amplitude + zero turbulence = laminar. A slow phase
   * offset per scanline (φ(y)) makes the sheet flow, not oscillate in lockstep.
   * The deformation is applied to the SAME sampled-path machinery cloth uses
   * (buildWaveData), but with a flow-tuned displacement instead of a flag whip.
   */
  _applyRiver(s, motion, t, intensity) {
    const wrap = s.wrap;
    if (!s._river) {
      s._river = buildWaveData(wrap);       // sampled pristine geometry per <path>
    }
    if (!s._river) { wrap.setAttribute('transform', ''); return; }
    const rv = s._river;
    const width = Math.max(1, rv.maxX - rv.minX);
    const height = Math.max(1, rv.maxY - rv.minY);
    const p = motion.params || {};
    // laminar tuning: long wavelength (few gentle crests across the width),
    // slow drift, shallow amplitude. Scale mildly by the preset's amplitude so
    // the Intensity slider still has a natural effect, but keep it calm.
    const A = 3.4 * (0.6 + (p.amplitude || 0.2)) * intensity;   // vertical swell (units)
    const k = 2 * Math.PI * 1.15 / width;                       // ~1 crest across the river
    const f = 0.28 * (0.6 + (p.frequency || 1.0) * 0.5);        // slow downstream speed
    const phase = 2 * Math.PI * f * t;
    const flow = 5.0 * intensity;                               // along-stream crest slide
    for (const pd of rv.paths) {
      let d = '';
      for (let i = 0; i < pd.pts.length; i++) {
        const [x0, y0] = pd.pts[i];
        // per-scanline phase offset → the surface flows downstream, not in lockstep
        const yPhase = (y0 - rv.minY) / height * Math.PI * 1.3;
        const arg = k * (x0 - rv.minX) - phase + yPhase;
        // primary swell + a small, slower second harmonic for organic surface
        const dy = A * Math.sin(arg) + A * 0.35 * Math.sin(arg * 0.5 + phase * 0.6);
        // gentle downstream shear so crests glide along the current
        const dx = flow * Math.cos(arg) * 0.5;
        d += (i ? 'L' : 'M') + (x0 + dx).toFixed(2) + ',' + (y0 + dy).toFixed(2);
      }
      pd.el.setAttribute('d', pd.closed ? d + 'Z' : d);
    }
    wrap.setAttribute('transform', '');
  }

  /*
   * Boat: a rigid hull shouldn't ripple like water — but it should FLOAT on the
   * ripples. Replicate the water-ripple rhythm as a gentle rigid BOB (rise/fall)
   * plus a slow ROCK (tilt about the waterline), like the moored boat in the
   * reference night clip. The bob/rock share the river's slow frequency and low
   * amplitude, so the boat reads as riding the same ripples the surface shows.
   * A small phase offset between bob and rock keeps it from looking mechanical.
   */
  _applyBoat(s, motion, t, intensity) {
    const wrap = s.wrap;
    if (!s._boat) {
      const b = wrap.getBBox();
      s._boat = {
        // pivot at the waterline: horizontal center, near the bottom of the hull
        px: b.x + b.width / 2,
        py: b.y + b.height * 0.82,
      };
    }
    const p = motion.params || {};
    // match the river's laminar cadence so boat + water feel coupled
    const f = 0.28 * (0.6 + (p.frequency || 1.0) * 0.5);   // same base as _applyRiver
    const amp = (0.6 + (p.amplitude || 0.2));
    const bob = 7.0 * amp * intensity * Math.sin(2 * Math.PI * f * t);          // vertical rise/fall
    const rock = 1.4 * amp * intensity * Math.sin(2 * Math.PI * f * 0.85 * t + 0.7); // tilt (deg)
    const bp = s._boat;
    wrap.setAttribute('transform',
      `translate(0 ${bob.toFixed(2)}) rotate(${rock.toFixed(3)} ${bp.px.toFixed(1)} ${bp.py.toFixed(1)})`);
  }

  _reset() {
    for (const s of this.sel.selections) {
      if (s._treeLeaves) {
        s._treeLeaves.el.removeAttribute('transform');
        s._treeLeaves = null;
      }
      if (s._leaves) {
        for (const lf of s._leaves) { lf.el.removeAttribute('transform'); lf.el.style.opacity = ''; }
      }
      if (s._birds) { for (const bd of s._birds) { bd.el.removeAttribute('transform'); bd.el.style.opacity = ''; } s._birds = null; s._birdsMotion = null; }
      if (s._clouds) { for (const cd of s._clouds) cd.el.removeAttribute('transform'); s._clouds = null; s._cloudsMotion = null; }
      if (s.kind === 'svg' && s.wrap) {
        s.wrap.setAttribute('transform', '');
        // restore pristine geometry for wave-deformed paths, and clear any
        // per-path transform used to ride detail elements (e.g. flag chakra)
        for (const el of s.wrap.querySelectorAll('path[data-ms-d0]')) {
          el.setAttribute('d', el.getAttribute('data-ms-d0'));
          el.removeAttribute('transform');
        }
        s._wave = null;
        s._river = null;
        s._boat = null;
        s._field = undefined;
        s._fieldMotion = null;
        // reset per-glyph transforms (glyph split itself is kept — harmless)
        if (s._text) {
          for (const it of s._text.items) it.el.removeAttribute('transform');
          s._text = undefined;
        }
      } else if (s.floatEl) s.floatEl.style.transform = '';
    }
    this.sel.syncHighlights();
  }
}

window.Animator = Animator;
window.computeMotion = computeMotion;
