/*
 * main.js — app controller.
 */

(() => {
'use strict';

const $ = id => document.getElementById(id);
const statusEl = $('status');
function status(msg, flash) {
  statusEl.textContent = msg;
  statusEl.classList.toggle('flash', !!flash);
  if (flash) setTimeout(() => statusEl.classList.remove('flash'), 2200);
}

const artContainer = $('artwork-container');
const overlay = $('selection-overlay');

// ---- overlay always matches the 800x500 viewBox basis ----
function syncOverlay() {
  const rect = artContainer.getBoundingClientRect();
  overlay.width = 800; overlay.height = 500;
  overlay.style.width = rect.width + 'px';
  overlay.style.height = rect.height + 'px';
}

// =========================================================================
//  Motion library
// =========================================================================
const library = new MotionLibrary();
const motionListEl = $('motion-list');

// ===========================================================================
//  Motion tiles: "Particle Swatch" — a Pantone-style chip of moving dots.
//  Presets: a 5x5 dot lattice driven by the REAL motion formula.
//  Captured motions: the dots replay 25 of the 144 real trajectories,
//  so the chip literally shows the recorded motion's spatial structure.
// ===========================================================================
const CHIP = 72, GRID_N = 5;
const chipTiles = [];   // [{canvas, ctx, motion, dots, tracks}]

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

function buildChipState(m) {
  if (m.trajectories && m.trajectories.length >= 25) {
    // subsample the 12x12 grid to 5x5, store drift-removed relative tracks
    const G = Math.round(Math.sqrt(m.trajectories.length));   // 12
    const idx = [];
    for (let gy = 0; gy < GRID_N; gy++)
      for (let gx = 0; gx < GRID_N; gx++)
        idx.push(Math.min(G - 1, Math.round(gy * (G - 1) / (GRID_N - 1))) * G
               + Math.min(G - 1, Math.round(gx * (G - 1) / (GRID_N - 1))));
    const tracks = idx.map(i => {
      const tr = m.trajectories[i];
      const x0 = tr[0][0], y0 = tr[0][1];
      return tr.map(p => [p[0] - x0, p[1] - y0]);
    });
    return { tracks };
  }
  return { tracks: null };
}

function renderMotionList() {
  chipTiles.length = 0;
  motionListEl.innerHTML = '';
  for (const m of library.getAll()) {
    const tile = document.createElement('div');
    tile.className = 'motion-chip' + (library.selectedId === m.id ? ' active' : '');
    const canvas = document.createElement('canvas');
    canvas.width = CHIP * 2; canvas.height = CHIP * 2;   // retina
    const label = document.createElement('div');
    label.className = 'chip-name';
    label.textContent = m.name;
    tile.appendChild(canvas);
    tile.appendChild(label);
    tile.title = m.desc || m.name;
    tile.onclick = () => {
      library.select(m.id);
      renderMotionList();
      const applied = applyMotionToActive();
      status(applied ? `Applied "${m.name}" to "${applied}".`
                     : `Motion "${m.name}" selected — now click an object to apply it.`, true);
    };
    motionListEl.appendChild(tile);
    chipTiles.push({ canvas, ctx: canvas.getContext('2d'), motion: m, ...buildChipState(m) });
  }
}

function chipLoop() {
  requestAnimationFrame(chipLoop);
  const t = performance.now() / 1000;
  for (const tile of chipTiles) {
    const { canvas, ctx, motion, tracks } = tile;
    if (!canvas.isConnected) continue;
    const S = canvas.width;
    const [r, g, b] = hexToRgb(motion.color || '#7c6cff');

    // motion-blur fade instead of clear → dots drag trails
    ctx.fillStyle = 'rgba(26,27,42,0.28)';
    ctx.fillRect(0, 0, S, S);

    const cell = S / (GRID_N + 1);
    ctx.fillStyle = `rgba(${r},${g},${b},0.16)`;
    for (let gy = 0; gy < GRID_N; gy++)          // faint rest lattice
      for (let gx = 0; gx < GRID_N; gx++)
        ctx.fillRect((gx + 1) * cell - 1, (gy + 1) * cell - 1, 2, 2);

    ctx.fillStyle = `rgb(${r},${g},${b})`;
    for (let gy = 0; gy < GRID_N; gy++) {
      for (let gx = 0; gx < GRID_N; gx++) {
        const hx = (gx + 1) * cell, hy = (gy + 1) * cell;
        let px, py;
        if (tracks) {
          // captured: replay real trajectory (ping-pong loop, drift stays visible)
          const tr = tracks[gy * GRID_N + gx];
          const n = tr.length;
          const f = (t * 12) % (2 * n);
          const i = f < n ? Math.floor(f) : (2 * n - 1 - Math.floor(f));
          px = hx + tr[i][0] * S * 1.1;
          py = hy + tr[i][1] * S * 1.1;
        } else {
          // preset: the actual animator formula, per-dot spatial seed
          const seed = (gx * 17 + gy * 31) * 0.02;
          const d = computeMotion(motion.params, seed, t, 1);
          px = hx + d.dx * 0.75;
          py = hy + d.dy * 0.75;
        }
        ctx.beginPath();
        ctx.arc(px, py, 2.6, 0, 7);
        ctx.fill();
      }
    }
  }
}
chipLoop();

// =========================================================================
//  Selection manager
// =========================================================================
const sel = new SelectionManager(overlay, artContainer);

sel.onCreated = (s, idx) => { renderChips(); showInspector(s); applyMotionToActive(); refreshHighlightsSoon(); };
sel.onSelected = (s, idx) => { renderChips(); showInspector(s); };

function applyMotionToActive() {
  const s = sel.getActive();
  const m = library.getSelected();
  if (s && m) {
    s.motionId = m.id;
    // captured motions carry a real trajectory field — geometry deformation
    // (field replay) looks far more real than rigid transforms, so default
    // wave mode ON for captures applied to SVG objects
    if (s.kind === 'svg' && m.trajectories && m.trajectories.length) {
      s.waveMode = true;
    }
    // switching motions invalidates deformation caches
    if (s.wrap) {
      for (const el of s.wrap.querySelectorAll('path[data-ms-d0]')) el.setAttribute('d', el.getAttribute('data-ms-d0'));
    }
    s._wave = null; s._field = undefined; s._fieldMotion = null; s._text = undefined;
    showInspector(s);
    if (sel.mode === 'svg') sel._renderSVGHighlights(); else sel.redraw();
    return s.name;
  }
  return null;
}

// =========================================================================
//  Artwork loading + scene tabs (Poster / Scenery)
// =========================================================================
let currentScene = 'poster';

// file-based scenes: fetched from disk and loaded through the same SVG path as
// an uploaded artwork (so the .layer[data-name] contract makes objects selectable)
const FILE_SCENES = {
  train: 'assets/scenes/train-window-adobe.svg',
};

async function loadScene(name) {
  currentScene = name;
  animator.pause();
  $('btn-play').textContent = '▶ Play';
  $('btn-play').classList.remove('playing');

  for (const b of document.querySelectorAll('.scene-tab'))
    b.classList.toggle('active', b.dataset.scene === name);

  if (FILE_SCENES[name]) {
    status('Loading scene…');
    try {
      const text = await fetch(FILE_SCENES[name]).then(r => r.text());
      loadUploadedSVG(text);
    } catch (e) {
      status('Could not load that scene.');
    }
    return;
  }

  artContainer.innerHTML = '';
  const svg = name === 'poster' ? createPosterSVG() : createScenerySVG();
  artContainer.appendChild(svg);
  syncOverlay();
  sel.attachSVG(svg);
  setModeUI('svg');
  renderChips(); hideInspector();

  status(name === 'poster' ? 'Poster loaded — click the flag or the title, then pick a motion.'
       : 'Scenery loaded — click any object, then pick a motion.');
}

for (const b of document.querySelectorAll('.scene-tab'))
  b.onclick = () => loadScene(b.dataset.scene);

function loadDefaultScenery() { loadScene('scenery'); }

function loadUploadedSVG(text) {
  artContainer.innerHTML = text;
  const svg = artContainer.querySelector('svg');
  if (!svg) { status('That SVG could not be parsed.'); return; }
  svg.style.width = '100%'; svg.style.height = '100%';
  if (!svg.getAttribute('viewBox')) {
    const w = svg.getAttribute('width') || 800, h = svg.getAttribute('height') || 500;
    svg.setAttribute('viewBox', `0 0 ${parseFloat(w)} ${parseFloat(h)}`);
  }
  syncOverlay();
  sel.attachSVG(svg);
  setModeUI('svg');
  renderChips(); hideInspector();
  const n = svg.querySelectorAll('.ms-wrap').length;
  status(`SVG loaded — ${n} selectable element(s). Click one to select, then pick a motion.`, true);
}

function loadRasterImage(dataUrl) {
  artContainer.innerHTML = `<img src="${dataUrl}" draggable="false">`;
  syncOverlay();
  sel.attachRaster();
  setModeUI('raster');
  renderChips(); hideInspector();
  status('Image loaded. Draw a rectangle around an object, name it, then pick a motion.', true);
}

function setModeUI(mode) {
  $('tool-controls').style.display = mode === 'raster' ? 'flex' : 'none';
  $('mode-hint').textContent = mode === 'svg'
    ? 'SVG mode — click objects to select'
    : 'Image mode — draw rectangles to select';
}

// =========================================================================
//  Region chips
// =========================================================================
function renderChips() {
  const el = $('region-chips');
  el.innerHTML = '';
  sel.selections.forEach((s, i) => {
    const chip = document.createElement('span');
    chip.className = 'region-chip' + (i === sel.activeIdx ? ' active' : '');
    chip.innerHTML = `<span class="dot" style="background:${s.color}"></span>${s.name}${s.motionId ? ' ✓' : ''}`;
    chip.onclick = () => sel.selectByIndex(i);
    el.appendChild(chip);
  });
}

// =========================================================================
//  Inspector
// =========================================================================
function showInspector(s) {
  $('inspector-section').hidden = false;
  $('inspector-content').hidden = false;
  $('insp-name').value = s.name;
  $('insp-speed').value = s.speed; $('insp-speed-val').textContent = s.speed.toFixed(1) + 'x';
  $('insp-intensity').value = s.intensity; $('insp-intensity-val').textContent = Math.round(s.intensity * 100) + '%';
  const badge = $('insp-motion-name');
  if (s.motionId) {
    const m = library.getById(s.motionId);
    badge.textContent = m ? m.name : 'Unknown';
    badge.classList.add('assigned');
  } else { badge.textContent = 'None — select a motion'; badge.classList.remove('assigned'); }
}
function hideInspector() { $('inspector-section').hidden = true; $('inspector-content').hidden = true; }

$('insp-name').addEventListener('change', () => { const s = sel.getActive(); if (s) { s.name = $('insp-name').value; renderChips(); if (sel.mode === 'svg') sel._renderSVGHighlights(); else sel.redraw(); } });
$('insp-speed').addEventListener('input', () => { const s = sel.getActive(); if (s) { s.speed = parseFloat($('insp-speed').value); $('insp-speed-val').textContent = s.speed.toFixed(1) + 'x'; } });
$('insp-intensity').addEventListener('input', () => { const s = sel.getActive(); if (s) { s.intensity = parseFloat($('insp-intensity').value); $('insp-intensity-val').textContent = Math.round(s.intensity * 100) + '%'; } });
$('btn-remove-motion').onclick = () => { const s = sel.getActive(); if (s) { s.motionId = null; if (s.wrap) s.wrap.setAttribute('transform',''); if (s.floatEl) s.floatEl.style.transform=''; showInspector(s); renderChips(); if (sel.mode==='svg') sel._renderSVGHighlights(); else sel.redraw(); status('Motion removed.'); } };
$('btn-delete-region').onclick = () => { sel.deleteActive(); renderChips(); const a = sel.getActive(); if (a) showInspector(a); else hideInspector(); status('Region deleted.'); };

// =========================================================================
//  Tools
// =========================================================================
$('btn-tool-rect').onclick = () => { sel.setTool('rect'); $('btn-tool-rect').classList.add('active'); };

// =========================================================================
//  Animator + play
// =========================================================================
const animator = new Animator(sel, library);
$('btn-play').onclick = () => {
  if (!sel.selections.some(s => s.motionId)) {
    status('Assign a motion to at least one object first.');
    return;
  }
  const playing = animator.toggle();
  sel.setHighlightsHidden(playing);   // outlines only in pause state
  $('btn-play').textContent = playing ? '⏸ Pause' : '▶ Play';
  $('btn-play').classList.toggle('playing', playing);
  status(playing ? 'Playing.' : 'Paused.');
};

// =========================================================================
//  Video export (for Reels)
// =========================================================================
$('btn-export-video').onclick = async () => {
  if (!sel.selections.some(s => s.motionId)) {
    status('Nothing is animated yet.'); return;
  }
  const btn = $('btn-export-video');
  btn.disabled = true;
  const wasPlaying = animator.playing;
  if (!wasPlaying) animator.play();
  try {
    const blob = await exportVideo(sel, animator, {
      seconds: 8, mode: 'flat',
      onProgress: p => { btn.textContent = p < 1 ? `Recording… ${Math.round(p * 100)}%` : 'Export video'; },
    });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'motion-swatch.' + (blob.type.includes('mp4') ? 'mp4' : 'webm');
    a.click(); URL.revokeObjectURL(a.href);
    status('Video exported — 1600×1000, matches the artwork size.', true);
  } catch (err) {
    status('Video export failed: ' + err.message);
  }
  if (!wasPlaying) animator.pause();
  btn.textContent = 'Export video';
  btn.disabled = false;
};

function refreshHighlightsSoon() { if (sel.mode === 'svg') requestAnimationFrame(() => sel._renderSVGHighlights()); }

// =========================================================================
//  Upload motion video
// =========================================================================
const capture = new MotionCapture();
$('btn-upload-motion').onclick = () => $('motion-input').click();
$('motion-input').onchange = async (e) => {
  const file = e.target.files[0]; if (!file) return;
  $('upload-status').textContent = 'Analyzing motion…';
  capture.onProgress = (p, msg) => {
    $('upload-status').textContent = msg || `Analyzing… ${Math.round(p * 100)}%`;
  };
  try {
    const motion = await capture.captureFromFile(file);
    if (motion) {
      // MULTI-MOTION BRANCH: if the service segmented ≥2 distinct motions,
      // show the picker so the user names and chooses which to save. Each
      // chosen region becomes its own Motion in the library (the whole-frame
      // `motion` variable is discarded — its trajectories/params are the
      // blended average, not what the user wants).
      if (motion.regions && motion.regions.length >= 2 && window.showMultiPick) {
        $('upload-status').textContent =
          `${motion.regions.length} motions detected — pick and name them.`;
        const picked = await showMultiPick(motion.videoUrl, motion.regions, {
          engine: motion.engine,
          framesAnalyzed: motion.framesAnalyzed,
          fps: motion.trajFps,
        });
        if (!picked.length) {
          // user cancelled or unchecked everything — release the shared
          // object URL and bail without adding anything
          URL.revokeObjectURL(motion.videoUrl);
          $('upload-status').textContent = 'No motions saved.';
        } else {
          for (const m of picked) library.add(m);
          renderMotionList();
          const names = picked.map(m => `"${m.name}"`).join(', ');
          $('upload-status').textContent =
            `Added ${picked.length} motion${picked.length > 1 ? 's' : ''}.`;
          status(`Added ${picked.length} motion${picked.length > 1 ? 's' : ''} — ${names}. Click one, then apply to an object.`, true);
        }
      } else {
        // SINGLE-MOTION PATH (unchanged): the "wow" extraction moment only
        // plays when there's really just one motion to celebrate.
        if (motion.trajectories && motion.videoUrl && window.showExtraction) {
          await showExtraction(motion.videoUrl, motion.trajectories, motion.params, motion.color);
        }
        library.add(motion); renderMotionList();
        $('upload-status').textContent = `Added "${motion.name}"`;
        status(`Motion "${motion.name}" captured from video — click it, then apply to an object.`, true);
      }
    } else {
      $('upload-status').textContent = 'Could not extract motion (need more movement / a longer clip).';
    }
  } catch (err) {
    $('upload-status').textContent = 'Video error: ' + err.message;
  }
  e.target.value = '';
};

// =========================================================================
//  Export animated SVG (self-contained: motion baked into CSS keyframes)
// =========================================================================
$('btn-export-svg').onclick = () => {
  if (sel.mode !== 'svg') { status('Export works with SVG artwork (the raster path has no vector scene to bake).'); return; }
  if (!sel.selections.some(s => s.motionId)) { status('Assign a motion to at least one object first.'); return; }
  const svgText = buildExportSVG(sel, library);
  if (!svgText) { status('Nothing to export.'); return; }
  const blob = new Blob([svgText], { type: 'image/svg+xml' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'motion-swatch-poster.svg';
  a.click(); URL.revokeObjectURL(a.href);
  status('Exported! Drop the .svg into any website — <img src="motion-swatch-poster.svg"> — it animates by itself.', true);
};

// =========================================================================
//  Upload artwork
// =========================================================================
$('btn-upload-art').onclick = () => $('art-input').click();
$('art-input').onchange = (e) => {
  const file = e.target.files[0]; if (!file) return;
  const reader = new FileReader();
  const isSvg = file.type === 'image/svg+xml' || file.name.toLowerCase().endsWith('.svg');
  reader.onload = () => { isSvg ? loadUploadedSVG(reader.result) : loadRasterImage(reader.result); };
  isSvg ? reader.readAsText(file) : reader.readAsDataURL(file);
  e.target.value = '';
};

// =========================================================================
//  Boot
// =========================================================================
window.addEventListener('resize', syncOverlay);
renderMotionList();
loadScene('poster');

// expose for automated testing
window.__ms = { sel, library, animator, loadScene, loadUploadedSVG, loadRasterImage };

})();
