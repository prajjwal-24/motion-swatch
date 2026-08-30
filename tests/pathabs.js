/*
 * Absolute coordinates of every POINT in a path `d`, in command order.
 *
 * Why this exists: Scene3's #Scarf is drawn entirely with RELATIVE commands (l, c, ...),
 * so the raw numbers in `d` are deltas. Any metric that reads consecutive number pairs
 * as positions measures nothing on that artwork. This walks the path the way the
 * renderer does and returns real positions.
 *
 * Because warpPathD preserves command structure exactly (same commands, same operand
 * counts, only the coordinates moved), the k-th point returned for the pristine `d0` is
 * the SAME material point as the k-th for the warped `d`. That exact correspondence is
 * what makes a local-distortion comparison valid.
 *
 * Arc radii/rotation/flags are operands, not positions, so they are skipped — only the
 * arc's endpoint is a point.
 */
const ARGC = { M: 2, L: 2, H: 1, V: 1, C: 6, S: 4, Q: 4, T: 2, A: 7, Z: 0 };

function pathPoints(d) {
  const toks = d.match(/[MmLlHhVvCcSsQqTtAaZz]|-?\d*\.?\d+(?:e[-+]?\d+)?/gi) || [];
  const pts = [];
  let i = 0, cmd = null, cx = 0, cy = 0, sx = 0, sy = 0;
  const num = () => parseFloat(toks[i++]);
  while (i < toks.length) {
    if (/[a-z]/i.test(toks[i])) { cmd = toks[i++]; }
    else if (cmd === 'M') cmd = 'L';                 // implicit repeat of M is L
    else if (cmd === 'm') cmd = 'l';
    if (cmd == null) break;
    const up = cmd.toUpperCase(), rel = cmd !== up, n = ARGC[up];
    if (n === undefined) break;
    if (up === 'Z') { cx = sx; cy = sy; continue; }
    if (i + n > toks.length) break;
    const ox = rel ? cx : 0, oy = rel ? cy : 0;
    if (up === 'H') { cx = ox + num(); pts.push(cx, cy); }
    else if (up === 'V') { cy = oy + num(); pts.push(cx, cy); }
    else if (up === 'A') {
      num(); num(); num(); num(); num();              // rx ry rot large-arc sweep
      cx = ox + num(); cy = oy + num(); pts.push(cx, cy);
    } else {
      // C/S/Q/T/M/L: every operand pair is a point (controls included)
      for (let k = 0; k < n; k += 2) {
        const x = ox + num(), y = oy + num();
        pts.push(x, y);
        if (k + 2 === n) { cx = x; cy = y; }          // last pair is the new current point
      }
      if (up === 'M') { sx = cx; sy = cy; }
    }
  }
  return pts;
}

module.exports = { pathPoints };
