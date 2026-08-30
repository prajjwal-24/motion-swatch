/*
 * Put an extracted swatch into the running app's library from the test side.
 *
 * The app deliberately ships NO pre-seeded swatch: "Extracted Motion" means motion
 * extracted in this session from a video the user supplied, so boot-loading a JSON file
 * into that panel would present a shipped asset as if it had just been measured. These
 * measurements still need the swatch though, so the harness installs it explicitly —
 * which also keeps it obvious in the test output that it came from a file.
 */
module.exports.installSwatch = async function installSwatch(page, url = 'assets/motion/flutter-flag-autumn.json') {
  return page.evaluate(async (url) => {
    const r = await fetch(url);
    if (!r.ok) return { ok: false, why: `HTTP ${r.status} for ${url}` };
    const m = await r.json();
    if (!m || !m.id) return { ok: false, why: `${url} has no id` };
    if (!window.__ms.library.getById(m.id)) window.__ms.library.add(m);
    return { ok: true, id: m.id };
  }, url);
};
