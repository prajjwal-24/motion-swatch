/* After removing the hand-tuned flag flutter: does the library still boot clean, and is
   the extracted swatch the only flag flutter on offer? */
const puppeteer = require('puppeteer-core');
(async () => {
  const b = await puppeteer.launch({ executablePath: '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome', args: ['--no-sandbox'] });
  const p = await b.newPage();
  const errs = [];
  p.on('pageerror', e => errs.push(String(e)));
  p.on('console', m => { if (m.type() === 'error') errs.push('console: ' + m.text()); });
  await p.goto('http://localhost:8000/index.html', { waitUntil: 'networkidle0' });
  await new Promise(r => setTimeout(r, 600));
  const out = await p.evaluate(() => {
    const all = window.__ms.library.getAll();
    const chips = (id) => Array.from(document.querySelectorAll(`#${id} .chip, #${id} > *`)).length;
    return {
      byId: !!window.__ms.library.getById('flag-flutter'),
      presets: all.filter(m => !m.fromUpload).map(m => m.id),
      extracted: all.filter(m => m.fromUpload).map(m => m.id),
      presetChips: chips('motion-list'),
      extractedChips: chips('extracted-list'),
      extractedVisible: !document.getElementById('extracted-section').hidden,
    };
  });
  await b.close();
  console.log(`flag-flutter still in library: ${out.byId}`);
  console.log(`presets (${out.presets.length} ids, ${out.presetChips} chips): ${out.presets.join(', ')}`);
  console.log(`extracted (${out.extracted.length} ids, ${out.extractedChips} chips, section visible=${out.extractedVisible}): ${out.extracted.join(', ')}`);
  console.log(errs.length ? `PAGE ERRORS:\n  ${errs.join('\n  ')}` : 'no page errors');
})();
