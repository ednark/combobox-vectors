// Headless smoke test of the demo wiring: the component, the static semantic
// table, and setDenseSource — under jsdom (installed with --no-save, not a
// declared dependency). Run: node /tmp/smoke.mjs  (or keep as a repo test)
let JSDOM;
try {
  ({ JSDOM } = await import('jsdom'));
} catch {
  console.error('jsdom not installed — run: npm install --no-save jsdom');
  process.exit(2);
}
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const at = (f) => join(ROOT, f);

const dom = new JSDOM(`<!doctype html><html><body>
  <label class="usa-label" for="agency">Find an agency</label>
  <div class="usa-combo-box" id="agency-box">
    <select id="agency" name="agency" class="usa-select"></select>
  </div>
</body></html>`);
globalThis.window = dom.window;
globalThis.document = dom.window.document;
// The component constructs events with the ambient constructors; under jsdom those
// must be the window's, or dispatchEvent rejects them (a harness artifact, not a bug).
globalThis.CustomEvent = dom.window.CustomEvent;
globalThis.Event = dom.window.Event;
globalThis.MouseEvent = dom.window.MouseEvent;

const { createVectorComboBox } = await import(at('vector-combobox.js'));
const { AGENCY_DATA } = await import(at('data-agencies.js'));
let TABLE;
try {
  TABLE = JSON.parse(readFileSync(at('data-agencies.semantic.json'), 'utf8'));
} catch (err) {
  console.error(`cannot load data-agencies.semantic.json (${err.message}) — run npm run build:tables`);
  process.exit(2);
}

const cb = createVectorComboBox(document.getElementById('agency-box'), {
  data: AGENCY_DATA,
  table: TABLE,
  placeholder: 'Describe your FOIA request…',
  threshold: 0.25,
  maxResults: 8,
});

const list = () => [...document.querySelectorAll('.usa-combo-box__list-option')]
  .map((li) => li.textContent.trim())
  .filter((t) => t && !t.includes('Vector search off') && !t.includes('Loading semantic') && !t.includes('unavailable'));

async function ask(q) {
  cb.input.value = q;
  cb.input.dispatchEvent(new dom.window.Event('input', { bubbles: true }));
  await new Promise((r) => setTimeout(r, 400));
  return list();
}

let failures = 0;
function expect(label, cond, detail) {
  if (!cond) { failures++; console.log(`  FAIL  ${label}${detail ? ' — ' + detail : ''}`); }
  else console.log(`  ok    ${label}`);
}

// 1. Table mode is ready on the first keystroke: no download, no pending state.
const first = await ask('who has my medical records');
expect('first keystroke returns results', first.length > 0, JSON.stringify(first.slice(0, 3)));
expect('VA is the top answer for medical records', /Veterans Affairs/.test(first[0] || ''), first[0]);
expect('no download-notice row in table mode', !first.some((t) => /Loading|unavailable/i.test(t)));

const storm = await ask('sandbags before the hurricane');
expect('FEMA tops a pure-paraphrase query', /Homeland Security|Federal Emergency/.test(storm[0] || ''), storm[0]);

// 2. Keyword-only facts still work: "October" has zero table coverage, so the
//    sparse channel must carry it.
const oct = await ask('tax return audit');
expect('keyword query still resolves', /Internal Revenue/.test(oct[0] || ''), oct[0]);

// 3. setDenseSource('off') -> keyword only, no exceptions, results still render.
// Baseline BM25 ranks TSA second for this query (see npm run eval:no-model: tier 1
// "airport security procedures" -> rank 2), so the check is membership, not first.
cb.setDenseSource('off');
const off = await ask('airport security procedures');
expect('off mode renders keyword results', off.length > 0, `first=${off[0]}`);
expect('off mode still surfaces TSA in the top 3', off.slice(0, 3).some((t) => /Transportation Security/.test(t)), off.slice(0, 3).join(' | '));

// 4. setDenseSource('table') restores semantic ranking.
cb.setDenseSource('table');
const back = await ask('who has my medical records');
expect('table mode restores after toggling off', /Veterans Affairs/.test(back[0] || ''), back[0]);

// 5. The status/notice row must not appear in table mode (nothing to wait for).
const notices = [...document.querySelectorAll('.usa-combo-box__list-option[aria-disabled="true"]')].map((n) => n.textContent);
expect('no stale progress notices', notices.length === 0, JSON.stringify(notices));

// 6. Selection still writes the underlying <select> (form submission path).
cb.input.value = 'nasa';
cb.input.dispatchEvent(new dom.window.Event('input', { bubbles: true }));
await new Promise((r) => setTimeout(r, 300));
const opts = [...document.querySelectorAll('.usa-combo-box__list-option')];
const target = opts.find((o) => /Aeronautics|NASA/i.test(o.textContent));
if (target) {
  target.dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true }));
  expect('picking an option fills the hidden select', cb.select.value !== '', cb.select.value);
} else {
  expect('picking an option fills the hidden select', false, 'no NASA row rendered');
}

console.log(failures ? `\n${failures} FAILURE(S)` : '\nall smoke checks passed');
process.exit(failures ? 1 : 0);
