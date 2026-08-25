// scripts/datasets.mjs
// Shared dataset loading + test-query definitions for the evaluation scripts.
// Fitness measure is corpus-relative discrimination (per project decision):
// a query passes when its intended option outranks the OTHER options in the
// same dataset — general embedding quality is explicitly out of scope.

import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

export function loadDataset(file) {
  const src = readFileSync(join(ROOT, file), 'utf8');
  const m = src.match(/export\s+const\s+[A-Za-z_]+\s*=\s*([\s\S]*?);?\s*$/);
  if (!m) throw new Error(`no export const array in ${file}`);
  return new Function(`return ${m[1]};`)();
}

export const policy = loadDataset('data.js');
export const agencies = loadDataset('data-agencies.js');

// Test queries. `targets` are value prefixes — the eval counts a hit when
// the best-ranked item whose value starts with any target is at rank N.
// Alternates list acceptable answers where two options are genuinely both
// correct (e.g. both accessibility docs, both health-record holders).
export const TESTS = [
  // ---- policy documents (data.js) ----
  { ds: 'policy', q: 'hackers stealing data', targets: ['OMB M-22-09'] },
  { ds: 'policy', q: 'usable for people with disabilities', targets: ['Section 508', '28 CFR Part 35'] },
  { ds: 'policy', q: 'risks of artificial intelligence', targets: ['Executive Order 14110', 'NIST AI Risk Management'] },
  { ds: 'policy', q: 'clear simple language', targets: ['Plain Writing Act'] },
  { ds: 'policy', q: 'publishing datasets as APIs', targets: ['Digital Government Strategy'] },
  { ds: 'policy', q: 'getting rid of paper records', targets: ['OMB M-19-21'] },
  { ds: 'policy', q: 'login and password security', targets: ['NIST SP 800-63-4'] },
  { ds: 'policy', q: 'better customer service online', targets: ['OMB Circular A-11', '21st Century'] },
  { ds: 'policy', q: 'what is cloud computing', targets: ['NIST SP 800-145'] },
  { ds: 'policy', q: 'protecting personally identifiable information', targets: ['Privacy Impact Assessment'] },

  // ---- agencies (data-agencies.js) ----
  { ds: 'agencies', q: 'I need my medical records from the VA', targets: ['dva'] },
  { ds: 'agencies', q: 'who has my medical records', targets: ['dva', 'hhs'] },
  { ds: 'agencies', q: 'airport security procedures', targets: ['tsa'] },
  { ds: 'agencies', q: 'workplace safety violations', targets: ['dol'] },
  { ds: 'agencies', q: 'border crossing logs', targets: ['cbp', 'dhs'] },
  { ds: 'agencies', q: 'drug seizure records', targets: ['doj'] },
  { ds: 'agencies', q: 'weather forecasts', targets: ['nws'] },
  { ds: 'agencies', q: 'wildfire management', targets: ['fs'] },
  { ds: 'agencies', q: 'drug approval process', targets: ['fdi'] },
  { ds: 'agencies', q: 'space exploration missions', targets: ['nasa'] },
  { ds: 'agencies', q: 'my tax return audit', targets: ['irs'] },
  { ds: 'agencies', q: 'national park camping', targets: ['nps'] },
  { ds: 'agencies', q: 'veterans disability benefits', targets: ['dva'] },
  { ds: 'agencies', q: 'car crash safety recalls', targets: ['nhtsa'] },
  { ds: 'agencies', q: 'retirement checks', targets: ['ssa'] },
  { ds: 'agencies', q: 'polluted drinking water', targets: ['epa'] },
  { ds: 'agencies', q: 'passport and visa applications', targets: ['dos'] },
  { ds: 'agencies', q: 'prison inmate records', targets: ['fboc'] },
];

export const DATASETS = { policy, agencies };
