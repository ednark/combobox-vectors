// scripts/datasets.mjs
// Shared dataset loading + test-query definitions for the evaluation scripts.
// Fitness measure is corpus-relative discrimination (per project decision):
// a query passes when its intended option outranks the OTHER options in the
// same dataset — general embedding quality is explicitly out of scope.
//
// The datasets are plain ES modules of data, so they are imported directly. This
// also means the eval scripts, the build scripts and the browser all read the
// same source array — no regex-parsed copy that can drift from what ships.
import { DATA } from '../data.js';
import { AGENCY_DATA } from '../data-agencies.js';

export const policy = DATA;
export const agencies = AGENCY_DATA;

// Test queries. `targets` are value prefixes — the eval counts a hit when
// the best-ranked item whose value starts with any target is at rank N.
// Alternates list acceptable answers where two options are genuinely both
// correct (e.g. both accessibility docs, both health-record holders).
//
// TIER 1 (TESTS): queries whose words are mostly ALREADY in the corpus
// (vectorText). They measure ranking quality GIVEN shared vocabulary — the
// lexical channel sees nearly the same tokens the embedder does, so this tier
// systematically UNDER-measures what the dense channel is for. See
// PARAPHRASE_TESTS below for the tier that does not.
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

// TIER 2 (PARAPHRASE_TESTS): how someone types when they do not know the
// vocabulary. Each query is deliberately phrased in words that are NOT in the
// dataset's vectorText — corpus-word coverage is roughly 20-50%, so BM25 is
// blind to most of the query. This is the tier that separates a semantic channel
// from a keyword channel, and it is the tier this component has never scored on.
//
// Authored by hand against the 42-agency and 15-document demo sets. Targets are
// best judgment of the right answer: review them before treating tier 2 as a
// gate rather than a measurement.
export const PARAPHRASE_TESTS = [
  // ---- agencies: everyday phrasing vs opaque agency labels ----
  { ds: 'agencies', q: 'sandbags before the hurricane', targets: ['fema'] },
  { ds: 'agencies', q: "grandma's monthly pension stub", targets: ['ssa'] },
  { ds: 'agencies', q: 'chemical smell in the creek behind my house', targets: ['epa'] },
  { ds: 'agencies', q: "grandfather's walker through the scanner", targets: ['tsa'] },
  { ds: 'agencies', q: 'tick bites at the campground', targets: ['nps'] },
  { ds: 'agencies', q: 'tinnitus claim rating', targets: ['dva'] },
  { ds: 'agencies', q: 'gelatin in my capsules', targets: ['fdi'] },
  { ds: 'agencies', q: 'drone over my neighborhood at night', targets: ['faa'] },
  { ds: 'agencies', q: 'insulating the attic for winter', targets: ['doe'] },
  { ds: 'agencies', q: 'hard hat on a scaffold', targets: ['dol'] },
  { ds: 'agencies', q: 'earthquake tremor map', targets: ['usgs'] },
  { ds: 'agencies', q: 'measles outbreak at the elementary school', targets: ['cdc'] },
  { ds: 'agencies', q: 'ATV trail permit on public land', targets: ['blm'] },
  { ds: 'agencies', q: 'refund got intercepted', targets: ['irs'] },

  // ---- policy documents: intent without the citation ----
  { ds: 'policy', q: 'fancy words on a benefits letter', targets: ['Plain Writing'] },
  { ds: 'policy', q: 'robot deciding who gets the loan', targets: ['Executive Order 14110', 'NIST AI Risk'] },
  { ds: 'policy', q: 'turning the water off on a breached server', targets: ['OMB M-22-09'] },
  { ds: 'policy', q: 'scrapbook of every dollar we spent', targets: ['OMB Circular A-11', '21st Century'] },
];

export const TIERS = [
  { name: 'tier 1: in-vocabulary queries', tests: TESTS },
  { name: 'tier 2: paraphrase queries (out-of-corpus vocabulary)', tests: PARAPHRASE_TESTS },
];
