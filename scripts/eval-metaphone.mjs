// scripts/eval-metaphone.mjs
// Corpus-relative eval of the BM25 sparse channel with and without phonetic
// (metaphone) token keys — the fuzzy-matching technique from govdoc-explainer's
// original hybrid search (Porter stem -> Metaphone -> set overlap), adapted
// here as an alternate BM25 tokenizer.
//
// Question answered: when the user typos a query, does the intended option
// still outrank the others? And does adding phonetic keys regress clean
// (correctly spelled) queries?
//
// Tokenizer variants:
//   baseline  — current production tokenizer (lowercase, alnum, naive stem)
//   union     — baseline tokens PLUS metaphone(stem(token)) keys, so exact
//               matches keep their IDF precision while typo'd tokens still
//               collide with the indexed phonetic key. Costs correctly spelled
//               words their precision: each exact hit is double-counted by its own
//               key, and 4-character keys collide with unrelated words.
//   phonetic  — metaphone keys ONLY (govdoc's approach, maximally fuzzy)
//   fallback  — exact tokens rank; a key is consulted ONLY for query tokens with
//               df == 0 in this corpus, which is the shape of a typo
//   fallback-half — same, with key matches weighted 0.5 of an exact term
//
// BM25 machinery is identical across variants (BM25Index injectable tokenizer).
// Dense channel is intentionally absent — this isolates the sparse channel.
//
// Usage: npm run eval:metaphone

import { BM25Index, createSparseIndex, tokenize, tokenizePhonetic } from '../vector-combobox.js';
import { metaphone } from '../metaphone.js';
import { DATASETS } from './datasets.mjs';

const metaphoneTokenizer = (text) => {
  const keys = tokenize(text).map(metaphone);
  return [...new Set(keys)];
};
const unionTokenizer = (text) => {
  const toks = tokenize(text);
  return [...toks, ...toks.map(metaphone)];
};

// Variants are index *options*, not just tokenizers: 'fallback' cannot be
// expressed as a tokenizer, because it needs to know which query tokens the index
// has never seen (df == 0) before consulting a phonetic key.
const VARIANTS = {
  baseline: { build: (items) => new BM25Index(items, { tokenize }) },
  union: { build: (items) => new BM25Index(items, { tokenize: unionTokenizer }) },
  phonetic: { build: (items) => new BM25Index(items, { tokenize: metaphoneTokenizer }) },
  // Exact tokens rank; metaphone keys are consulted only for tokens the corpus
  // does not contain at all.
  fallback: { build: (items) => createSparseIndex(items, { phonetic: 'fallback' }) },
  // Fallback, but keys carry half a term's weight — collisions are weaker evidence
  // than an exact spelling match.
  'fallback-half': { build: (items) => createSparseIndex(items, { phonetic: 'fallback', phoneticWeight: 0.5 }) },
};

// Typo queries: realistic combobox-speed misspellings. Targets as in datasets.mjs.
const TYPO_TESTS = [
  { ds: 'agencies', q: 'bording crossing logs', targets: ['cbp', 'dhs'] },
  { ds: 'agencies', q: 'veterens disability benifits', targets: ['dva'] },
  { ds: 'agencies', q: 'airport secuirty procedurs', targets: ['tsa'] },
  { ds: 'agencies', q: 'wether forecasts', targets: ['nws'] },
  { ds: 'agencies', q: 'tax reurn audit', targets: ['irs'] },
  { ds: 'agencies', q: 'spase exploration', targets: ['nasa'] },
  { ds: 'agencies', q: 'medicle records', targets: ['dva', 'hhs'] },
  { ds: 'agencies', q: 'prision inmate recods', targets: ['fboc'] },
  { ds: 'agencies', q: 'car crach safety recals', targets: ['nhtsa'] },
  { ds: 'agencies', q: 'polution in drinkng watter', targets: ['epa'] },
  { ds: 'policy', q: 'geting rid of paper recods', targets: ['OMB M-19-21'] },
  { ds: 'policy', q: 'clear simple langauge', targets: ['Plain Writing Act'] },
  { ds: 'policy', q: 'hackers steling data', targets: ['OMB M-22-09'] },
  { ds: 'policy', q: 'riscks of artifical inteligence', targets: ['Executive Order 14110', 'NIST AI Risk Management'] },
];

// Clean lexical queries: regression guard — union must not hurt exact matches.
const CLEAN_TESTS = [
  { ds: 'agencies', q: 'border crossing logs', targets: ['cbp', 'dhs'] },
  { ds: 'agencies', q: 'veterans disability benefits', targets: ['dva'] },
  { ds: 'agencies', q: 'airport security procedures', targets: ['tsa'] },
  { ds: 'agencies', q: 'weather forecasts', targets: ['nws'] },
  { ds: 'agencies', q: 'tax return audit', targets: ['irs'] },
  { ds: 'agencies', q: 'space exploration', targets: ['nasa'] },
  { ds: 'agencies', q: 'passport and visa applications', targets: ['dos'] },
  { ds: 'policy', q: 'records management electronic', targets: ['OMB M-19-21'] },
  { ds: 'policy', q: 'zero trust', targets: ['OMB M-22-09'] },
  { ds: 'policy', q: 'October', targets: ['Executive Order 14110'] },
];

function rankOf(index, items, query, targets) {
  const scores = index.scores(query);
  const ranked = items
    .map((item, i) => ({ i, s: scores[i], target: targets.some((t) => item.value.startsWith(t)) }))
    .sort((a, b) => b.s - a.s);
  const hit = ranked.findIndex((x) => x.target && x.s > 0);
  return hit === -1 ? Infinity : hit + 1;
}

function runSet(name, tests, indexes, itemsByDs) {
  console.log(`\n---- ${name} ----`);
  const summary = {};
  for (const [variant, indexesByDs] of Object.entries(indexes)) {
    const rows = tests.map((t) => ({
      q: t.q,
      rank: rankOf(indexesByDs[t.ds], itemsByDs[t.ds], t.q, t.targets),
    }));
    const top1 = rows.filter((r) => r.rank === 1).length;
    const top3 = rows.filter((r) => r.rank <= 3).length;
    const misses = rows.filter((r) => r.rank === Infinity).length;
    summary[variant] = { top1, top3, misses, n: rows.length, rows };
    console.log(
      `${variant.padEnd(9)} top1 ${top1}/${rows.length}  top3 ${top3}/${rows.length}  no-match ${misses}`
    );
  }
  // per-query diff: baseline rank -> variant rank
  for (let i = 0; i < tests.length; i++) {
    const parts = Object.entries(summary).map(([v, s]) => `${v}=${s.rows[i].rank === Infinity ? '-' : s.rows[i].rank}`);
    console.log(`  ${tests[i].q.slice(0, 34).padEnd(34)} ${parts.join('  ')}`);
  }
  return summary;
}

function main() {
  // One index per (variant, dataset). Union doubles doc length; BM25's
  // length normalization absorbs that.
  const indexes = {};
  for (const [variant, v] of Object.entries(VARIANTS)) {
    indexes[variant] = {};
    for (const [ds, items] of Object.entries(DATASETS)) {
      indexes[variant][ds] = v.build(items);
    }
  }

  const typoSummary = runSet('TYPO QUERIES (improvement test)', TYPO_TESTS, indexes, DATASETS);
  const cleanSummary = runSet('CLEAN QUERIES (regression guard)', CLEAN_TESTS, indexes, DATASETS);

  console.log('\n================ VERDICT INPUT ================');
  for (const [variant, s] of Object.entries(typoSummary)) {
    const c = cleanSummary[variant];
    console.log(
      `${variant.padEnd(9)} typo top1 ${s.top1}/${s.n} top3 ${s.top3}/${s.n} | clean top1 ${c.top1}/${c.n} top3 ${c.top3}/${c.n}`
    );
  }
}

main();
