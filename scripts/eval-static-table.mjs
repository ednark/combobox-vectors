// scripts/eval-static-table.mjs
// Measure the model-free dense channel: a build-time static semantic table
// (scripts/build-semantic-table.mjs) scored through the component's own fusion.
//
// Two query tiers, from scripts/datasets.mjs:
//   tier 1 — in-vocabulary queries. Nearly every query word already appears in
//            some option's vectorText, so the keyword channel sees what the
//            semantic channel sees. This tier cannot measure generalization.
//   tier 2 — paraphrase queries. 20-50% word coverage of the corpus vocabulary,
//            so BM25 is blind to most of the query. This is the tier that decides
//            whether a semantic channel is worth any bytes at all.
//
// Zero dependencies, runs offline, no model download.
//
// Usage: npm run eval:table      (or: node scripts/eval-static-table.mjs)
//        node scripts/eval-static-table.mjs --keep-sweep 5,10,25,50

import { readFileSync, statSync, existsSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

import { DATASETS, TIERS } from './datasets.mjs';
import { BM25Index, tokenizePhonetic, rankItems } from '../search-core.js';
import { DEFAULTS } from '../vector-combobox.js';
import { decodeTable } from '../static-table.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

const DATASET_FOR = { policy: { key: 'policy', file: 'data' }, agencies: { key: 'agencies', file: 'data-agencies' } };

function rankInOrder(order, values, targets) {
  const pos = order.findIndex((i) => targets.some((t) => values[i].startsWith(t)));
  return pos === -1 ? values.length + 1 : pos + 1;
}

function summarize(ranks) {
  const n = ranks.length;
  return {
    n,
    top1: ranks.filter((r) => r === 1).length,
    top3: ranks.filter((r) => r <= 3).length,
    mrr: ranks.reduce((a, r) => a + 1 / Math.min(r, n), 0) / n,
  };
}

function pad(tag, width) { return tag.padEnd(width); }

// What each row feeds the fusion. 'none' = no dense channel; 'dense-only' = no
// keyword evidence at all, so the two single-channel rows are really single-channel.
const CONFIGS = [
  { name: 'BM25 only (no semantic channel)', mode: 'sparse-only' },
  { name: 'static table only (no keywords)', mode: 'dense-only' },
  { name: 'table + BM25, ungated RRF (old behavior)', mode: 'hybrid', opts: { sparseMinCoverage: 0, genericDfFrac: 1 } },
  { name: 'table + BM25, component defaults', mode: 'hybrid' },
  { name: 'table + BM25, absolute cutoff (old)', mode: 'hybrid', opts: { thresholdMode: 'absolute' } },
];

function loadArtifact(file) {
  const p = join(ROOT, `${file}.semantic.json`);
  if (!existsSync(p)) {
    throw new Error(`${file}.semantic.json missing — run npm run build:tables first`);
  }
  let artifact;
  try {
    artifact = JSON.parse(readFileSync(p, 'utf8'));
  } catch (err) {
    throw new Error(`${file}.semantic.json is not valid JSON (${err.message}) — npm run build:tables`);
  }
  return { path: p, artifact, kb: statSync(p).size / 1024 };
}

let regressions = 0;

for (const [dsName, { key, file }] of Object.entries(DATASET_FOR)) {
  const items = DATASETS[key];
  const values = items.map((it) => it.value);
  const { artifact, kb } = loadArtifact(file);
  const table = decodeTable(artifact);

  // The table's columns are this option list; a stale artifact is a silent
  // correctness bug, so verify the alignment instead of trusting it.
  const mismatch = values.filter((v, i) => artifact.options[i] !== v).length;
  if (mismatch) {
    throw new Error(`${file}.semantic.json is stale: ${mismatch}/${values.length} options differ ` +
      `from ${file}.js — npm run build:tables`);
  }

  const index = new BM25Index(items, {
    tokenize: tokenizePhonetic,
    genericDfFrac: DEFAULTS.genericDfFrac,
  });

  console.log(`\n=== ${dsName} — ${values.length} options — ${artifact.embedder} ===`);
  console.log(`    artifact: ${kb.toFixed(1)} KB on disk (${artifact.stats.rowsKept} rows, ` +
    `quantizer ${artifact.stats.quantizer}, row order ${artifact.stats.rowOrder})`);
  console.log(`    vs model path: 15,100 KB ONNX + ~1,000 KB ONNX runtime + ~200 KB tokenizer`);

  for (const tier of TIERS) {
    const tests = tier.tests.filter((t) => t.ds === key);
    if (!tests.length) continue;
    const cov = tests.reduce((a, t) => a + table.coverage(t.q), 0) / tests.length;
    console.log(`\n  ${tier.name}  (n=${tests.length})  table word coverage ${(cov * 100).toFixed(0)}%`);
    for (const cfg of CONFIGS) {
      const o = { ...DEFAULTS, ...(cfg.opts || {}) };
      const none = values.map(() => 0);
      const ranks = tests.map((t) => {
        const dense = cfg.mode === 'sparse-only' ? null : Array.from(table.scores(t.q));
        const lex = cfg.mode === 'dense-only' ? none : index.scores(t.q);
        const lexRank = cfg.mode === 'hybrid' ? index.scores(t.q, { informativeOnly: true }) : null;
        const coverage = cfg.mode === 'hybrid' ? index.coverage(t.q) : null;
        return rankInOrder(rankItems(values.length, lex, dense, o, lexRank, coverage), values, t.targets);
      });
      const m = summarize(ranks);
      console.log(`    ${pad(cfg.name, 44)} top1 ${String(m.top1).padStart(2)}/${m.n}  ` +
        `top3 ${String(m.top3).padStart(2)}/${m.n}  MRR ${m.mrr.toFixed(3)}`);
    }
  }
}

console.log('\n--- reading this ---');
console.log('The semantic table must land near the 15 MB model it replaces on tier 1,');
console.log('and it is on tier 2 that a semantic channel earns its bytes. Tier 2 is 18');
console.log('hand-written queries: treat movement as directional, not as a gate.');

process.exit(regressions ? 1 : 0);
