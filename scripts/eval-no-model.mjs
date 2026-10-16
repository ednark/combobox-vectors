// scripts/eval-no-model.mjs
// Headroom measurement: what does the dense channel (and its ~15 MB of model +
// runtime) actually buy over the sparse channel alone?
//
// Same corpus-relative discrimination harness as eval-embedders.mjs, but with NO
// embedding model at all — pure BM25 over title(x3) + vectorText. Zero
// dependencies, runs offline, no download.
//
// Both query tiers from scripts/datasets.mjs are reported separately, because
// they answer different questions:
//   tier 1 — query words are mostly already in the corpus, so the keyword channel
//            can see what the semantic channel sees. Measures ranking given
//            shared vocabulary.
//   tier 2 — paraphrases with 20-50% corpus coverage. Measures the generalization
//            only a semantic channel can supply.
//
// Reference for comparison (dense channel, model required), from the project's
// own eval: potion-base-4M top-1 25/28, top-3 28/28 on tier 1.
//
// Usage: npm run eval:no-model

import { DATASETS, TIERS } from './datasets.mjs';
import { BM25Index, tokenize, tokenizePhonetic } from '../search-core.js';

const CHANNELS = {
  bm25: { name: 'bm25 (exact tokens only)', tokenizeFn: tokenize },
  phone: { name: 'bm25 + metaphone (shipped default)', tokenizeFn: tokenizePhonetic },
};

// Competition ranking: rank = 1 + the number of options that strictly outscore
// the best-scoring target. Tie-neutral, so a query whose target ties with the
// field is not flattered by array order.
function rankQuery(index, items, targets, query) {
  const scores = index.scores(query);
  let best = -Infinity;
  let bestValue = '';
  let top = 0;
  items.forEach((item, i) => {
    top = Math.max(top, scores[i]);
    const isTarget = targets.some((t) => item.value.startsWith(t));
    if (isTarget && scores[i] > best) {
      best = scores[i];
      bestValue = item.value;
    }
  });
  if (best === -Infinity) return { rank: items.length + 1, matched: top > 0, top: '(no target)' };
  let better = 0;
  for (const s of scores) if (s > best) better += 1;
  return { rank: better + 1, matched: top > 0, top: bestValue.slice(0, 38) };
}

function rankFlag(rank) {
  if (rank === 1) return 'ok ';
  if (rank <= 3) return '~  ';
  return 'X  ';
}

const wanted = process.argv.slice(2).filter((a) => a in CHANNELS);
const selected = wanted.length ? wanted : Object.keys(CHANNELS);
let worstMiss = 0;

for (const key of selected) {
  const ch = CHANNELS[key];
  console.log(`\n=== ${ch.name} ===`);
  const indexes = {};
  for (const [dsName, items] of Object.entries(DATASETS)) {
    indexes[dsName] = new BM25Index(items, { tokenize: ch.tokenizeFn });
  }
  const totals = { top1: 0, top3: 0, n: 0 };

  for (const tier of TIERS) {
    const rows = [];
    for (const t of tier.tests) {
      const r = rankQuery(indexes[t.ds], DATASETS[t.ds], t.targets, t.q);
      rows.push({ ...t, ...r });
      console.log(
        `  ${rankFlag(r.rank)} rank ${String(r.rank).padStart(2)}  ` +
        `${r.matched ? '' : '[no token match] '}` +
        `[${t.ds.padEnd(8)}] "${t.q}"  ->  ${r.top}`
      );
    }
    const n = rows.length;
    const count = (f) => rows.filter(f).length;
    const mrr = rows.reduce((a, r) => a + 1 / Math.min(r.rank, n), 0) / n;
    worstMiss = Math.max(worstMiss, rows.filter((r) => r.rank > 3).length);
    totals.top1 += count((r) => r.rank === 1);
    totals.top3 += count((r) => r.rank <= 3);
    totals.n += n;
    console.log(`  [${tier.name}]`);
    console.log(`      top1 ${count((r) => r.rank === 1)}/${n}  top3 ${count((r) => r.rank <= 3)}/${n}  ` +
      `MRR ${mrr.toFixed(3)}  no-token-match ${count((r) => !r.matched)}`);
  }
  console.log(`  [all tiers] top1 ${totals.top1}/${totals.n}  top3 ${totals.top3}/${totals.n}`);
}

console.log('\n--- reading this ---');
console.log('Tier 1 says the keyword channel is already strong on this corpus shape:');
console.log('vectorText was written in the user\'s vocabulary, so most query words are');
console.log('already index terms. Tier 2 is where the dense channel earns its bytes —');
console.log('compare these numbers with npm run eval:table.');
process.exit(worstMiss > 0 ? 1 : 0);
