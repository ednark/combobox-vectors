// scripts/eval-embedders.mjs
// Corpus-relative discrimination eval for dense embedders.
//
// Question answered: for each test query, does the intended option outrank
// the other options in the SAME dataset? NOT a general embedding benchmark —
// a model that is globally mediocre can still perfectly separate 42 agency
// descriptions, and that is all this component needs.
//
// Models compared:
//   Xenova/paraphrase-MiniLM-L3-v2  (current default, 17.5 MB int8) — baseline
//   minishlab/potion-base-2M        ( 7.6 MB fp32, 64-dim static)
//   minishlab/potion-base-4M        (15.1 MB fp32, 128-dim static)
//   minishlab/potion-base-8M        (30.2 MB fp32, 256-dim static; quality anchor)
//
// Metrics per model: top-1/top-3 hit rate, MRR, mean rank, mean margin of the
// best target over the best non-target.
//
// Usage: npm run eval:embedders

import { writeFileSync, existsSync, readFileSync, mkdirSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { pipeline, AutoTokenizer, env } from '@huggingface/transformers';
import ort from 'onnxruntime-node';
import { DATASETS, TESTS } from './datasets.mjs';

env.allowLocalModels = false;

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const CACHE = join(ROOT, '.cache');
mkdirSync(CACHE, { recursive: true });

async function download(modelId) {
  const file = join(CACHE, modelId.replaceAll('/', '--') + '.onnx');
  if (!existsSync(file)) {
    console.log(`  downloading ${modelId} onnx …`);
    const res = await fetch(`https://huggingface.co/${modelId}/resolve/main/onnx/model.onnx`);
    if (!res.ok) throw new Error(`download failed for ${modelId}: ${res.status}`);
    writeFileSync(file, Buffer.from(await res.arrayBuffer()));
  }
  return file;
}

// --- Embedder factories. Both return async (texts: string[]) => number[][] ---
// and both produce L2-normalized vectors (cosine == dot product).

async function pipelineEmbedder(modelId) {
  const extractor = await pipeline('feature-extraction', modelId);
  return async (texts) => {
    const out = await extractor(texts, { pooling: 'mean', normalize: true });
    const dim = out.dims.at(-1);
    return Array.from({ length: texts.length }, (_, i) =>
      Array.from(out.data.subarray(i * dim, (i + 1) * dim))
    );
  };
}

async function potionEmbedder(modelId) {
  const [tokenizer, onnxPath] = await Promise.all([
    AutoTokenizer.from_pretrained(modelId),
    download(modelId),
  ]);
  const session = await ort.InferenceSession.create(readFileSync(onnxPath));
  return async (texts) => {
    // model2vec protocol (per MinishLab export script): per-text token ids
    // (no special tokens, no unk), flattened, with cumulative offsets —
    // the ONNX graph is an EmbeddingBag(mean) + L2 normalize.
    const encodings = texts.map((t) =>
      tokenizer.encode(t, { add_special_tokens: false })
        .filter((id) => id !== tokenizer.unk_token_id)
    );
    const flat = encodings.flatMap((ids) => ids);
    if (!flat.length) return texts.map(() => {
      // No usable tokens: the graph needs a valid input; feed unk once and
      // treat the result as a zero-ish vector (norm-1 but meaningless).
      return null; // replaced below
    });
    const offsets = [0];
    for (let i = 0; i < encodings.length - 1; i++) {
      offsets.push(offsets[i] + encodings[i].length);
    }
    const feeds = {
      input_ids: new ort.Tensor('int64', BigInt64Array.from(flat.map(BigInt)), [flat.length]),
      offsets: new ort.Tensor('int64', BigInt64Array.from(offsets.map(BigInt)), [offsets.length]),
    };
    const out = await session.run(feeds);
    const emb = out[session.outputNames[0]];
    const dim = emb.dims.at(-1);
    const vecs = Array.from({ length: texts.length }, (_, i) =>
      Array.from(emb.data.subarray(i * dim, (i + 1) * dim))
    );
    // Empty encodings were fed garbage positions; mark them as zero vectors
    // (score 0 against everything — same practical effect as no signal).
    return vecs.map((v, i) => (encodings[i].length ? v : new Array(dim).fill(0)));
  };
}

const MODELS = [
  { name: 'MiniLM-L3 (baseline)', make: () => pipelineEmbedder('Xenova/paraphrase-MiniLM-L3-v2') },
  { name: 'potion-2M (7.6 MB)', make: () => potionEmbedder('minishlab/potion-base-2M') },
  { name: 'potion-4M (15.1 MB)', make: () => potionEmbedder('minishlab/potion-base-4M') },
  { name: 'potion-8M (30.2 MB, anchor)', make: () => potionEmbedder('minishlab/potion-base-8M') },
];

const dot = (a, b) => {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += a[i] * b[i];
  return s;
};

async function main() {
  const results = {};

  for (const model of MODELS) {
    console.log(`\n=== ${model.name} ===`);
    const embed = await model.make();
    const rows = [];

    // Embed every option once per dataset (vectorText only — that is what
    // the dense channel ranks against).
    const docVecs = {};
    for (const [dsName, items] of Object.entries(DATASETS)) {
      docVecs[dsName] = await embed(items.map((i) => i.vectorText || i.text));
    }

    for (const t of TESTS) {
      const items = DATASETS[t.ds];
      const vecs = docVecs[t.ds];
      const [q] = await embed([t.q]);
      const scored = items.map((item, i) => ({
        value: item.value,
        target: t.targets.some((tg) => item.value.startsWith(tg)),
        s: dot(q, vecs[i]),
      })).sort((a, b) => b.s - a.s);

      const firstTarget = scored.findIndex((x) => x.target);
      const rank = firstTarget + 1;
      const targetScore = scored[firstTarget].s;
      const bestNonTarget = scored.find((x) => !x.target)?.s ?? 0;
      const margin = targetScore - bestNonTarget;
      rows.push({ ...t, rank, margin, top: scored[0].value.slice(0, 38) });
      console.log(
        `  rank ${String(rank).padStart(2)}  margin ${margin >= 0 ? '+' : ''}${margin.toFixed(3)}  ` +
        `[${t.ds.padEnd(8)}] "${t.q}"`
      );
    }

    const n = rows.length;
    const top1 = rows.filter((r) => r.rank === 1).length;
    const top3 = rows.filter((r) => r.rank <= 3).length;
    const mrr = rows.reduce((s, r) => s + 1 / r.rank, 0) / n;
    const meanRank = rows.reduce((s, r) => s + r.rank, 0) / n;
    const meanMargin = rows.reduce((s, r) => s + r.margin, 0) / n;
    results[model.name] = { top1, top3, n, mrr, meanRank, meanMargin, rows };
    console.log(`  -> top1 ${top1}/${n}  top3 ${top3}/${n}  MRR ${mrr.toFixed(3)}  meanRank ${meanRank.toFixed(1)}  meanMargin ${meanMargin.toFixed(3)}`);
  }

  // ---- summary table ----
  console.log('\n================ SUMMARY ================');
  console.log('model                          top1  top3   MRR   meanRank  meanMargin');
  for (const [name, r] of Object.entries(results)) {
    console.log(
      `${name.padEnd(30)} ${String(r.top1).padStart(3)}/${r.n} ${String(r.top3).padStart(4)}/${r.n}  ${r.mrr.toFixed(3).padStart(6)}  ${r.meanRank.toFixed(1).padStart(8)}  ${r.meanMargin.toFixed(3).padStart(10)}`
    );
  }

  // Per-query rank comparison across models (the interesting diff view).
  console.log('\n---- per-query rank by model ----');
  const names = Object.keys(results);
  const first = results[names[0]].rows;
  for (let i = 0; i < first.length; i++) {
    const ranks = names.map((nm) => results[nm].rows[i].rank);
    const flag = ranks.some((r, j) => j > 0 && r > 3 && ranks[0] <= 3) ? '  <-- REGRESSION' : '';
    console.log(
      `${first[i].q.slice(0, 38).padEnd(38)} ${ranks.map((r) => String(r).padStart(3)).join(' ')}${flag}`
    );
  }
  console.log(`  (columns: ${names.map((n, i) => `${i + 1}=${n}`).join('  ')}`);
}

main().catch((err) => {
  console.error('eval failed:', err);
  process.exit(1);
});
