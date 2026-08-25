// scripts/precompute.mjs
// Build-time embedding precomputation for combobox-vectors.
//
// Embeds each option's `vectorText` with the SAME model the front-end uses
// for the query (Xenova/paraphrase-MiniLM-L3-v2) via Transformers.js in Node,
// so precomputed vectors and runtime query vectors share one space.
// Model parity is a hard requirement — a different model produces a
// different vector space and silently garbage similarity.
//
// Usage:
//   npm run precompute                 # embed all datasets, writes *.embeddings.json
//   npm run precompute -- data.js      # embed a single dataset
//
// Output: for each input `X.js` exporting DATA-like arrays, writes
// `X.embeddings.json` mapping value -> L2-normalized 384-dim vector.
//
// Inspired by govdoc-explainer's govdoc_explainer/embeddings.py (fastembed
// at build time), adapted to JS so model parity is guaranteed by
// construction: the exact same Transformers.js pipeline code path.

import { readFileSync, writeFileSync, existsSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, resolve } from 'node:path';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));

// Keep in sync with DEFAULTS.model in vector-combobox.js. A mismatch here
// invalidates every precomputed vector (see validateEmbedding in the module).
const MODEL = 'minishlab/potion-base-4M';

const DATASETS = process.argv.length > 2
  ? process.argv.slice(2)
  : ['data.js', 'data-agencies.js'];

function loadDataset(file) {
  const path = join(ROOT, file);
  if (!existsSync(path)) {
    console.error(`skip ${file}: not found`);
    return null;
  }
  const src = readFileSync(path, 'utf8');
  // Static ES-module data files; evaluate the export without a bundler.
  // `export const NAME = ...` -> `module.exports = ...` (handles leading
  // comment lines and the optional trailing semicolon).
  const m = src.match(/export\s+const\s+[A-Za-z_]+\s*=\s*([\s\S]*?);?\s*$/);
  if (!m) {
    console.error(`skip ${file}: no \`export const\` array found`);
    return null;
  }
  const mod = { exports: null };
  new Function('module', `module.exports = ${m[1]};`)(mod);
  const data = mod.exports;
  if (!Array.isArray(data)) {
    console.error(`skip ${file}: export is not an array`);
    return null;
  }
  return { path, data };
}

async function main() {
  // IMPORTANT: precompute uses the SAME embedMany() the browser runs —
  // imported from vector-combobox.js — so build-time and query-time vectors
  // share one code path and one model. Parity by construction.
  const { embedMany } = await import('../vector-combobox.js');

  console.log(`loading ${MODEL} …`);

  for (const file of DATASETS) {
    const loaded = loadDataset(file);
    if (!loaded) continue;
    const { path, data } = loaded;

    // Batch all vectorTexts in ONE call.
    const texts = data.map((item) => item.vectorText || item.text);
    console.log(`${file}: embedding ${texts.length} items …`);
    const vectors = await embedMany(texts, MODEL);

    const index = {};
    data.forEach((item, i) => {
      index[item.value] = vectors[i];
    });

    const outPath = path.replace(/\.js$/, '.embeddings.json');
    writeFileSync(outPath, JSON.stringify(index));
    console.log(`${file}: wrote ${outPath} (${Object.keys(index).length} vectors)`);

    // Validation per ROADMAP-precompute.md: L2 norm ~1, dim reported per model.
    const dims = new Set(Object.values(index).map((v) => v.length));
    const norms = Object.values(index).map((v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0)));
    const minNorm = Math.min(...norms), maxNorm = Math.max(...norms);
    console.log(`${file}: dim=${[...dims].join('/')}, norms ${minNorm.toFixed(4)}..${maxNorm.toFixed(4)} ${
      minNorm > 0.99 && maxNorm < 1.01 ? '(OK, L2-normalized)' : '(FAIL: not normalized!)'
    }`);
  }
}

main().catch((err) => {
  console.error('precompute failed:', err);
  process.exit(1);
});
