// scripts/build-semantic-table.mjs
// Build-time generation of a *static semantic table*: the dense channel
// flattened into a word -> per-option integer score matrix, so the browser needs
// no embedding model, no ONNX runtime, no WASM, and no tokenizer.
//
// The derivation (repeated in static-table.js where it is consumed):
//   cos(query, option_j) = ( Σ_i <E_i, V_j> ) / ‖ Σ_i E_i ‖
// The denominator is the same positive number for every option j, so it cannot
// change the ranking — and RRF fusion consumes ranks. What ships is only the
// numerator: S[i,j] = <E_i, V_j>, quantized to int8 with a single global scale.
//
// Pruning rule: a word can only move a rank through the SPREAD of its score row
// (max_j - min_j). A word whose spread is 0 adds a constant to every option and
// provably cannot change any order, so dropping low-spread words is
// rank-preserving in principle. The real cost is coverage: a query whose words
// were all dropped scores nothing in the dense channel and falls back to
// keywords. The build reports measured coverage per eval tier.
//
// Inputs: the static embedder's raw lookup table (fetched, cached in .cache/)
// and the precomputed option vectors from `npm run precompute`.
// Dependencies: none — Node fetch + zlib only. No Transformers.js, no ONNX.
//
// Usage:
//   node scripts/build-semantic-table.mjs [--keep 25] [--model minishlab/potion-base-4M]
//       [data] [data-agencies]

import { readFileSync, writeFileSync, existsSync, mkdirSync, statSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { gzipSync, brotliCompressSync } from 'node:zlib';

import { DATASETS, TESTS, PARAPHRASE_TESTS, TIERS } from './datasets.mjs';
import { decodeTable, tableWords } from '../static-table.js';
import { BM25Index, tokenizePhonetic, rankItems } from '../search-core.js';

const ROOT = dirname(dirname(fileURLToPath(import.meta.url)));
const CACHE = join(ROOT, '.cache');
mkdirSync(CACHE, { recursive: true });

const argv = process.argv.slice(2);
const flag = (name, dflt) => {
  const i = argv.indexOf(`--${name}`);
  return i === -1 ? dflt : argv[i + 1];
};
const MODEL = flag('model', 'minishlab/potion-base-4M');
const HUB = flag('hub', 'https://huggingface.co');
const KEEP_PCT = Number(flag('keep', 25));
if (!Number.isFinite(KEEP_PCT) || KEEP_PCT <= 0 || KEEP_PCT > 100) {
  throw new Error(`--keep must be a percentage in (0,100], got ${flag('keep', 25)}`);
}
const DATASET_NAMES = argv.filter((a) => !a.startsWith('--') && !isValueOfFlag(a));
function isValueOfFlag(a) {
  return a === MODEL || a === HUB || a === flag('keep', 25) || /^\d+$/.test(a);
}

// Must match the component's defaults so measured numbers are what ships.
const RANK_OPTS = {
  threshold: 0.3, maxResults: 10, hybrid: true, fusion: 'rrf', rrfK: 60, lexicalWeight: 0.5,
  thresholdMode: 'relative',
};
// The historical absolute cutoff, measured alongside for comparison.
const RANK_OPTS_ABS = { ...RANK_OPTS, thresholdMode: 'absolute' };
// Sparse-channel informativeness gates to sweep (fraction of options a query
// token may appear in and still count as evidence for ranking).
const GATE_FRACS = [0.2, 0.34, 0.5];
// Score-normalized fusion comparison (same gate, same cutoff).
const WEIGHT_OPTS = { ...RANK_OPTS, fusion: 'weighted', lexicalWeight: 0.5 };

const DS_KEY = { data: 'policy', 'data-agencies': 'agencies', policy: 'policy', agencies: 'agencies' };

// ---------- inputs ----------
async function cached(url, file) {
  const p = join(CACHE, file);
  if (!existsSync(p) || statSync(p).size === 0) {
    console.log(`  fetching ${url}`);
    const res = await fetch(url);
    if (!res.ok) throw new Error(`fetch ${url} -> HTTP ${res.status}`);
    writeFileSync(p, Buffer.from(await res.arrayBuffer()));
  }
  return p;
}

function parseJson(buffer, what) {
  try {
    return JSON.parse(buffer.toString('utf8'));
  } catch (err) {
    throw new Error(`${what} is not valid JSON: ${err.message}`);
  }
}

async function loadEmbedderTable(model) {
  const tensorPath = await cached(
    `${HUB}/${model}/resolve/main/model.safetensors`,
    `${model.replaceAll('/', '--')}.safetensors`
  );
  const vocabPath = await cached(
    `${HUB}/${model}/resolve/main/vocab.txt`,
    `${model.replaceAll('/', '--')}.vocab.txt`
  );
  const buf = readFileSync(tensorPath);
  let header;
  try {
    header = JSON.parse(buf.subarray(8, 8 + Number(buf.readBigUInt64LE(0))).toString('utf8'));
  } catch (err) {
    throw new Error(`${tensorPath}: not a safetensors file (${err.message})`);
  }
  const entries = Object.entries(header).filter(([k]) => k !== '__metadata__');
  if (entries.length !== 1) throw new Error(`${tensorPath}: expected one tensor, found ${entries.length}`);
  const [name, meta] = entries[0];
  if (meta.dtype !== 'F32') throw new Error(`${name}: dtype ${meta.dtype} unsupported (expected F32)`);
  if (meta.shape.length !== 2) throw new Error(`${name}: expected [rows, dim], got [${meta.shape}]`);
  const start = 8 + Number(buf.readBigUInt64LE(0)) + meta.data_offsets[0];
  const end = 8 + Number(buf.readBigUInt64LE(0)) + meta.data_offsets[1];
  const copy = buf.subarray(start, end).slice(); // own buffer: alignment-safe view
  const vocab = readFileSync(vocabPath, 'utf8').split('\n').map((s) => s.replace(/\r$/, ''));
  if (vocab.length < meta.shape[0]) {
    throw new Error(`vocab.txt has ${vocab.length} lines but the table has ${meta.shape[0]} rows`);
  }
  return {
    name,
    rows: meta.shape[0],
    dim: meta.shape[1],
    data: new Float32Array(copy.buffer, copy.byteOffset, copy.length / 4),
    vocab,
  };
}

function loadDataset(name) {
  const key = DS_KEY[name];
  if (!key) throw new Error(`unknown dataset "${name}" (use: data | data-agencies)`);
  const embPath = join(ROOT, `${name}.embeddings.json`);
  if (!existsSync(embPath)) {
    throw new Error(`${name}.embeddings.json missing — run npm run precompute first`);
  }
  const emb = parseJson(readFileSync(embPath), `${name}.embeddings.json`);
  const items = DATASETS[key];
  const missing = items.filter((it) => !Array.isArray(emb[it.value]) || !emb[it.value].length);
  if (missing.length) {
    throw new Error(`${name}: ${missing.length}/${items.length} options have no precomputed ` +
      `embedding (e.g. "${String(missing[0]?.value).slice(0, 40)}") — npm run precompute`);
  }
  return { key, items, vectors: items.map((it) => emb[it.value]) };
}

// ---------- ranking metrics ----------
function rankOf(scores, values, targets) {
  let best = -Infinity;
  for (let i = 0; i < values.length; i++) {
    if (targets.some((t) => values[i].startsWith(t)) && scores[i] > best) best = scores[i];
  }
  if (best === -Infinity) return values.length + 1;
  let better = 0;
  for (const s of scores) if (s > best) better += 1;
  return better + 1;
}

// Rank of the intended option inside an already-ordered list of indices.
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

function line(tag, m) {
  return `  ${tag.padEnd(42)} top1 ${String(m.top1).padStart(2)}/${String(m.n).padStart(2)}  ` +
    `top3 ${String(m.top3).padStart(2)}/${m.n}  MRR ${m.mrr.toFixed(3)}`;
}

// ---------- build ----------
async function buildTable(datasetName, embedder) {
  const { key, items, vectors: O } = loadDataset(datasetName);
  const values = items.map((it) => it.value);
  const nOpt = values.length;
  console.log(`\n=== ${datasetName} (${key}): ${nOpt} options, ${O[0].length}-dim vectors ===`);

  const { rows: nTok, dim, data: E, vocab } = embedder;
  if (dim !== O[0].length) throw new Error(`embedder dim ${dim} != option vector dim ${O[0].length}`);

  const wordAt = [];
  const vocabIndexOf = new Map();
  for (let i = 0; i < nTok; i++) {
    const w = vocab[i];
    // Whole words only: subword continuations ("##ing") are unreachable from a
    // plain word lookup, so a row for one would be a row the runtime can never use.
    if (/^[a-z][a-z0-9]*$/.test(w) && !vocabIndexOf.has(w)) {
      vocabIndexOf.set(w, i);
      wordAt.push(i);
    }
  }
  console.log(`  ${wordAt.length} whole-word vocabulary entries (of ${nTok} rows)`);

  // S[i,j] = <E_i, V_j>, and the per-word spread that decides whether to keep it.
  const S = new Float64Array(wordAt.length * nOpt);
  const spread = new Float64Array(wordAt.length);
  for (let r = 0; r < wordAt.length; r++) {
    const base = wordAt[r] * dim;
    let lo = Infinity, hi = -Infinity;
    for (let j = 0; j < nOpt; j++) {
      const vj = O[j];
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += E[base + d] * vj[d];
      S[r * nOpt + j] = dot;
      if (dot < lo) lo = dot;
      if (dot > hi) hi = dot;
    }
    spread[r] = hi - lo;
  }

  const sortedSpread = Float64Array.from(spread).sort();
  const cut = sortedSpread[Math.min(sortedSpread.length - 1,
    Math.floor(sortedSpread.length * (1 - KEEP_PCT / 100)))];
  const keep = [];
  for (let r = 0; r < wordAt.length; r++) if (spread[r] >= cut) keep.push(r);
  // word -> index into the `keep` list, for calibration and verification lookups
  const rowOfWord = new Map();
  keep.forEach((r, i) => {
    rowOfWord.set(vocab[wordAt[r]], i);
  });
  console.log(`  keep top ${KEEP_PCT}% by spread: ${keep.length} rows (spread floor ${cut.toFixed(3)}, ` +
    `max ${sortedSpread[sortedSpread.length - 1].toFixed(1)})`);

  // int8 with ONE global scale. A per-column scale would rescale each option
  // differently — that is NOT rank-preserving (verified the hard way).
  let absMax = 0;
  for (const r of keep) for (let j = 0; j < nOpt; j++) absMax = Math.max(absMax, Math.abs(S[r * nOpt + j]));
  const gscale = 127 / absMax;
  const Q = new Int8Array(keep.length * nOpt);
  for (let r = 0; r < keep.length; r++) {
    for (let j = 0; j < nOpt; j++) {
      Q[r * nOpt + j] = Math.max(-128, Math.min(127, Math.round(S[keep[r] * nOpt + j] * gscale)));
    }
  }

  // Reference the table must reproduce: exact fp64 word-mean cosine, unpruned,
  // unquantized — i.e. the static embedder itself for whole-word queries.
  const reference = (query) => {
    const words = tableWords(query).map((w) => vocabIndexOf.get(w)).filter((i) => i !== undefined);
    if (!words.length) return null;
    const acc = new Float64Array(dim);
    for (const i of words) {
      const base = i * dim;
      for (let d = 0; d < dim; d++) acc[d] += E[base + d];
    }
    let norm = 0;
    for (let d = 0; d < dim; d++) norm += acc[d] * acc[d];
    norm = Math.sqrt(norm);
    if (!norm) return null;
    const out = Array(nOpt);
    for (let j = 0; j < nOpt; j++) {
      let dot = 0;
      for (let d = 0; d < dim; d++) dot += acc[d] * O[j][d];
      out[j] = dot / norm;
    }
    return out;
  };

  // Calibrate int units -> approximately cosine units, because `threshold` filters
  // on the absolute value (ranking does not care). Median ratio over eval queries;
  // the spread is reported so the approximation is not mistaken for cosine.
  const calib = [];
  for (const q of [...TESTS, ...PARAPHRASE_TESTS].map((t) => t.q)) {
    const ref = reference(q);
    const rows = tableWords(q).map((w) => rowOfWord.get(w)).filter((r) => r !== undefined);
    if (!ref || !rows.length) continue;
    let maxRef = -Infinity, maxTab = -Infinity;
    for (let j = 0; j < nOpt; j++) {
      let acc = 0;
      for (const r of rows) acc += Q[r * nOpt + j];
      const tab = acc / rows.length;
      maxRef = Math.max(maxRef, ref[j]);
      maxTab = Math.max(maxTab, tab);
    }
    // Only positive ratios: a negative scale would silently invert the ranking.
    if (maxTab > 1e-9 && maxRef > 0) calib.push(maxRef / maxTab);
  }
  calib.sort((a, b) => a - b);
  const scale = calib.length ? calib[Math.floor(calib.length / 2)] : 1 / gscale;
  console.log(`  calibrated scale ${scale.toFixed(6)} (${calib.length} queries, ` +
    `p5 ${calib.length ? calib[0].toFixed(4) : 'n/a'} .. p95 ` +
    `${calib.length ? calib[Math.floor(calib.length * 0.95)].toFixed(4) : 'n/a'})`);

  // Q row r corresponds to keep[r] (an index into wordAt), i.e. to the word
  // vocab[wordAt[keep[r]]]. Every later step must use the SAME Q-row index, or the
  // word list and the score matrix silently disagree — which is a ranking bug that
  // looks like a quality problem, not like a crash.
  const wordOfRow = keep.map((w) => vocab[wordAt[w]]);
  const spreadOfRow = keep.map((w) => spread[w]);
  const nRows = keep.length;

  // Row order that compresses best. Adjacent rows in vocabulary (frequency) order
  // are numerically similar; alphabetical groups by stem; spread order clusters
  // magnitudes. Measure, don't guess.
  const rowMatrix = (order) => {
    const m = new Int8Array(order.length * nOpt);
    for (let r = 0; r < order.length; r++) {
      const src = order[r];
      for (let j = 0; j < nOpt; j++) m[r * nOpt + j] = Q[src * nOpt + j];
    }
    return m;
  };
  const rowsSeq = Array.from({ length: nRows }, (_, r) => r);
  const orderings = {
    vocab: rowsSeq, // keep[] was built in vocabulary order
    alpha: rowsSeq.slice().sort((a, b) => (wordOfRow[a] < wordOfRow[b] ? -1 : 1)),
    spread: rowsSeq.slice().sort((a, b) => spreadOfRow[b] - spreadOfRow[a]),
  };
  let best = null;
  for (const [name, order] of Object.entries(orderings)) {
    const m = rowMatrix(order);
    const bytes = brotliCompressSync(Buffer.from(m.buffer)).length;
    console.log(`    row order ${name.padEnd(7)} ${m.length} B raw, ${bytes} B brotli`);
    if (!best || bytes < best.bytes) best = { name, order, matrix: m, bytes };
  }
  const words = best.order.map((r) => wordOfRow[r]);
  // The words string and the matrix rows are the same permutation; assert it, since
  // a mismatch here degrades ranking silently.
  if (best.order.some((r, i) => words[i] !== wordOfRow[r])) {
    throw new Error('internal: word list and score matrix are misaligned');
  }

  const artifact = {
    format: 'combobox-semantic-table/1',
    embedder: MODEL,
    dim,
    options: values,
    words: words.join('\n'),
    rowCount: words.length,
    scores: Buffer.from(best.matrix.buffer, best.matrix.byteOffset, best.matrix.byteLength).toString('base64'),
    scale,
    stats: {
      dataset: datasetName,
      keepPct: KEEP_PCT,
      wholeWordVocab: wordAt.length,
      rowsKept: words.length,
      spreadFloor: Number(cut.toFixed(6)),
      quantizer: 'int8, single global scale',
      rowOrder: best.name,
      rawBytes: best.matrix.length,
      brotliBytes: best.bytes,
      wordsGzipBytes: gzipSync(Buffer.from(words.join('\n'))).length,
    },
  };
  return { artifact, reference, key, values, items };
}

// ---------- verification: measure the artifact that was just written ----------
function topAgreement(a, b) {
  const first = (s) => {
    let bi = 0, bv = -Infinity;
    for (let i = 0; i < s.length; i++) if (s[i] > bv) { bv = s[i]; bi = i; }
    return bi;
  };
  return first(a) === first(b);
}

/**
 * Score the emitted artifact through the SAME fusion the component runs.
 * Three rows per tier: the exact static embedder in fp64 (the reference the
 * table claims to replace), the int8 table alone, and the table fused with
 * BM25 by RRF. Also reports word coverage — the honest cost of pruning — and
 * how often the table's top-1 is identical to the reference's.
 */
function verify({ artifact, reference, values, items, key }) {
  const table = decodeTable(artifact);
  const bm25 = new BM25Index(items, { tokenize: tokenizePhonetic });
  const denseScores = (q) => Array.from(table.scores(q));
  const fusedRanks = (q) => rankItems(values.length, bm25.scores(q), denseScores(q), RANK_OPTS);
  const fusedRanksAbs = (q) => rankItems(values.length, bm25.scores(q), denseScores(q), RANK_OPTS_ABS);
  // Gated fusion: full BM25 scores decide ELIGIBILITY, informative-token-only
  // scores decide the sparse RANK, so generic-word matches appear without being
  // boosted above a clear semantic hit. One index per gate fraction, because the
  // df ceiling is an index-construction parameter.
  const gateIdx = new Map(GATE_FRACS.map((frac) => [
    frac, new BM25Index(items, { tokenize: tokenizePhonetic, genericDfFrac: frac }),
  ]));
  // Term-coverage sweep is computed per tier (see verify loop below).
  const gated = (frac, q, dense, fusionOpts = RANK_OPTS, minCoverage = 0) => {
    const idx = gateIdx.get(frac);
    return rankItems(values.length, idx.scores(q), dense, { ...fusionOpts, sparseMinCoverage: minCoverage },
      idx.scores(q, { informativeOnly: true }), idx.coverage(q));
  };

  const covSweepFor = (tests) => {
    const configs = [
      { name: 'rrf, no weighting', f: RANK_OPTS, cov: 0, conf: null },
      { name: 'rrf, cov>=0.67 cliff', f: RANK_OPTS, cov: 0.67, conf: null },
      { name: 'rrf, alpha=evidence', f: RANK_OPTS, cov: 0, conf: 'coverage' },
      { name: 'weighted, no weighting', f: WEIGHT_OPTS, cov: 0, conf: null },
      { name: 'weighted, cov>=0.67 cliff', f: WEIGHT_OPTS, cov: 0.67, conf: null },
      { name: 'weighted, alpha=evidence', f: WEIGHT_OPTS, cov: 0, conf: 'coverage' },
      { name: 'weighted, ramp .5->1', f: WEIGHT_OPTS, cov: 0, conf: 'ramp', ramp: [0.5, 1] },
      { name: 'weighted, ramp .34->1', f: WEIGHT_OPTS, cov: 0, conf: 'ramp', ramp: [0.34, 1] },
      { name: 'rrf, ramp .5->1', f: RANK_OPTS, cov: 0, conf: 'ramp', ramp: [0.5, 1] },
      { name: 'rrf, ramp .34->1', f: RANK_OPTS, cov: 0, conf: 'ramp', ramp: [0.34, 1] },
      { name: 'w+alpha, cov>=0.67 also', f: WEIGHT_OPTS, cov: 0.67, conf: 'coverage' },
    ];
    const rows = [];
    for (const c of configs) {
      const opts = { ...c.f, sparseMinCoverage: c.cov, sparseConfidence: c.conf, sparseRamp: c.ramp };
      const denseOf = (t) => denseScores(t.q);
      const modelOf = (t) => reference(t.q) ?? values.map(() => 0);
      rows.push({
        label: c.name,
        table: summarize(tests.map((t) => rankInOrder(
          rankItems(values.length, gateIdx.get(0.34).scores(t.q), denseOf(t), opts,
            gateIdx.get(0.34).scores(t.q, { informativeOnly: true }), gateIdx.get(0.34).coverage(t.q)),
          values, t.targets))),
        model: summarize(tests.map((t) => rankInOrder(
          rankItems(values.length, gateIdx.get(0.34).scores(t.q), modelOf(t), opts,
            gateIdx.get(0.34).scores(t.q, { informativeOnly: true }), gateIdx.get(0.34).coverage(t.q)),
          values, t.targets))),
      });
    }
    return rows;
  };

  console.log('\n  --- measured through the shipped fusion (search-core.js) ---');
  for (const tier of TIERS) {
    const tests = tier.tests.filter((t) => t.ds === key);
    if (!tests.length) continue;

    const denseRanks = tests.map((t) => rankOf(reference(t.q) ?? [], values, t.targets));
    const tabRanks = tests.map((t) => rankOf(denseScores(t.q), values, t.targets));
    const fusRanks = tests.map((t) => rankInOrder(fusedRanks(t.q), values, t.targets));
    const fusRanksAbs = tests.map((t) => rankInOrder(fusedRanksAbs(t.q), values, t.targets));
    const gateRows = GATE_FRACS.map((frac) => ({
      frac,
      table: summarize(tests.map((t) => rankInOrder(gated(frac, t.q, denseScores(t.q)), values, t.targets))),
      model: summarize(tests.map((t) => rankInOrder(
        gated(frac, t.q, reference(t.q) ?? values.map(() => 0)), values, t.targets))),
      // Weighted (score-normalized) fusion, same gate: a strong dense margin can
      // outrank a weak keyword hit, which RRF's flat 1/(60+rank) cannot.
      wtable: summarize(tests.map((t) => rankInOrder(
        gated(frac, t.q, denseScores(t.q), WEIGHT_OPTS), values, t.targets))),
      wmodel: summarize(tests.map((t) => rankInOrder(
        gated(frac, t.q, reference(t.q) ?? values.map(() => 0), WEIGHT_OPTS), values, t.targets))),
    }));
    // Same fusion, but with the EXACT embedder as the dense channel: if the
    // thresholded eligibility rule costs the model path the same ground, the loss
    // belongs to the rule, not to the table.
    const refFusRanks = tests.map((t) => rankInOrder(
      rankItems(values.length, bm25.scores(t.q), reference(t.q) ?? values.map(() => 0), RANK_OPTS),
      values, t.targets));
    const refFusAbsRanks = tests.map((t) => rankInOrder(
      rankItems(values.length, bm25.scores(t.q), reference(t.q) ?? values.map(() => 0), RANK_OPTS_ABS),
      values, t.targets));
    const scaleCheck = tests.map((t) => {
      const r = reference(t.q); if (!r) return null;
      const d = denseScores(t.q);
      let mr = -Infinity, md = -Infinity;
      for (let j = 0; j < values.length; j++) { mr = Math.max(mr, r[j]); md = Math.max(md, d[j]); }
      return { refMax: mr, tabMax: md };
    }).filter(Boolean);
    const agree = tests.filter((t) => {
      const r = reference(t.q);
      return r && topAgreement(r, denseScores(t.q));
    }).length;
    const cov = tests.reduce((a, t) => a + table.coverage(t.q), 0) / tests.length;

    console.log(`  ${tier.name} — table word coverage ${(cov * 100).toFixed(0)}%`);
    console.log(line('    reference: exact fp64 static embedder', summarize(denseRanks)));
    console.log(line('    reference: model + BM25, absolute cutoff', summarize(refFusAbsRanks)));
    console.log(line('    reference: model + BM25, relative cutoff', summarize(refFusRanks)));
    console.log(line('    artifact : int8 static table', summarize(tabRanks)));
    console.log(line('    artifact : table + BM25, ungated fusion', summarize(fusRanksAbs)));
    console.log(line('    artifact : table + BM25, ungated (rel cutoff)', summarize(fusRanks)));
    for (const row of gateRows) {
      console.log(line(`    artifact : RRF gated, df<=${row.frac.toFixed(2)}`, row.table));
      console.log(line(`    reference: RRF gated, df<=${row.frac.toFixed(2)}`, row.model));
    }
    console.log('    — sparse-channel weighting sweep (df<=0.34 gate) —');
    for (const row of covSweepFor(tests)) {
      console.log(line(`    artifact : ${row.label}`, row.table));
      console.log(line(`    reference: ${row.label}`, row.model));
    }
    const below = scaleCheck.filter((x) => x.refMax < RANK_OPTS.threshold).length;
    console.log(`    top-1 identical to reference: ${agree}/${tests.length}   ` +
      `queries whose best true cosine < threshold ${RANK_OPTS.threshold}: ${below}/${tests.length} ` +
      `(ref max ${(scaleCheck.reduce((a, x) => a + x.refMax, 0) / scaleCheck.length).toFixed(3)} / ` +
      `table max ${(scaleCheck.reduce((a, x) => a + x.tabMax, 0) / scaleCheck.length).toFixed(3)})`);
  }
}

async function main() {
  const embedder = await loadEmbedderTable(MODEL);
  console.log(`embedder ${MODEL}: "${embedder.name}" ${embedder.rows} x ${embedder.dim} ` +
    `(${(embedder.data.byteLength / 1048576).toFixed(1)} MB fp32)`);

  const names = DATASET_NAMES.length ? DATASET_NAMES : ['data', 'data-agencies'];
  for (const name of names) {
    const built = await buildTable(name, embedder);
    const out = join(ROOT, `${name}.semantic.json`);
    writeFileSync(out, JSON.stringify(built.artifact));
    const kb = (statSync(out).size / 1024).toFixed(1);
    const gz = (gzipSync(readFileSync(out)).length / 1024).toFixed(1);
    console.log(`\n  wrote ${name}.semantic.json — ${kb} KB on disk, ${gz} KB gzipped`);
    console.log('  replaces: 15.1 MB model + ~1 MB ONNX runtime + tokenizer, with no inference');
    verify(built);
  }
}

main().catch((err) => {
  console.error('build failed:', err.message || err);
  process.exit(1);
});
