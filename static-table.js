// static-table.js
// Model-free dense channel: a build-time *static semantic table*.
//
// Why this works: the default embedder (minishlab/potion-base-4M) is a Model2Vec
// StaticModel — a lookup table of per-token vectors, mean-pooled and L2-normalized.
// The dense score is therefore a sum of dot products that can be precomputed:
//
//   cos(query, option_j) = < normalize(mean_i E_i), V_j >
//                        = ( Σ_i <E_i, V_j> ) / ‖ Σ_i E_i ‖
//
// The denominator is the same positive number for every option j, so it cannot
// change the RANKING — and RRF fusion (the component's default) consumes ranks,
// not scores. What ships is therefore just S[i,j] = <E_i, V_j>: one integer row
// per vocabulary word, one column per option. Query-time work is a word lookup
// and an integer add. No ONNX, no WASM, no tokenizer, no inference.
//
// The table is a projection of the embedder ONTO a specific option list, so it
// must be regenerated when the options change (`npm run build:tables`). It
// replaces the dense channel only: the BM25 sparse channel is unchanged and the
// two still fuse with RRF.
//
// Limit worth stating plainly: this does not remove the need for semantic
// KNOWLEDGE, only the need to run a model. Word → option affinities that the
// embedder knew are now data. A table built from corpus statistics alone cannot
// know that "sandbags" implies a disaster agency, because nothing in the corpus
// pairs them.
//
// Artifacts come from scripts/build-semantic-table.mjs.

/** Split a query into surface words for table lookup.
 *  Deliberately NOT the BM25 tokenizer: no stemming (the vocabulary rows are keyed
 *  on whole words, "records" and "record" are different entries) and no phonetic
 *  keys (typo tolerance is the sparse channel's job). */
export function tableWords(text) {
  return (text || '').toLowerCase().normalize('NFKD').replace(/[̀-ͯ]/g, '')
    .split(/[^a-z0-9]+/).filter((w) => w.length > 1);
}

function decodeBase64(b64) {
  if (typeof Buffer !== 'undefined') {
    const buf = Buffer.from(b64, 'base64');
    return new Int8Array(buf.buffer, buf.byteOffset, buf.byteLength);
  }
  const bin = atob(b64);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return new Int8Array(out.buffer);
}

/**
 * Decode an artifact from scripts/build-semantic-table.mjs.
 * @returns {{
 *   options: string[], words: string[], rows: Int8Array,
 *   scores(query: string): Float64Array, stats: object, ready: true
 * }}
 */
export function decodeTable(artifact) {
  if (!artifact || artifact.format !== 'combobox-semantic-table/1') {
    throw new Error(`static-table: unknown artifact format ${artifact && artifact.format}`);
  }
  const nOptions = artifact.options.length;
  const words = artifact.words.split('\n');
  if (words.length !== artifact.rowCount) {
    throw new Error(`static-table: ${words.length} words vs rowCount ${artifact.rowCount}`);
  }
  const rows = decodeBase64(artifact.scores);
  if (rows.length !== words.length * nOptions) {
    throw new Error(`static-table: matrix is ${rows.length} bytes, expected ${words.length * nOptions}`);
  }
  const index = new Map();
  words.forEach((w, i) => index.set(w, i));

  const out = new Float64Array(nOptions);

  return {
    format: artifact.format,
    embedder: artifact.embedder,
    options: artifact.options,
    words,
    rows,
    scale: artifact.scale,
    stats: artifact.stats || {},
    ready: true,

    /** Number of query words the table can actually score (diagnostic: this is
     *  what pruning costs — unseen words contribute nothing). */
    coverage(query) {
      const ws = tableWords(query);
      if (!ws.length) return 0;
      let hit = 0;
      for (const w of ws) if (index.has(w)) hit += 1;
      return hit / ws.length;
    },

    /**
     * Dense score per option, in approximately cosine units (calibrated at build
     * time; ranking is exact, the absolute value is only used by `threshold`).
     */
    scores(query) {
      out.fill(0);
      const ws = tableWords(query);
      let hit = 0;
      for (const w of ws) {
        const r = index.get(w);
        if (r === undefined) continue;
        hit += 1;
        const base = r * nOptions;
        for (let j = 0; j < nOptions; j++) out[j] += rows[base + j];
      }
      if (!hit) return new Float64Array(nOptions).fill(0);
      // Divide by the number of matched words: the reference model mean-pools, and
      // the (constant) divisor does not change the ranking, but it keeps the value
      // comparable across queries of different length for the threshold.
      for (let j = 0; j < nOptions; j++) out[j] = (out[j] / hit) * artifact.scale;
      return out;
    },
  };
}
