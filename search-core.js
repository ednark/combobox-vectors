// search-core.js
// The pure retrieval logic shared by the component and the evaluation scripts.
//
// Extracted from vector-combobox.js so that an eval measures the ranking the
// library actually performs, rather than a re-implementation that can silently
// drift from it. No DOM, no model, no dependencies: a caller supplies the two
// channels (a BM25 index and, optionally, an array of dense scores) and gets
// ranked item indices back.
//
// Channels:
//   sparse — BM25 over title(boosted) + vectorText, optionally with metaphone
//            phonetic keys. Always available, builds synchronously.
//   dense  — either a model's cosine similarities (vector-combobox.js +
//            Transformers.js) or a build-time static semantic table
//            (static-table.js, no inference). Both are just numbers per option.

import { metaphone } from './metaphone.js';

// English function words. Needed because IDF-style rarity does NOT identify
// stopwords at this corpus size: in 42 options, "the" appears in only 8 of them,
// so a df-fraction gate calls it informative — and a query like "sandbags before
// the hurricane" then gets its keyword ranking decided by the word "the" alone
// (measured: it put the Department of Defense above FEMA). Production BM25 hides
// this because df("the") approaches 1 in a real corpus. Filtering them from BOTH
// the query and the documents is the honest fix at 15-100 options.
const STOPWORDS = new Set([
  'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'but', 'by', 'can', 'cannot',
  'could', 'did', 'do', 'does', 'for', 'from', 'get', 'got', 'had', 'has', 'have',
  'he', 'her', 'here', 'him', 'his', 'how', 'i', 'if', 'in', 'into', 'is', 'it',
  'its', 'me', 'might', 'more', 'most', 'must', 'my', 'no', 'not', 'of', 'on', 'or',
  'our', 'out', 'over', 'she', 'should', 'so', 'some', 'than', 'that', 'the', 'their',
  'them', 'then', 'there', 'these', 'they', 'this', 'those', 'to', 'too', 'under',
  'up', 'us', 'was', 'we', 'were', 'what', 'when', 'where', 'which', 'who', 'why',
  'will', 'with', 'would', 'you', 'your',
]);

function stem(w) {
  // Naive suffix stripper (NOT Porter) — enough for plural/verb variants on a
  // small corpus: "records"->"record", "publishing"->"publish".
  if (w.length > 5 && w.endsWith('ing')) return w.slice(0, -3);
  if (w.length > 4 && w.endsWith('ed')) return w.slice(0, -2);
  if (w.length > 4 && w.endsWith('es')) return w.slice(0, -2);
  if (w.length > 3 && w.endsWith('s') && !w.endsWith('ss')) return w.slice(0, -1);
  return w;
}

export function tokenize(text) {
  return (text || '')
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((w) => w.length > 1)
    .map(stem)
    .filter((w) => !STOPWORDS.has(w));
}

// Union tokenizer: exact stemmed tokens PLUS metaphone phonetic keys. Exact
// tokens keep their IDF precision; phonetic keys let typo'd query tokens
// still collide with the intended term. (Evaluated in
// scripts/eval-metaphone.mjs — typo top-3 14/14, clean no regression.)
export function tokenizePhonetic(text) {
  const toks = tokenize(text);
  return [...toks, ...toks.map(metaphone)];
}

// --- BM25 sparse retrieval channel (the classic keyword-matching ranker) ---
// Sparse counterpart to the dense embedding: IDF-weighted exact-term matching
// over each option's title (boosted) + vectorText. Catches dates, numbers,
// and proper nouns the dense channel can't see ("October" -> "(October 2023)").
// ~40 lines, no dependencies.

// The tokenizer is injectable so evaluations (scripts/eval-metaphone.mjs) and
// downstream users can subclass with e.g. phonetic keys while keeping the
// IDF/BM25 machinery identical.
export class BM25Index {
  constructor(items, { k1 = 1.5, b = 0.75, titleBoost = 3, genericDfFrac = 0.34, tokenize: tokenizeFn = tokenize } = {}) {
    this.k1 = k1;
    this.b = b;
    this.titleBoost = titleBoost;
    // A token appearing in more than this fraction of the options carries almost
    // no discriminative information ("before", "at", "behind", "record"). Used by
    // scores(query, { informativeOnly: true }) to gate the sparse channel.
    this.genericDfFrac = genericDfFrac;
    this._tokenize = tokenizeFn;
    this.tfs = [];
    this.dls = [];
    this.df = new Map();
    for (const item of items) {
      // Title tokens repeated `titleBoost` times so title hits dominate.
      const title = this._tokenize(item.text);
      const tokens = [];
      for (let r = 0; r < this.titleBoost; r++) tokens.push(...title);
      tokens.push(...this._tokenize(item.vectorText || item.text));
      const tf = new Map();
      for (const t of tokens) tf.set(t, (tf.get(t) || 0) + 1);
      for (const t of tf.keys()) this.df.set(t, (this.df.get(t) || 0) + 1);
      this.tfs.push(tf);
      this.dls.push(tokens.length);
    }
    this.N = items.length;
    this.avgdl = this.dls.reduce((a, c) => (a + c), 0) / (this.N || 1);
  }

  /**
   * Raw BM25 scores of `query` against every indexed item.
   * @param {string} query
   * @param {{ informativeOnly?: boolean }} [opts] with informativeOnly, query
   *        tokens that appear in more than `genericDfFrac` of the options
   *        contribute nothing, so matching only generic words scores 0.
   */
  scores(query, { informativeOnly = false } = {}) {
    const out = new Array(this.N).fill(0);
    const qTokens = [...new Set(this._tokenize(query))];
    if (!qTokens.length) return out;
    const dfCeiling = this.genericDfFrac * this.N;
    for (const t of qTokens) {
      const df = this.df.get(t) || 0;
      if (!df) continue;
      if (informativeOnly && df > dfCeiling) continue;
      const idf = Math.log(1 + (this.N - df + 0.5) / (df + 0.5));
      for (let i = 0; i < this.N; i++) {
        const tf = this.tfs[i].get(t) || 0;
        if (!tf) continue;
        const denom = tf + this.k1 * (1 - this.b + (this.b * this.dls[i]) / this.avgdl);
        out[i] += idf * ((tf * (this.k1 + 1)) / denom);
      }
    }
    return out;
  }
  /**
   * Per-item fraction of the query's INFORMATIVE tokens that appear in the item.
   * Scale-free lexical evidence: a query whose meaningful words are all present in
   * one option is a keyword query; a query where only one of three words appears is
   * not, however good that single match's BM25 score looks in isolation.
   * @returns {number[]|null} coverage per item, or null when the query has no
   *          informative tokens at all — nothing for the sparse channel to assert,
   *          which is the normal shape of a paraphrase
   */
  coverage(query) {
    const informative = this.informativeTokens(query);
    if (!informative.length) return null;
    const out = Array(this.N);
    for (let i = 0; i < this.N; i++) {
      let hit = 0;
      for (const t of informative) if (this.tfs[i].get(t)) hit += 1;
      out[i] = hit / informative.length;
    }
    return out;
  }

  // Query tokens that exist in the index and are rare enough to carry information.
  informativeTokens(query) {
    const dfCeiling = this.genericDfFrac * this.N;
    return [...new Set(this._tokenize(query))].filter((t) => {
      const df = this.df.get(t) || 0;
      return df > 0 && df <= dfCeiling;
    });
  }
}

// Keyword-only ranking (no download required). Used before the dense channel is
// ready and permanently if it fails.
export function bm25Ranked(count, lexRaw, maxResults) {
  const hits = [];
  for (let i = 0; i < count; i++) if (lexRaw[i] > 0) hits.push({ i, lex: lexRaw[i] });
  hits.sort((a, b) => b.lex - a.lex);
  return hits.slice(0, maxResults).map((x) => x.i);
}

// How much the sparse channel is credited for a given query, from the fraction of
// the query's informative tokens it can account for (maxCoverage, 0..1):
//   'coverage' — credit = maxCoverage (linear, no cliff)
//   'ramp'     — credit = (maxCoverage - lo) / (hi - lo), clamped to 0..1. A query
//                the keyword channel explains barely at all contributes nothing,
//                so the dense channel stands alone; a keyword query gets full credit.
//   anything else — credit = 1 (the historical always-on hybrid)
function sparseCredit(o, sparseCoverage) {
  if (!sparseCoverage || !o.sparseConfidence) return 1;
  const maxCov = Math.max(...sparseCoverage);
  if (o.sparseConfidence === 'coverage') return maxCov;
  if (o.sparseConfidence === 'ramp') {
    const [lo, hi] = o.sparseRamp || [0.5, 1];
    if (hi <= lo) return maxCov >= hi ? 1 : 0;
    return Math.max(0, Math.min(1, (maxCov - lo) / (hi - lo)));
  }
  return 1;
}

/**
 * Rank options from the two channels.
 *
 * @param {number} count        number of options
 * @param {number[]} lexRaw     BM25 score per option (0 = no keyword match)
 * @param {number[]|null} denseRaw dense score per option, or null when the dense
 *                              channel is disabled/unavailable -> keyword-only
 * @param {object} o            { threshold, maxResults, hybrid, fusion, rrfK, lexicalWeight,
 *                                thresholdMode }
 * @param {number[]} [lexRankRaw] scores used to RANK the sparse channel; defaults
 *        to `lexRaw`. Pass informative-token-only scores so a query's generic
 *        words ("before", "at", "behind") cannot buy RRF rank credit and drag a
 *        clear semantic match down the list. Ungated, fusion measured WORSE than
 *        dense alone on the paraphrase tier (table 10/14 -> 6/14 top-1).
 * @param {number[]|null} [sparseCoverage] per-item fraction of the query's
 *        informative tokens present in the item (index.coverage(query)). Items
 *        below `o.sparseMinCoverage` get no sparse rank credit, and when NO item
 *        clears it the sparse channel is skipped for that query entirely — that is
 *        a paraphrase, and the keyword channel has no evidence to contribute.
 * @returns {number[]} option indices, best first
 *
 * Eligibility: a dense hit at or above `threshold` OR any keyword match, so
 * "October" surfaces even when its semantic score is near zero, and a purely
 * semantic query is never blocked by the absence of a token match.
 *
 * `thresholdMode` decides what `threshold` is measured against:
 *   'relative' (default) — a fraction of the query's BEST dense score. Scale-free,
 *      so the same value works for cosine, for a static table's calibrated units,
 *      and across embedders. An absolute cosine cutoff silently evicts the dense
 *      channel's own top pick whenever a query has only weak affinity for every
 *      option — which is exactly the paraphrase case the dense channel exists for
 *      (measured: 8/14 paraphrase agency queries have best true cosine < 0.3).
 *   'absolute' — the historical behaviour: `threshold` compared against the raw
 *      dense score. Only meaningful when the dense channel really is cosine.
 */
export function rankItems(count, lexRaw, denseRaw, o, lexRankRaw = null, sparseCoverage = undefined) {
  const sparseSrc = lexRankRaw ?? lexRaw;
  const hasDense = denseRaw !== null && denseRaw !== undefined;
  // The coverage cliff exists to protect semantic evidence from weak lexical
  // noise, so it applies only when a dense channel is present. In keyword-only
  // mode there is nothing to protect: demoting a partial match would just hide
  // options the user could still have picked.
  let gated = sparseSrc;
  if (hasDense && o.sparseMinCoverage > 0) {
    // sparseCoverage === null: the index found no informative token in this query,
    // so the keyword channel has nothing to assert. (undefined = caller opted out of
    // gating, i.e. the historical always-on hybrid.)
    if (sparseCoverage === null) {
      gated = Array(count).fill(0);
    } else if (sparseCoverage) {
      let anyClears = false;
      for (let i = 0; i < count; i++) if (sparseCoverage[i] >= o.sparseMinCoverage) anyClears = true;
      if (!anyClears) {
        gated = Array(count).fill(0); // pure paraphrase: no keyword evidence
      } else {
        gated = sparseSrc.map((score, i) => (sparseCoverage[i] >= o.sparseMinCoverage ? score : 0));
      }
    }
  }
  if (!hasDense) {
    return bm25Ranked(count, gated, o.maxResults);
  }
  const items = [];
  let denseMax = -Infinity;
  let denseAny = false;
  for (let i = 0; i < count; i++) {
    items.push({ i, s: denseRaw[i], lex: lexRaw[i] });
    if (denseRaw[i] > 0) denseAny = true;
    if (denseRaw[i] > denseMax) denseMax = denseRaw[i];
  }
  // No dense signal at all (e.g. every query word was pruned from the table):
  // keyword-only, rather than ranking noise from an all-zero channel.
  if (!denseAny) return bm25Ranked(count, lexRaw, o.maxResults);

  const cutoff = o.thresholdMode === 'absolute' || denseMax <= 0
    ? o.threshold
    : o.threshold * denseMax;
  const eligible = items.filter((x) => x.s >= cutoff || x.lex > 0);

  if (o.fusion === 'weighted') {
    // Score-based fusion (needs per-query normalization of BM25).
    // Evidence-proportional weighting: the keyword channel is credited in terms of
    // how much of the query's informative vocabulary it can explain (maxCoverage).
    // A keyword query explains all of it (alpha 1) and behaves like the historical
    // hybrid; a paraphrase explains a third of it, so the keyword channel stops
    // outranking a clear semantic match. No threshold cliff to tune.
    const alpha = sparseCredit(o, sparseCoverage);
    let lexMax = 0;
    for (const x of eligible) if (gated[x.i] > lexMax) lexMax = gated[x.i];
    return eligible
      .map((x) => ({ ...x, combined: x.s + alpha * o.lexicalWeight * (lexMax > 0 ? gated[x.i] / lexMax : 0) }))
      .sort((a, b) => b.combined - a.combined)
      .slice(0, o.maxResults)
      .map((x) => x.i);
  }

  // Reciprocal Rank Fusion: rank-based, no normalization or tuning.
  // Dense ranks over ALL items (the dense score is always defined); sparse ranks
  // only over items with BM25 > 0 (others contribute no term).
  const denseRank = new Map(
    [...items].sort((a, b) => b.s - a.s).map((x, pos) => [x.i, pos + 1])
  );
  const sparseRanked = items.filter((x) => gated[x.i] > 0)
    .sort((a, b) => gated[b.i] - gated[a.i]);
  const sparseRank = new Map(sparseRanked.map((x, pos) => [x.i, pos + 1]));
  // Same evidence-proportional weighting for rank-based fusion: scale the sparse
  // RRF term by the fraction of the query's informative tokens the keyword channel
  // can account for. RRF's flat 1/(k+rank) otherwise hands a partial generic match
  // the same credit as a full one.
  const sparseAlpha = sparseCredit(o, sparseCoverage);
  return eligible
    .map((x) => {
      let rrf = 0;
      if (denseRank.has(x.i)) rrf += 1 / (o.rrfK + denseRank.get(x.i));
      if (sparseRank.has(x.i)) rrf += sparseAlpha * (1 / (o.rrfK + sparseRank.get(x.i)));
      return { ...x, combined: rrf };
    })
    .sort((a, b) => b.combined - a.combined)
    .slice(0, o.maxResults)
    .map((x) => x.i);
}
