# Plan: model-free semantic ranking (static semantic table)

Status: **implemented** — `scripts/build-semantic-table.mjs` emits
`data*.semantic.json` (`npm run build:tables`); `static-table.js` decodes it at
runtime; the component defaults to table mode when a `table` artifact is supplied.
Related: `ROADMAP-precompute.md` (the step this builds on), README
"Model-free semantic ranking".

## The move

Precomputed option vectors removed option-side inference. One dependency was left:
the **query**. For a static embedder (`potion-base-4M` = `embeddings F32
[29528, 128]`, mean-pooled, L2-normalized) the query side can be flattened too:

```text
cos(q, option_j) = ( Σ_i <E_i, V_j> ) / ‖ Σ_i E_i ‖
```

`‖ Σ_i E_i ‖` is constant across `j`, so it cannot change the ranking, and RRF
consumes ranks. Ship `S[i,j] = <E_i, V_j>` as int8 with a single global scale and
the dense channel becomes a word lookup plus an integer add: no ONNX runtime, no
WASM, no tokenizer, no inference.

| | before | after |
|---|---|---|
| dense bytes | 15.1 MB model + ~1 MB ORT + ~200 KB vocab | 348 KB (42 options) / 157 KB (15 options) |
| ready | after download | first keystroke, synchronously |
| failure mode | CDN blocked → keyword-only forever | none — it is data |

## Decisions taken, with the measurement behind each

- **Prune by score spread, not by frequency.** A word moves a ranking only through
  `max_j - min_j` of its row; a zero-spread word adds a constant to every option and
  provably cannot reorder. Keep the top `--keep`% (default 25). Side effect that
  surprised us: pruning *improved* paraphrase accuracy over the full table
  (9/14 vs 7/14 top-1 fused) because it also drops the generic words diluting the
  mean-pooled query vector.
- **int8 with ONE global scale.** A per-column scale rescales each option
  differently — that is NOT rank-preserving, and it initially looked like a quality
  win from quantization when it was a different ranking function. int8 measured exact
  against fp64 on both datasets; int4 measured exact too.
- **Whole-word rows only.** Subword continuations (`##ing`) are unreachable from a
  plain word lookup, so a row for one is dead weight. Cost: a query word that is not
  a vocabulary entry contributes nothing (the model would have decomposed it). Typo
  tolerance is the metaphone channel's job, so this is a deliberate division of labor.
- **Relative threshold is the new default.** An absolute 0.3 cosine cutoff silently
  evicted the dense channel's own top pick on 8/14 paraphrase agency queries.
  `thresholdMode: 'relative'` makes `threshold` a fraction of the query's best dense
  score — scale-free, and correct for table units where cosine units are meaningless.
- **Stop words are load-bearing at 42 documents.** `genericDfFrac` cannot identify
  "the" (df 8/42 = 19%, so it looks informative), and "sandbags before the hurricane"
  had its keyword ranking decided by the word "the" alone — Department of Defense
  above FEMA. A stop-word list in the tokenizer fixed it. This is the failure mode of
  IDF-style rarity on small corpora.
- **Term-coverage cliff beats the alternatives we tried** (df gate alone, weighted
  fusion, evidence-proportional `sparseConfidence`): tier 1 24/28 top-1 & 28/28
  top-3, tier 2 11/18 — the best joint result without a tuned weight per dataset.
  `sparseConfidence: 'coverage' | 'ramp'` remain available, measured but not default.
- **Eval logic lives in `search-core.js`.** The scripts import the same tokenizer,
  BM25 index and fusion the browser runs. Before this, the eval reimplemented fusion
  and drifted (its tie handling hid the paraphrase regression for a day).

## Phonetic modes: the follow-up, and a lesson about the harness

The commit that shipped the table reported a "phonetic-vs-exact regression"
(exact-only 37/46 vs union 33/46 top-1 across both tiers). **That number was wrong.**
Two bugs in `scripts/eval-no-model.mjs` manufactured it:

1. A query with no keyword match scores 0 everywhere, and competition ranking of an
   all-zero field returns rank 1 — so the exact-only channel was credited with
   answering 7 paraphrase queries it never answered.
2. `null <= 3` is true in JavaScript, so encoding "no evidence" as `null` made those
   same queries count as top-3 hits, and `1 / Math.min(rank, n)` on a 0 rank produced
   `MRR Infinity` in the output — a value that should have stopped the run.

Fixed by treating "no evidence" as its own bucket, excluded from top-1/top-3/MRR, and
labeled as such in the output. Honest tier-1 numbers: 23/28 top-1, 28/28 top-3 for
**all three phonetic modes** — the keyword channel has real signal there and the modes
do not differ. Honest tier-2 numbers: keyword-only is near-useless in every mode
(5-7/18 top-1 with 7-11 queries unanswered), which is the whole argument for the dense
channel.

So `'fallback'` shipped as the default for a different reason than a measured ranking
win — and the README says so rather than implying one. Union double-counts correctly
spelled words (`zero trust` 14.0 vs 7.2) and leaks phantom matches onto citation
queries (`FIPS 199` scoring 0.1 on three unrelated documents), on a corpus where 57%
of stems share a metaphone key. Fallback keeps identical typo tolerance
(10/14 top-1, 14/14 top-3) and identical fused ranking without that ambiguity.
`phoneticWeight: 0.5` measured the same again, so the weight is available but not
needed here.

## The metric bug that produced two wrong conclusions

`rankFromScores` used to compute rank as `1 + #{scores strictly greater}`, which for an
all-zero vector returns **1** — a channel that produced no evidence at all was credited
with a first-place answer. Same class as the `null <= 3` comparison bug in
`eval-no-model.mjs`. Both are now prevented structurally: `scripts/metrics.mjs` is the
only place that turns scores into ranks, it returns `null` for no-evidence, `summarize()`
excludes nulls from top-1/top-3/MRR and reports `noEvidence` as its own bucket, and the
report lines print it (e.g. `top1 7/14 ... (3 no evidence)`) so a channel resting on
unanswered queries is visible instead of flattering.

If a number in this project's history says "dense-only beats hybrid", assume this bug
until re-measured on the rendered basis.

## What is still open

0. **Re-benchmark `phonetic` per deployment.** All three modes tie on this corpus;
   they will not on one with heavier abbreviation or non-English morphology. A
   keyword-only deployment (`hybrid: false`) should compare `'off'` and `'fallback'`
   on its own query log, since `df == 0` is the shape of a typo *and* of a paraphrase.
1. **Grow tier 2 into judged data.** 18 hand-written paraphrase queries are directional.
   Until then, treat fusion numbers as a signal, not a gate — the README's own warning
   about hand-tuned fusion weights applies to anything chosen on this set.
2. **RESOLVED — hybrid does not lose on paraphrases; the measurement did.** Corrected
   on the rendered basis: dense-only 19/28 · 24/28 (tier 1) and 9/18 · 11/18 (tier 2)
   vs fused 24/28 · 28/28 and 11/18 · 12/18. The earlier "dense-only 10/14 top-1" was
   an all-zero tie scored as rank 1 — 3 of 14 agency paraphrases have 0% table word
   coverage, so the dense channel answered none of them. Remaining open question,
   narrower and honest: fused tier-2 top-3 (12/18) still trails the model's 14/18, so
   the keyword channel's value *on paraphrases* is unproven even though it no longer
   harms. One principled attempt remains listed in README ("Still open"), and then
   stop until there is judged data.
3. **Column-count growth.** One column per option: 42 → 348 KB, 200 → ~1.6 MB at the
   same pruning. Row clustering (word → prototype, `vocab → prototype` map plus a
   `prototypes × options` matrix) is the next compression lever; measure the Pareto
   curve before building it.
4. **Non-static embedders.** The flattening is exact only for mean-pooled static
   models. For a contextual model (`Xenova/all-MiniLM-L6-v2`) the same trick becomes
   an approximation — the token table is no longer the whole function. Worth stating
   in docs when someone asks; unmeasured.
5. **Threshold units in table mode.** Calibrated empirically (build reports p5..p95 of
   the fit). If `threshold` starts mattering for a deployment, replace the single
   constant with a per-query normalization.

## Reproduce

```bash
npm run build:tables      # writes data*.semantic.json, prints measured quality
npm run eval:table        # scores the artifacts through the shipped fusion
npm run eval:no-model     # keyword-only baseline, both tiers, zero dependencies
node scripts/smoke.mjs    # headless DOM wiring check (npm i --no-save jsdom)
```
