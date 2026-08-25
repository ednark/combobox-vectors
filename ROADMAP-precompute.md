# Plan: build-time precomputed vectors

Status: **implemented** — `scripts/precompute.mjs` generates
`*.embeddings.json` indexes (`npm run precompute`); the client attaches them
via the `embedding` field on data items and only embeds the query at runtime.
The notes below were the original design; kept for context.

## What shipped

- `scripts/precompute.mjs` — batches all `vectorText`s per dataset into ONE
  pipeline call (vs. the client's sequential loop), validates L2 norms ≈ 1
  and dim = 384, writes `data.embeddings.json` / `data-agencies.embeddings.json`.
- `vector-combobox.js` — items with an `embedding` field skip runtime
  embedding; missing items fall back to runtime embedding. The pipeline is
  still warmed at init because the query always needs runtime inference.
- `index.html` — imports the JSON indexes (import attributes) and merges
  them onto the data arrays.
- Parity verified: recomputing an item's embedding at runtime matches the
  precomputed vector exactly (max component diff 0.0 — same model, same
  pooling, same dtype).

## Motivation

- Cut first-open latency (currently: download ~17 MB model + embed 15
  `vectorText`s before the first ranked result renders).
- Reduce client compute/battery on low-end devices.
- Allow larger datasets (hundreds of options) without a slow first open.

## Design

### 1. Build-time step (script, run when `data.js` changes)

A small Node script (`scripts/precompute.js`, to be written) that:

1. Reads the curated dataset (the same source array used to generate
   `data.js`).
2. Embeds each option's `vectorText` with the **same model** the client uses
   for the query — `Xenova/paraphrase-MiniLM-L3-v2` (int8 ONNX) via
   Transformers.js in Node — with `pooling: 'mean', normalize: true`.
   Model parity is a hard requirement: mixing models (e.g. govdoc-explainer's
   build-time fastembed `all-MiniLM-L6-v2` with a different client model)
   produces incompatible vector spaces and silently garbage similarity.
3. Writes `data.precomputed.js` (or augments `data.js`) with each item as:
   `{ value, text, vectorText, embedding: number[384] }` (L2-normalized, so
   client-side cosine = dot product).

### 2. Client changes (vector-combobox.js)

- Accept precomputed `embedding` on data items; skip runtime embedding for any
  item that has one.
- Only the **query** embedding is computed in the browser.
- Keep `refresh()` semantics: `refresh()` re-embeds items missing/fresh
  `vectorText` only.

### 3. Fallbacks

- If an item lacks `embedding`, embed it on first open (current MVP behavior).
- If the model is swapped via the `model` option, treat all precomputed
  vectors as invalid (dimension/source mismatch) and re-embed at runtime.

## Validation

- Assert precomputed vectors are L2-normalized (norm ≈ 1) and 384-dim.
- Spot-check a few known queries (the demo queries in `data.js` comments) rank
  the same with precomputed vs runtime vectors (should be identical — same
  model, same pooling).

## Notes

- govdoc-explainer precedent: `govdoc_explainer/embeddings.py` uses fastembed
  (`sentence-transformers/all-MiniLM-L6-v2`) at build time and the Xenova JS
  port client-side; same-model pairing is what keeps the spaces compatible.
- A batched build script embedding N items in one pipeline call is ~N times
  faster than N single calls; the current MVP embeds items sequentially in a
  loop and could also batch for faster first-open.
