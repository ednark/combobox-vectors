# combobox-vectors

A combo box whose filtering is driven by semantic vector search instead of
substring typeahead. Users type what they *mean*; the component ranks options
by what they *are*.

The gap it closes: a FOIA request form asks the user to pick an agency from
forty. The user's mental model is *"who has my medical records?"* — the label
is *"Department of Veterans Affairs."* Substring typeahead cannot bridge that
gap. There is no substring of one in the other. This component can, because it
ranks by vector similarity between the query and text the user never sees.

Framework-agnostic: keep any combo box's existing markup, styling, and
accessibility semantics; replace only the filtering controller. The demo uses
the U.S. Web Design System's combo box classes, but the controller swap works
against any combo box that follows the same `input` + `list` + `select`
pattern.

## The core idea: vectorize the hidden text

The trick is not embedding the option label — the label is the least
informative text an option has. Each option carries a `vectorText` field:
a description of what the option actually *is*, what data it collects, what
it handles. For the FOIA selector, that is a paragraph describing each
agency's jurisdiction and typical request categories. For a policy-document
picker, it is the document's summary and keyword list.

Both the `vectorText` and the user's query pass through the same model, so
they land in the same vector space. Ranking is a dot product. That is the
whole mechanism.

```js
{ value: "dva", text: "Department of Veterans Affairs (VA)",
  vectorText: "VA provides healthcare, education benefits, disability
    compensation... FOIA requests often involve disability evaluations,
    mental health care access..." }
```

## Why not just vectors? The hybrid problem

Pure dense retrieval has a blind spot. Ask for "October" and semantic ranking
has nothing to work with — but one option is titled *"Executive Order 14110
(October 2023)."* The query is a lexical fact, not a semantic one.

The solution is boring and well-established: run both channels.

- **Dense** — cosine similarity between the query embedding and each option's
  `vectorText` embedding. Captures topical similarity: "hackers stealing
  data" → `OMB M-22-09`.
- **Sparse** — BM25 with IDF weighting over the title (boosted ×3) and
  `vectorText`. Captures exact tokens the dense channel is blind to: "October"
  → `EO 14110`. Each token is also indexed as a **metaphone phonetic key**
  (`phonetic: true`, default), so typo'd queries ("recods", "secuirty") still
  collide with the intended term — measured at typo top-1 6/14 → 10/14, top-3
  12/14 → 14/14, clean queries no regression.

The two ranked lists are fused with **Reciprocal Rank Fusion** (`1/(60 + rank)`
per channel — the same default Elasticsearch 8, OpenSearch, and Azure AI
Search use for hybrid queries). Rank-based fusion needs no score
normalization: BM25 is unbounded, cosine is 0..1, and RRF sidesteps the
mismatch entirely.

The sparse channel is always live; the dense channel is a progressive
enhancement. An option qualifies if its vector score clears a threshold *or*
it has any keyword match — so "October" surfaces even when its semantic score
is near zero, and a purely semantic query is never blocked by the absence of a
token match.

## Choosing the model: a narrower fitness measure

The default model is `minishlab/potion-base-4M` — a Model2Vec "static"
embedder: a distilled embedding table with mean pooling and L2 normalization,
no transformer layers at all. 15.1 MB fp32 ONNX, 128-dim, inference is a single
embedding lookup.

That choice came from an evaluation with a deliberately narrow fitness
measure: the model does not need to be good at embedding in general, it needs
to correctly distinguish *the options in the list* from each other. On that
corpus-relative test — a battery of intent queries with known intended targets
across the two demo datasets — the 15.1 MB static model beat the 17.5 MB
contextual transformer it replaced (top-1 25/28 vs 24/28, top-3 28/28 vs
27/28). A 2M-parameter variant at 7.6 MB measurably degraded (top-1 20/28),
which is where the size/quality frontier sits for this corpus.

Override `model` with any Transformers.js feature-extraction id (e.g.
`Xenova/all-MiniLM-L6-v2`) for contextual embeddings when the option list
demands it.

## The 15 MB problem

The model is 15 MB. For a combo box. Browser cache partitioning means the
download is not shared across origins — each site re-pays the cost. The
mitigations are all variations on one principle: **the dense channel is a
progressive enhancement, never a dependency.**

- **BM25 runs unconditionally.** The sparse index builds synchronously at init.
  The combo box is fully functional on first paint, not after model load.
- **Results re-rank transparently.** Keyword results render immediately; when
  the model finishes loading, an open list is re-ranked with RRF fusion.
- **The dense channel is opt-out by design.** Disabled through configuration,
  it short-circuits to BM25-only and skips model warmup entirely — a page
  deployed with vectors off downloads zero model bytes.
- **Failure degrades, never breaks.** If the CDN is blocked or the user is
  offline, the component settles permanently into keyword-only mode rather
  than an eternally-loading dropdown.

**Strong fit:** high-frequency internal tools (15 MB amortizes to zero
across daily sessions); opaque-label domains (program codes, legal
citations, clinical terminology); privacy or compliance boundaries (HIPAA,
CJIS, air-gapped networks — local inference keeps the data path on-device);
zero-backend deployments (static sites, offline-first PWAs).

**Poor fit:** single-visit public pages (taxing the 99% who won't return);
small option sets (below ~10, users scan the full list faster than they can
type); bandwidth-constrained contexts.

## Build-time precomputed vectors

For larger datasets, option embeddings can be precomputed at build time and
shipped as JSON — only the query is embedded at runtime. The same
`embedMany()` function powers the browser and the build script, so
build/runtime model parity holds by construction (verified at 0.0 max
component difference). This is how the demo's `*.embeddings.json` files are
generated:

```bash
npm run precompute    # writes data.embeddings.json, data-agencies.embeddings.json
```

## Usage

```html
<label for="cb">Find a policy document</label>
<div class="combo-box" id="cb-box">
  <select id="cb" name="cb"></select>
</div>

<script type="module">
  import { createVectorComboBox } from './vector-combobox.js';
  import { DATA } from './data.js';

  createVectorComboBox(document.getElementById('cb-box'), {
    data: DATA,                       // [{ value, text, vectorText }, ...]
    placeholder: 'Search by topic…',
    threshold: 0.25,
    maxResults: 15,
    onChange: (value, item) => {},
  });
</script>
```

## Run the demo

```bash
python3 -m http.server 8000    # → http://localhost:8000/
```

The demo includes two side-by-side combo boxes: a 15-document policy picker
and a 42-agency FOIA selector, with clickable example queries and a vector
search toggle for A/B testing dense vs keyword-only ranking.

## Configuration

| option            | default                                                    | meaning                                              |
|-------------------|------------------------------------------------------------|------------------------------------------------------|
| `data`            | `[]`                                                       | Array of `{ value, text, vectorText }`              |
| `model`           | `'minishlab/potion-base-4M'`                               | Embedding model (Model2Vec via ORT, or any Transformers.js feature-extraction id) |
| `threshold`       | `0.3`                                                      | Minimum cosine similarity for an option to show     |
| `maxResults`      | `10`                                                       | Max options rendered in the dropdown                |
| `debounceMs`      | `150`                                                      | Debounce for re-ranking after a keystroke           |
| `phonetic`        | `true`                                                     | Index metaphone phonetic keys alongside exact tokens (typo tolerance) |
| `hybrid`          | `true`                                                     | Fuse BM25 keyword matching with vector similarity   |
| `fusion`          | `'rrf'`                                                    | `'rrf'` (rank-based, no tuning) or `'weighted'` (score-based) |
| `rrfK`            | `60`                                                       | RRF rank constant                                   |
| `lexicalWeight`   | `0.5`                                                      | BM25 weight (only when `fusion: 'weighted'`)        |
| `placeholder`     | `''`                                                       | Input placeholder                                   |
| `onChange`        | `null`                                                     | `onChange(value, item)` callback                    |

## API

`createVectorComboBox(boxEl, options)` → `{ el, select, input, setQuery, setDenseEnabled, setUsePrecomputed, refresh, destroy }`

- `setQuery(query)` — programmatically fill the input and rank (as if typed)
- `setDenseEnabled(bool)` — toggle the dense channel on/off (BM25 stays live)
- `setUsePrecomputed(bool)` — switch between build-time and on-load vectors

## Keyboard

`↓`/`↑` move focus · `Enter` select · `Esc` close & revert · `Home`/`End` jump

## Files

- `vector-combobox.js` — the reusable ES module (imports `metaphone.js` +
  Transformers.js at runtime; no bundler)
- `metaphone.js` — phonetic encoder (ported from govdoc-explainer)
- `data.js` / `data-agencies.js` — demo datasets
- `data.*.embeddings.json` — precomputed vectors (`npm run precompute`)
- `scripts/precompute.mjs` — build-time embedding index generation
- `scripts/eval-embedders.mjs` — corpus-relative model evaluation
- `scripts/eval-metaphone.mjs` — phonetic tokenizer evaluation

## What's still open

- **Static-model frontier**: potion-2M degraded (20/28), potion-4M was the
  sweet spot. Where exactly does discrimination break, and does it break
  differently for different option-list domains? Answerable with per-domain
  evaluation data, not a general benchmark.
- **Fusion tuning**: RRF with k=60 is deliberately boring. Channel weighting
  is worth revisiting per-domain, but only with relevance-judged evaluation
  data — hand-tuned fusion weights are where search systems decay silently.
- **Scaling**: BM25 builds synchronously and the fused re-rank is O(n) per
  keystroke — instant at 42, fine at a few hundred, worth profiling at a
  thousand. Beyond that, the sparse channel needs an inverted index and the
  dense channel might need approximate nearest-neighbor search.
