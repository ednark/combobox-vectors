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
  → `EO 14110`. A token can also be looked up by its **metaphone phonetic key**, so
  typo'd queries ("recods", "secuirty") still collide with the intended term —
  measured at typo top-1 6/14 → 10/14, top-3 12/14 → 14/14, clean queries no
  regression. How the keys are consulted is configurable, and the modes are not
  equivalent (see `phonetic` below and "Phonetic modes").

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

- **A static table replaces the model entirely.** See the next section: ~200–350 KB
  of JSON, no inference, correct on the first keystroke. This is now the demo's
  default dense source.
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

## Model-free semantic ranking: the static semantic table

The default embedder is not a neural network at runtime. `minishlab/potion-base-4M`
is a lookup table — `embeddings F32 [29528, 128]`, 14.4 MB of rows, mean-pooled and
L2-normalized. And because the option vectors are already precomputed, the entire
dense channel can be flattened algebraically:

```text
cos(query, option_j) = < normalize(mean_i E_i), V_j >
                     = ( Σ_i <E_i, V_j> ) / ‖ Σ_i E_i ‖
                     └──── per-word, per-option: precomputable ────┘
```

The denominator is the same positive number for every option `j`, so it cannot
change the ranking — and RRF fusion consumes ranks. What ships is therefore a
`words × options` matrix of integers: **no ONNX runtime, no WASM, no tokenizer, no
inference.** Query-time work is a word lookup and an integer accumulation.

```bash
npm run build:tables    # scripts/build-semantic-table.mjs — zero dependencies
```

Pruning: a word can only move a ranking through the **spread** of its score row
(`max_j - min_j`). A word whose spread is 0 adds a constant to every option and
provably cannot change an order, so the build keeps the highest-spread words and
drops the rest — 25% of the vocabulary by default (`--keep`). Measured cost is
coverage, not correctness: a query whose surviving words are absent from the table
scores nothing in the dense channel and falls back to keywords.

Measured on the two demo datasets, 46 queries (component defaults, `npm run eval:table`):

| dense source                  | bytes                 | tier 1 top-1 | tier 1 top-3 | tier 2 top-1 | tier 2 top-3 |
|-------------------------------|-----------------------|--------------|--------------|--------------|--------------|
| none (BM25 only) | 0 | 23/28 | 28/28 | 7/18 | 9/18 |
| static table only | 348 KB + 157 KB | 19/28 | 24/28 | 9/18 | 11/18 |
| **table + BM25 (shipped default)** | ~505 KB | **24/28** | **28/28** | **11/18** | **13/18** |
| 15 MB model + BM25 | ~16.3 MB | 24/28 | 27/28 | 10/18 | 14/18 |

The static table matches the model it replaces on the in-vocabulary tier and beats it
on the paraphrase tier, at roughly 1/45th the bytes — because spread-pruning also
removes the generic words that were diluting the mean-pooled query vector. int8
quantization is exact for ranking, and int4 measured exact too.

**What this does not do.** It does not remove the need for semantic *knowledge*, only
the need to *compute* it: word→option affinities the embedder knew are now data. A
table derived from corpus statistics alone cannot know that "sandbags" implies a
disaster-response agency, because nothing in the corpus pairs them — that knowledge
has to arrive from a pretrained embedder (any size), from hand-authored paraphrases,
or from a server. Two structural consequences:

- The table is a projection onto **one option list**. Add, remove or reorder options
  and rebuild (`npm run build:tables`); the eval refuses a stale artifact rather than
  ranking against misaligned columns.
- Query-time scores are not cosine. Ranking is unaffected (RRF uses ranks), but
  `threshold` filters on magnitude, so the build calibrates a single scale factor and
  reports its spread. Use `thresholdMode: 'relative'` (default) and it stops mattering.

## Phonetic modes

Metaphone keys are four characters long, so many different words share one. In the
demo corpora that is not a corner case: **57% of the distinct stems in `data.js`
live in a key bucket shared with another stem** (45% in `data-agencies.js`), and the
worst buckets are exactly what this component is for — `22, 09, 2022, 199, 800, 53,
2002` collapse together, and `international, interview, undercover, interagency,
internal, interstate, interception` share a key.

Three modes, measured on the same corpora:

| mode | what it does | typo top-1 / top-3 | clean top-1 | fused tier 1 | fused tier 2 |
|---|---|---|---|---|---|
| `'off'` | exact stemmed tokens only | 6/14 · 12/14 | 8/10 | 24/28 · 28/28 | 10/18 · 12/18 |
| `'union'` (historical `true`) | keys indexed beside exact tokens, in documents **and** queries | 10/14 · 14/14 | 8/10 | 24/28 · 28/28 | 11/18 · 13/18 |
| **`'fallback'`** (default) | exact tokens rank; a key is consulted **only** when the exact spelling is absent from the index (`df == 0`) | 10/14 · 14/14 | 8/10 | 24/28 · 28/28 | 11/18 · 12/18 |

Read honestly: on this corpus **the ranking is the same for all three**, so switching
the default is not a quality win and is not claimed as one. It is a correctness and
robustness change. Union double-counts every correctly spelled word — its own key
fires too — which shows up directly in scores even when the order survives:
`zero trust` scores 14.0 under union vs 7.2 under fallback/exact, and `FIPS 199`
leaks a 0.1 onto Section 508, OMB M-19-21 and SP 800-63-4 under union while
fallback/exact match one document. That is phantom evidence for a citation query, and
a corpus with more abbreviations or a different language has more of it. Fallback keeps
typo tolerance (its premise is `df == 0`, which is what a typo looks like) without
making correctly spelled words ambiguous, and it keeps the term-coverage gate honest:
a key hit only counts as lexical evidence when the word genuinely isn't in the corpus.

One caveat worth knowing: `df == 0` is also the shape of a *paraphrase* word —
"tinnitus", "sandbag", "gelatin". So in fallback mode those get a phonetic lookup that
is usually wrong, not absent. Under fusion the term-coverage gate suppresses most of
that, which is why the fused numbers hold; a keyword-only deployment (`hybrid: false`)
should benchmark `'off'` against `'fallback'` on its own query log.

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
| `table`           | `null`                                                     | Static semantic table artifact — enables model-free dense ranking (no download) |
| `model`           | `'minishlab/potion-base-4M'`                               | Embedding model (Model2Vec via ORT, or any Transformers.js feature-extraction id) |
| `threshold`       | `0.3`                                                      | Minimum dense score for an option to show            |
| `thresholdMode`   | `'relative'`                                               | `'relative'`: fraction of the query's best dense score (scale-free). `'absolute'`: compare against the raw dense score (cosine only) |
| `maxResults`      | `10`                                                       | Max options rendered in the dropdown                |
| `debounceMs`      | `150`                                                      | Debounce for re-ranking after a keystroke           |
| `phonetic`        | `'fallback'`                                               | Phonetic matching mode: `'fallback'` (keys only for query tokens the corpus has never seen — the shape of a typo), `'union'` (historical `true`: keys indexed alongside exact tokens in documents and queries), `false`/`'off'` (exact only) |
| `phoneticWeight`  | `1`                                                        | Weight of a phonetic-key match relative to an exact term, in `'fallback'` mode |
| `hybrid`          | `true`                                                     | Fuse BM25 keyword matching with vector similarity   |
| `fusion`          | `'rrf'`                                                    | `'rrf'` (rank-based, no tuning) or `'weighted'` (score-based) |
| `rrfK`            | `60`                                                       | RRF rank constant                                   |
| `lexicalWeight`   | `0.5`                                                      | BM25 weight (only when `fusion: 'weighted'`)        |
| `genericDfFrac`   | `0.34`                                                     | A query token in more than this fraction of options is not informative and earns no sparse rank credit |
| `sparseMinCoverage` | `0.67`                                                   | Fraction of the query's informative tokens an option must contain before the keyword channel influences the order (dense mode only) |
| `sparseConfidence` | `null`                                                    | Weight the sparse channel by query evidence instead of a cliff: `'coverage'` or `'ramp'` |
| `placeholder`     | `''`                                                       | Input placeholder                                   |
| `onChange`        | `null`                                                     | `onChange(value, item)` callback                    |

## API

`createVectorComboBox(boxEl, options)` → `{ el, select, input, setQuery, setDenseSource, setDenseEnabled, setUsePrecomputed, refresh, destroy }`

- `setQuery(query)` — programmatically fill the input and rank (as if typed)
- `setDenseSource('table' | 'model' | false)` — choose the dense channel. `'table'`
  needs the `table` artifact and downloads nothing; `'model'` is what starts the
  ~16 MB embedder download; `false` is keyword-only
- `setDenseEnabled(bool)` — toggle the dense channel on/off (BM25 stays live)
- `setUsePrecomputed(bool)` — switch between build-time and on-load vectors (model mode)

## Keyboard

`↓`/`↑` move focus · `Enter` select · `Esc` close & revert · `Home`/`End` jump

## Files

- `vector-combobox.js` — the reusable ES module (USWDS combo-box controller; imports
  `search-core.js` and, for model mode, Transformers.js at runtime; no bundler)
- `search-core.js` — the pure retrieval logic: tokenizer, BM25 index, term-coverage
  gating, and the eligibility + fusion ranking. Shared by the component and every
  eval script, so a measurement is of the code that ships
- `static-table.js` — decoder for the model-free static semantic table (no deps;
  works in the browser and in Node)
- `metaphone.js` — phonetic encoder (ported from govdoc-explainer)
- `data.js` / `data-agencies.js` — demo datasets
- `data.*.embeddings.json` — precomputed option vectors (`npm run precompute`)
- `data.*.semantic.json` — static semantic tables (`npm run build:tables`)
- `scripts/build-semantic-table.mjs` — build the model-free table and verify it
- `scripts/precompute.mjs` — build-time embedding index generation
- `scripts/eval-embedders.mjs` — corpus-relative model evaluation
- `scripts/eval-metaphone.mjs` — phonetic tokenizer evaluation
- `scripts/eval-no-model.mjs` — the keyword-only baseline, both query tiers
- `scripts/eval-static-table.mjs` — score the static table through the real fusion
- `scripts/smoke.mjs` — headless DOM check of wiring (needs `npm i --no-save jsdom`)

## What's still open

- **The evaluation had one tier, and it was the wrong one.** All 28 published test
  queries are phrased in words that already appear in `vectorText`, so the keyword
  channel sees what the embedder sees: BM25 alone scores 23/28 top-1 there and the
  15 MB model 24-25/28. Paraphrase queries (tier 2 in `scripts/datasets.mjs`, 20-50%
  corpus-word coverage) separate the two: BM25 7/18, the static table fused 11/18.
  Every model/size/pruning decision made before this was made blind to that gap.
- **Tier 2 is 18 hand-written queries.** Directions, not a gate. Growing it into
  relevance-judged data is the prerequisite for the fusion-tuning item below.
- **Hybrid still loses ground on paraphrases.** Dense-only table measures 9/18 top-1
  on tier 2; fused it is 11/18 overall but the keyword channel is never *helping*
  there. The df gate, the term-coverage cliff and evidence-proportional weighting
  (`sparseConfidence`) were all measured; the cliff won, and none of them closes the
  gap. This is fusion-design territory that wants judged data, not another sweep.
- **Stop words are doing real work at this corpus size.** `genericDfFrac` cannot
  identify "the" in 42 documents (it appears in only 8), and before the stop-word
  list a query's keyword ranking could be decided by that one word. A corpus with
  different function-word statistics may want a different list, or a real stemmer.
- **Static-model frontier**: potion-2M degraded (20/28), potion-4M was the
  sweet spot. Where exactly does discrimination break, and does it break
  differently for different option-list domains? Answerable with per-domain
  evaluation data, not a general benchmark.
- **Fusion tuning**: RRF with k=60 is deliberately boring. Channel weighting
  is worth revisiting per-domain, but only with relevance-judged evaluation
  data — hand-tuned fusion weights are where search systems decay silently.
- **Scaling**: BM25 builds synchronously and the fused re-rank is O(n) per
  keystroke — instant at 42, fine at a few hundred, worth profiling at a
  thousand. The table's row count is fixed by the vocabulary, but its column count
  grows with options: 42 columns is 348 KB, 200 options would be ~1.6 MB at the same
  pruning — at which point the word→prototype clustering idea gets interesting.
