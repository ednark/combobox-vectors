// vector-combobox.js
// A USWDS combo-box extension whose filtering is driven by client-side
// semantic vector search instead of substring typeahead.
//
// Reuses the USWDS combo-box markup/CSS (usa-combo-box classes) but ships its
// own controller JS, so it does NOT require uswds.min.js. Embeddings are
// computed in the browser with Transformers.js (same model used for the query
// and for each option's "vector text"), adapted from govdoc-explainer's
// assets/semantic_search.js.
//
// Usage:
//   import { createVectorComboBox } from './vector-combobox.js';
//   createVectorComboBox(document.getElementById('my-box'), {
//     data: [{ value, text, vectorText }, ...],
//     placeholder: 'Search by topic…',
//     threshold: 0.3,        // min cosine similarity to show (0..1)
//     maxResults: 10,        // max options rendered in the dropdown
//     hybrid: true,          // fuse BM25 keyword matching with vectors
//     fusion: 'rrf',         // 'rrf' (best practice) or 'weighted'
//     onChange: (value, item) => {},
//   });
//
// Required HTML (USWDS combo-box input markup):
//   <label class="usa-label" for="cb">Label</label>
//   <div class="usa-combo-box" id="cb-box">
//     <select id="cb" name="cb" class="usa-select"></select>
//   </div>

import { BM25Index, createSparseIndex, rankItems, tokenize, tokenizePhonetic } from './search-core.js';
import { decodeTable } from './static-table.js';

// The sparse channel and the fusion logic live in search-core.js so the
// evaluation scripts measure the ranking this module actually performs. Re-exported
// here because they were previously imported from this module.
export { BM25Index, createSparseIndex, tokenize, tokenizePhonetic };

// Exported so the evaluation scripts measure the component's actual defaults
// rather than a copy that can drift.
export const DEFAULTS = {
  // Default dense embedder: minishlab/potion-base-4M — a Model2Vec
  // "StaticModel" (distilled from bge-base into a pure embedding table,
  // 128-dim, mean-pooled, L2-normalized in-graph). 15.1 MB fp32 ONNX.
  // Chosen by corpus-relative evaluation (scripts/eval-embedders.mjs):
  // it matched or beat the previous contextual baseline
  // (Xenova/paraphrase-MiniLM-L3-v2, 17.5 MB int8) at discriminating the
  // demo datasets — top-1 25/28 vs 24/28, top-3 28/28 vs 27/28 — while
  // running inference as a single embedding lookup (no transformer layers).
  // The fitness measure is separation WITHIN the option list, not general
  // embedding quality; for datasets where contextuality matters more,
  // override `model` with any Transformers.js feature-extraction model.
  model: 'minishlab/potion-base-4M',
  transformersUrl: 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3',
  // onnxruntime-web ESM bundle (WASM inlined) for running Model2Vec ONNX
  // graphs in the browser; Node uses onnxruntime-node instead.
  ortWebUrl: 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.21.0/dist/ort.bundle.min.mjs',
  // Hugging Face Hub host for Model2Vec ONNX/tokenizer files.
  hubUrl: 'https://huggingface.co',
  threshold: 0.3,
  maxResults: 10,
  debounceMs: 150,
  // Phonetic (metaphone) keys in the BM25 tokenizer: each token is indexed
  // both exactly (stemmed) and as a 4-char phonetic key, so typo'd queries
  // still collide with the intended token ("recods" -> RKRT -> "records").
  // Corpus-relative eval (scripts/eval-metaphone.mjs): typo queries top-1
  // 6/14 -> 10/14, top-3 12/14 -> 14/14, clean queries no regression.
  // Phonetic matching mode for the sparse channel. 'fallback' (default): exact
  // tokens rank, and a query token is only looked up by metaphone key when its
  // exact spelling is absent from the index — the shape of a typo. 'union' (the
  // historical `true`) indexes keys alongside exact tokens in both documents and
  // queries, which double-counts correctly spelled words and dilutes ranking with
  // 4-character key collisions. false/'off' is exact-only.
  phonetic: 'fallback',
  // Hybrid search fuses a BM25 sparse keyword index (title boosted + vectorText)
  // with dense vector similarity, so queries like "October" (in a title but
  // semantically absent from the vectorText) still surface the right option.
  // An item is kept if vectorScore >= threshold OR it has any BM25 match.
  // Fusion of the two ranked lists (best practice, cf. Elasticsearch/OpenSearch
  // hybrid search):
  //   'rrf'       — Reciprocal Rank Fusion: score = sum of 1/(rrfK + rank)
  //                 across channels. Rank-based, needs no score normalization
  //                 or weight tuning. Default.
  //   'weighted'  — vectorScore + lexicalWeight * normalizedBM25 (score-based;
  //                 lets you deliberately bias toward one channel).
  hybrid: true,
  fusion: 'rrf',
  rrfK: 60,
  lexicalWeight: 0.5,
  // --- Dense source -------------------------------------------------------
  // 'table' (when `table` is supplied): a build-time static semantic table
  // (scripts/build-semantic-table.mjs) — a word -> per-option score matrix. Zero
  // inference: no ONNX runtime, no WASM, no tokenizer, ~200 KB instead of the
  // ~15 MB model, and results are correct on the first keystroke.
  // 'model': Transformers.js + the `model` embedder. Needed for a contextual
  // embedder, or when options change too often to regenerate a table.
  table: null,                  // artifact from build-semantic-table.mjs
  // --- Eligibility --------------------------------------------------------
  // 'relative' (default): `threshold` is a fraction of the query's BEST dense
  // score, so it is scale-free across embedders and table units. An absolute cosine
  // cutoff silently evicts the dense channel's own top pick whenever a query only
  // weakly matches everything — which is the paraphrase case the dense channel
  // exists for (measured: 8/14 paraphrase agency queries have a best true cosine
  // under the old 0.3 absolute cutoff).
  thresholdMode: 'relative',    // 'relative' | 'absolute'
  // --- Sparse-channel gating (measured by scripts/build-semantic-table.mjs) -----
  // A query token appearing in more than this fraction of the options carries
  // almost no information, so it cannot earn the sparse channel rank credit.
  genericDfFrac: 0.34,
  // How much of the query's informative vocabulary an option must contain before
  // the keyword channel influences the order at all. Ungated, a generic word match
  // ("before", "at", "behind") gives a distractor the same RRF credit as a full
  // keyword match, and fusion measured WORSE than dense alone on paraphrase queries
  // (table alone 13/18 -> fused 9/18 top-1; this gate recovers 10/18 and also
  // improves the in-vocabulary tier). 0 disables the gate.
  sparseMinCoverage: 0.67,
  // Weight the sparse channel by how much of the query it explains instead of
  // using a cliff: 'coverage' | 'ramp' | null (always-on hybrid).
  sparseConfidence: null,
  placeholder: '',
  defaultValue: '',
  onChange: null,
};

const PREFIX = 'usa-combo-box';
const PRISTINE = `${PREFIX}--pristine`;
const CL = {
  select: `${PREFIX}__select`,
  input: `${PREFIX}__input`,
  clear: `${PREFIX}__clear-input`,
  clearWrap: `${PREFIX}__clear-input__wrapper`,
  sep: `${PREFIX}__input-button-separator`,
  toggle: `${PREFIX}__toggle-list`,
  toggleWrap: `${PREFIX}__toggle-list__wrapper`,
  list: `${PREFIX}__list`,
  option: `${PREFIX}__list-option`,
  focused: `${PREFIX}__list-option--focused`,
  selected: `${PREFIX}__list-option--selected`,
  noResults: `${PREFIX}__list-option--no-results`,
  status: `${PREFIX}__status`,
};

// --- Dense embedding backends ------------------------------------------------
// Two backends, one interface: async (texts: string[]) => number[][] with
// L2-normalized rows (cosine == dot product).
//
//   pipeline  — any Transformers.js feature-extraction model (contextual
//               transformers like MiniLM). Handled by transformers.js.
//   model2vec — minishlab potion "StaticModel"s: an EmbeddingBag(mean) +
//               L2-normalize ONNX graph, driven manually via ONNX Runtime
//               (transformers.js has no model2vec backbone). Inputs are
//               flattened per-text token ids (no special tokens, no unk)
//               plus cumulative offsets — the protocol from MinishLab's
//               export script.
//
// The same embedMany() powers the browser (query + on-load option vectors)
// and scripts/precompute.mjs (build-time vectors), so build/runtime model
// parity holds by construction.

let _pipeline = null;
let _pipelineModel = null;
let _m2v = null; // { model, tokenizer, session, run(texts) }

async function loadTransformers(transformersUrl) {
  // Browser: import the CDN ESM bundle. Node (precompute script, tests):
  // https imports aren't supported by the default loader, so use the locally
  // installed package instead.
  const mod =
    typeof process !== 'undefined' && process.versions?.node
      ? await import('@huggingface/transformers')
      : await import(transformersUrl);
  if (mod.env) mod.env.allowLocalModels = false;
  return mod;
}

// Model2Vec detection: potion/StaticModel repos are cheap to recognize by id
// prefix; anything else falls through to the transformers.js pipeline path.
function isModel2Vec(modelId) {
  return /^minishlab\/(potion|m2v)/.test(modelId);
}

async function getModel2Vec(model, { transformersUrl, ortWebUrl, hubUrl }) {
  if (_m2v && _m2v.model === model) return _m2v;
  const mod = await loadTransformers(transformersUrl);
  const tokenizer = await mod.AutoTokenizer.from_pretrained(model);

  const ort =
    typeof process !== 'undefined' && process.versions?.node
      ? await import('onnxruntime-node')
      : await import(/* @vite-ignore */ ortWebUrl);

  const res = await fetch(`${hubUrl}/${model}/resolve/main/onnx/model.onnx`);
  if (!res.ok) throw new Error(`model2vec: failed to fetch ONNX for ${model} (${res.status})`);
  const bytes = new Uint8Array(await res.arrayBuffer());
  const session = await ort.InferenceSession.create(bytes);

  const run = async (texts) => {
    const encodings = texts.map((t) =>
      tokenizer.encode(t, { add_special_tokens: false })
        .filter((id) => id !== tokenizer.unk_token_id)
    );
    const dim = session.outputNames.length ? null : null; // dim read from result
    // Empty encodings would misalign offsets; give them a single pad token so
    // every text yields a row, then zero those rows afterward.
    const safe = encodings.map((ids) => (ids.length ? ids : [tokenizer.pad_token_id ?? 0]));
    const flat = safe.flatMap((ids) => ids);
    const offsets = [0];
    for (let i = 0; i < safe.length - 1; i++) offsets.push(offsets[i] + safe[i].length);
    const feeds = {
      input_ids: new ort.Tensor('int64', BigInt64Array.from(flat.map(BigInt)), [flat.length]),
      offsets: new ort.Tensor('int64', BigInt64Array.from(offsets.map(BigInt)), [offsets.length]),
    };
    const out = await session.run(feeds);
    const emb = out[session.outputNames[0]];
    const d = emb.dims.at(-1);
    return texts.map((_, i) =>
      encodings[i].length
        ? Array.from(emb.data.subarray(i * d, (i + 1) * d))
        : new Array(d).fill(0)
    );
  };

  _m2v = { model, run };
  return _m2v;
}

// Batch embed texts with the configured model. Exported so the build-time
// precompute script uses the identical code path as the browser.
export async function embedMany(texts, model, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  if (isModel2Vec(model)) {
    const m2v = await getModel2Vec(model, o);
    return m2v.run(texts);
  }
  if (!_pipeline || _pipelineModel !== model) {
    const mod = await loadTransformers(o.transformersUrl);
    _pipeline = await mod.pipeline('feature-extraction', model);
    _pipelineModel = model;
  }
  const out = await _pipeline(texts, { pooling: 'mean', normalize: true });
  const dim = out.dims.at(-1);
  return Array.from({ length: texts.length }, (_, i) =>
    Array.from(out.data.subarray(i * dim, (i + 1) * dim))
  );
}

// Single-text convenience wrapper (query embedding).
async function embed(text, model, opts) {
  const [vec] = await embedMany([text], model, opts);
  return vec;
}

function cosine(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // vectors are L2-normalized, so dot == cosine
}

function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}

function changeValue(el, value = '') {
  el.value = value;
  el.dispatchEvent(new CustomEvent('change', { bubbles: true, cancelable: true, detail: { value } }));
}

export function createVectorComboBox(boxEl, opts = {}) {
  const o = { ...DEFAULTS, ...opts };
  const selectEl = boxEl.querySelector('select');
  if (!selectEl) throw new Error('vector-combobox: wrapper must contain a <select>');

  const selectId = selectEl.id;
  const labelEl = document.querySelector(`label[for="${selectId}"]`);
  if (!labelEl) throw new Error(`vector-combobox: missing <label for="${selectId}">`);

  const listId = `${selectId}--list`;
  const labelId = `${selectId}-label`;
  const hintId = `${selectId}--assistiveHint`;

  // --- Build hidden <select> options from data (for form submission) ---
  selectEl.replaceChildren();
  const placeholderOpt = document.createElement('option');
  placeholderOpt.value = '';
  placeholderOpt.textContent = o.placeholder ? `Select…` : '';
  selectEl.appendChild(placeholderOpt);
  for (const item of o.data) {
    const op = document.createElement('option');
    op.value = item.value;
    op.textContent = item.text;
    selectEl.appendChild(op);
  }
  selectEl.classList.add('usa-sr-only', CL.select);
  selectEl.setAttribute('aria-hidden', 'true');
  selectEl.setAttribute('tabindex', '-1');
  selectEl.removeAttribute('id');
  selectEl.value = '';
  labelEl.setAttribute('id', labelId);

  // --- Enhance the wrapper into the full USWDS combo-box structure ---
  const inputEl = document.createElement('input');
  inputEl.id = selectId;
  inputEl.className = CL.input;
  inputEl.type = 'text';
  inputEl.setAttribute('role', 'combobox');
  inputEl.setAttribute('aria-owns', listId);
  inputEl.setAttribute('aria-controls', listId);
  inputEl.setAttribute('aria-autocomplete', 'list');
  inputEl.setAttribute('aria-describedby', hintId);
  inputEl.setAttribute('aria-expanded', 'false');
  inputEl.setAttribute('autocapitalize', 'off');
  inputEl.setAttribute('autocomplete', 'off');
  if (o.placeholder) inputEl.setAttribute('placeholder', o.placeholder);

  boxEl.insertAdjacentElement('beforeend', inputEl);
  boxEl.insertAdjacentHTML(
    'beforeend',
    `<span class="${CL.clearWrap}" tabindex="-1">
       <button type="button" class="${CL.clear}" aria-label="Clear the select contents">&nbsp;</button>
     </span>
     <span class="${CL.sep}">&nbsp;</span>
     <span class="${CL.toggleWrap}" tabindex="-1">
       <button type="button" tabindex="-1" class="${CL.toggle}" aria-label="Toggle the dropdown list">&nbsp;</button>
     </span>
     <ul tabindex="-1" id="${listId}" class="${CL.list}" role="listbox" aria-labelledby="${labelId}" hidden></ul>
     <div class="${CL.status} usa-sr-only" role="status"></div>
     <span id="${hintId}" class="usa-sr-only">When autocomplete results are available use up and down arrows to review and enter to select. Touch device users, explore by touch or with swipe gestures.</span>`
  );

  const listEl = boxEl.querySelector(`.${CL.list}`);
  const statusEl = boxEl.querySelector(`.${CL.status}`);
  const toggleEl = boxEl.querySelector(`.${CL.toggle}`);
  const clearEl = boxEl.querySelector(`.${CL.clear}`);

  // --- State ---
  const bm25Index = createSparseIndex(o.data, o); // sparse keyword channel (instant)

  // Two embedding sources, each computed exactly once:
  //   - precomputed: build-time vectors from item.embedding (scripts/precompute.mjs)
  //   - runtime: every option's vectorText embedded on page load (background)
  // The precomputed toggle just switches which source array the search reads;
  // both are cached, so switching is instant after the initial page-load pass.
  let usePrecomputed = true;         // which source the search reads
  let runtimeVecs = null;            // Promise<number[][]> — on-load embeddings
  let activeVecs = null;             // Promise<number[][]> — currently selected source
  let ready = false;                 // model + active embeddings available
  let denseFailed = false;           // model download/inference failed
  let denseEnabled = o.hybrid;       // runtime switch: dense channel on/off
  // A static semantic table answers queries synchronously, so the dense channel is
  // ready before the first keystroke and nothing is ever downloaded.
  const staticTable = o.table ? decodeTable(o.table) : null;
  let denseSource = staticTable ? 'table' : 'model';
  if (staticTable) ready = true;
  let highlightedIndex = -1;
  let renderedItems = [];            // current ranked results

  // Embed every option's vectorText at runtime. Memoized: the pass runs once
  // per component lifetime (kicked off eagerly at init) and is awaited — never
  // re-run — by every later toggle.
  const embedAll = () =>
    (runtimeVecs ??= (async () => {
      const vecs = [];
      for (const item of o.data) {
        vecs.push(await embed(item.vectorText || item.text, o.model, o));
      }
      return vecs;
    })());

  // Assemble the source array for the current usePrecomputed setting.
  // Precomputed items are used when selected; holes (items without a
  // precomputed vector) are always filled from the runtime set.
  function computeVecs() {
    if (usePrecomputed && o.data.every((item) => item.embedding && item.embedding.length)) {
      return Promise.resolve(o.data.map((item) => item.embedding));
    }
    return (async () => {
      const runtime = await embedAll();
      return o.data.map((item, i) =>
        usePrecomputed && item.embedding && item.embedding.length
          ? item.embedding
          : runtime[i]
      );
    })();
  }

  async function ensureReady() {
    if (ready) return;
    activeVecs ??= computeVecs();
    setStatus('Loading search model…');
    await activeVecs;
    ready = true;
  }

  function setStatus(text) { statusEl.textContent = text; }

  function rank(query) {
    // Returns a promise of ranked item indices.
    return (async () => {
      if (!query.trim()) {
        return o.data.map((_, i) => i).slice(0, o.maxResults);
      }
      const lexRaw = bm25Index.scores(query);
      // Informative-token scores rank the sparse channel; full scores still decide
      // eligibility, so a generic-only match can appear without being boosted.
      const lexInfo = bm25Index.scores(query, { informativeOnly: true });
      const coverage = bm25Index.coverage(query);
      const fuse = (denseRaw) => rankItems(o.data.length, lexRaw, denseRaw, o, lexInfo, coverage);

      // Dense channel disabled at runtime (e.g. the demo's "vector search"
      // toggle switched off): keyword-only ranking, no model required.
      if (!denseEnabled) {
        return fuse(null);
      }
      // Static semantic table: synchronous, no download, no inference.
      if (denseSource === 'table') {
        return fuse(Array.from(staticTable.scores(query)));
      }

      // Degraded mode: dense model not ready (or failed). The sparse BM25
      // index needs no download, so keyword search works instantly.
      if (!ready) {
        if (!denseFailed) {
          // Upgrade the visible results to full hybrid once the model lands.
          ensureReady()
            .then(() => {
              if (denseEnabled && isOpen() && inputEl.value.trim()) {
                rank(inputEl.value).then(renderList);
              }
            })
            .catch(() => {});
          if (!o.hybrid) return []; // vector-only config: nothing to show yet
        }
        return fuse(null);
      }

      const q = await embed(query, o.model, o);
      const vecs = await activeVecs;
      return fuse(o.data.map((item, i) => cosine(q, vecs[i])));

    })();
  }

  function clearHighlight() {
    const cur = listEl.querySelector(`.${CL.focused}`);
    if (cur) {
      cur.classList.remove(CL.focused);
      cur.setAttribute('tabindex', '-1');
    }
    inputEl.setAttribute('aria-activedescendant', '');
  }

  function highlight(index) {
    clearHighlight();
    const optEls = listEl.querySelectorAll(`.${CL.option}`);
    if (index < 0 || index >= optEls.length) {
      highlightedIndex = -1;
      return;
    }
    const el = optEls[index];
    el.classList.add(CL.focused);
    el.setAttribute('tabindex', '0');
    inputEl.setAttribute('aria-activedescendant', el.id);
    const bottom = el.offsetTop + el.offsetHeight;
    if (bottom > listEl.scrollTop + listEl.offsetHeight) {
      listEl.scrollTop = bottom - listEl.offsetHeight;
    } else if (el.offsetTop < listEl.scrollTop) {
      listEl.scrollTop = el.offsetTop;
    }
    highlightedIndex = index;
  }

  // Non-interactive notice row appended to the list (loading/degraded states).
  function appendStatusLi(text) {
    const li = document.createElement('li');
    li.className = CL.option;
    li.setAttribute('aria-disabled', 'true');
    li.textContent = text;
    listEl.appendChild(li);
  }

  function renderList(indices) {
    listEl.replaceChildren();
    renderedItems = indices;
    if (!indices.length) {
      const li = document.createElement('li');
      li.className = CL.noResults;
      li.textContent = 'No results found';
      listEl.appendChild(li);
      setStatus('No results.');
    } else {
      const currentVal = selectEl.value;
      indices.forEach((idx, pos) => {
        const item = o.data[idx];
        const li = document.createElement('li');
        li.id = `${listId}--option-${pos}`;
        li.className = CL.option;
        li.setAttribute('role', 'option');
        li.setAttribute('tabindex', '-1');
        li.setAttribute('aria-setsize', indices.length);
        li.setAttribute('aria-posinset', pos + 1);
        li.setAttribute('aria-selected', String(item.value === currentVal));
        li.setAttribute('data-value', item.value);
        li.setAttribute('data-index', String(idx));
        li.textContent = item.text;
        if (item.value === currentVal) li.classList.add(CL.selected);
        listEl.appendChild(li);
      });
      setStatus(`${indices.length} result${indices.length > 1 ? 's' : ''} available.`);
      highlight(0);
    }
    // Progressive-enhancement notices: keyword results are shown while the
    // dense model downloads, and the combobox keeps working (keyword-only)
    // forever if the model can never load.
    if (!denseEnabled) {
      appendStatusLi('Vector search off — keyword search only.');
    } else if (!ready && !denseFailed) {
      appendStatusLi('Loading semantic ranking — keyword matches shown…');
      setStatus('Keyword results shown while the semantic model loads.');
    } else if (denseFailed) {
      appendStatusLi('Semantic ranking unavailable — keyword search only.');
    }
  }

  async function open(query) {
    listEl.hidden = false;
    inputEl.setAttribute('aria-expanded', 'true');
    const indices = await rank(query);
    renderList(indices);
  }

  function close() {
    listEl.hidden = true;
    listEl.scrollTop = 0;
    inputEl.setAttribute('aria-expanded', 'false');
    clearHighlight();
    setStatus('');
  }

  function isOpen() { return !listEl.hidden; }

  function selectByDataIndex(idx) {
    const item = o.data[idx];
    changeValue(selectEl, item.value);
    changeValue(inputEl, item.text);
    boxEl.classList.add(PRISTINE);
    close();
    inputEl.focus();
    if (typeof o.onChange === 'function') o.onChange(item.value, item);
  }

  function resetSelection() {
    if (selectEl.value) {
      const item = o.data.find((d) => d.value === selectEl.value);
      if (item && inputEl.value.toLowerCase() !== item.text.toLowerCase()) {
        changeValue(inputEl, item.text);
      }
      boxEl.classList.add(PRISTINE);
      return;
    }
    if (inputEl.value) changeValue(inputEl, '');
    boxEl.classList.remove(PRISTINE);
  }

  // --- Event wiring ---
  const debouncedRank = debounce(async (q) => {
    if (!isOpen()) return;
    const indices = await rank(q);
    renderList(indices);
  }, o.debounceMs);

  inputEl.addEventListener('input', () => {
    boxEl.classList.remove(PRISTINE);
    if (!isOpen()) open(inputEl.value);
    debouncedRank(inputEl.value);
  });

  inputEl.addEventListener('focus', () => {
    if (!isOpen()) open(inputEl.value);
  });

  inputEl.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      if (!isOpen()) return open(inputEl.value);
      highlight(Math.min(highlightedIndex + 1, renderedItems.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      if (!isOpen()) return open(inputEl.value);
      highlight(Math.max(highlightedIndex - 1, 0));
    } else if (e.key === 'Enter') {
      if (isOpen() && highlightedIndex >= 0 && renderedItems[highlightedIndex] != null) {
        e.preventDefault();
        selectByDataIndex(renderedItems[highlightedIndex]);
      } else if (isOpen()) {
        e.preventDefault();
        close();
        resetSelection();
      }
    } else if (e.key === 'Escape') {
      if (isOpen()) {
        e.preventDefault();
        close();
        resetSelection();
        inputEl.focus();
      }
    } else if (e.key === 'Home') {
      e.preventDefault();
      if (isOpen()) highlight(0);
    } else if (e.key === 'End') {
      e.preventDefault();
      if (isOpen()) highlight(renderedItems.length - 1);
    }
  });

  listEl.addEventListener('click', (e) => {
    const li = e.target.closest(`.${CL.option}`);
    if (!li || li.classList.contains(CL.noResults)) return;
    const idx = Number(li.dataset.index);
    if (Number.isNaN(idx)) return;
    selectByDataIndex(idx);
  });

  listEl.addEventListener('mousemove', (e) => {
    const li = e.target.closest(`.${CL.option}`);
    if (!li) return;
    const optEls = Array.from(listEl.querySelectorAll(`.${CL.option}`));
    const pos = optEls.indexOf(li);
    if (pos >= 0 && pos !== highlightedIndex) highlight(pos);
  });

  toggleEl.addEventListener('click', () => {
    if (isOpen()) close();
    else {
      inputEl.focus();
      open(inputEl.value);
    }
  });

  clearEl.addEventListener('click', () => {
    if (selectEl.value) changeValue(selectEl);
    if (inputEl.value) changeValue(inputEl, '');
    boxEl.classList.remove(PRISTINE);
    if (isOpen()) open('');
    inputEl.focus();
  });

  // Close on outside click. Uses mousedown so it runs BEFORE any outside
  // element's click handler — e.g. a demo "example phrase" button that calls
  // setQuery() — so the programmatic open isn't immediately re-closed by
  // this listener when the click bubbles up to document.
  document.addEventListener('mousedown', (e) => {
    if (!boxEl.contains(e.target)) {
      if (isOpen()) {
        close();
        resetSelection();
      }
    }
  });

  // Programmatically set the search query: fills the input, opens the
  // dropdown, and ranks options against the query (as if the user typed it).
  // Used for "example phrase" buttons on the demo page.
  function setQuery(query) {
    inputEl.value = query;
    boxEl.classList.remove(PRISTINE);
    inputEl.focus(); // focus listener opens + ranks when the list was closed
    if (!isOpen()) open(query);
  }

  // Pre-seed default value if requested.
  if (o.defaultValue) {
    const item = o.data.find((d) => d.value === o.defaultValue);
    if (item) {
      changeValue(selectEl, item.value);
      changeValue(inputEl, item.text);
      boxEl.classList.add(PRISTINE);
    }
  }

  // Runtime switch for the dense (vector) channel. Turning it off makes the
  // combobox keyword-only instantly — no model download or inference at all
  // (useful for testing, comparisons, or slow connections). Turning it back on
  // resumes the warmup/load path and re-ranks the visible query. The sparse
  // BM25 channel is always available.
  // Switch the dense channel between the static semantic table, the in-browser
  // embedder, and off. Table mode needs no download and is ready immediately;
  // model mode resumes the warmup path. The sparse BM25 channel is always live.
  function setDenseSource(source) {
    const want = source === false || source === 'off' ? null : source;
    if (want === null) {
      denseEnabled = false;
    } else {
      if (want === 'model' && !o.model) throw new Error('vector-combobox: model mode needs o.model');
      if (want === 'table' && !staticTable) throw new Error('vector-combobox: no `table` artifact was supplied');
      denseEnabled = true;
      denseSource = want;
      if (want === 'table') { ready = true; }
      else if (!ready && !denseFailed) {
        ensureReady().catch((err) => {
          console.error('vector-combobox: failed to load embedding model', err);
          denseFailed = true;
        });
      }
    }
    rerank();
  }

  function rerank() {
    if (!isOpen()) return;
    if (inputEl.value.trim()) rank(inputEl.value).then(renderList);
    else renderList(o.data.map((_, i) => i).slice(0, o.maxResults));
  }

  function setDenseEnabled(enabled) {
    denseEnabled = !!enabled;
    if (denseEnabled && !ready && !denseFailed) {
      ensureReady()
        .then(() => {
          if (isOpen() && inputEl.value.trim()) rank(inputEl.value).then(renderList);
        })
        .catch(() => {}); // failure path handled by the init catch below
    }
    // Re-render the open list so the notice row reflects the new mode.
    if (isOpen()) {
      if (inputEl.value.trim()) rank(inputEl.value).then(renderList);
      else renderList(o.data.map((_, i) => i).slice(0, o.maxResults));
    }
  }

  // Runtime switch between the two embedding sources. Both arrays exist by
  // this point (precomputed from build time, runtime computed on page load),
  // so this is a source-array swap plus a re-rank — no re-embedding, no
  // invalidation. If the page-load pass is still running, the re-rank simply
  // awaits it. (Both sources use the same model, so rankings are identical;
  // this is a demonstration/testing control.)
  function setUsePrecomputed(enabled) {
    usePrecomputed = !!enabled;
    activeVecs = computeVecs();
    if (isOpen()) {
      if (inputEl.value.trim()) rank(inputEl.value).then(renderList);
      else renderList(o.data.map((_, i) => i).slice(0, o.maxResults));
    }
  }

  // Kick off the page-load embedding pass and model warmup in the background
  // (non-blocking). Skipped entirely when the dense channel starts disabled
  // (o.hybrid false or setDenseEnabled(false) before first interaction). On
  // failure the combobox degrades permanently to keyword-only search instead
  // of breaking. The runtime pass always runs — it is the on-load source the
  // precomputed toggle switches to — even when every item has a precomputed
  // vector, and the pipeline warmup is needed for the query regardless.
  // With a static semantic table the dense channel is already ready and nothing
  // must be downloaded, so warmup is skipped entirely.
  if (denseEnabled && denseSource === 'model') {
    embedAll();
    ensureReady()
      .then(() => embed('warmup', o.model, o))
      .catch((err) => {
        console.error('vector-combobox: failed to load embedding model', err);
        denseFailed = true;
        setStatus('Semantic model unavailable; keyword search only.');
        if (isOpen() && inputEl.value.trim()) {
          rank(inputEl.value).then(renderList);
        }
      });
  }

  return {
    el: boxEl,
    select: selectEl,
    input: inputEl,
    setQuery,
    setDenseEnabled,
    setDenseSource,
    setUsePrecomputed,
    refresh: () => {
      activeVecs = null;
      runtimeVecs = null;
      ready = false;
      denseFailed = false;
      return ensureReady();
    },
    destroy: () => { boxEl.replaceChildren(); boxEl.classList.remove(PRISTINE); },
  };
}
