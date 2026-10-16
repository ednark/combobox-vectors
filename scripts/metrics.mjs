// scripts/metrics.mjs
// Shared ranking metrics for the evaluation scripts. Centralized because the same
// bug produced two wrong conclusions in one session: a channel that produced NO
// evidence scores 0 against every option, and naive competition ranking of an
// all-zero field returns rank 1 — which reads as "the keyword channel answered 7
// paraphrases it never answered" and "dense-only beats hybrid". `null <= 3` is also
// true in JavaScript, so a null sentinel must never reach a comparison filter.
//
// Rule enforced here: **no evidence is not rank 1.** It is its own bucket, excluded
// from top-1/top-3/MRR and counted separately so the reader can see how much of a
// score rests on the channel simply declining to answer.

/**
 * Rank of the intended option among scored candidates.
 * @param {number[]} scores one score per option
 * @param {string[]} values option values, aligned with scores
 * @param {string[]} targets value prefixes that count as a correct answer
 * @returns {number|null} 1-based rank, or null when nothing scored above 0
 */
export function rankFromScores(scores, values, targets) {
  let best = -Infinity;
  let any = false;
  for (let i = 0; i < values.length; i++) {
    if (scores[i] > 0) any = true;
    if (targets.some((t) => values[i].startsWith(t)) && scores[i] > best) best = scores[i];
  }
  if (!any || best <= 0) return null; // no evidence, not a win
  let better = 0;
  for (const s of scores) if (s > best) better += 1;
  return better + 1;
}

/**
 * Rank of the intended option inside an ordered list of indices — what the
 * component actually renders. A target absent from the returned order is a real
 * miss (the list was produced and excluded it), so it scores n + 1.
 */
export function rankFromOrder(order, values, targets) {
  const pos = order.findIndex((i) => targets.some((t) => values[i].startsWith(t)));
  return pos === -1 ? values.length + 1 : pos + 1;
}

/**
 * @returns {{n:number, answered:number, noEvidence:number, top1:number, top3:number,
 *             mrr:number}} mrr averages over answered queries only.
 */
export function summarize(ranks) {
  const n = ranks.length;
  const answered = ranks.filter((r) => typeof r === 'number' && r > 0);
  const noEvidence = n - answered.length;
  return {
    n,
    answered: answered.length,
    noEvidence,
    top1: answered.filter((r) => r === 1).length,
    top3: answered.filter((r) => r <= 3).length,
    mrr: answered.length ? answered.reduce((a, r) => a + 1 / Math.min(r, n), 0) / answered.length : 0,
  };
}

/** One aligned report line; `extra` appends the no-evidence bucket when nonzero. */
export function metricLine(tag, m, width = 44) {
  const extra = m.noEvidence ? `  (${m.noEvidence} no evidence)` : '';
  return `  ${tag.padEnd(width)} top1 ${String(m.top1).padStart(2)}/${m.n}  ` +
    `top3 ${String(m.top3).padStart(2)}/${m.n}  MRR ${m.mrr.toFixed(3)}${extra}`;
}
