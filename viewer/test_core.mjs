// Checks viewer/core.js against an export's fixtures (viewer_export.py --fixtures):
//   node viewer/test_core.mjs results/viewer/<method>_t<t>
// Exits 1 on any failure.
import { createRequire } from "module";
import fs from "fs";
import path from "path";

const core = createRequire(import.meta.url)("./core.js");
const dir = process.argv[2];
if (!dir) { console.error("usage: node viewer/test_core.mjs <export folder>"); process.exit(2); }
const read = (p) => JSON.parse(fs.readFileSync(path.join(dir, p), "utf8"));
const bytes = (p) => new Uint8Array(fs.readFileSync(path.join(dir, p)));
const float64 = (p) => { const b = fs.readFileSync(path.join(dir, p)); return new Float64Array(b.buffer, b.byteOffset, b.length / 8); };

const meta = read("meta.json");
const model = new core.Model(meta);
const enc = meta.encoding;
let failed = 0;
const check = (ok, msg) => { console.log(`${ok ? "PASS" : "FAIL"}  ${msg}`); if (!ok) failed++; };

const shardCache = new Map();
function shard(kind, d) {
  const { name } = model.shardOf(d);
  const key = `${kind}/${name}`;
  if (!shardCache.has(key)) {
    shardCache.set(key, kind === "centers" ? float64(`${key}.bin`)
      : new core.StreetStrategy(model, bytes(`${key}.bin`), enc.shapes[kind][1], enc.shapes[kind][2]));
  }
  return shardCache.get(key);
}
function centersOf(d) {
  const { pos } = model.shardOf(d);
  const n = meta.k * meta.dim;
  return shard("centers", d).subarray(pos * n, (pos + 1) * n);
}
const flop = new core.StreetStrategy(model, bytes("flop.bin"), 1, model.H);
function boardsOf(d, river) {
  const [tA, tB] = meta.turn_cards[d];
  return [meta.flops[0].concat([tA, river[0]]), meta.flops[1].concat([tB, river[1]])];
}

// (a) Evaluator.
{
  const ev = read("fixtures/evaluator.json");
  let bad = 0;
  ev.cards.forEach((c, i) => { if (core.handValue(c) !== ev.values[i]) bad++; });
  check(bad === 0, `evaluator: ${ev.cards.length - bad}/${ev.cards.length} 7-card values equal Python's ` +
    `(categories ${ev.categories.join(" ")})`);
}

// (b) Equity and bucket ids at random river deals.
{
  const bk = read("fixtures/buckets.json");
  let eqBad = 0, idBad = 0, kappaBad = 0, possBad = 0, maxEqDiff = 0, n = 0;
  for (const deal of bk.deals) {
    const kappa = core.riverKappa(model, deal.turn_deal);
    if (kappa !== deal.kappa) kappaBad++;
    const { eq, possible } = core.riverEquity(model, boardsOf(deal.turn_deal, deal.river), kappa);
    const ids = core.bucketIds(model, eq, possible, centersOf(deal.turn_deal));
    for (let h = 0; h < model.H; h++) {
      if (possible[h] !== deal.possible[h]) possBad++;
      for (let b = 0; b < 2; b++) {
        const diff = Math.abs(eq[h * 2 + b] - deal.eq[h][b]);
        if (diff !== 0) eqBad++;
        maxEqDiff = Math.max(maxEqDiff, diff);
      }
      for (let p = 0; p < 2; p++) if (ids[h] !== deal.ids[p][h]) idBad++;
      n++;
    }
  }
  check(kappaBad === 0, `river kappa: ${bk.deals.length - kappaBad}/${bk.deals.length} bit-identical`);
  check(possBad === 0, `possible hands: ${possBad} mismatches`);
  check(eqBad === 0, `equity: ${eqBad} of ${2 * n} values differ from Python's (largest ${maxEqDiff})`);
  check(idBad === 0, `bucket ids: ${idBad} mismatches over ${bk.deals.length} river deals x ${model.H} hands x both players`);
  console.log(`      (solver's sampling scale vs scale 1 in Python: ${bk.solver_scale_id_mismatches} id mismatches)`);
}

// (c) Range weights along the path to each point, against forward() on the decoded strategy.
{
  const rc = read("fixtures/reach.json");
  let maxDiff = 0, blockedBad = 0, count = 0;
  const compare = (got, want, possible) => {
    for (let h = 0; h < model.H; h++) {
      if (!possible[h]) { if (got[h] !== 0) blockedBad++; continue; }
      maxDiff = Math.max(maxDiff, Math.abs(got[h] - want[h]) / Math.max(1, Math.abs(want[h])));
      count++;
    }
  };
  for (const pt of rc.turn) {
    const { decisions, at } = core.walk(model, core.pathTo(model, 1, pt.line, pt.node));
    if (at.kind !== "node" || at.t !== 1 || at.k !== pt.node || at.line !== pt.line) throw new Error("bad path");
    const [tA, tB] = meta.turn_cards[pt.turn_deal];
    const possible = model.possible(meta.flops.flat().concat([tA, tB]));
    const ctx = { flop, turn: shard("turn", pt.turn_deal), river: null, d: pt.turn_deal, ids: null };
    for (let p = 0; p < 2; p++) compare(core.rangeWeights(model, ctx, decisions, p, possible), pt.reach[p], possible);
  }
  for (const pt of rc.river) {
    const { decisions, at } = core.walk(model, core.pathTo(model, 2, pt.line, pt.node));
    if (at.kind !== "node" || at.t !== 2 || at.k !== pt.node || at.line !== pt.line) throw new Error("bad path");
    const d = pt.turn_deal;
    const { eq, possible } = core.riverEquity(model, boardsOf(d, pt.river), core.riverKappa(model, d));
    const ids = core.bucketIds(model, eq, possible, centersOf(d));
    const ctx = { flop, turn: shard("turn", d), river: shard("river", d), d, ids };
    for (let p = 0; p < 2; p++) compare(core.rangeWeights(model, ctx, decisions, p, possible), pt.reach[p], possible);
  }
  check(maxDiff <= 1e-12 && blockedBad === 0,
    `range weights: ${rc.turn.length} turn and ${rc.river.length} river points, ${count} hand-player values, ` +
    `largest difference ${maxDiff.toExponential(2)} (tolerance 1e-12), blocked hands nonzero: ${blockedBad}`);
  const q = Math.max(...rc.turn.map((p) => p.quantization_effect));
  console.log(`      (uint8 storage moves a turn range weight by at most ${q.toFixed(5)} from the float strategy's)`);
}

// Betting amounts derived from the template and bet size equal the template's own chip counts.
{
  let bad = 0, n = 0;
  model.templates.forEach((tp, t) => {
    tp.closes.forEach(([hist, added]) => {
      const c = core.streetChips(model, t, hist); n++;
      if (c[0] !== added || c[1] !== added) bad++;
    });
    tp.children.forEach((row, k) => row.forEach((child, a) => {
      if (child[0] !== "fold") return;
      const c = core.streetChips(model, t, tp.hist[k]); n++;
      if (c[child[1]] !== child[2]) bad++;
    }));
  });
  check(bad === 0, `betting: ${n - bad}/${n} closes and folds have the template's chip counts`);
  const contribOk = meta.contrib.every((c, t) => t === 0 || c.every((x, l) => {
    const prev = meta.contrib[t - 1][Math.floor(l / 5)];
    return x === prev + core.streetChips(model, t - 1, model.templates[t - 1].closes[l % 5][0])[0];
  }));
  check(contribOk, "contributions: each street's start equals the previous start plus the closing line's chips");
}

// The flop report in the checkpoint's metadata (the solve log's flop_hands line).
{
  const want = { KcKh: 0.022, AcAd: 0.023, "7d7h": 0.0561, "9c9d": 0.0264, "5s6s": 0.0393, "3c4d": 0.0219 };
  const ckptReport = JSON.parse(fs.readFileSync(path.join(dir, "check.json"), "utf8")).flop_p0_bet;
  let worst = 0;
  for (const [name, v] of Object.entries(ckptReport)) {
    if (name === "overall") continue;
    const h = model.handIdx(core.parseCard(name.slice(0, 2)), core.parseCard(name.slice(2)));
    worst = Math.max(worst, Math.abs(flop.prob(0, 0, h, 0, 1) - v));
  }
  check(worst < 1 / 255 + 5e-5, `flop root: P0's bet frequency for the report's hands within ${worst.toFixed(5)} of the export's float values`);
  if (meta.checkpoint.t === 1008 && meta.method === "kmeans_2d") {
    const h = model.handIdx(core.parseCard("7d"), core.parseCard("7h"));
    check(Math.abs(flop.prob(0, 0, h, 0, 1) - want["7d7h"]) < 1 / 255 + 5e-5, `7d7h bets ${flop.prob(0, 0, h, 0, 1).toFixed(4)} at the flop root (solve.log: 0.0561)`);
  }
}

console.log(failed ? `${failed} check(s) failed` : "all checks passed");
process.exit(failed ? 1 : 0);
