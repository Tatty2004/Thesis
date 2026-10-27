// The strategy viewer's core, without the DOM: cards, the hand evaluator, river equity and bucket
// ids, the betting template walk, strategy decoding and range weights. The page loads it as a
// classic script (window.ViewerCore); Node requires it (viewer/test_core.mjs checks it against
// fixtures written by bombpot/experiments/viewer_export.py --fixtures).
(function (root) {
  "use strict";

  const RANKS = "23456789TJQKA";
  const SUITS = "cdhs";
  const NUM_SUITS = 4;

  function cardStr(c) {
    return RANKS[(c / NUM_SUITS) | 0] + SUITS[c % NUM_SUITS];
  }

  function parseCard(s) {
    const r = RANKS.indexOf(s[0]), u = SUITS.indexOf(s[1]);
    if (s.length !== 2 || r < 0 || u < 0) throw new Error(`not a card: ${s}`);
    return r * NUM_SUITS + u;
  }

  // Hand values: a port of holdem/evaluator._nb_value (category, then five ranks, 4 bits each).
  const HIGH_CARD = 0, PAIR = 1, TWO_PAIR = 2, TRIPS = 3, STRAIGHT = 4, FLUSH = 5, FULL_HOUSE = 6, QUADS = 7,
    STRAIGHT_FLUSH = 8;
  const ACE = 12;

  function straightTop(mask) {
    for (let top = ACE; top > 3; top--) if (((mask >> (top - 4)) & 0x1f) === 0x1f) return top;
    if (mask & (1 << ACE) && (mask & 0xf) === 0xf) return 3;
    return -1;
  }

  function pack(category, r, n) {
    let v = category;
    for (let i = 0; i < 5; i++) v = v * 16 + (i < n ? r[i] : 0);
    return v;
  }

  function handValue(cards) {
    const n = cards.length;
    const counts = new Array(13).fill(0), suitCount = [0, 0, 0, 0], suitMask = [0, 0, 0, 0];
    let mask = 0;
    for (const c of cards) {
      const r = (c / NUM_SUITS) | 0, s = c % NUM_SUITS;
      counts[r]++;
      suitCount[s]++;
      suitMask[s] |= 1 << r;
      mask |= 1 << r;
    }
    const out = [0, 0, 0, 0, 0];
    let flush = -1;
    if (n >= 5) {
      for (let s = 0; s < 4; s++) if (suitCount[s] >= 5) flush = s;
      if (flush >= 0) {
        const top = straightTop(suitMask[flush]);
        if (top >= 0) { out[0] = top; return pack(STRAIGHT_FLUSH, out, 1); }
      }
    }
    let quad = -1, trip = -1;
    for (let r = 12; r >= 0; r--) {
      if (counts[r] === 4 && quad < 0) quad = r;
      if (counts[r] === 3 && trip < 0) trip = r;
    }
    let m;
    if (quad >= 0) {
      out[0] = quad; m = 1;
      for (let r = 12; r >= 0; r--) if (r !== quad && counts[r] > 0 && m < 1 + Math.min(1, n - 4)) out[m++] = r;
      return pack(QUADS, out, m);
    }
    if (trip >= 0 && n >= 5) {
      for (let r = 12; r >= 0; r--) {
        if (r !== trip && counts[r] >= 2) { out[0] = trip; out[1] = r; return pack(FULL_HOUSE, out, 2); }
      }
    }
    if (flush >= 0) {
      m = 0;
      for (let r = 12; r >= 0; r--) if ((suitMask[flush] >> r) & 1 && m < 5) out[m++] = r;
      return pack(FLUSH, out, 5);
    }
    if (n >= 5) {
      const top = straightTop(mask);
      if (top >= 0) { out[0] = top; return pack(STRAIGHT, out, 1); }
    }
    if (trip >= 0) {
      out[0] = trip; m = 1;
      for (let r = 12; r >= 0; r--) {
        for (let u = 0; u < (r !== trip ? counts[r] : 0); u++) if (m < 1 + Math.min(2, n - 3)) out[m++] = r;
      }
      return pack(TRIPS, out, m);
    }
    let pa = -1, pb = -1;
    for (let r = 12; r >= 0; r--) {
      if (counts[r] === 2) { if (pa < 0) pa = r; else if (pb < 0) pb = r; }
    }
    if (pb >= 0) {
      out[0] = pa; out[1] = pb; m = 2;
      for (let r = 12; r >= 0; r--) {
        for (let u = 0; u < (r !== pa && r !== pb ? counts[r] : 0); u++) if (m < 2 + Math.min(1, n - 4)) out[m++] = r;
      }
      return pack(TWO_PAIR, out, m);
    }
    if (pa >= 0) {
      out[0] = pa; m = 1;
      for (let r = 12; r >= 0; r--) {
        for (let u = 0; u < (r !== pa ? counts[r] : 0); u++) if (m < 1 + Math.min(3, n - 2)) out[m++] = r;
      }
      return pack(PAIR, out, m);
    }
    m = 0;
    for (let r = 12; r >= 0; r--) for (let u = 0; u < counts[r]; u++) if (m < 5) out[m++] = r;
    return pack(HIGH_CARD, out, m);
  }

  const CATEGORY_NAMES = ["high card", "pair", "two pair", "trips", "straight", "flush", "full house", "quads",
    "straight flush"];

  function category(value) {
    return Math.floor(value / 1048576);
  }

  // The exported model: meta.json plus lookups. --------------------------------------------------

  class Model {
    constructor(meta) {
      this.meta = meta;
      this.hands = meta.hands;
      this.H = meta.hands.length;
      this.handIndex = new Map(meta.hands.map((h, i) => [h[0] * 52 + h[1], i]));
      this.turnIndex = new Map(meta.turn_cards.map((p, d) => [p[0] * 52 + p[1], d]));
      this.templates = meta.templates;
      this.enc = meta.encoding;
      this.A = meta.templates[0].actions.map((a) => a.length);
      this.flopCards = new Set(meta.flops.flat());
    }

    hand(i) { return this.hands[i]; }

    handIdx(c1, c2) { return this.handIndex.get(Math.min(c1, c2) * 52 + Math.max(c1, c2)); }

    turnDeal(turnA, turnB) {
      const d = this.turnIndex.get(turnA * 52 + turnB);
      if (d === undefined) throw new Error(`no turn deal ${cardStr(turnA)} ${cardStr(turnB)}`);
      return d;
    }

    shardOf(d) {
      const g = this.enc.turn_group;
      return { name: this.meta.shards[Math.floor(d / g)], pos: d % g };
    }

    // Which hands avoid every public card.
    possible(publicCards) {
      const blocked = new Uint8Array(52);
      for (const c of publicCards) blocked[c] = 1;
      const out = new Uint8Array(this.H);
      for (let i = 0; i < this.H; i++) out[i] = blocked[this.hands[i][0]] || blocked[this.hands[i][1]] ? 0 : 1;
      return out;
    }
  }

  // One street's decoded strategy file: rows (pos, line, row) of `width` bytes.
  class StreetStrategy {
    constructor(model, bytes, lines, rows) {
      this.bytes = bytes;
      this.lines = lines;
      this.rows = rows;
      this.width = model.enc.width;
      this.off = model.enc.offsets;
      this.A = model.A;
      this.scale = model.enc.scale;
      if (bytes.length % (lines * rows * this.width)) throw new Error("strategy file has the wrong size");
    }

    // The stored values decode as b / 255, and the dropped last action as (255 - sum) / 255.
    prob(pos, line, row, k, a) {
      const base = ((pos * this.lines + line) * this.rows + row) * this.width + this.off[k];
      const n = this.A[k] - 1;
      if (a < n) return this.bytes[base + a] / this.scale;
      let s = 0;
      for (let u = 0; u < n; u++) s += this.bytes[base + u];
      return (this.scale - s) / this.scale;
    }

    mix(pos, line, row, k) {
      const out = new Float64Array(this.A[k]);
      for (let a = 0; a < out.length; a++) out[a] = this.prob(pos, line, row, k, a);
      return out;
    }
  }

  // River equity and buckets (CardStreet.equity, EquityBuckets.ids). ------------------------------

  // boards: [[5 cards of A], [5 cards of B]]; kappa: the river deal's chance weight, used exactly as
  // Python uses it: (wins + ties / 2) * kappa / (count * kappa).
  function riverEquity(model, boards, kappa) {
    const H = model.H, hands = model.hands;
    const possible = model.possible(boards[0].concat(boards[1]));
    const eq = new Float64Array(H * 2);
    const strength = [new Int32Array(H), new Int32Array(H)];
    for (let b = 0; b < 2; b++) {
      for (let i = 0; i < H; i++) {
        if (possible[i]) strength[b][i] = handValue([hands[i][0], hands[i][1]].concat(boards[b]));
      }
    }
    const live = [];
    for (let i = 0; i < H; i++) if (possible[i]) live.push(i);
    for (const i of live) {
      const [a1, a2] = hands[i];
      let count = 0;
      const win = [0, 0], tie = [0, 0];
      for (const j of live) {
        const [b1, b2] = hands[j];
        if (b1 === a1 || b1 === a2 || b2 === a1 || b2 === a2) continue;
        count++;
        for (let b = 0; b < 2; b++) {
          if (strength[b][j] < strength[b][i]) win[b]++;
          else if (strength[b][j] === strength[b][i]) tie[b]++;
        }
      }
      for (let b = 0; b < 2; b++) {
        eq[i * 2 + b] = count > 0 ? ((win[b] + 0.5 * tie[b]) * kappa) / (count * kappa) : 0;
      }
    }
    return { eq, possible, strength };
  }

  // centers: Float64Array (k * dim) for this turn deal, weakest first. Impossible hands get id k.
  function bucketIds(model, eq, possible, centers) {
    const { k, dim } = model.meta;
    const H = model.H;
    const ids = new Int32Array(H);
    for (let i = 0; i < H; i++) {
      if (!possible[i]) { ids[i] = k; continue; }
      const x = dim === 1 ? [(eq[i * 2] + eq[i * 2 + 1]) / 2] : [eq[i * 2], eq[i * 2 + 1]];
      let best = Infinity, arg = 0;
      for (let c = 0; c < k; c++) {
        let d2 = 0;
        for (let u = 0; u < dim; u++) {
          const z = x[u] - centers[c * dim + u];
          d2 += z * z;
        }
        if (d2 < best) { best = d2; arg = c; }
      }
      ids[i] = arg;
    }
    return ids;
  }

  function riverKappa(model, d) {
    return model.meta.turn_kappa[d] * model.meta.river_scale * 1.0;
  }

  // Betting -------------------------------------------------------------------------------------

  // Chips each player has added on street t after the betting string `hist`, from the template's
  // players and the street's bet size: a bet or raise tops up to the larger amount plus one bet.
  function streetChips(model, t, hist) {
    const tp = model.templates[t], bet = model.meta.bet_sizes[t];
    const chips = [0, 0];
    let k = 0;
    for (const ch of hist) {
      const p = tp.players[k];
      const top = Math.max(chips[0], chips[1]);
      if (ch === "r") chips[p] = top + bet;
      else if (ch === "c") chips[p] = top;
      const a = tp.actions[k].indexOf(ch);
      const child = tp.children[k][a];
      if (child[0] !== "node") break;
      k = child[1];
    }
    return chips;
  }

  function actionLabel(model, t, k, a) {
    const tp = model.templates[t], bet = model.meta.bet_sizes[t];
    const chips = streetChips(model, t, tp.hist[k]);
    const p = tp.players[k], top = Math.max(chips[0], chips[1]), owe = top - chips[p];
    const ch = tp.actions[k][a];
    if (ch === "f") return { kind: "fold", text: "Fold", amount: 0 };
    if (ch === "c") return owe > 0 ? { kind: "call", text: `Call ${fmt(owe)}`, amount: owe }
      : { kind: "check", text: "Check", amount: 0 };
    if (top === 0) return { kind: "bet", text: `Bet ${fmt(bet)}`, amount: bet };
    return { kind: "raise", text: `Raise to ${fmt(top + bet)}`, amount: top + bet - chips[p] };
  }

  function fmt(x) {
    return Number.isInteger(x) ? String(x) : x.toFixed(1);
  }

  // Walk a list of action indices from the flop root. Returns the decisions taken and where the
  // walk stands: {kind: "node", t, k, line}, {kind: "fold", t, folder, line}, or
  // {kind: "showdown", line} after the last street closes.
  function walk(model, actions) {
    const T = model.templates.length;
    let t = 0, k = 0, line = 0;
    const decisions = [];
    for (const a of actions) {
      const tp = model.templates[t];
      decisions.push({ t, k, line, player: tp.players[k], a });
      const child = tp.children[k][a];
      if (child[0] === "node") { k = child[1]; continue; }
      if (child[0] === "fold") return { decisions, at: { kind: "fold", t, folder: child[1], line, k } };
      const nextLine = line * tp.closes.length + child[1];
      if (t + 1 === T) return { decisions, at: { kind: "showdown", line: nextLine, t, close: child[1] } };
      t += 1; k = 0; line = nextLine;
    }
    return { decisions, at: { kind: "node", t, k, line } };
  }

  // The action indices that reach street t's node k on entering line `line` (lines are mixed-radix
  // over each earlier street's closes).
  function pathTo(model, t, line, k) {
    const closes = [];
    let l = line;
    for (let u = t - 1; u >= 0; u--) {
      const n = model.templates[u].closes.length;
      closes.unshift(l % n);
      l = Math.floor(l / n);
    }
    const out = [];
    const follow = (u, hist) => {
      const tp = model.templates[u];
      let kk = 0;
      for (const ch of hist) {
        const a = tp.actions[kk].indexOf(ch);
        out.push(a);
        const child = tp.children[kk][a];
        if (child[0] === "node") kk = child[1];
      }
    };
    closes.forEach((c, u) => follow(u, model.templates[u].closes[c][0]));
    follow(t, model.templates[t].hist[k]);
    return out;
  }

  // Strategy access for one spot: ctx = {flop, turn, river: StreetStrategy | null, d, ids}.
  function rowOf(ctx, t, h) {
    return t === 2 ? ctx.ids[h] : h;
  }

  function posOf(model, ctx, t) {
    return t === 0 ? 0 : ctx.d % model.enc.turn_group;
  }

  function streetOf(ctx, t) {
    const s = [ctx.flop, ctx.turn, ctx.river][t];
    if (!s) throw new Error(`street ${t}'s strategy isn't loaded`);
    return s;
  }

  // Range weight of every hand for `player`: the product of the probabilities of the player's own
  // actions along the path, in path order, and 0 for hands that share a card with the board.
  function rangeWeights(model, ctx, decisions, player, possible) {
    const H = model.H;
    const w = new Float64Array(H);
    for (let h = 0; h < H; h++) w[h] = possible[h] ? 1.0 : 0.0;
    for (const dec of decisions) {
      if (dec.player !== player) continue;
      const s = streetOf(ctx, dec.t), pos = posOf(model, ctx, dec.t);
      for (let h = 0; h < H; h++) {
        if (w[h] !== 0) w[h] = w[h] * s.prob(pos, dec.line, rowOf(ctx, dec.t, h), dec.k, dec.a);
      }
    }
    return w;
  }

  // Every hand's action mix at a node: Float64Array (H * A).
  function nodeMixes(model, ctx, t, k, line) {
    const s = streetOf(ctx, t), pos = posOf(model, ctx, t), A = model.A[k], H = model.H;
    const out = new Float64Array(H * A);
    for (let h = 0; h < H; h++) for (let a = 0; a < A; a++) out[h * A + a] = s.prob(pos, line, rowOf(ctx, t, h), k, a);
    return out;
  }

  // The 13 x 13 grid: rows and columns from ace down, pairs on the diagonal, suited above it.
  function cellOf(c1, c2) {
    const r1 = (c1 / NUM_SUITS) | 0, r2 = (c2 / NUM_SUITS) | 0;
    const hi = Math.max(r1, r2), lo = Math.min(r1, r2);
    if (hi === lo) return { row: 12 - hi, col: 12 - hi };
    if (c1 % NUM_SUITS === c2 % NUM_SUITS) return { row: 12 - hi, col: 12 - lo };
    return { row: 12 - lo, col: 12 - hi };
  }

  function cellLabel(row, col) {
    const a = RANKS[12 - Math.min(row, col)], b = RANKS[12 - Math.max(row, col)];
    if (row === col) return a + b;
    return a + b + (col > row ? "s" : "o");
  }

  // Aggregate per cell: combos possible, range weight sum, and the weighted mix (mixes may be null).
  function grid(model, possible, weights, mixes, A) {
    const cells = [];
    for (let i = 0; i < 169; i++) {
      const row = Math.floor(i / 13), col = i % 13;
      cells.push({ row, col, label: cellLabel(row, col), hands: [], possible: 0, weight: 0, mix: new Float64Array(A) });
    }
    let total = 0, combos = 0;
    const summary = new Float64Array(A);
    for (let h = 0; h < model.H; h++) {
      const [c1, c2] = model.hands[h];
      const { row, col } = cellOf(c1, c2);
      const cell = cells[row * 13 + col];
      cell.hands.push(h);
      if (!possible[h]) continue;
      cell.possible++;
      combos++;
      const w = weights[h];
      cell.weight += w;
      total += w;
      if (mixes) for (let a = 0; a < A; a++) { cell.mix[a] += w * mixes[h * A + a]; summary[a] += w * mixes[h * A + a]; }
    }
    for (const cell of cells) if (cell.weight > 0) for (let a = 0; a < A; a++) cell.mix[a] /= cell.weight;
    if (total > 0) for (let a = 0; a < A; a++) summary[a] /= total;
    return { cells, summary, inRange: total, combos };
  }

  const api = {
    RANKS, SUITS, cardStr, parseCard, handValue, category, CATEGORY_NAMES, Model, StreetStrategy, riverEquity,
    bucketIds, riverKappa, streetChips, actionLabel, walk, pathTo, rangeWeights, nodeMixes, cellOf, cellLabel, grid,
    fmt,
  };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.ViewerCore = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
