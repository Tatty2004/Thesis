# Strategy viewer

A page for walking an M1 checkpoint's average strategy action by action: a 13x13 hand grid colored
by the action mix, range weights, and a combo inspector with river buckets and equities.

- `index.html`: the page (published as a private claude.ai artifact).
- `core.js`: everything but the DOM: cards, the evaluator, river equity and bucket ids, the betting
  template walk, decoding and range weights. The page loads it as a script and Node requires it.
- `test_core.mjs`: checks `core.js` against the fixtures an export writes.
- The data comes from `bombpot/experiments/viewer_export.py` and lives in `results/viewer/` (scratch),
  never in git.

Published: kmeans_2d, https://claude.ai/artifact/Wc2yVzkqvTwhUKgUx8Pjr7 (private).

## Refresh to a newer checkpoint

1. Export, with fixtures (about 2.5 min, 7 GB):
   ```
   cd /n/fs/ugrad/ug27/rb5709/Thesis
   ls results/m1/m1_kmeans2d/solve/          # the newest ckpt_t<T>.npz
   NUMBA_NUM_THREADS=8 nice .venv/bin/python -m bombpot.experiments.viewer_export \
       --checkpoint results/m1/m1_kmeans2d/solve/ckpt_t<T>.npz --fixtures
   ```
   It stops with an error if any decoded probability is 1/255 or more from the float average, or
   if the flop report differs from the checkpoint's.
2. Test: `node viewer/test_core.mjs results/viewer/kmeans_2d_t<T>`. Every line must say PASS. (Node was
   removed from rinse on 2026-10-27; use a host that has it.)
3. Publish. Artifacts serve no raw binary type, so the export's `web/` folder holds `meta.json` and every
   `.bin` as base64 of its gzip (`.txt`). Copy `web/` plus `viewer/index.html` and `viewer/core.js` into
   one folder under the session's scratchpad (artifact sources must be there or under the working
   directory, and `results/` is on scratch), then publish `index.html` to the URL above with `files`, in
   three batches under the 64 MB limit:
   1. `core.js`, `meta.json`, `flop.txt`, `centers/*.txt`, `turn/*.txt` (about 56 MB);
   2. the first 23 `river/*.txt` (2c through 8c);
   3. the other 23 (8d through As).

   A publish replaces files with the same path and keeps the rest, so every shard must be republished:
   a stale shard would mix two checkpoints.

avg_1d would need its own artifact (a version holds at most 256 MB).
