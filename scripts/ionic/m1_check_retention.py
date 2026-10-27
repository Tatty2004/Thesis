"""Usage: .venv/bin/python scripts/ionic/m1_check_retention.py results/m1/<config>/solve[-<tag>]

Check retention exactly from a solve folder's log: every saved checkpoint's hours of solving
are in its 'saved' line, so the rule (newest 2 + first at or after each milestone) can be
recomputed after every save and compared with what the log says was deleted and what is on disk."""
import json
import re
import sys
from pathlib import Path

from bombpot.experiments.m1 import keep_set

folder = Path(sys.argv[1])
log = (folder / "solve.log").read_text()
milestones = json.loads((folder / "runs.jsonl").read_text().splitlines()[0])["run"]["milestone_hours"]
events = re.findall(r"(saved|retention: deleted) ckpt_t(\d+)\.npz \(([\d.]+) h of solving\)", log)
alive, hours, ok = {}, {}, True
for kind, t, h in events:
    t, h = int(t), float(h)
    if kind == "saved":
        alive[t] = h
        hours[t] = h
        expected_keep = keep_set(list(alive.items()), milestones)
        expected_deleted = set(alive) - expected_keep
    else:
        if t not in expected_deleted or abs(hours[t] - h) > 1e-6:
            ok = False
            print(f"unexpected deletion of {t}")
        del alive[t]
        expected_deleted.discard(t)
on_disk = {int(p.name[6:-4]) for p in folder.glob("ckpt_t*.npz")}
final = keep_set(list(hours.items()), milestones)
print(f"milestones {milestones}")
print("saved (t: hours):", {t: round(h, 6) for t, h in sorted(hours.items())})
print(f"rule keeps {sorted(final)}; on disk {sorted(on_disk)}; deletions all as the rule says: {ok}; "
      f"match: {final == on_disk and set(alive) == on_disk}")
print("milestones crossed:", {m: min((t for t, h in hours.items() if h >= m), default=None) for m in milestones})
