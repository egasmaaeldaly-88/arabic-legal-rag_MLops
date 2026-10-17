"""Move half-scored questions out of the final RAGAS score cache so the next run re-scores them.

Old runs saved a question as final as soon as ONE metric had a score, so questions whose other
metrics hit a 429 stayed incomplete forever. Complete entries are kept untouched.

Usage (repo root):  python scripts/fix_score_cache.py [outputs/ragas_scores.json]
A backup is written next to the file (.bak).
"""
import json
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/ragas_scores.json")
cache = json.loads(path.read_text(encoding="utf-8"))
shutil.copy(path, str(path) + ".bak")
moved = kept = 0
for key in [k for k in cache if not k.endswith("|partial")]:
    scores = cache[key]
    if isinstance(scores, dict) and any(v is None for v in scores.values()):
        cache[key + "|partial"] = {"attempts": 1, "scores": scores}
        del cache[key]
        moved += 1
    else:
        kept += 1
path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"complete (kept): {kept}; incomplete (will be re-scored): {moved}")
