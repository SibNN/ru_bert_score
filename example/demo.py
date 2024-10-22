import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from bert_score.score import score

with open("hyps.txt") as f:
    cands = [line.strip() for line in f]

with open("refs.txt") as f:
    refs = [line.strip() for line in f]

(P, R, F), hashname = score(cands, refs, model_type="kazzand/ru-longformer-tiny-16384", return_hash=True)
print(
    f"{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={F.mean().item():.6f}"
)
