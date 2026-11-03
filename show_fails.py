import csv
from evaluate import RESULTS

with open(RESULTS, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

failed = [r for r in rows if r["answer_ok"] != "True"]
print(f"{len(failed)} failed out of {len(rows)}\n")

for r in failed:
    print(f"Q{r['id']} [{r['type']}] {r['question']}")
    print(f"  retrieval hit : {r['retrieval_hit']}")
    print(f"  expected      : {r['expected']}")
    print(f"  model said    : {r['answer']}\n")