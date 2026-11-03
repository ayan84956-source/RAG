import csv
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "Result.csv"

with open(path, encoding="utf-8-sig", newline="") as f:
    for row in csv.DictReader(f):
        print(f"Q{row['id']} [{row['type']}]  {row['question']}")
        print(f"  expected : {row['expected']}")
        print(f"  answer   : {row['answer']}")
        print(f"  scored   : retrieval={row['retrieval_hit']}  answer_ok={row['answer_ok']}\n")