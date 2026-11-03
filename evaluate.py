import csv
import re
import sys
from collections import defaultdict
import ollama
from ask import LLM, NOT_FOUND, SYSTEM_PROMPT, build_context, retrieve

QUESTIONS = "evals_questions.csv"
RESULTS = "evals_results.csv"

def normalize(text):
    return text.lower().replace(",", "")

def contains(text, needle):
    """Whole-token match, so '420' does not match inside '3420'."""
    pattern = r"(?<![\w.])" + re.escape(normalize(needle)) + r"(?![\w])"
    return re.search(pattern, normalize(text)) is not None

def facts_present(text, expected):
    return all(contains(text, part.strip()) for part in expected.split(";"))

def generate(question, hits):
    user_message = f"Sources:\n\n{build_context(hits)}\n\nQuestion: {question}"
    response = ollama.chat(
        model=LLM,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": user_message}],
        options={"temperature": 0, "num_ctx": 4096, "num_predict": 250},
    )
    return response["message"]["content"].strip()

def main():
    with open(QUESTIONS, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(sys.argv) > 1:                     
        rows = rows[:int(sys.argv[1])]

    results = []
    for row in rows:
        hits = retrieve(row["question"])
        context_text = " ".join(doc for doc, _meta, _dist in hits)
        answer = generate(row["question"], hits)

        if row["expected"] == "NOT_FOUND":
            retrieval_hit = None        
            answer_ok = "could not find" in normalize(answer)
        else:
            retrieval_hit = facts_present(context_text, row.get("context_needed") or row["expected"])
            answer_ok = facts_present(answer, row["expected"])

        results.append({**row, "retrieval_hit": retrieval_hit, "answer_ok": answer_ok,
                        "answer": answer.replace("\n", " ")})
        r = "n/a" if retrieval_hit is None else ("hit " if retrieval_hit else "MISS")
        print(f"Q{row['id']:>2} [{row['type']:<12}] retrieval={r}  answer={'OK  ' if answer_ok else 'FAIL'}")

    with open(RESULTS, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)

    print("\nBy question type:")
    by_type = defaultdict(list)
    for r in results:
        by_type[r["type"]].append(r)
    for kind, rs in by_type.items():
        print(f"  {kind:<13} answers correct: {sum(r['answer_ok'] for r in rs)}/{len(rs)}")

    scored = [r for r in results if r["retrieval_hit"] is not None]
    print(f"\nRetrieval hit rate : {sum(r['retrieval_hit'] for r in scored)}/{len(scored)}")
    print(f"Answers correct    : {sum(r['answer_ok'] for r in results)}/{len(results)}")
    print(f"Full answers saved to {RESULTS}")

if __name__ == "__main__":
    main()