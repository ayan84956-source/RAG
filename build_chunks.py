import json
from pathlib import Path

SRC = Path("cache/parsed_captioned.json")
OUT = Path("cache/chunks.json")
MAX_WORDS = 300
SKIP_LABELS = {"page_header", "page_footer"}
HEADER_LABELS = {"section_header", "title"}

def n_words(text: str) -> int:
    return len(text.split())

def build_chunks(items):
    chunks = []
    section = "Document start"
    buf, buf_pages = [], []        

    def add(kind, pages, body):
        chunks.append({
            "id": f"c{len(chunks):04d}",
            "type": kind,
            "section": section,
            "pages": sorted(set(pages)),
            "text": f"Section: {section}\n{body}",
        })

    def flush():
        nonlocal buf, buf_pages
        if buf:
            add("text", buf_pages, "\n".join(buf))
        buf, buf_pages = [], []

    for item in items:
        kind, page = item["type"], item.get("page")

        if kind == "text":
            label, text = item.get("label", ""), item["content"]
            if label in SKIP_LABELS:
                continue
            if label in HEADER_LABELS:
                flush()                 
                section = text
                continue
            if sum(n_words(t) for t in buf) + n_words(text) > MAX_WORDS:
                flush()
            buf.append(text)
            buf_pages.append(page)

        elif kind == "table":
            add("table", [page], f"Table on page {page}:\n{item['content']}")

        elif kind == "image":
            add("image", [page], item.get("caption", ""))

    flush()
    return chunks

def main():
    items = json.loads(SRC.read_text(encoding="utf-8"))
    chunks = build_chunks(items)
    OUT.write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    print(f"{len(chunks)} chunks saved to {OUT}\n")
    for kind in ("text", "table", "image"):
        sizes = [n_words(c["text"]) for c in chunks if c["type"] == kind]
        if sizes:
            print(f"  {kind:5}: {len(sizes):3} chunks, words per chunk min={min(sizes)} max={max(sizes)}")

    checks = {
        "Q1  cost per watt 2020": "2.20",
        "Q3  inverter cost": "1,080",
        "Q4  HJT efficiency": "24.0",
        "Q5  fault code E04": "E04",
        "Q6  June peak": "880",
        "Q7  wind speed (safety image)": "30 km/h",
        "Q8  hotline (scan)": "555-0142",
        "Q9  inverter extension (scan)": "420",
    }
    print("\nWhere is each answer?")
    for label, needle in checks.items():
        hits = [f"{c['type']} p{c['pages']}" for c in chunks if needle in c["text"]]
        print(f"  {label:32} {'FOUND in ' + ', '.join(hits[:4]) if hits else 'NOT FOUND'}")

if __name__ == "__main__":
    main()