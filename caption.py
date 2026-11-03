import json
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image

PARSED = Path("cache/parsed.json")      
OUT = Path("cache/parsed_captioned.json")  
TMP_IMAGE = Path("cache/_ocr_tmp.png")
MAX_SIDE = 2000     
LINE_TOLERANCE = 18   

try:
    from rapidocr import RapidOCR     
    NEW_API = True
except ImportError:
    try:
        from rapidocr_onnxruntime import RapidOCR 
        NEW_API = False
    except ImportError:
        raise SystemExit("RapidOCR is not installed. Run: pip install rapidocr onnxruntime")

engine = RapidOCR()


def save_json(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def label_chart_values(positioned):
    """positioned: list of (x_center, y_center, text).
    Matches decimal value labels to the years on the x-axis by horizontal position.
    Returns a dict {year: value}, or {} if the image has no year axis."""
    years = [(x, int(t)) for x, _y, t in positioned if re.fullmatch(r"20\d\d", t)]
    if len(years) < 2:
        return {}
    xs, ys = zip(*years)
    slope, intercept = np.polyfit(xs, ys, 1)        
    lo, hi = min(ys) - 1.6, max(ys) + 1.6

    found = {}
    for x, _y, t in positioned:
        if re.fullmatch(r"\d+\.\d+", t): 
            year = slope * x + intercept
            if abs(year - round(year)) <= 0.3 and lo <= round(year) <= hi:
                found.setdefault(int(round(year)), t)
    return dict(sorted(found.items()))


def ocr_image(path):
    """Returns (text, labelled) where text is a string and labelled is a dict."""
    img = Image.open(path).convert("RGB")
    if max(img.size) > MAX_SIDE:
        s = MAX_SIDE / max(img.size)
        img = img.resize((int(img.width * s), int(img.height * s)))
    img.save(TMP_IMAGE)

    if NEW_API:
        res = engine(str(TMP_IMAGE))
        if res.txts is None or len(res.txts) == 0:
            return "", {}
        boxes, texts = res.boxes, res.txts
    else:
        res, _ = engine(str(TMP_IMAGE))
        if not res:
            return "", {}
        boxes = [r[0] for r in res]
        texts = [r[1] for r in res]

    words, positioned = [], []
    for box, text in zip(boxes, texts):
        xs = [p[0] for p in box]
        ys = [p[1] for p in box]
        words.append((sum(ys) / len(ys), min(xs), text))
        positioned.append((sum(xs) / len(xs), sum(ys) / len(ys), text.strip()))
    words.sort()

    lines, current, last_y = [], [], None
    for y, x, text in words:
        if last_y is not None and abs(y - last_y) > LINE_TOLERANCE:
            lines.append(sorted(current))
            current = []
        current.append((x, text))
        last_y = y
    if current:
        lines.append(sorted(current))

    text_out = "\n".join("  ".join(t for _, t in line) for line in lines)
    return text_out, label_chart_values(positioned)


def main():
    items = json.loads(PARSED.read_text(encoding="utf-8"))
    images = [i for i in items if i["type"] == "image"]
    print(f"{len(images)} images (API: {'new' if NEW_API else 'old'})\n")

    for n, item in enumerate(images, 1):
        if item["page"] == 1:
            item["ocr"] = ""
            item["caption"] = "Decorative cover illustration of rooftop solar panels."
            continue

        try:
            item["ocr"], labelled = ocr_image(item["image_path"])
        except Exception as e:
            print(f"[{n}/{len(images)}] page {item['page']}: ERROR {type(e).__name__}: {e}")
            item["ocr"], labelled = "", {}

        parts = [f"Image on page {item['page']}."]
        if item["ocr"]:
            parts.append(f"Text read from the image (OCR):\n{item['ocr']}")
        else:
            parts.append("No readable text.")
        if labelled:
            pairs = "; ".join(f"{year}: {value}" for year, value in labelled.items())
            parts.append(f"Values matched to years by position on the chart: {pairs}")
        item["caption"] = "\n".join(parts)

        print(f"[{n}/{len(images)}] page {item['page']}: {len(item['ocr'])} characters, "
              f"{len(labelled)} year-value pairs")
        save_json(OUT, items)

    save_json(OUT, items)
    with_text = sum(1 for i in images if i.get("ocr"))
    print(f"Done. {with_text} of {len(images)} images have OCR text. Saved to {OUT}")


if __name__ == "__main__":
    main()