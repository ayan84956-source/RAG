import json
from pathlib import Path
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import PictureItem, TableItem, TextItem

PDF_PATH = Path("data/solar_energy_test_report.pdf")
OUT_DIR = Path("cache")
IMG_DIR = OUT_DIR / "images"
IMG_DIR.mkdir(parents=True, exist_ok=True)

MIN_IMG_SIDE = 150 

def build_converter() -> DocumentConverter:
    opts = PdfPipelineOptions()
    opts.do_ocr = True              
    opts.do_table_structure = True    
    opts.generate_picture_images = True 
    opts.images_scale = 1.0
    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=opts)}
    )

def page_of(item):
    return item.prov[0].page_no if item.prov else None


def main():
    print("Parsing PDF...")
    doc = build_converter().convert(PDF_PATH).document

    items, skipped = [], 0

    for item, _level in doc.iterate_items():
        page = page_of(item)

        if isinstance(item, TableItem):
            items.append({
                "type": "table",
                "page": page,
                "content": item.export_to_markdown(doc=doc),
            })

        elif isinstance(item, PictureItem):
            img = item.get_image(doc)
            if img is None or min(img.size) < MIN_IMG_SIDE:
                skipped += 1
                continue
            path = IMG_DIR / f"page{page}_pic{len(items)}.png"
            img.save(path)
            items.append({
                "type": "image",
                "page": page,
                "image_path": str(path),
                "size": list(img.size),
            })

        elif isinstance(item, TextItem):
            text = item.text.strip()
            if text:
                items.append({
                    "type": "text",
                    "label": str(item.label.value), 
                    "page": page,
                    "content": text,
                })

    (OUT_DIR / "parsed.json").write_text(json.dumps(items, indent=2), encoding="utf-8")

    print(f"\nSaved {len(items)} items to cache/parsed.json (skipped {skipped} tiny images)")
    for kind in ("text", "table", "image"):
        print(f"  {kind}: {sum(1 for i in items if i['type'] == kind)}")

    print("\nItems per page:")
    for p in sorted({i["page"] for i in items if i["page"]}):
        row = [sum(1 for i in items if i["page"] == p and i["type"] == k) for k in ("text", "table", "image")]
        print(f"  page {p:>2}: text={row[0]:>2}  tables={row[1]}  images={row[2]}")

    first_table = next((i for i in items if i["type"] == "table"), None)
    if first_table:
        print(f"\nFirst table (page {first_table['page']}):\n{first_table['content']}")

if __name__ == "__main__":
    main()