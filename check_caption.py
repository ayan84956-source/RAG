import json

items = json.load(open("cache/parsed_captioned.json", encoding="utf-8"))

for item in items:
    if item["type"] == "image" and item["page"] in (5, 11, 12):
        print(f"--- page {item['page']} ---")
        print(item["caption"])
        print()