"""Local PDF review utilities. Never imports the Flask app or connects to a service."""
from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import fitz
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
AUDIT = ROOT / "book_audit"


def book_paths():
    return {int(re.search(r"第\s*([14])\s*册", p.name).group(1)): p
            for p in ROOT.glob("*.pdf") if re.search(r"第\s*([14])\s*册", p.name)}


def render_pages(pages, output_name, tile_width=680, columns=2):
    output = AUDIT / "pages"
    output.mkdir(parents=True, exist_ok=True)
    paths = book_paths()
    tiles = []
    for book, number in pages:
        with fitz.open(paths[book]) as document:
            page = document[number - 1]
            pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        filename = output / f"book{book}-p{number:04}.png"
        pix.save(filename)
        image = Image.open(filename).convert("RGB")
        image.thumbnail((tile_width, 1000))
        tile = Image.new("RGB", (tile_width + 12, 1036), "#eeeeee")
        ImageDraw.Draw(tile).text((10, 8), f"Book {book} / PDF page {number}", fill="black")
        tile.paste(image, ((tile.width - image.width) // 2, 32))
        tiles.append(tile)
    sheet = Image.new("RGB", (columns * (tile_width + 12), ((len(tiles) + columns - 1) // columns) * 1036), "white")
    for i, tile in enumerate(tiles):
        sheet.paste(tile, ((i % columns) * tile.width, (i // columns) * tile.height))
    target = AUDIT / f"{output_name}.jpg"
    sheet.save(target, quality=92)
    return target


def audit_original_warnings():
    baseline = json.loads((AUDIT / "original_621_report.json").read_text(encoding="utf-8"))
    entries = json.loads((AUDIT / "parsed_entries.json").read_text(encoding="utf-8"))
    inventory = json.loads((AUDIT / "inventory.json").read_text(encoding="utf-8"))
    pages = {(p["book"], p["page"]): p for p in inventory}
    appendix_start = {
        book: min((p["page"] for p in inventory if p["book"] == book
                   and any(re.match(r"^附录", line["text"]) for line in p["lines"][:5])), default=None)
        for book in book_paths()
    }
    by_image = {}
    for item in entries:
        for evidence in item.get("image_evidence", []):
            key = (item["source_file"], evidence["page"], evidence["path"])
            by_image.setdefault(key, []).append({"entry_id": item["_id"], "title": item["entry_title"],
                                               "start_page": item["page"], "evidence": evidence})
    rows = []
    for index, warning in enumerate(baseline["review_items"], 1):
        book = int(re.search(r"第\s*([14])\s*册", warning["file"]).group(1))
        number = int(re.search(r"第\s*(\d+)\s*页", warning["reason"]).group(1))
        original_images = warning.get("images", [])
        owners = [owner for path in original_images for owner in by_image.get((warning["file"], number, path), [])]
        row = {"warning_id": f"pdf-warning-{index:04}", "book": book, "page": number,
               "original": warning, "decision": "unresolved", "matching_entries": owners,
               "verification": "source_coordinates_and_text; not an expert review of sign correctness"}
        if owners and len(owners) == len(original_images) and all(o["evidence"]["boundary_confirmed"] for o in owners):
            row["decision"] = "resolved_image_ownership"
            row["reason"] = "图片已按正文标题和跨页阅读顺序归入唯一词条，并保存原页坐标"
        elif "缺少动作说明" in warning["reason"]:
            title = warning["reason"].split("页，", 1)[-1]
            matches = [e for e in entries if e["source_file"] == warning["file"] and e.get("page") == number
                       and re.sub(r"\s+", "", e["entry_title"]).startswith(re.sub(r"\s+", "", title))]
            if len(matches) == 1:
                row["decision"] = "resolved_short_or_wrapped_entry"
                row["reason"] = "原文说明实际存在，已恢复短说明或合并拆行标题"
                row["matching_entries"] = [{"entry_id": e["_id"], "title": e["entry_title"]} for e in matches]
        if row["decision"] == "unresolved":
            page = pages[(book, number)]
            headings = [line["text"] for line in page["lines"] if line["size"] >= 24]
            if appendix_start[book] is not None and number >= appendix_start[book]:
                row["decision"] = "excluded_non_dictionary_material"
                row["reason"] = "位于原书附录或后续索引；不作为词典动作说明导入"
            elif book == 1 and 105 <= number <= 112:
                row["decision"] = "excluded_non_dictionary_material"
                row["reason"] = "原书图解说明、国歌示意或笔画索引，不作为动作词条导入"
            elif warning["reason"].endswith("，其他"):
                row["decision"] = "excluded_section_heading"
                row["reason"] = "其他是章节名；其下数字词条已分别解析"
            row["source_headings"] = headings
        rows.append(row)
    from collections import Counter
    result = {"total": len(rows), "counts": dict(Counter(row["decision"] for row in rows)), "items": rows}
    (AUDIT / "warning_decisions.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["counts"], ensure_ascii=True))
    return rows


def render_warning_sheets(rows):
    locations = sorted({(row["book"], row["page"]) for row in rows})
    paths = book_paths()
    folder = AUDIT / "warning_sheets"
    folder.mkdir(exist_ok=True)
    font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
    tile_w, tile_h, columns, count = 460, 710, 4, 12
    docs = {book: fitz.open(path) for book, path in paths.items()}
    manifest = []
    try:
        for start in range(0, len(locations), count):
            subset = locations[start:start + count]
            sheet = Image.new("RGB", (tile_w * columns, tile_h * 3), "#dddddd")
            for i, (book, number) in enumerate(subset):
                pix = docs[book][number - 1].get_pixmap(matrix=fitz.Matrix(0.9, 0.9), alpha=False)
                page_image = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                page_image.thumbnail((tile_w - 10, tile_h - 60))
                x, y = (i % columns) * tile_w, (i // columns) * tile_h
                draw = ImageDraw.Draw(sheet)
                draw.text((x + 6, y + 4), f"Book {book} / PDF page {number}", fill="black", font=font)
                matches = [row for row in rows if row["book"] == book and row["page"] == number]
                title = next((m["matching_entries"][0]["title"] for m in matches if m["matching_entries"]), "非词条/待查")
                draw.text((x + 6, y + 28), "核对：" + title[:23], fill="black", font=font)
                sheet.paste(page_image, (x + (tile_w - page_image.width) // 2, y + 56))
            filename = f"sheet-{start // count + 1:02}.jpg"
            sheet.save(folder / filename, quality=92)
            manifest.append({"sheet": filename, "pages": subset, "visual_reviewed": False})
            print(filename, flush=True)
    finally:
        for doc in docs.values():
            doc.close()
    (AUDIT / "visual_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def validate_parsed_entries():
    """Check local source evidence; this does not approve or upload any entry."""
    from collections import Counter
    from book_knowledge import keyword_search

    entries = json.loads((AUDIT / "parsed_entries.json").read_text(encoding="utf-8"))
    pdf = [e for e in entries if e["source_file"].lower().endswith(".pdf")]
    failures, owners = [], {}
    for entry in pdf:
        label = entry["_id"]
        # The text cleaner collapses spaces; the source title retains them.
        first_line, separator, body = entry["text"].partition("\n")
        if " ".join(first_line.split()) != " ".join(entry["entry_title"].split()) or not separator or not body:
            failures.append([label, "text does not start with its own title and description"])
        if not entry["image_paths"]:
            failures.append([label, "missing illustration"])
        if entry["book"] == 4 and entry["page_end"] >= 422:
            failures.append([label, "appendix/index entered dictionary"])
        if entry["book"] == 1 and entry["page"] < 113:
            failures.append([label, "prefatory material entered dictionary"])
        for evidence in entry.get("image_evidence", []):
            key = (entry["source_file"], evidence["page"], tuple(round(v, 2) for v in evidence["bbox"]))
            owners.setdefault(key, []).append(label)
            media = ROOT / "book_media" / evidence["path"].removeprefix("/book-media/")
            if not media.is_file():
                failures.append([label, "extracted image missing", evidence["path"]])
            if not evidence["boundary_confirmed"]:
                failures.append([label, "unconfirmed image boundary"])
            if evidence.get("association_method") != "reviewed_layout_exception":
                center = (evidence["bbox"][1] + evidence["bbox"][3]) / 2
                if not any(s["page"] == evidence["page"] and s["bbox"][1] <= center < s["bbox"][3]
                           for s in entry["source_segments"]):
                    failures.append([label, "image outside source segment"])
    duplicate_owners = [value for value in owners.values() if len(value) != 1]
    if duplicate_owners:
        failures.append(["duplicate image placements", duplicate_owners])
    if len({e["_id"] for e in entries}) != len(entries):
        failures.append(["duplicate entry IDs"])
    # Approved copies are used only in memory for retrieval checks. Disk stays pending.
    candidates = [dict(e, review_status="approved") for e in pdf]
    queries = {}
    for query, expected in [("豆腐", ["豆腐①", "豆腐②"]), ("豆腐①", ["豆腐①"]),
                            ("豆腐②", ["豆腐②"]), ("不存在的手语词条XYZ", [])]:
        titles = [e["entry_title"].split()[0] for e in keyword_search(candidates, query)]
        queries[query] = titles
        if titles != expected:
            failures.append(["retrieval", query, titles, expected])
    for name in ["毒辣", "毒品", "独裁"]:
        entry = next(e for e in pdf if e["entry_title"].startswith(name))
        if len(entry["image_paths"]) != 1 or any(other in entry["text"] for other in ["毒辣", "毒品", "独裁"] if other != name):
            failures.append([name, "neighboring entry contamination"])
    result = {"entry_count": len(entries), "pdf_entry_count": len(pdf),
              "counts_by_book": dict(Counter(e["book"] for e in entries)),
              "pdf_image_placements": len(owners), "retrieval_examples": queries,
              "failures": failures, "passed": not failures,
              "scope": "Local structural and retrieval checks, not blanket review approval or cloud validation"}
    (AUDIT / "structural_checks.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=True))
    if failures:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pages", nargs="+", help="Book:PDF-page pairs, e.g. 1:634 4:55")
    parser.add_argument("--sheet", default="inspection")
    parser.add_argument("--inventory", action="store_true")
    parser.add_argument("--reparse", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--render-warnings", action="store_true")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    AUDIT.mkdir(exist_ok=True)
    baseline = AUDIT / "original_621_report.json"
    if not baseline.exists():
        shutil.copy2(ROOT / "book_review_report_v2.json", baseline)
    if args.reparse:
        from book_knowledge import import_books
        result = import_books()
        (AUDIT / "parsed_entries.json").write_text(
            json.dumps(result["chunks"], ensure_ascii=False, indent=2), encoding="utf-8",
        )
        print("Parsed", len(result["chunks"]), "entries; warnings", result["report"]["parse_warning_count"])
    if args.inventory:
        inventory = []
        for book, path in book_paths().items():
            with fitz.open(path) as document:
                for number, page in enumerate(document, 1):
                    lines = []
                    for block in page.get_text("dict")["blocks"]:
                        for line in block.get("lines", []):
                            spans = line.get("spans", [])
                            text = "".join(s["text"] for s in spans).strip()
                            if text:
                                lines.append({"text": text, "bbox": line["bbox"],
                                              "size": max(s["size"] for s in spans)})
                    inventory.append({"book": book, "page": number, "lines": lines,
                                      "images": [{k: v for k, v in info.items() if k != "digest"}
                                                 for info in page.get_image_info(xrefs=True)]})
        (AUDIT / "inventory.json").write_text(json.dumps(inventory, ensure_ascii=False), encoding="utf-8")
        print("Inventoried", len(inventory), "PDF pages locally")
    if args.audit or args.render_warnings:
        rows = audit_original_warnings()
        if args.render_warnings:
            render_warning_sheets(rows)
    if args.validate:
        validate_parsed_entries()
    if args.pages:
        pairs = [tuple(map(int, pair.split(":"))) for pair in args.pages]
        print(render_pages(pairs, args.sheet))
        details = []
        for book, number in pairs:
            with fitz.open(book_paths()[book]) as doc:
                page = doc[number - 1]
                lines = []
                for block in page.get_text("dict")["blocks"]:
                    for line in block.get("lines", []):
                        spans = line.get("spans", [])
                        lines.append({"text": "".join(s["text"] for s in spans),
                                      "size": max((s["size"] for s in spans), default=0),
                                      "bbox": line["bbox"]})
                details.append({"book": book, "page": number, "lines": lines,
                                "images": [{k: v for k, v in info.items() if k != "digest"}
                                           for info in page.get_image_info(xrefs=True)]})
        (AUDIT / f"{args.sheet}.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
