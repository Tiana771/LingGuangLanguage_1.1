from __future__ import annotations

import hashlib
import html
import json
import posixpath
import re
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parent
BOOKS_DIR = ROOT
MEDIA_DIR = ROOT / "book_media"
REPORT_PATH = ROOT / "book_review_report_v2.json"
COLLECTION = "SignKnowledgeChunks"
STAGING_COLLECTION = "SignKnowledgeChunks_v2"


def _safe_member(name: str) -> str | None:
    normalized = unquote(name).replace("\\", "/")
    path = PurePosixPath(normalized)
    if path.is_absolute() or any(part in {"..", ""} for part in path.parts):
        return None
    return str(path)


def _clean_text(value: str) -> str:
    value = html.unescape(value).replace("\u3000", " ")
    lines = [re.sub(r"\s+", " ", line).strip() for line in value.splitlines()]
    return "\n".join(line for line in lines if line)


def _book_no(path: Path) -> int:
    match = re.search(r"第\s*([1-4])\s*册", path.name)
    return int(match.group(1)) if match else 0


def _entry_id(book: int, source: str, location: str, title: str) -> str:
    digest = hashlib.sha256(f"v2:{book}:{source}:{location}:{title}".encode("utf-8")).hexdigest()[:24]
    return f"book{book}-v2-{digest}"


def _entry_document(book: int, source: str, title: str, text: str, images: list[str], *,
                    page: int | None = None, anchor: str | None = None,
                    confidence: str = "high", review_reason: str = "",
                    review_status: str = "pending") -> dict[str, Any]:
    text = _clean_text(text)
    location = f"page:{page}" if page is not None else f"anchor:{anchor or ''}"
    image_paths = list(dict.fromkeys(images))
    content_hash = hashlib.sha256(json.dumps({
        "title": title, "text": text, "images": image_paths, "page": page, "anchor": anchor,
    }, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return {
        "_id": _entry_id(book, source, location, title),
        "book": book,
        "source_file": source,
        "source_title": f"第{book}册",
        "page": page,
        "anchor": anchor,
        "entry_title": title,
        "text": text,
        "image_paths": image_paths,
        "content_hash": content_hash,
        "parser_version": 2,
        "image_association": "entry_structure" if page is None else "pdf_vertical_band",
        "parse_confidence": confidence,
        "review_reason": review_reason,
        "review_status": review_status,
        "embedding": None,
    }


def _extract_epub_image(archive: zipfile.ZipFile, member: str, href: str,
                        names: set[str], media_root: Path, prefix: str,
                        source: str, review: list[dict[str, str]]) -> str | None:
    parsed = urlsplit(href)
    if parsed.scheme or parsed.netloc or not parsed.path:
        review.append({"file": source, "reason": f"非本地 EPUB 图片引用：{href[:100]}"})
        return None
    combined = posixpath.normpath(posixpath.join(str(PurePosixPath(member).parent), unquote(parsed.path)))
    safe_image = _safe_member(combined)
    if not safe_image or safe_image not in names:
        review.append({"file": source, "reason": f"词条图片路径无法匹配：{href[:100]}"})
        return None
    raw = archive.read(safe_image)
    suffix = Path(safe_image).suffix.lower()
    if len(raw) > 12 * 1024 * 1024 or suffix not in {".jpg", ".jpeg", ".png", ".webp", ".gif"}:
        review.append({"file": source, "reason": f"词条图片不支持或超过 12 MB：{safe_image}"})
        return None
    relative = Path(prefix) / f"{hashlib.sha256(safe_image.encode()).hexdigest()[:20]}{suffix}"
    target = media_root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(raw)
    return "/book-media/" + relative.as_posix()


def parse_epub(path: Path, media_root: Path = MEDIA_DIR) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    book = _book_no(path)
    review: list[dict[str, str]] = []
    chunks: list[dict[str, Any]] = []
    prefix = hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:12]
    try:
        archive = zipfile.ZipFile(path)
    except (OSError, zipfile.BadZipFile) as exc:
        return [], [{"file": path.name, "reason": f"EPUB 无法打开：{type(exc).__name__}"}]
    with archive:
        names = archive.namelist()
        if sum(item.file_size for item in archive.infolist()) > 1_000_000_000:
            return [], [{"file": path.name, "reason": "EPUB 解压后体积超过 1 GB，已停止解析"}]
        container = "META-INF/container.xml"
        if container not in names:
            return [], [{"file": path.name, "reason": "缺少 EPUB container.xml"}]
        try:
            rootfile = ElementTree.fromstring(archive.read(container)).find(".//{*}rootfile")
            opf_name = rootfile.attrib.get("full-path", "") if rootfile is not None else ""
            opf_safe = _safe_member(opf_name)
            if not opf_safe:
                raise ValueError("OPF 路径非法")
            opf_root = ElementTree.fromstring(archive.read(opf_safe))
        except Exception as exc:
            return [], [{"file": path.name, "reason": f"EPUB 目录信息无法解析：{type(exc).__name__}"}]
        opf_dir = PurePosixPath(opf_safe).parent
        manifest: dict[str, str] = {}
        nav_ids = set()
        for item in opf_root.findall(".//{*}manifest/{*}item"):
            href = item.attrib.get("href", "")
            safe = _safe_member(str(opf_dir / href))
            if safe:
                manifest[item.attrib.get("id", "")] = safe
                if "nav" in item.attrib.get("properties", "").split():
                    nav_ids.add(item.attrib.get("id", ""))
        toc_targets: dict[tuple[str, str], str] = {}
        for name in names:
            if not name.lower().endswith((".xhtml", ".html", ".htm")):
                continue
            try:
                toc_root = ElementTree.fromstring(archive.read(name))
            except ElementTree.ParseError:
                continue
            is_toc = PurePosixPath(name).name.lower().startswith(("toc", "nav")) or name in {
                manifest.get(item_id) for item_id in nav_ids
            }
            if not is_toc:
                continue
            for link in toc_root.iter():
                href = link.attrib.get("href", "")
                if not href:
                    continue
                parsed_link = urlsplit(href)
                target_file = posixpath.normpath(posixpath.join(str(PurePosixPath(name).parent), unquote(parsed_link.path)))
                target_safe = _safe_member(target_file)
                fragment = unquote(parsed_link.fragment)
                link_text = _clean_text("".join(link.itertext()))
                if target_safe and fragment and link_text:
                    toc_targets[(target_safe, fragment)] = link_text
        spine = opf_root.findall(".//{*}spine/{*}itemref")
        if not spine:
            review.append({"file": path.name, "reason": "EPUB 未声明 spine 阅读顺序"})
        for spine_index, itemref in enumerate(spine, 1):
            item_id = itemref.attrib.get("idref", "")
            member = manifest.get(item_id)
            if not member or member not in names:
                review.append({"file": path.name, "reason": f"spine 第 {spine_index} 项缺少章节文件"})
                continue
            if item_id in nav_ids or PurePosixPath(member).name.lower().startswith(("toc", "nav")):
                continue
            if archive.getinfo(member).file_size > 32 * 1024 * 1024:
                review.append({"file": path.name, "reason": f"章节超过 32 MB，已跳过：{member}"})
                continue
            try:
                root = ElementTree.fromstring(archive.read(member))
            except (ElementTree.ParseError, UnicodeDecodeError):
                review.append({"file": path.name, "reason": f"章节解析失败：{member}"})
                continue
            body = root.find(".//{*}body")
            if body is None:
                review.append({"file": path.name, "reason": f"章节没有 body：{member}"})
                continue
            current: dict[str, Any] | None = None
            records: list[dict[str, Any]] = []
            for node in list(body):
                tag = node.tag.rsplit("}", 1)[-1].lower()
                classes = set(node.attrib.get("class", "").split())
                if tag in {"h2", "h3"} and ("sect2" in classes or node.attrib.get("id")):
                    if current:
                        records.append(current)
                    anchor = node.attrib.get("id", "")
                    raw_title = _clean_text("".join(node.itertext()))
                    toc_title = toc_targets.get((member, anchor))
                    current = {"title": raw_title or toc_title or "", "anchor": anchor, "parts": [], "images": []}
                    if not raw_title:
                        review.append({"file": path.name, "reason": f"目录锚点缺少词条标题：{member}#{anchor}"})
                    continue
                if current is None or any(token in classes for token in ("toc", "index", "navigation")):
                    continue
                text_part = _clean_text("".join(node.itertext()))
                if text_part:
                    current["parts"].append(text_part)
                for image_node in node.iter():
                    if image_node.tag.rsplit("}", 1)[-1].lower() != "img":
                        continue
                    image_path = _extract_epub_image(
                        archive, member, image_node.attrib.get("src", ""), set(names),
                        media_root, prefix, path.name, review,
                    )
                    if image_path:
                        current["images"].append(image_path)
            if current:
                records.append(current)
            for record in records:
                title = record["title"]
                text = "\n".join([title, *record["parts"]]).strip()
                if not title or not re.search(r"[\u3400-\u9fff]", title) or len(text) < len(title) + 1:
                    review.append({"file": path.name, "reason": f"词条内容不完整，需审核：{member}#{record['anchor']}"})
                    continue
                toc_title = toc_targets.get((member, record["anchor"]))
                if toc_title and _canonical_entry_title(toc_title) != _canonical_entry_title(title):
                    review.append({"file": path.name, "reason": f"目录文字与正文标题不同：{member}#{record['anchor']}"})
                chunks.append(_entry_document(
                    book, path.name, title, text, record["images"], anchor=f"{member}#{record['anchor']}",
                    confidence="high" if toc_title else "medium",
                    review_reason="目录链接和正文锚点已对应" if toc_title else "正文标题未在目录链接中核实",
                    review_status="pending",
                ))
    return chunks, review


def _pdf_lines(page) -> list[dict[str, Any]]:
    lines = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = line.get("spans", [])
            text = "".join(span.get("text", "") for span in spans).strip()
            if text:
                lines.append({"text": text, "bbox": tuple(line["bbox"]),
                              "size": max((span.get("size", 0) for span in spans), default=0)})
    return sorted(lines, key=lambda line: (round(line["bbox"][1], 1), line["bbox"][0]))


def _pdf_heading_groups(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The dictionary uses 25 pt entry titles; 27 pt section names are not entries."""
    groups = []
    index = 0
    while index < len(lines):
        first = lines[index]
        if not 24 <= first["size"] <= 26:
            index += 1
            continue
        selected = [index]
        end = index
        while end + 1 < len(lines):
            candidate = lines[end + 1]
            previous = lines[end]
            gap = candidate["bbox"][1] - previous["bbox"][3]
            overlap = min(candidate["bbox"][3], previous["bbox"][3]) - max(candidate["bbox"][1], previous["bbox"][1])
            same_row = overlap >= 0.6 * min(candidate["bbox"][3] - candidate["bbox"][1],
                                          previous["bbox"][3] - previous["bbox"][1])
            if not (24 <= candidate["size"] <= 26 and (same_row or -6 <= gap <= 8)):
                break
            selected.append(end + 1)
            end += 1
        # Chinese and pinyin may be separate text blocks at the same baseline,
        # with fractionally different top coordinates. Restore left-to-right order.
        rows: list[list[int]] = []
        for i in selected:
            if rows and abs(lines[i]["bbox"][1] - lines[rows[-1][0]]["bbox"][1]) < 8:
                rows[-1].append(i)
            else:
                rows.append([i])
        title = " ".join(lines[i]["text"] for row in rows
                         for i in sorted(row, key=lambda n: lines[n]["bbox"][0]))
        # Includes number and Latin entries (0, CT, KTV, QQ, WIFI). A Chinese-only
        # appendix/index heading has no pronunciation and must not start an entry.
        if re.search(r"[A-Za-z\u00c0-\u024f]", title) and not re.match(r"^[一二三四五六七八九十]+画$", title):
            groups.append({"title": title, "indices": selected, "top": min(lines[i]["bbox"][1] for i in selected),
                           "bottom": max(lines[i]["bbox"][3] for i in selected)})
        index = end + 1
    return groups


def _pdf_non_entry_section(lines: list[dict[str, Any]]) -> bool:
    return any(
        re.match(r"^(?:附录|(?:词目|笔画|拼音|音序)索引|目\s*录)", line["text"])
        for line in lines[:5]
    )


def _pdf_reviewed_layout(path: Path) -> dict[int, list[dict[str, Any]]]:
    """Apply only local, visually checked exceptions for the exact source bytes."""
    config_path = ROOT / "book_pdf_layout_overrides.json"
    if not config_path.exists():
        return {}
    config = json.loads(config_path.read_text(encoding="utf-8"))
    source = next((s for s in config["sources"] if s["source_file"] == path.name), None)
    if source is None or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
        return {}
    grouped: dict[int, list[dict[str, Any]]] = {}
    for exception in source["images"]:
        grouped.setdefault(exception["page"], []).append(exception)
    return grouped


def parse_pdf(path: Path, media_root: Path = MEDIA_DIR) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read entry bands in reading order, including continuations on the next page.

    Ownership comes from headings and source rectangles, never image file order.
    Every new entry remains pending; the local audit records evidence separately.
    """
    try:
        import fitz
    except ImportError:
        return [], [{"file": path.name, "reason": "缺少 PyMuPDF，PDF 未解析"}]
    book = _book_no(path)
    chunks: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    active: dict[str, Any] | None = None
    reviewed_layout = _pdf_reviewed_layout(path)

    def finish():
        nonlocal active
        if active is None:
            return
        body = _clean_text("\n".join(active["parts"]))
        if not body or not re.search(r"[\u3400-\u9fff]", body):
            review.append({"file": path.name, "page": active["page"],
                           "entry_title": active["title"], "images": active["images"],
                           "reason": f"词条没有可核实的动作说明：PDF 第 {active['page']} 页，{active['title']}"})
        else:
            reason = "按标题字号、连续阅读顺序和图片坐标完成分条"
            if len(active["segments"]) > 1:
                reason += "；已包含跨页续文和续图"
            if active["ambiguous"]:
                reason = "图片跨越词条区域边界，仍需确认"
            item = _entry_document(
                book, path.name, active["title"], active["title"] + "\n" + body,
                active["images"], page=active["page"],
                confidence="medium" if active["ambiguous"] else "high", review_reason=reason,
            )
            item["source_segments"] = active["segments"]
            item["image_evidence"] = active["image_evidence"]
            item["page_end"] = active["segments"][-1]["page"]
            chunks.append(item)
        active = None

    with fitz.open(path) as document:
        for page_number, page in enumerate(document, 1):
            lines = _pdf_lines(page)
            if _pdf_non_entry_section(lines):
                finish()
                # The actual dictionary ends before its appendix. Later grammar
                # examples and alphabetical/stroke indexes are not entry data.
                if chunks and any(re.match(r"^附录", line["text"]) for line in lines[:5]):
                    break
                continue
            headings = _pdf_heading_groups(lines)
            if not headings and active is None:
                continue
            # Do not propagate an entry through blank pages or non-dictionary sections.
            if active and (not lines or page_number > active["page"] + 2):
                finish()
                if not headings:
                    continue
            placements = []
            for image_index, info in enumerate(page.get_image_info(xrefs=True), 1):
                bbox = list(info["bbox"])
                try:
                    if not info.get("xref"):
                        raise ValueError("inline image without xref")
                    extracted = document.extract_image(info["xref"])
                    data = extracted.get("image", b"")
                    suffix = "." + extracted.get("ext", "")
                    if not data or len(data) > 12 * 1024 * 1024 or suffix not in {".jpg", ".jpeg", ".png", ".webp"}:
                        raise ValueError("unsupported image")
                    relative = Path(f"book{book}") / (hashlib.sha256(data).hexdigest()[:24] + suffix)
                    target = media_root / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists():
                        target.write_bytes(data)
                    placement = {"bbox": bbox, "url": "/book-media/" + relative.as_posix()}
                    for exception in reviewed_layout.get(page_number, []):
                        if (any(h["title"] == exception["entry_title"] for h in headings)
                                and all(abs(a - b) < 0.5 for a, b in zip(bbox, exception["bbox"]))):
                            placement["reviewed_owner"] = exception["entry_title"]
                            placement["review_note"] = exception["reason"]
                            break
                    placements.append(placement)
                except Exception as exc:
                    review.append({"file": path.name, "page": page_number, "bbox": bbox,
                                   "reason": f"PDF 第 {page_number} 页图片 {image_index} 提取失败：{type(exc).__name__}"})

            # The first band belongs to the previous page's still-active entry.
            bands = [(None, 0.0, headings[0]["top"] if headings else page.rect.height)]
            bands.extend((heading, heading["top"], headings[i + 1]["top"] if i + 1 < len(headings) else page.rect.height)
                         for i, heading in enumerate(headings))
            for heading, top, bottom in bands:
                if heading:
                    finish()
                    active = {"title": heading["title"], "page": page_number, "parts": [], "images": [],
                              "segments": [], "image_evidence": [], "ambiguous": False}
                owned_images = [p for p in placements
                                if ((heading and p.get("reviewed_owner") == heading["title"])
                                    or (not p.get("reviewed_owner")
                                        and top <= (p["bbox"][1] + p["bbox"][3]) / 2 < bottom))]
                if active is None:
                    for placement in owned_images:
                        review.append({"file": path.name, "page": page_number, "images": [placement["url"]],
                                       "bbox": placement["bbox"],
                                       "reason": f"图片前方未找到相邻正文词条：PDF 第 {page_number} 页"})
                    continue
                excluded_lines = set(heading["indices"]) if heading else set()
                parts = [line["text"] for i, line in enumerate(lines)
                         if top <= line["bbox"][1] < bottom and i not in excluded_lines
                         and line["size"] < 24]
                if not parts and not owned_images and heading is None:
                    continue
                active["parts"].extend(parts)
                active["segments"].append({"page": page_number, "bbox": [0, top, page.rect.width, bottom]})
                for placement in owned_images:
                    bbox = placement["bbox"]
                    ambiguous = not placement.get("reviewed_owner") and (bbox[1] < top - 8 or bbox[3] > bottom + 8)
                    active["images"].append(placement["url"])
                    active["image_evidence"].append({
                        "page": page_number, "bbox": bbox, "path": placement["url"],
                        "role": "inline_symbol" if bbox[3] - bbox[1] < 30 else "illustration",
                        "boundary_confirmed": not ambiguous,
                        "association_method": "reviewed_layout_exception" if placement.get("reviewed_owner") else "heading_band",
                        **({"review_note": placement["review_note"]} if placement.get("review_note") else {}),
                    })
                    if ambiguous:
                        active["ambiguous"] = True
                        review.append({"file": path.name, "page": page_number, "images": [placement["url"]],
                                       "entry_title": active["title"], "bbox": bbox,
                                       "reason": f"图片跨越词条边界：PDF 第 {page_number} 页，{active['title']}"})
        finish()
    return chunks, review


def discover_books(directory: Path = BOOKS_DIR) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".pdf", ".epub"}) if directory.exists() else []


def _link_review_candidates(reviews: list[dict[str, str]], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    linked = []
    for warning in reviews:
        source = warning.get("file", "")
        reason = warning.get("reason", "")
        page_match = re.search(r"第\s*(\d+)\s*页", reason)
        anchor_match = re.search(r"([^\s：:]+\.x?html?)#([^\s，。；;]+)", reason, re.IGNORECASE)
        candidates = [item for item in chunks if item.get("source_file") == source]
        if page_match:
            candidates = [item for item in candidates if item.get("page") == int(page_match.group(1))]
        elif anchor_match:
            anchor = f"{anchor_match.group(1)}#{anchor_match.group(2)}"
            candidates = [item for item in candidates if item.get("anchor") == anchor]
        else:
            candidates = []
        linked.append({
            **warning,
            "related_chunks": [
                {"id": item["_id"], "title": item["entry_title"], "page": item.get("page"),
                 "anchor": item.get("anchor"), "images": item.get("image_paths", []),
                 "excerpt": item.get("text", "")[:400]}
                for item in candidates
            ],
        })
    return linked


def import_books(paths: list[Path] | None = None) -> dict[str, Any]:
    all_chunks: list[dict[str, Any]] = []
    reviews: list[dict[str, str]] = []
    for path in paths or discover_books():
        parsed, report = parse_epub(path) if path.suffix.lower() == ".epub" else parse_pdf(path)
        all_chunks.extend(parsed)
        reviews.extend(report)
    report = {
        "chunk_count": len(all_chunks),
        "approved_count": sum(item["review_status"] == "approved" for item in all_chunks),
        "review_count": sum(item["review_status"] == "pending" for item in all_chunks),
        "parse_warning_count": len(reviews),
        "review_items": _link_review_candidates(reviews, all_chunks),
        "pending_chunks": [
            {"id": item["_id"], "book": item["book"], "title": item["entry_title"],
             "source_file": item["source_file"], "page": item["page"], "images": item["image_paths"],
             "anchor": item.get("anchor"), "confidence": item["parse_confidence"],
             "reason": item["review_reason"], "excerpt": item["text"][:600]}
            for item in all_chunks if item["review_status"] == "pending"
        ],
    }
    REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"chunks": all_chunks, "report": report}


def keyword_terms(query: str) -> list[str]:
    terms = set(re.findall(r"[A-Za-z0-9]{2,}", query.casefold()))
    for sequence in re.findall(r"[\u3400-\u9fff]+", query):
        compact_sequence = re.sub(r"\s+", "", sequence)
        if len(compact_sequence) == 1:
            terms.add(compact_sequence)
        elif len(compact_sequence) <= 4:
            terms.add(compact_sequence)
        else:
            terms.update(compact_sequence[index:index + 2] for index in range(len(compact_sequence) - 1))
    compact_query = re.sub(r"\s+", "", query.casefold())
    if compact_query:
        terms.add(compact_query)
    return sorted(terms, key=len, reverse=True)[:20]


_ENTRY_QUERY_SUFFIXES = (
    "有关的手语", "相关的手语", "有关手语", "相关手语", "有关的词条", "相关的词条",
    "相关的词", "有关的词", "有关的动作", "相关的动作",
    "是什么意思", "什么意思", "怎么做", "怎么说", "如何做", "是什么", "怎么比",
    "怎么打", "如何比", "如何说", "的手语", "手语怎么做", "手语",
)

_ENTRY_QUERY_PREFIXES = (
    "请告诉我", "我想知道", "我想查找", "我想查询", "我想查", "帮我查找", "帮我查询",
    "帮我找", "搜索一下", "查询一下", "查找一下", "搜索", "查询", "查找", "关于", "和", "跟",
)


def _entry_query_term(query: str) -> str:
    value = re.sub(r"\s+", "", query.casefold())
    changed = True
    while changed:
        changed = False
        for suffix in sorted(_ENTRY_QUERY_SUFFIXES, key=len, reverse=True):
            if value.endswith(suffix) and len(value) > len(suffix):
                value = value[:-len(suffix)]
                changed = True
                break
        if changed:
            continue
        for prefix in sorted(_ENTRY_QUERY_PREFIXES, key=len, reverse=True):
            if value.startswith(prefix) and len(value) > len(prefix):
                value = value[len(prefix):]
                changed = True
                break
    return value.strip("，。！？?！、：:")


def _canonical_entry_title(title: str) -> str:
    title = re.sub(r"\s+", "", title.casefold())
    match = re.match(r"[\u3400-\u9fff]+[①②③④⑤⑥⑦⑧⑨⑩]?(?:[（(][一二三四五六七八九十0-9]+[）)])?", title)
    return match.group(0) if match else title


def _exact_entry_matches(chunks: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
    term = _entry_query_term(query)
    if not term:
        return []
    query_key = _canonical_entry_title(term)
    numbered = bool(re.search(r"[①②③④⑤⑥⑦⑧⑨⑩]$", query_key))
    return [
        chunk for chunk in chunks
        if (
            _canonical_entry_title(str(chunk.get("entry_title", ""))) == query_key
            or (not numbered and re.fullmatch(re.escape(query_key) + r"[①②③④⑤⑥⑦⑧⑨⑩]", _canonical_entry_title(str(chunk.get("entry_title", "")))))
        )
    ]


def _query_evidence(chunk: dict[str, Any], query: str) -> dict[str, Any] | None:
    term = _entry_query_term(query)
    text = str(chunk.get("text", ""))
    if not term:
        return None
    match = re.search(re.escape(term), text, re.IGNORECASE)
    if not match:
        return None

    lines = text.splitlines()
    matching_lines = [index for index, line in enumerate(lines) if re.search(re.escape(term), line, re.IGNORECASE)]
    descriptive_lines = [
        index for index in matching_lines
        if not re.fullmatch(rf"\s*{re.escape(term)}\s*[：:]?\s*\d{{1,4}}\s*", lines[index], re.IGNORECASE)
    ]
    if not descriptive_lines:
        return None

    line_index = descriptive_lines[0]
    start_line = line_index
    end_line = min(len(lines), line_index + 3)
    excerpt = "\n".join(lines[start_line:end_line]).strip()
    if len(excerpt) > 700:
        start = max(0, match.start() - 180)
        excerpt = text[start:match.end() + 420].strip()

    result = {**chunk, "text": excerpt}
    title = str(chunk.get("entry_title", ""))
    if _canonical_entry_title(title) != term:
        result["entry_title"] = f"{term}（原文片段）"
    return result


def keyword_search(chunks: list[dict[str, Any]], query: str, limit: int = 5) -> list[dict[str, Any]]:
    exact_matches = _exact_entry_matches(chunks, query)
    if exact_matches:
        results = []
        for chunk in exact_matches:
            evidence = _query_evidence(chunk, query)
            results.append({**chunk, "score": 1000})
        return results[:limit]

    terms = keyword_terms(query)
    normalized = re.sub(r"\s+", "", query.casefold())
    simple_entry_query = bool(re.fullmatch(r"[\u3400-\u9fff]+[①②③④⑤⑥⑦⑧⑨⑩]?", _canonical_entry_title(_entry_query_term(query))))
    scored = []
    for chunk in chunks:
        haystack = f"{chunk.get('entry_title', '')} {chunk.get('text', '')}".casefold()
        compact = re.sub(r"\s+", "", haystack)
        score = sum(haystack.count(term) for term in terms)
        if normalized and normalized in compact:
            score += max(2, len(terms))
        if score and (not simple_entry_query or _entry_query_term(query) in _canonical_entry_title(str(chunk.get("entry_title", "")))):
            evidence = _query_evidence(chunk, query)
            if evidence:
                scored.append((score, evidence))
    scored.sort(key=lambda pair: (-pair[0], pair[1].get("book", 0), pair[1].get("page") or 0))
    return [{**chunk, "score": score} for score, chunk in scored[:limit]]


def merge_knowledge_results(
    vector_results: list[dict[str, Any]], keyword_results: list[dict[str, Any]],
    query: str, limit: int = 5,
) -> list[dict[str, Any]]:
    exact_title_matches = _exact_entry_matches(keyword_results, query)
    if exact_title_matches:
        return [{**item, "score": max(item.get("score", 0), 1000)} for item in exact_title_matches[:limit]]

    remaining_keywords = []
    vector_ids = {str(item.get("_id", "")) for item in vector_results}
    for item in keyword_results:
        if str(item.get("_id", "")) not in vector_ids:
            remaining_keywords.append(item)

    ordered = vector_results + remaining_keywords
    deduplicated = []
    seen = set()
    for item in ordered:
        item_id = str(item.get("_id", ""))
        if item_id in seen:
            continue
        seen.add(item_id)
        deduplicated.append(item)
        if len(deduplicated) >= limit:
            break
    return deduplicated
