"""user_prompt_builder.py — 主模型 user prompt 总装

统一组装主模型每轮调用的 user content。
当前包括：
- <memory> 块
- <goals> 块
- <style> 块
- <social_tips> 块
- <world> 顶层包裹
- <current_time> 块
- <unread_info> 块
- <qq> 内层包裹
- 聊天记录 XML / 多模态内容
- <system_reminder> 末尾附加块
"""

import base64
import html
from typing import cast
from urllib.parse import urlparse, ParseResult

from .final_reminder import append_final_reminder
from .history_window import has_previous_messages, load_history_window
from .unread_builder import build_unread_info_xml
from .xml_builder import build_forward_browser_content, build_multimodal_content
from ..media.outbound_image import make_data_url
from ..compression.config import (
    DEFAULT_WORLD_MULTIMODAL_IMAGE_LIMIT,
    normalize_generation_config,
    normalize_world_multimodal_image_limit,
)
from ..session import sessions
from browser_adapter.config import (
    DEFAULT_BROWSER_MULTIMODAL_IMAGE_LIMIT,
    browser_multimodal_image_limit as _config_browser_multimodal_image_limit,
)

BROWSER_CLOSED_WORLD_TAG = '<browser active="false" state="closed"/>'


def _build_prompt_block(tag: str, content: str) -> str:
    """构建一个简单的 XML 文本块。"""
    normalized = content.strip()
    if normalized:
        return f"<{tag}>\n{normalized}\n</{tag}>"
    return f"<{tag}>\n</{tag}>"


def _prepend_text_block(content: "str | list", text: str) -> "str | list":
    """给 user prompt 前部插入纯文本块。"""
    if isinstance(content, str):
        return text + "\n" + content
    return [{"type": "text", "text": text + "\n"}] + content


def _append_text_part(parts: list, text: str) -> None:
    if not text:
        return
    if parts and isinstance(parts[-1], dict) and parts[-1].get("type") == "text":
        parts[-1] = {**parts[-1], "text": parts[-1].get("text", "") + text}
    else:
        parts.append({"type": "text", "text": text})


def _xml_text(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=False)


def _xml_attr(value: object) -> str:
    return html.escape("" if value is None else str(value), quote=True)


def _is_image_url_part(part: dict) -> bool:
    return isinstance(part, dict) and part.get("type") == "image_url"


def _limit_multimodal_image_parts(content: "str | list", limit: int) -> "str | list":
    """Keep at most the last ``limit`` real image_url parts in the prompt."""
    if isinstance(content, str) or limit < 0:
        return content
    image_count = sum(1 for part in content if _is_image_url_part(part))
    overflow = image_count - limit
    if overflow <= 0:
        return content

    limited: list = []
    remaining_to_drop = overflow
    for part in content:
        if _is_image_url_part(part) and remaining_to_drop > 0:
            remaining_to_drop -= 1
            continue
        limited.append(part)
    return limited


def _browser_image_part(image: dict) -> dict | None:
    data = image.get("data")
    if isinstance(data, bytes):
        b64 = base64.b64encode(data).decode("ascii")
    elif isinstance(data, str):
        b64 = data
    else:
        return None
    data_url = make_data_url(b64, str(image.get("mime_type") or "image/jpeg"))
    if not data_url:
        return None
    return {"type": "image_url", "image_url": {"url": data_url}}


def _rect_attr(rect: object) -> str:
    if not isinstance(rect, dict):
        return "0,0,0,0"
    return ",".join(_xml_attr(rect.get(key, 0)) for key in ("x", "y", "width", "height"))


def _target_key(value: object) -> str:
    return "" if value is None else str(value)


def _has_attr_value(value: object) -> bool:
    return value is not None and str(value) != ""


def _state_attr(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return _xml_attr(value)


def _target_tag_adds_semantics(item: dict) -> bool:
    tag = str(item.get("tag") or "")
    if not tag:
        return False
    role = str(item.get("role") or "")
    input_type = str(item.get("input_type") or "")
    if tag == role:
        return False
    if role == "link" and tag in {"a", "area"}:
        return False
    if role == "button" and tag == "button":
        return False
    if role == "textbox" and tag in {"input", "textarea"}:
        return False
    if tag == "input" and input_type:
        return False
    if role == "select" and tag == "select":
        return False
    if role == "combobox" and tag in {"input", "select"}:
        return False
    return True


def _text_tag_adds_semantics(block: dict) -> bool:
    tag = str(block.get("tag") or "").strip().lower()
    if not tag:
        return False
    kind = str(block.get("kind") or "").strip().lower()
    if len(tag) == 2 and tag[0] == "h" and tag[1].isdigit():
        return True
    if "-" in tag:
        return True
    if tag == "a":
        return not bool(_collect_referenced_targets_from_parts(block.get("parts")))

    represented_by_kind = {
        ("button", "button"),
        ("label", "label"),
        ("label", "legend"),
        ("list_item", "li"),
        ("paragraph", "p"),
        ("pre", "pre"),
        ("quote", "blockquote"),
        ("section", "article"),
        ("section", "section"),
        ("table_cell", "td"),
        ("table_cell", "th"),
        ("time", "time"),
    }
    if (kind, tag) in represented_by_kind:
        return False
    if tag in {"div", "span", "p", "li"}:
        return False
    return tag in {
        "abbr",
        "caption",
        "cite",
        "code",
        "dd",
        "dfn",
        "dt",
        "em",
        "figcaption",
        "kbd",
        "mark",
        "q",
        "s",
        "samp",
        "small",
        "strong",
        "sub",
        "summary",
        "sup",
        "var",
    }


def _href_target_attrs(href: object, page_url: object) -> list[str]:
    raw = str(href or "").strip()
    if not raw:
        return []
    try:
        parsed: ParseResult = urlparse(raw)
    except Exception:
        return [f'href="{_xml_attr(raw)}"']

    try:
        page = urlparse(str(page_url or ""))
    except Exception:
        page = None
    same_origin = (
        page is not None
        and parsed.scheme == page.scheme
        and parsed.netloc == page.netloc
    )
    same_document = (
        same_origin
        and page is not None
        and (parsed.path or "/") == (page.path or "/")
    )

    attrs: list[str] = []
    if parsed.scheme and parsed.scheme not in {"http", "https", "file"}:
        attrs.append(f'href_scheme="{_xml_attr(parsed.scheme)}"')
    if parsed.netloc and not same_origin:
        attrs.append(f'href_host="{_xml_attr(parsed.netloc)}"')

    path = parsed.path or ""
    if parsed.fragment and same_document:
        attrs.append(f'href_anchor="{_xml_attr(parsed.fragment)}"')
    elif path and path != "/":
        if _opaque_href_path(path):
            attrs.append('href_path_kind="opaque"')
        else:
            attrs.append(f'href_path="{_xml_attr(path)}"')
    if attrs:
        return attrs
    if same_origin:
        return []
    return [f'href="{_xml_attr(raw)}"']


def _opaque_href_path(path: str) -> bool:
    if len(path) <= 240:
        return False
    segments = [segment for segment in path.split("/") if segment]
    if len(segments) >= 2 and segments[0] in {"x", "sspa", "gp"}:
        return True
    return any(len(segment) > 96 for segment in segments)


def _href_is_opaque(href: object) -> bool:
    raw = str(href or "").strip()
    if not raw:
        return False
    try:
        parsed = urlparse(raw)
    except Exception:
        return len(raw) > 240
    return _opaque_href_path(parsed.path or "")


def _rect_tuple(rect: object) -> tuple[float, float, float, float] | None:
    if not isinstance(rect, dict):
        return None
    try:
        x = float(rect.get("x") or 0)
        y = float(rect.get("y") or 0)
        width = float(rect.get("width") or 0)
        height = float(rect.get("height") or 0)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def _rect_area(rect: tuple[float, float, float, float]) -> float:
    return max(0.0, rect[2]) * max(0.0, rect[3])


def _intersection_area(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> float:
    left = max(first[0], second[0])
    top = max(first[1], second[1])
    right = min(first[0] + first[2], second[0] + second[2])
    bottom = min(first[1] + first[3], second[1] + second[3])
    return max(0.0, right - left) * max(0.0, bottom - top)


def _rect_covers(
    container: tuple[float, float, float, float],
    child: tuple[float, float, float, float],
    *,
    minimum_ratio: float = 0.65,
) -> bool:
    child_area = _rect_area(child)
    if child_area <= 0:
        return False
    return _intersection_area(container, child) / child_area >= minimum_ratio


def _collect_referenced_targets_from_parts(parts: object) -> set[str]:
    refs: set[str] = set()
    if not isinstance(parts, list):
        return refs
    for part in parts:
        if isinstance(part, dict) and part.get("type") == "ref":
            refs.add(_target_key(part.get("target")))
    return refs


def _collect_referenced_targets(tables: list[dict], text_blocks: object) -> set[str]:
    refs: set[str] = set()
    for table in tables:
        for row in table.get("rows") or []:
            if not isinstance(row, dict):
                continue
            for cell in row.get("cells") or []:
                if isinstance(cell, dict):
                    refs.update(_collect_referenced_targets_from_parts(cell.get("parts")))
    if isinstance(text_blocks, list):
        for block in text_blocks:
            if isinstance(block, dict):
                refs.update(_collect_referenced_targets_from_parts(block.get("parts")))
    return refs


def _has_equivalent_visible_href(
    item: dict,
    click_targets: list,
    referenced_targets: set[str],
) -> bool:
    href = str(item.get("href") or "").strip()
    if not href:
        return False
    item_key = _target_key(item.get("index"))
    for other in click_targets:
        if not isinstance(other, dict):
            continue
        if _target_key(other.get("index")) == item_key:
            continue
        if _target_key(other.get("index")) not in referenced_targets:
            continue
        if other.get("role") != "link" or other.get("source") != "visible":
            continue
        if str(other.get("href") or "").strip() == href:
            return True
    return False


def _covers_referenced_visible_link(
    item: dict,
    click_targets: list,
    referenced_targets: set[str],
) -> bool:
    item_rect = _rect_tuple(item.get("rect"))
    if item_rect is None:
        return False
    item_key = _target_key(item.get("index"))
    for other in click_targets:
        if not isinstance(other, dict):
            continue
        if _target_key(other.get("index")) == item_key:
            continue
        if _target_key(other.get("index")) not in referenced_targets:
            continue
        if other.get("role") != "link" or other.get("source") != "visible":
            continue
        other_rect = _rect_tuple(other.get("rect"))
        if other_rect is not None and _rect_covers(item_rect, other_rect):
            return True
    return False


def _equivalent_covered_targets(
    click_targets: list,
    referenced_targets: set[str],
) -> set[str]:
    covered: set[str] = set()
    for item in click_targets:
        if not isinstance(item, dict) or item.get("role") != "link":
            continue
        source = str(item.get("source") or "")
        key = _target_key(item.get("index"))
        if source in {"alt", "graphic", "visual", "href_tail"} and _has_equivalent_visible_href(
            item,
            click_targets,
            referenced_targets,
        ):
            covered.add(key)
            continue
        rect = _rect_tuple(item.get("rect"))
        thin_visible_sliver = rect is not None and (rect[2] <= 5 or rect[3] <= 5)
        if source == "href_tail" and (
            thin_visible_sliver
            or (
                _href_is_opaque(item.get("href"))
                and _covers_referenced_visible_link(item, click_targets, referenced_targets)
            )
        ):
            covered.add(key)
    return covered


def _render_mixed_text_parts(
    parts: object,
    targets_by_index: dict[str, dict],
    inlined_visible_targets: set[str],
    referenced_targets: set[str],
    suppressed_targets: set[str],
) -> str:
    if not isinstance(parts, list):
        return ""
    labeled_ref_ids = {
        _target_key(part.get("target"))
        for part in parts
        if isinstance(part, dict)
        and part.get("type") == "ref"
        and _target_key(part.get("target")) not in suppressed_targets
        and isinstance(targets_by_index.get(_target_key(part.get("target"))), dict)
        and targets_by_index[_target_key(part.get("target"))].get("source") == "visible"
        and str(targets_by_index[_target_key(part.get("target"))].get("name") or "").strip()
    }
    rendered: list[str] = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "ref":
            target_id = _target_key(part.get("target"))
            if target_id in suppressed_targets:
                continue
            target = targets_by_index.get(target_id)
            if isinstance(target, dict):
                referenced_targets.add(target_id)
            label = ""
            if isinstance(target, dict) and target.get("source") == "visible":
                label = str(target.get("name") or "").strip()
            if label:
                inlined_visible_targets.add(target_id)
                rendered.append(f'<ref target="{_xml_attr(part.get("target"))}">{_xml_text(label)}</ref>')
            elif labeled_ref_ids:
                continue
            else:
                rendered.append(f'<ref target="{_xml_attr(part.get("target"))}"/>')
        elif part.get("type") == "text":
            rendered.append(_xml_text(part.get("text")))
    return "".join(rendered)


_REFERENCED_LINK_DETAIL_KEYS = {
    "aria_busy",
    "aria_checked",
    "aria_controls",
    "aria_disabled",
    "aria_expanded",
    "aria_haspopup",
    "aria_pressed",
    "aria_selected",
    "blocked",
    "checked",
    "deleted",
    "description",
    "error_message",
    "indeterminate",
    "inserted",
    "marked",
    "open",
    "popover_action",
    "popover_open",
    "popover_target",
    "text_tone",
}


def _meaningful_current_attr(value: object) -> bool:
    if not _has_attr_value(value):
        return False
    return str(value).strip().lower() not in {"false", "none"}


def _include_state_attr(key: str, value: object) -> bool:
    if key == "aria_current":
        return _meaningful_current_attr(value)
    return _has_attr_value(value)


def _target_detail_needed(item: dict, referenced_targets: set[str]) -> bool:
    if _target_key(item.get("index")) not in referenced_targets:
        return True
    if item.get("role") != "link":
        return True
    if item.get("group") == "control":
        return True
    if item.get("source") != "visible":
        return True
    if not str(item.get("name") or "").strip():
        return True
    if item.get("disabled"):
        return True
    if _meaningful_current_attr(item.get("aria_current")):
        return True
    return any(_has_attr_value(item.get(key)) for key in _REFERENCED_LINK_DETAIL_KEYS)


def _browser_image_xml_line(image: dict, *, embedded: bool | None = None) -> str:
    attrs = [
        f'kind="{_xml_attr(image.get("kind"))}"',
        f'ref="{_xml_attr(image.get("ref"))}"',
        f'alt="{_xml_attr(image.get("alt"))}"',
        f'width="{_xml_attr(image.get("width"))}"',
        f'height="{_xml_attr(image.get("height"))}"',
        f'x="{_xml_attr(image.get("x"))}"',
        f'y="{_xml_attr(image.get("y"))}"',
    ]
    if embedded is not None:
        attrs.append(f'embedded="{str(bool(embedded)).lower()}"')
    if _has_attr_value(image.get("frame")):
        attrs.append(f'frame="{_xml_attr(image.get("frame"))}"')
    if _has_attr_value(image.get("pseudo")):
        attrs.append(f'pseudo="{_xml_attr(image.get("pseudo"))}"')
    if image.get("loaded") is False:
        attrs.append('loaded="false"')
    return "    <image " + " ".join(attrs) + "/>"


def _browser_viewport_image_xml_line(viewport: dict, viewport_part: dict | None) -> str:
    return (
        '  <viewport_image '
        f'ref="{_xml_attr(viewport.get("ref"))}" '
        f'embedded="{str(viewport_part is not None).lower()}" '
        'overlay="click_index"/>'
    )


def _render_browser_world_content(
    snapshot: dict | None,
    multimodal_image_limit: int | None = None,
) -> "str | list":
    if not isinstance(snapshot, dict) or not snapshot.get("active"):
        return BROWSER_CLOSED_WORLD_TAG

    if multimodal_image_limit is None:
        multimodal_image_limit = _browser_multimodal_image_limit()
    image_budget = normalize_world_multimodal_image_limit(multimodal_image_limit)
    scroll_data = snapshot.get("scroll")
    scroll: dict = cast(dict, scroll_data) if isinstance(scroll_data, dict) else {}
    page_url = snapshot.get("url")
    viewport = snapshot.get("viewport") if isinstance(snapshot.get("viewport"), dict) else None
    viewport_part = _browser_image_part(viewport) if viewport else None
    if image_budget == 0:
        viewport_part = None
    remaining_image_budget = None if image_budget < 0 else max(0, image_budget - (1 if viewport_part else 0))
    all_images = [image for image in (snapshot.get("images") or []) if isinstance(image, dict)]

    embedded_image_parts: dict[int, dict] = {}
    for index, image in enumerate(all_images):
        part = _browser_image_part(image)
        if part is None:
            continue
        if remaining_image_budget is not None:
            if remaining_image_budget <= 0:
                continue
            remaining_image_budget -= 1
        embedded_image_parts[index] = part
    image_parts_count = len(embedded_image_parts) + (1 if viewport_part is not None else 0)

    lines = [
        '<browser active="true" format="browser_world_v1">',
        f'  <page url="{_xml_attr(snapshot.get("url"))}" title="{_xml_attr(snapshot.get("title"))}"/>',
    ]
    viewport_size = snapshot.get("viewport_size") if isinstance(snapshot.get("viewport_size"), dict) else {}
    if viewport_size:
        lines.append(
            '  <viewport '
            f'width="{_xml_attr(viewport_size.get("width", 0))}" '
            f'height="{_xml_attr(viewport_size.get("height", 0))}" '
            'space="viewport_css_px"/>'
        )
    lines.append(
        (
            '  <scroll '
            f'y="{_xml_attr(scroll.get("y", 0))}" '
            f'viewport_height="{_xml_attr(scroll.get("viewport_height", 0))}" '
            f'page_height="{_xml_attr(scroll.get("page_height", 0))}" '
            f'can_scroll_up="{str(bool(scroll.get("can_scroll_up"))).lower()}" '
            f'can_scroll_down="{str(bool(scroll.get("can_scroll_down"))).lower()}"/>'
        )
    )
    if pending := snapshot.get("pending_click"):
        if isinstance(pending, dict):
            lines.append(
                f'  <pending_click x="{_xml_attr(pending.get("x"))}" y="{_xml_attr(pending.get("y"))}"/>'
            )

    frames = [item for item in (snapshot.get("frames") or []) if isinstance(item, dict)]
    if frames:
        lines.append(
            f'  <frames items="{len(frames)}" '
            'scope="viewport" space="viewport_css_px">'
        )
        for frame in frames:
            attrs = [
                f'index="{_xml_attr(frame.get("index"))}"',
                f'tag="{_xml_attr(frame.get("tag") or "iframe")}"',
                f'rect="{_rect_attr(frame.get("rect"))}"',
            ]
            for key in ("name", "source", "url"):
                if _has_attr_value(frame.get(key)):
                    attrs.append(f'{key}="{_xml_attr(frame.get(key))}"')
            if _has_attr_value(frame.get("title")) and str(frame.get("title")) != str(frame.get("name")):
                attrs.append(f'title="{_xml_attr(frame.get("title"))}"')
            lines.append("    <frame " + " ".join(attrs) + "/>")
        lines.append("  </frames>")

    scroll_regions = [item for item in (snapshot.get("scroll_regions") or []) if isinstance(item, dict)]
    if scroll_regions:
        lines.append(
            f'  <scroll_regions items="{len(scroll_regions)}" '
            'order="viewport" space="viewport_css_px" action="browser_control.scroll_region(index,pixels)">'
        )
        for region in scroll_regions:
            attrs = [
                f'index="{_xml_attr(region.get("index"))}" '
                f'role="{_xml_attr(region.get("role"))}" '
                f'rect="{_rect_attr(region.get("rect"))}" '
                f'scroll_y="{_xml_attr(region.get("scroll_y", 0))}" '
                f'viewport_height="{_xml_attr(region.get("viewport_height", 0))}" '
                f'scroll_height="{_xml_attr(region.get("scroll_height", 0))}" '
                f'can_scroll_up="{str(bool(region.get("can_scroll_up"))).lower()}" '
                f'can_scroll_down="{str(bool(region.get("can_scroll_down"))).lower()}"'
            ]
            if region.get("tag") and region.get("tag") != region.get("role"):
                attrs.append(f'tag="{_xml_attr(region.get("tag"))}"')
            if region.get("name"):
                attrs.append(f'name="{_xml_attr(region.get("name"))}"')
            if _has_attr_value(region.get("frame")):
                attrs.append(f'frame="{_xml_attr(region.get("frame"))}"')
            try:
                has_horizontal_scroll = float(region.get("scroll_width") or 0) > float(region.get("viewport_width") or 0) + 2
            except (TypeError, ValueError):
                has_horizontal_scroll = False
            if has_horizontal_scroll:
                attrs.append(f'scroll_x="{_xml_attr(region.get("scroll_x", 0))}"')
                attrs.append(f'viewport_width="{_xml_attr(region.get("viewport_width", 0))}"')
                attrs.append(f'scroll_width="{_xml_attr(region.get("scroll_width", 0))}"')
                attrs.append(f'can_scroll_left="{str(bool(region.get("can_scroll_left"))).lower()}"')
                attrs.append(f'can_scroll_right="{str(bool(region.get("can_scroll_right"))).lower()}"')
            lines.append("    <region " + " ".join(attrs) + "/>")
        lines.append("  </scroll_regions>")

    click_targets = snapshot.get("click_targets") or []
    targets_by_index = {
        _target_key(item.get("index")): item
        for item in click_targets
        if isinstance(item, dict)
    }
    inlined_visible_targets: set[str] = set()
    referenced_targets: set[str] = set()

    tables = [item for item in (snapshot.get("tables") or []) if isinstance(item, dict)]
    text_blocks = snapshot.get("text_blocks") or []
    raw_referenced_targets = _collect_referenced_targets(tables, text_blocks)
    equivalent_covered_targets = _equivalent_covered_targets(click_targets, raw_referenced_targets)
    if tables:
        lines.append(
            f'  <tables items="{len(tables)}" '
            'scope="viewport" space="viewport_css_px" inline_refs="target_index">'
        )
        for table in tables:
            table_attrs = [
                f'role="{_xml_attr(table.get("role"))}" '
                f'tag="{_xml_attr(table.get("tag"))}" '
                f'rect="{_rect_attr(table.get("rect"))}"'
            ]
            if table.get("name"):
                table_attrs.append(f'name="{_xml_attr(table.get("name"))}"')
            if _has_attr_value(table.get("frame")):
                table_attrs.append(f'frame="{_xml_attr(table.get("frame"))}"')
            lines.append("    <table " + " ".join(table_attrs) + ">")
            for row in table.get("rows") or []:
                if not isinstance(row, dict):
                    continue
                row_attrs = [
                    f'rect="{_rect_attr(row.get("rect"))}"',
                ]
                if _has_attr_value(row.get("row_index")):
                    row_attrs.append(f'row_index="{_xml_attr(row.get("row_index"))}"')
                lines.append("      <row " + " ".join(row_attrs) + ">")
                for cell in row.get("cells") or []:
                    if not isinstance(cell, dict):
                        continue
                    cell_attrs = [
                        f'row="{_xml_attr(cell.get("row"))}" '
                        f'col="{_xml_attr(cell.get("col"))}" '
                        f'tag="{_xml_attr(cell.get("tag"))}" '
                        f'header="{str(bool(cell.get("header"))).lower()}" '
                        f'rect="{_rect_attr(cell.get("rect"))}"'
                    ]
                    if cell.get("role"):
                        cell_attrs.append(f'role="{_xml_attr(cell.get("role"))}"')
                    if _has_attr_value(cell.get("row_index")):
                        cell_attrs.append(f'row_index="{_xml_attr(cell.get("row_index"))}"')
                    if _has_attr_value(cell.get("col_index")):
                        cell_attrs.append(f'col_index="{_xml_attr(cell.get("col_index"))}"')
                    if _has_attr_value(cell.get("aria_sort")):
                        cell_attrs.append(f'aria_sort="{_xml_attr(cell.get("aria_sort"))}"')
                    if _has_attr_value(cell.get("aria_busy")):
                        cell_attrs.append(f'aria_busy="{_xml_attr(cell.get("aria_busy"))}"')
                    for key in ("deleted", "inserted", "marked", "background_tone", "text_tone"):
                        if _include_state_attr(key, cell.get(key)):
                            cell_attrs.append(f'{key}="{_state_attr(cell.get(key))}"')
                    if _has_attr_value(cell.get("rowspan")) and str(cell.get("rowspan")) != "1":
                        cell_attrs.append(f'rowspan="{_xml_attr(cell.get("rowspan"))}"')
                    if _has_attr_value(cell.get("colspan")) and str(cell.get("colspan")) != "1":
                        cell_attrs.append(f'colspan="{_xml_attr(cell.get("colspan"))}"')
                    lines.append(
                        "        <cell "
                        + " ".join(cell_attrs)
                        + ">"
                        + _render_mixed_text_parts(
                            cell.get("parts"),
                            targets_by_index,
                            inlined_visible_targets,
                            referenced_targets,
                            equivalent_covered_targets,
                        )
                        + "</cell>"
                    )
                lines.append("      </row>")
            lines.append("    </table>")
        lines.append("  </tables>")

    indicators = [item for item in (snapshot.get("indicators") or []) if isinstance(item, dict)]
    if indicators:
        lines.append(
            f'  <indicators items="{len(indicators)}" '
            'scope="viewport" space="viewport_css_px">'
        )
        for item in indicators:
            attrs = [
                f'kind="{_xml_attr(item.get("kind"))}" '
                f'tag="{_xml_attr(item.get("tag"))}" '
                f'rect="{_rect_attr(item.get("rect"))}"'
            ]
            if _has_attr_value(item.get("frame")):
                attrs.append(f'frame="{_xml_attr(item.get("frame"))}"')
            for key in (
                "role",
                "name",
                "source",
                "variant",
                "text",
                "input_type",
                "value",
                "placeholder",
                "checked",
                "selected_text",
                "selected_options",
                "visible_options",
                "disabled_options",
                "multiple",
                "aria_checked",
                "aria_controls",
                "aria_current",
                "aria_expanded",
                "aria_haspopup",
                "aria_invalid",
                "invalid",
                "invalid_reason",
                "aria_autocomplete",
                "aria_busy",
                "deleted",
                "inserted",
                "marked",
                "background_tone",
                "text_tone",
                "tone",
                "color",
                "aria_level",
                "aria_posinset",
                "aria_setsize",
                "aria_pressed",
                "aria_required",
                "aria_sort",
                "aria_selected",
                "value_min",
                "value_max",
                "value_low",
                "value_high",
                "value_optimum",
                "aria_value_now",
                "aria_value_min",
                "aria_value_max",
                "aria_value_text",
                "description",
                "indeterminate",
                "controls",
                "paused",
                "muted",
                "current_time",
                "duration",
                "loop",
                "autoplay",
                "playback_rate",
                "volume",
                "required",
                "read_only",
                "blocked",
                "inert",
                "aria_disabled",
                "disabled",
            ):
                if _include_state_attr(key, item.get(key)):
                    attrs.append(f'{key}="{_state_attr(item.get(key))}"')
            lines.append("    <indicator " + " ".join(attrs) + "/>")
        lines.append("  </indicators>")

    lines.append(
        f'  <text_blocks items="{len(text_blocks)}" '
        'scope="viewport" space="viewport_css_px" inline_refs="target_index">'
    )
    for block in text_blocks:
        if not isinstance(block, dict):
            continue
        attrs = [
            f'kind="{_xml_attr(block.get("kind"))}"',
            f'rect="{_rect_attr(block.get("rect"))}"',
        ]
        if _text_tag_adds_semantics(block):
            attrs.insert(1, f'tag="{_xml_attr(block.get("tag"))}"')
        for key in (
            "aria_current",
            "aria_expanded",
            "aria_invalid",
            "aria_pressed",
            "aria_sort",
            "aria_selected",
            "aria_checked",
            "aria_busy",
            "deleted",
            "inserted",
            "marked",
            "background_tone",
            "text_tone",
            "aria_level",
            "aria_posinset",
            "aria_setsize",
            "aria_live",
            "datetime",
            "level",
            "describes",
        ):
            if _include_state_attr(key, block.get(key)):
                attrs.append(f'{key}="{_state_attr(block.get(key))}"')
        if _has_attr_value(block.get("frame")):
            attrs.append(f'frame="{_xml_attr(block.get("frame"))}"')
        lines.append(
            '    <text '
            + " ".join(attrs)
            + ">"
            f'{_render_mixed_text_parts(block.get("parts"), targets_by_index, inlined_visible_targets, referenced_targets, equivalent_covered_targets)}</text>'
        )
    lines.append("  </text_blocks>")

    detailed_click_targets: list[dict] = []
    covered_by_refs = 0
    covered_by_equivalents = 0
    for item in click_targets:
        if not isinstance(item, dict):
            continue
        if _target_key(item.get("index")) in equivalent_covered_targets:
            covered_by_equivalents += 1
        elif _target_detail_needed(item, referenced_targets):
            detailed_click_targets.append(item)
        else:
            covered_by_refs += 1
    click_target_attrs = [
        f'items="{len(click_targets)}"',
        f'details="{len(detailed_click_targets)}"',
        f'covered_by_refs="{covered_by_refs}"',
    ]
    if covered_by_equivalents:
        click_target_attrs.append(f'covered_by_equivalents="{covered_by_equivalents}"')
    click_target_attrs.extend([
        'order="viewport"',
        'space="viewport_css_px"',
        'action="browser_control.click(index)"',
    ])
    lines.append("  <click_targets " + " ".join(click_target_attrs) + ">")
    for item in detailed_click_targets:
        if not isinstance(item, dict):
            continue
        attrs = [
            f'index="{_xml_attr(item.get("index"))}" '
            f'group="{_xml_attr(item.get("group"))}" '
            f'role="{_xml_attr(item.get("role"))}" '
            f'rect="{_rect_attr(item.get("rect"))}"'
        ]
        label_in_text = _target_key(item.get("index")) in inlined_visible_targets and item.get("source") == "visible"
        if not label_in_text:
            if _has_attr_value(item.get("name")):
                attrs.append(f'name="{_xml_attr(item.get("name"))}"')
            if _has_attr_value(item.get("source")) and str(item.get("source")) != "visible":
                attrs.append(f'source="{_xml_attr(item.get("source"))}"')
        if _target_tag_adds_semantics(item):
            attrs.append(f'tag="{_xml_attr(item.get("tag"))}"')
        if item.get("input_type"):
            attrs.append(f'input_type="{_xml_attr(item.get("input_type"))}"')
        if _has_attr_value(item.get("frame")):
            attrs.append(f'frame="{_xml_attr(item.get("frame"))}"')
        for key in (
            "value",
            "placeholder",
            "checked",
            "indeterminate",
            "selected_text",
            "selected_options",
            "selected_files",
            "file_count",
            "visible_options",
            "disabled_options",
            "multiple",
            "focused",
            "selection_start",
            "selection_end",
            "text_selection",
            "aria_checked",
            "aria_controls",
            "aria_current",
            "aria_expanded",
            "aria_haspopup",
            "aria_invalid",
            "invalid",
            "invalid_reason",
            "aria_autocomplete",
            "aria_busy",
            "deleted",
            "inserted",
            "marked",
            "background_tone",
            "text_tone",
            "aria_level",
            "aria_posinset",
            "aria_setsize",
            "aria_pressed",
            "aria_required",
            "aria_sort",
            "aria_selected",
            "aria_value_now",
            "aria_value_min",
            "aria_value_max",
            "aria_value_text",
            "value_min",
            "value_max",
            "required",
            "read_only",
            "contenteditable",
            "open",
            "popover_target",
            "popover_action",
            "popover_open",
            "tab_index",
            "context_role",
            "context",
            "active_descendant",
            "description",
            "error_message",
        ):
            if _include_state_attr(key, item.get(key)):
                attrs.append(f'{key}="{_state_attr(item.get(key))}"')
        attrs.extend(_href_target_attrs(item.get("href"), page_url))
        if item.get("disabled"):
            attrs.append('disabled="true"')
        lines.append("    <target " + " ".join(attrs) + "/>")
    lines.append("  </click_targets>")

    image_attrs = [
        f'visible="{len(all_images)}"',
        f'embedded="{len(embedded_image_parts)}"',
    ]
    omitted_images = max(0, len(all_images) - len(embedded_image_parts))
    if omitted_images:
        image_attrs.append(f'omitted="{omitted_images}"')
    images_open_line = "  <images " + " ".join(image_attrs) + ">"
    if image_parts_count <= 0:
        lines.append(images_open_line)
        for image in all_images:
            lines.append(_browser_image_xml_line(image, embedded=False))
        lines.append("  </images>")
        if viewport:
            lines.append(_browser_viewport_image_xml_line(viewport, viewport_part))
        lines.append("</browser>")
        return "\n".join(lines)

    parts: list[dict] = [{"type": "text", "text": "\n".join([*lines, images_open_line]) + "\n"}]
    for index, image in enumerate(all_images):
        part = embedded_image_parts.get(index)
        parts.append({
            "type": "text",
            "text": _browser_image_xml_line(image, embedded=False if part is None else None) + "\n",
        })
        if part is not None:
            parts.append(part)
    if viewport:
        parts.append({"type": "text", "text": "  </images>\n" + _browser_viewport_image_xml_line(viewport, viewport_part) + "\n"})
        if viewport_part is not None:
            parts.append(viewport_part)
        parts.append({"type": "text", "text": "</browser>"})
    else:
        parts.append({"type": "text", "text": "  </images>\n</browser>"})
    return parts


def _build_browser_world_content() -> "str | list":
    try:
        from browser_adapter.session import browser_world_snapshot

        return _render_browser_world_content(browser_world_snapshot())
    except Exception:
        return ""


def _browser_multimodal_image_limit() -> int:
    """Read the browser-only cap for real multimodal image parts."""
    try:
        import app_state

        return _config_browser_multimodal_image_limit(getattr(app_state, "config", {}) or {})
    except Exception:
        return DEFAULT_BROWSER_MULTIMODAL_IMAGE_LIMIT


def _world_multimodal_image_limit() -> int:
    """Read the runtime cap for real multimodal images inside <world>."""
    try:
        import app_state

        cfg = getattr(app_state, "config", {}) or {}
        if not bool(cfg.get("vision", True)):
            return -1
        gen = getattr(app_state, "GEN", None) or cfg.get("generation")
        return normalize_generation_config(gen).get(
            "world_multimodal_image_limit",
            DEFAULT_WORLD_MULTIMODAL_IMAGE_LIMIT,
        )
    except Exception:
        return DEFAULT_WORLD_MULTIMODAL_IMAGE_LIMIT


def _chat_log_multimodal_image_hint(limit: int) -> int:
    """Return the legacy per-chat-log hint that avoids avoidable old image work."""
    return -1 if limit < 0 else limit


def _wrap_chat_log_with_world(
    chat_log: "str | list",
    unread_xml: str,
    current_time: str,
    forward_content: "str | list" = "",
    browser_content: "str | list" = "",
) -> "str | list":
    """将聊天记录用 <world><qq> 包裹，并在前面插入 unread_info 块。"""
    unread_block = unread_xml if unread_xml else "<unread_info/>"
    current_time_block = f"<current_time>{current_time}</current_time>"
    if (
        isinstance(chat_log, str)
        and not isinstance(forward_content, list)
        and not isinstance(browser_content, list)
    ):
        forward_block = f"\n{forward_content}" if forward_content else ""
        browser_block = f"\n{browser_content}" if browser_content else ""
        return f"<world>\n{current_time_block}\n<qq>\n{unread_block}\n{chat_log}{forward_block}\n</qq>{browser_block}\n</world>"

    new_parts: list = [{"type": "text", "text": f"<world>\n{current_time_block}\n<qq>\n{unread_block}\n"}]
    if isinstance(chat_log, str):
        _append_text_part(new_parts, chat_log)
    else:
        new_parts.extend(chat_log)
    if forward_content:
        _append_text_part(new_parts, "\n")
        if isinstance(forward_content, str):
            _append_text_part(new_parts, forward_content)
        else:
            new_parts.extend(forward_content)
    _append_text_part(new_parts, "\n</qq>")
    if browser_content:
        _append_text_part(new_parts, "\n")
        if isinstance(browser_content, str):
            _append_text_part(new_parts, browser_content)
        else:
            new_parts.extend(browser_content)
    _append_text_part(new_parts, "\n</world>")
    return new_parts


def _strip_world_close(content: "str | list") -> tuple["str | list", str]:
    suffix = "\n</world>"
    if isinstance(content, str):
        if content.endswith(suffix):
            return content[: -len(suffix)], suffix
        if content.endswith("</world>"):
            return content[: -len("</world>")], "</world>"
        return content, ""

    parts = list(content)
    for index in range(len(parts) - 1, -1, -1):
        part = parts[index]
        if not isinstance(part, dict) or part.get("type") != "text":
            continue
        text = str(part.get("text", ""))
        marker = text.rfind("</world>")
        if marker < 0 or text[marker + len("</world>"):].strip():
            continue
        before = text[:marker]
        if before.endswith("\n"):
            before = before[:-1]
            close = "\n</world>"
        else:
            close = "</world>"
        parts = parts[: index + 1]
        if before:
            parts[index] = {**part, "text": before}
        else:
            parts = parts[:index]
        return parts, close
    return parts, ""


def _append_browser_content_to_world(
    content: "str | list",
    browser_content: "str | list",
) -> "str | list":
    if not browser_content:
        return content

    opened, close = _strip_world_close(content)
    close = close or "\n</world>"
    if isinstance(opened, str) and isinstance(browser_content, str):
        return f"{opened}\n{browser_content}{close}"

    parts: list = [{"type": "text", "text": opened}] if isinstance(opened, str) else list(opened)
    _append_text_part(parts, "\n")
    if isinstance(browser_content, str):
        _append_text_part(parts, browser_content)
    else:
        parts.extend(browser_content)
    _append_text_part(parts, close)
    return parts


def _build_unread_bubble_text(unread: int) -> str:
    """构建浏览态底部未读气泡文案。"""
    if unread <= 0:
        return ""
    unread_text = "99+" if unread > 99 else str(unread)
    return f"当前会话有 {unread_text} 条未读新消息"


def _build_current_chat_log(session) -> "str | list":
    """最新窗口聊天记录构建：统一输出 current 模式与 has_previous 状态。"""
    conv_meta = session._get_conv_meta()
    world_image_limit = _world_multimodal_image_limit()
    return build_multimodal_content(
        session.context_messages,
        conv_meta,
        max_images=_chat_log_multimodal_image_hint(world_image_limit),
        quoted_extra=session.quoted_extra,
        chat_logs_mode="current",
        has_previous=has_previous_messages(session, browsing=False),
    )


def _build_browsing_chat_log(session) -> "str | list":
    """浏览态聊天记录构建：统一输出 history 模式、has_previous 与未读气泡。"""
    view = session.chat_window_view
    top_db_id = view.get("top_db_id")
    page_size = int(view.get("page_size", 10))
    if not top_db_id:
        # 状态异常：兜底回 live 渲染，避免空 prompt
        return _build_current_chat_log(session)

    msgs = load_history_window(session, int(top_db_id), page_size)
    if not msgs:
        return _build_current_chat_log(session)

    unread = session.consume_visible_unread_messages(msgs)

    conv_meta = session._get_conv_meta()
    world_image_limit = _world_multimodal_image_limit()
    return build_multimodal_content(
        msgs,
        conv_meta,
        max_images=_chat_log_multimodal_image_hint(world_image_limit),
        quoted_extra=session.quoted_extra,
        chat_logs_mode="history",
        has_previous=has_previous_messages(session, browsing=True, top_db_id=int(top_db_id)),
        bubble_text=_build_unread_bubble_text(unread),
    )


def build_main_user_prompt(session, *, consume_unread: bool = True) -> "str | list":
    """组装主模型本轮 user prompt。

    浏览态（session.is_browsing_history() 为真）下：
    - 聊天记录 XML 统一输出 <chat_logs mode="..." has_previous="...">
    - 浏览态不消费 unread_count，未读新消息以 <bubble> 出现在 <chat_logs> 内
    - 聊天记录从 DB 加载历史窗口，而非渲染最新 context
    """
    current_key = f"{session.conv_type}_{session.conv_id}" if session.conv_type else ""
    browsing = session.is_browsing_history()

    if consume_unread and not browsing:
        session.clear_unread_messages()

    unread_xml = build_unread_info_xml(sessions, current_key)
    if browsing:
        chat_log = _build_browsing_chat_log(session)
    else:
        chat_log = _build_current_chat_log(session)
    forward_content = build_forward_browser_content(session)
    dynamic_blocks = session.build_dynamic_prompt_blocks()
    browser_content = _build_browser_world_content()
    user_prompt = _wrap_chat_log_with_world(
        chat_log,
        unread_xml,
        dynamic_blocks["current_time"],
        forward_content,
    )
    user_prompt = _limit_multimodal_image_parts(
        user_prompt,
        normalize_world_multimodal_image_limit(_world_multimodal_image_limit()),
    )
    user_prompt = _append_browser_content_to_world(user_prompt, browser_content)
    prefix = "\n".join([
        _build_prompt_block("memory", dynamic_blocks["memory"]),
        _build_prompt_block("goals", dynamic_blocks["goals"]),
        _build_prompt_block("style", session._style_prompt),
        _build_prompt_block("social_tips", session.get_social_tips()),
    ])
    user_prompt = _prepend_text_block(user_prompt, prefix)
    return append_final_reminder(user_prompt, session)
