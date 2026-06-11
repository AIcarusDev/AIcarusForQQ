from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from browser.world_prompt import render_browser_world_content  # noqa: E402
from tools.browser_control import execute  # noqa: E402
from browser.session import browser_world_snapshot  # noqa: E402

OUT = ROOT / "output" / "browser_world_samples"

SAMPLES: list[tuple[str, str]] = [
    ("example", "https://example.com/"),
    ("hacker_news", "https://news.ycombinator.com/"),
    ("wikipedia", "https://www.wikipedia.org/"),
    ("visual_variants", "visual_variants.html"),
    ("partial_visual_clip", "partial_visual_clip.html"),
    ("svg_text", "svg_text.html"),
    ("gradient_background", "gradient_background.html"),
    ("semantic_background_visual", "semantic_background_visual.html"),
    ("visual_occlusion", "visual_occlusion.html"),
    ("clickable_visual_target", "clickable_visual_target.html"),
    ("image_map", "image_map.html"),
    ("data_image", "data_image.html"),
    ("broken_image", "broken_image.html"),
    ("embedded_media", "embedded_media.html"),
    ("media_state", "media_state.html"),
    ("masked_visual", "masked_visual.html"),
    ("semantic_graphics", "semantic_graphics.html"),
    ("disabled_control", "disabled_control.html"),
    ("text_state", "text_state.html"),
    ("icon_controls", "icon_controls.html"),
    ("interaction_blocking", "interaction_blocking.html"),
    ("iframe_sample", "iframe_sample.html"),
    ("shadow_sample", "shadow_sample.html"),
    ("orphan_text", "orphan_text.html"),
    ("form_state", "form_state.html"),
    ("form_error_state", "form_error_state.html"),
    ("native_validation", "native_validation.html"),
    ("placeholder_state", "placeholder_state.html"),
    ("multi_select", "multi_select.html"),
    ("file_input_state", "file_input_state.html"),
    ("focus_selection", "focus_selection.html"),
    ("contenteditable", "contenteditable.html"),
    ("native_disclosure", "native_disclosure.html"),
    ("custom_controls", "custom_controls.html"),
    ("display_contents_target", "display_contents_target.html"),
    ("self_link_text", "self_link_text.html"),
    ("equivalent_card_links", "equivalent_card_links.html"),
    ("pseudo_content", "pseudo_content.html"),
    ("pseudo_visual", "pseudo_visual.html"),
    ("css_counters", "css_counters.html"),
    ("css_attr_content", "css_attr_content.html"),
    ("text_transform", "text_transform.html"),
    ("text_visual_state", "text_visual_state.html"),
    ("color_state", "color_state.html"),
    ("clip_sample", "clip_sample.html"),
    ("clip_path_visibility", "clip_path_visibility.html"),
    ("transparent_overlay", "transparent_overlay.html"),
    ("scroll_region", "scroll_region.html"),
    ("active_descendant", "active_descendant.html"),
    ("aria_widgets", "aria_widgets.html"),
    ("aria_names", "aria_names.html"),
    ("state_attrs", "state_attrs.html"),
    ("input_types", "input_types.html"),
    ("noisy_focusables", "noisy_focusables.html"),
    ("semantic_dialog", "semantic_dialog.html"),
    ("tooltip_state", "tooltip_state.html"),
    ("modal_backdrop", "modal_backdrop.html"),
    ("group_context", "group_context.html"),
    ("indicators", "indicators.html"),
    ("busy_state", "busy_state.html"),
    ("loading_placeholder", "loading_placeholder.html"),
    ("long_semantics", "long_semantics.html"),
    ("ordered_list", "ordered_list.html"),
    ("table_sort_state", "table_sort_state.html"),
    ("table_sample", "table_sample.html"),
    ("aria_grid", "aria_grid.html"),
]

VISIBLE_IMAGE_WAIT_SAMPLES = {
    "visual_variants",
    "visual_occlusion",
    "image_map",
    "data_image",
    "broken_image",
    "iframe_sample",
    "shadow_sample",
    "wikipedia",
}

HIDDEN_FIXTURE_PATTERNS = [
    "Offscreen",
    "Clipped hidden text should not appear",
    "Clipped image should not appear",
    "Hidden result item",
    "secret should not appear",
    "Opaque covered image should not appear",
    "Hidden visual",
    "Hidden visual action should not appear",
    "Offscreen visual action should not appear",
    "Hidden clipped visual should not appear",
    "Offscreen clipped visual should not appear",
    "Hidden semantic background should not appear",
    "Offscreen semantic background should not appear",
    "Decorative small background should not appear",
    "Clip-path hidden text should not appear",
    "Clip-path hidden image should not appear",
    "Offscreen gradient background text should not appear",
    "Hidden editor text should not appear",
    "Hidden closed details content should not appear",
    "Hidden help popover text should not appear",
    "Hidden collapsed suggestion should not appear",
    "Offscreen active descendant text should not appear",
    "Hidden ARIA collection item should not appear",
    "Offscreen ARIA collection item should not appear",
    "Hidden describedby text should not appear",
    "Offscreen image map text should not appear",
    "Hidden broken image should not appear",
    "Offscreen broken image should not appear",
    "Offscreen embedded media text should not appear",
    "Offscreen media control should not appear",
    "Offscreen masked visual text should not appear",
    "Offscreen semantic graphic should not appear",
    "Decorative semantic graphic should not appear",
    "Hidden text state should not appear",
    "Offscreen text state should not appear",
    "Hidden semantic heading should not appear",
    "Offscreen semantic heading should not appear",
    "Offscreen icon semantics should not appear",
    "Decorative sparkle should not appear",
    "Hidden CSS counter should not appear",
    "Offscreen CSS counter should not appear",
    "Hidden CSS attr should not appear",
    "Offscreen CSS attr should not appear",
    "Hidden transformed text should not appear",
    "Hidden transformed action should not appear",
    "Hidden deleted text should not appear",
    "Offscreen deleted text should not appear",
    "Hidden marked text should not appear",
    "Offscreen marked text should not appear",
    "Hidden deleted action should not appear",
    "Offscreen deleted action should not appear",
    "Hidden color state should not appear",
    "Offscreen color state should not appear",
    "Hidden color swatch should not appear",
    "Offscreen color swatch should not appear",
    "Hidden SVG label should not appear",
    "Offscreen SVG label should not appear",
    "Collapsed hidden option should not appear",
    "Hidden ordered item should not appear",
    "Hidden table row should not appear",
    "Hidden grid row should not appear",
    "Hidden sort state should not appear",
    "Offscreen sort state should not appear",
    "Offscreen input type text should not appear",
    "Offscreen focus selection text should not appear",
    "Offscreen placeholder text should not appear",
    "Hidden filled placeholder should not appear",
    "Hidden form error should not appear",
    "Offscreen form error should not appear",
    "Hidden native validation should not appear",
    "Offscreen native validation should not appear",
    "hidden-native-validation",
    "offscreen-native-validation",
    "Offscreen form text should not appear",
    "Hidden indeterminate option should not appear",
    "Offscreen indeterminate option should not appear",
    "Offscreen semantic dialog text should not appear",
    "Hidden tooltip should not appear",
    "Offscreen tooltip should not appear",
    "Behind modal text should not appear",
    "Behind modal action should not appear",
    "Behind modal image should not appear",
    "Offscreen group context text should not appear",
    "Offscreen indicator text should not appear",
    "Hidden busy state should not appear",
    "Offscreen busy state should not appear",
    "Hidden loading placeholder should not appear",
    "Offscreen loading placeholder should not appear",
    "Hidden spinner should not appear",
    "Offscreen spinner should not appear",
    "Hidden CSS disabled action should not appear",
    "Offscreen CSS disabled action should not appear",
    "Hidden pseudo visual should not appear",
    "Offscreen pseudo visual should not appear",
    "Offscreen frame text should not appear",
    "Hidden frame text should not appear",
    "Hidden child frame should not appear",
    "Offscreen child frame should not appear",
    "Offscreen child frame text should not appear",
    "Main offscreen text should not appear",
    "Hidden display contents action should not appear",
    "Offscreen display contents action should not appear",
    "Hidden self link should not appear",
]


def _sample_url(raw: str) -> str:
    if raw.startswith(("http://", "https://", "file://")):
        return raw
    return (OUT / raw).as_uri()


def _sanitize(value: Any) -> Any:
    if isinstance(value, bytes):
        return {"_bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items() if key != "_frame"}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def _xml_text_and_image_count(rendered: str | list) -> tuple[str, int]:
    if isinstance(rendered, list):
        text = "".join(
            part.get("text", "")
            for part in rendered
            if isinstance(part, dict) and part.get("type") == "text"
        )
        image_count = sum(
            1
            for part in rendered
            if isinstance(part, dict) and part.get("type") == "image_url"
        )
        return text, image_count
    return str(rendered or ""), 0


def _rects(snapshot: dict[str, Any]):
    for key in ("click_targets", "text_blocks", "scroll_regions", "frames", "indicators"):
        for item in snapshot.get(key) or []:
            if isinstance(item, dict) and isinstance(item.get("rect"), dict):
                yield key, item["rect"]
    for table in snapshot.get("tables") or []:
        if not isinstance(table, dict):
            continue
        if isinstance(table.get("rect"), dict):
            yield "tables", table["rect"]
        for row in table.get("rows") or []:
            if not isinstance(row, dict):
                continue
            if isinstance(row.get("rect"), dict):
                yield "table_rows", row["rect"]
            for cell in row.get("cells") or []:
                if isinstance(cell, dict) and isinstance(cell.get("rect"), dict):
                    yield "table_cells", cell["rect"]
    for item in snapshot.get("images") or []:
        if isinstance(item, dict):
            yield "images", {
                "x": item.get("x", 0),
                "y": item.get("y", 0),
                "width": item.get("width", 0),
                "height": item.get("height", 0),
            }


def _out_of_viewport(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    viewport = snapshot.get("viewport_size") or snapshot.get("viewport") or {}
    width = float(viewport.get("width") or 0)
    height = float(viewport.get("height") or 0)
    bad: list[dict[str, Any]] = []
    for key, rect in _rects(snapshot):
        try:
            x = float(rect.get("x") or 0)
            y = float(rect.get("y") or 0)
            w = float(rect.get("width") or 0)
            h = float(rect.get("height") or 0)
        except (TypeError, ValueError):
            bad.append({"key": key, "rect": rect})
            continue
        if x < -1 or y < -1 or x + w > width + 1 or y + h > height + 1:
            bad.append({"key": key, "rect": rect})
    return bad


def _target_attr_count(snapshot: dict[str, Any], attr: str, value: object = None) -> int:
    count = 0
    for item in snapshot.get("click_targets") or []:
        if not isinstance(item, dict) or attr not in item:
            continue
        if value is None or str(item.get(attr)) == str(value):
            count += 1
    return count


def _role_count(snapshot: dict[str, Any], role: str) -> int:
    return sum(
        1
        for item in snapshot.get("click_targets") or []
        if isinstance(item, dict) and item.get("role") == role
    )


def _text_block_count(snapshot: dict[str, Any], *, kind: str | None = None, tag: str | None = None) -> int:
    count = 0
    for item in snapshot.get("text_blocks") or []:
        if not isinstance(item, dict):
            continue
        if kind is not None and item.get("kind") != kind:
            continue
        if tag is not None and item.get("tag") != tag:
            continue
        count += 1
    return count


def _text_block_attr_count(snapshot: dict[str, Any], attr: str, value: object = None) -> int:
    count = 0
    for item in snapshot.get("text_blocks") or []:
        if not isinstance(item, dict) or attr not in item:
            continue
        if value is None or str(item.get(attr)) == str(value):
            count += 1
    return count


def _table_counts(snapshot: dict[str, Any]) -> dict[str, int]:
    tables = [item for item in snapshot.get("tables") or [] if isinstance(item, dict)]
    rows = 0
    cells = 0
    for table in tables:
        table_rows = [item for item in table.get("rows") or [] if isinstance(item, dict)]
        rows += len(table_rows)
        for row in table_rows:
            cells += sum(1 for item in row.get("cells") or [] if isinstance(item, dict))
    return {"tables": len(tables), "rows": rows, "cells": cells}


def _table_cell_attr_count(snapshot: dict[str, Any], attr: str, value: object = None) -> int:
    count = 0
    for table in snapshot.get("tables") or []:
        if not isinstance(table, dict):
            continue
        for row in table.get("rows") or []:
            if not isinstance(row, dict):
                continue
            for cell in row.get("cells") or []:
                if not isinstance(cell, dict) or attr not in cell:
                    continue
                if value is None or str(cell.get(attr)) == str(value):
                    count += 1
    return count


def _table_attr_count(snapshot: dict[str, Any], attr: str, value: object = None) -> int:
    count = 0
    for item in snapshot.get("tables") or []:
        if not isinstance(item, dict) or attr not in item:
            continue
        if value is None or str(item.get(attr)) == str(value):
            count += 1
    return count


def _indicator_count(snapshot: dict[str, Any], kind: str | None = None) -> int:
    return sum(
        1
        for item in snapshot.get("indicators") or []
        if isinstance(item, dict) and (kind is None or item.get("kind") == kind)
    )


def _image_kind_count(snapshot: dict[str, Any], kind: str) -> int:
    return sum(
        1
        for item in snapshot.get("images") or []
        if isinstance(item, dict) and item.get("kind") == kind
    )


def _image_attr_count(snapshot: dict[str, Any], attr: str, value: object = None) -> int:
    count = 0
    for item in snapshot.get("images") or []:
        if not isinstance(item, dict) or attr not in item:
            continue
        if value is None or str(item.get(attr)) == str(value):
            count += 1
    return count


def _image_pixel_mismatches(snapshot: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        from PIL import Image
    except Exception:
        return []
    bad: list[dict[str, Any]] = []
    for index, item in enumerate(snapshot.get("images") or []):
        if not isinstance(item, dict) or not isinstance(item.get("data"), bytes):
            continue
        try:
            image = Image.open(io.BytesIO(item["data"]))
            actual = image.size
            expected = (int(item.get("width") or 0), int(item.get("height") or 0))
        except Exception:
            bad.append({"index": index, "kind": item.get("kind"), "reason": "unreadable"})
            continue
        if expected[0] > 0 and expected[1] > 0 and actual != expected:
            bad.append({
                "index": index,
                "kind": item.get("kind"),
                "expected": expected,
                "actual": actual,
            })
    return bad


def _indicator_attr_count(snapshot: dict[str, Any], attr: str, value: object = None) -> int:
    count = 0
    for item in snapshot.get("indicators") or []:
        if not isinstance(item, dict) or attr not in item:
            continue
        if value is None or str(item.get(attr)) == str(value):
            count += 1
    return count


def _frame_count(snapshot: dict[str, Any], tag: str | None = None) -> int:
    return sum(
        1
        for item in snapshot.get("frames") or []
        if isinstance(item, dict) and (tag is None or item.get("tag") == tag)
    )


def _href_suffix_count(snapshot: dict[str, Any], suffix: str) -> int:
    return sum(
        1
        for item in snapshot.get("click_targets") or []
        if isinstance(item, dict) and str(item.get("href") or "").endswith(suffix)
    )


def _target_name_contains(snapshot: dict[str, Any], needle: str) -> int:
    return sum(
        1
        for item in snapshot.get("click_targets") or []
        if isinstance(item, dict) and needle in str(item.get("name") or "")
    )


def _target_attr_contains(snapshot: dict[str, Any], attr: str, needle: str) -> int:
    return sum(
        1
        for item in snapshot.get("click_targets") or []
        if isinstance(item, dict) and needle in str(item.get(attr) or "")
    )


def _summarize(slug: str, snapshot: dict[str, Any], xml: str, image_parts: int) -> dict[str, Any]:
    roles: list[str] = []
    for item in snapshot.get("click_targets") or []:
        role = item.get("role") if isinstance(item, dict) else None
        if role and str(role).startswith(("tab", "switch", "slider", "option", "treeitem", "menuitem")):
            roles.append(str(role))

    table_counts = _table_counts(snapshot)
    return {
        "slug": slug,
        "url": snapshot.get("url"),
        "title": snapshot.get("title"),
        "viewport_size": snapshot.get("viewport_size") or {},
        "scroll": snapshot.get("scroll") or {},
        "click_targets": len(snapshot.get("click_targets") or []),
        "scroll_regions": len(snapshot.get("scroll_regions") or []),
        "frames": _frame_count(snapshot),
        "iframe_frames": _frame_count(snapshot, "iframe"),
        "framed_targets": _target_attr_count(snapshot, "frame"),
        "framed_text_blocks": _text_block_attr_count(snapshot, "frame"),
        "framed_tables": _table_attr_count(snapshot, "frame"),
        "framed_indicators": _indicator_attr_count(snapshot, "frame"),
        "framed_images": _image_attr_count(snapshot, "frame"),
        "text_blocks": len(snapshot.get("text_blocks") or []),
        "tables": table_counts["tables"],
        "table_rows": table_counts["rows"],
        "table_cells": table_counts["cells"],
        "table_aria_sort_cells": _table_cell_attr_count(snapshot, "aria_sort"),
        "table_aria_busy_cells": _table_cell_attr_count(snapshot, "aria_busy"),
        "indicators": _indicator_count(snapshot),
        "progressbar_indicators": _indicator_count(snapshot, "progressbar"),
        "meter_indicators": _indicator_count(snapshot, "meter"),
        "audio_indicators": _indicator_count(snapshot, "audio"),
        "video_indicators": _indicator_count(snapshot, "video"),
        "graphic_indicators": _indicator_count(snapshot, "graphic"),
        "color_swatch_indicators": _indicator_count(snapshot, "color_swatch"),
        "loading_placeholder_indicators": _indicator_count(snapshot, "loading_placeholder"),
        "busy_indicators": _indicator_count(snapshot, "busy"),
        "disabled_control_indicators": _indicator_count(snapshot, "disabled_control"),
        "css_disabled_control_indicators": _indicator_attr_count(snapshot, "blocked", "css_disabled"),
        "inert_control_indicators": _indicator_count(snapshot, "inert_control"),
        "table_cell_text_blocks": _text_block_count(snapshot, tag="td") + _text_block_count(snapshot, tag="th"),
        "label_text_blocks": _text_block_count(snapshot, tag="label"),
        "legend_text_blocks": _text_block_count(snapshot, tag="legend"),
        "time_text_blocks": _text_block_count(snapshot, kind="time"),
        "heading_text_blocks": _text_block_count(snapshot, kind="heading"),
        "heading_level_text_blocks": _text_block_attr_count(snapshot, "level"),
        "deleted_text_blocks": _text_block_attr_count(snapshot, "deleted"),
        "inserted_text_blocks": _text_block_attr_count(snapshot, "inserted"),
        "marked_text_blocks": _text_block_attr_count(snapshot, "marked"),
        "background_tone_text_blocks": _text_block_attr_count(snapshot, "background_tone"),
        "text_tone_text_blocks": _text_block_attr_count(snapshot, "text_tone"),
        "datetime_text_blocks": _text_block_attr_count(snapshot, "datetime"),
        "text_aria_current_blocks": _text_block_attr_count(snapshot, "aria_current"),
        "text_aria_live_blocks": _text_block_attr_count(snapshot, "aria_live"),
        "svg_text_blocks": _text_block_count(snapshot, kind="svg_text"),
        "visible_images": len(snapshot.get("images") or []),
        "unloaded_images": _image_attr_count(snapshot, "loaded", False),
        "canvas_images": _image_kind_count(snapshot, "canvas"),
        "background_images": _image_kind_count(snapshot, "background"),
        "mask_images": _image_kind_count(snapshot, "mask"),
        "input_image_images": _image_kind_count(snapshot, "input_image"),
        "pseudo_visual_images": (
            _image_kind_count(snapshot, "pseudo_image")
            + _image_kind_count(snapshot, "pseudo_background")
            + _image_kind_count(snapshot, "pseudo_mask")
        ),
        "object_images": _image_kind_count(snapshot, "object"),
        "embed_images": _image_kind_count(snapshot, "embed"),
        "image_kinds": sorted({
            str(item.get("kind") or "")
            for item in snapshot.get("images") or []
            if isinstance(item, dict)
        }),
        "image_parts": image_parts,
        "image_pixel_mismatches": len(_image_pixel_mismatches(snapshot)),
        "xml_chars": len(xml),
        "out_of_viewport_rects": len(_out_of_viewport(snapshot)),
        "contains_hidden_fixture_text": any(pattern in xml for pattern in HIDDEN_FIXTURE_PATTERNS),
        "has_visible_text_node": "visible_text" in xml or "<visible_text" in xml,
        "contains_password_value": "secret-password" in xml or "super-secret" in xml,
        "contains_zero_origin_text": "0,0,0,0" in xml,
        "aria_roles": roles,
        "labelledby_targets": _target_attr_count(snapshot, "source", "labelledby"),
        "graphic_source_targets": _target_attr_count(snapshot, "source", "graphic"),
        "described_targets": _target_attr_count(snapshot, "description"),
        "error_message_targets": _target_attr_count(snapshot, "error_message"),
        "native_invalid_targets": _target_attr_count(snapshot, "invalid"),
        "type_mismatch_targets": _target_attr_contains(snapshot, "invalid_reason", "type_mismatch"),
        "range_overflow_targets": _target_attr_contains(snapshot, "invalid_reason", "range_overflow"),
        "pattern_mismatch_targets": _target_attr_contains(snapshot, "invalid_reason", "pattern_mismatch"),
        "deleted_targets": _target_attr_count(snapshot, "deleted"),
        "inserted_targets": _target_attr_count(snapshot, "inserted"),
        "marked_targets": _target_attr_count(snapshot, "marked"),
        "deleted_table_cells": _table_cell_attr_count(snapshot, "deleted"),
        "inserted_table_cells": _table_cell_attr_count(snapshot, "inserted"),
        "marked_table_cells": _table_cell_attr_count(snapshot, "marked"),
        "background_tone_table_cells": _table_cell_attr_count(snapshot, "background_tone"),
        "text_tone_table_cells": _table_cell_attr_count(snapshot, "text_tone"),
        "background_tone_targets": _target_attr_count(snapshot, "background_tone"),
        "text_tone_targets": _target_attr_count(snapshot, "text_tone"),
        "tone_indicators": _indicator_attr_count(snapshot, "tone"),
        "color_indicators": _indicator_attr_count(snapshot, "color"),
        "variant_indicators": _indicator_attr_count(snapshot, "variant"),
        "busy_state_targets": _target_attr_count(snapshot, "aria_busy"),
        "indeterminate_targets": _target_attr_count(snapshot, "indeterminate"),
        "contenteditable_targets": _target_attr_count(snapshot, "contenteditable"),
        "summary_targets": _role_count(snapshot, "summary"),
        "open_state_targets": _target_attr_count(snapshot, "open"),
        "popover_targets": _target_attr_count(snapshot, "popover_target"),
        "focusable_targets": _role_count(snapshot, "focusable"),
        "fallback_focusable_targets": sum(
            1
            for item in snapshot.get("click_targets") or []
            if isinstance(item, dict) and item.get("role") == "focusable" and item.get("source") == "fallback"
        ),
        "slider_targets": _role_count(snapshot, "slider"),
        "treeitem_targets": _role_count(snapshot, "treeitem"),
        "spinbutton_targets": _role_count(snapshot, "spinbutton"),
        "color_targets": _role_count(snapshot, "color"),
        "file_targets": _role_count(snapshot, "file"),
        "selected_file_targets": _target_attr_count(snapshot, "selected_files"),
        "file_count_targets": _target_attr_count(snapshot, "file_count"),
        "focused_targets": _target_attr_count(snapshot, "focused"),
        "text_selection_targets": _target_attr_count(snapshot, "text_selection"),
        "placeholder_targets": _target_attr_count(snapshot, "placeholder"),
        "placeholder_source_targets": _target_attr_count(snapshot, "source", "placeholder"),
        "multiple_select_targets": _target_attr_count(snapshot, "multiple"),
        "selected_options_targets": _target_attr_count(snapshot, "selected_options"),
        "visible_options_targets": _target_attr_count(snapshot, "visible_options"),
        "disabled_options_targets": _target_attr_count(snapshot, "disabled_options"),
        "active_descendant_targets": _target_attr_count(snapshot, "active_descendant"),
        "aria_autocomplete_targets": _target_attr_count(snapshot, "aria_autocomplete"),
        "aria_level_targets": _target_attr_count(snapshot, "aria_level"),
        "aria_posinset_targets": _target_attr_count(snapshot, "aria_posinset"),
        "aria_setsize_targets": _target_attr_count(snapshot, "aria_setsize"),
        "aria_posinset_text_blocks": _text_block_attr_count(snapshot, "aria_posinset"),
        "aria_setsize_text_blocks": _text_block_attr_count(snapshot, "aria_setsize"),
        "dialog_context_targets": _target_attr_count(snapshot, "context_role", "dialog"),
        "group_context_targets": _target_attr_count(snapshot, "context_role", "group"),
        "radiogroup_context_targets": _target_attr_count(snapshot, "context_role", "radiogroup"),
        "toolbar_context_targets": _target_attr_count(snapshot, "context_role", "toolbar"),
        "shipping_context_targets": _target_attr_contains(snapshot, "context", "Shipping method"),
        "payment_context_targets": _target_attr_contains(snapshot, "context", "Payment method"),
        "editor_context_targets": _target_attr_contains(snapshot, "context", "Editor tools"),
        "session_timeout_context_targets": _target_attr_contains(snapshot, "context", "Session timeout"),
        "confirm_transfer_context_targets": _target_attr_contains(snapshot, "context", "Confirm transfer"),
        "behind_modal_targets": _target_name_contains(snapshot, "Behind modal action"),
        "status_text_blocks": _text_block_count(snapshot, kind="status"),
        "alert_text_blocks": _text_block_count(snapshot, kind="alert"),
        "tooltip_text_blocks": _text_block_count(snapshot, kind="tooltip"),
        "describing_text_blocks": _text_block_attr_count(snapshot, "describes"),
        "busy_text_blocks": _text_block_attr_count(snapshot, "aria_busy"),
        "blocked_overlay_targets": _href_suffix_count(snapshot, "#blocked"),
        "pass_through_overlay_targets": _href_suffix_count(snapshot, "#pass"),
        "blocked_interaction_targets": sum(
            _target_name_contains(snapshot, needle)
            for needle in (
                "Disabled fieldset action",
                "Inert link",
                "Inert button",
                "ARIA disabled child",
            )
        ),
        "active_interaction_targets": _href_suffix_count(snapshot, "#active"),
        "contains_long_semantic_tails": all(
            tail in xml
            for tail in (
                "UNIQUE_ARIA_LABEL_TAIL_SHOULD_APPEAR",
                "UNIQUE_ARIA_DESCRIPTION_TAIL_SHOULD_APPEAR",
                "UNIQUE_PSEUDO_CONTENT_TAIL_SHOULD_APPEAR",
            )
        ),
        "contains_ordered_markers": all(marker in xml for marker in ("3. Prepare viewport", "4. <ref target=", "C. Review sample")),
        "contains_css_counter_content": all(
            marker in xml
            for marker in (
                "Step 3: Generated counter label",
                "Step 4: <ref target=",
                "Task B: Custom marker item",
            )
        ),
        "contains_css_attr_content": all(
            marker in xml
            for marker in (
                "Status: Ready for review",
                "Build channel: Canary",
                "Approve 7 pending",
            )
        ),
        "contains_text_transform_content": all(
            marker in xml
            for marker in (
                "EXPORT REPORT",
                "Quarterly Revenue Summary",
                "compliance notice",
                'name="DOWNLOAD CSV"',
            )
        ) and not any(
            marker in xml
            for marker in (
                "export report",
                "quarterly revenue summary",
                "COMPLIANCE NOTICE",
                'name="download csv"',
            )
        ),
        "contains_text_visual_state": all(
            marker in xml
            for marker in (
                'deleted="true"',
                'marked="true"',
                'inserted="true"',
                "Legacy plan $99",
                "Deprecated endpoint",
                "Highlighted SLA window",
                'name="Old billing portal"',
                "Superseded price",
            )
        ),
        "contains_color_state": all(
            marker in xml
            for marker in (
                'background_tone="green"',
                'text_tone="red"',
                'kind="color_swatch"',
                'tone="green"',
                'color="#16a34a"',
                'name="Connection online"',
                'background_tone="red"',
                'background_tone="yellow"',
            )
        ),
        "contains_svg_text_structure": all(
            marker in xml
            for marker in (
                'kind="svg_text"',
                "Revenue Q1",
                "North region",
                "$120k",
                "Series A",
            )
        ),
        "contains_broken_image_state": all(
            marker in xml
            for marker in (
                'kind="image"',
                'alt="Quarterly chart image"',
                'loaded="false"',
                'alt="Loaded reference image"',
            )
        ) and "Hidden broken image should not appear" not in xml and "Offscreen broken image should not appear" not in xml,
        "contains_partial_visual_clip": all(
            marker in xml
            for marker in (
                'kind="image"',
                'alt="Partially visible wide chart"',
                'width="181"',
                'height="120"',
            )
        ) and "Hidden clipped visual should not appear" not in xml and "Offscreen clipped visual should not appear" not in xml,
        "contains_semantic_background_visual": all(
            marker in xml
            for marker in (
                'kind="background"',
                'alt="Ada Lovelace profile avatar"',
                'width="64"',
                'height="64"',
            )
        ) and not any(
            marker in xml
            for marker in (
                "Hidden semantic background should not appear",
                "Offscreen semantic background should not appear",
                "Decorative small background should not appear",
            )
        ),
        "contains_clip_path_visibility_filter": (
            "Visible clip-path control text." in xml
            and "Clip-path hidden text should not appear" not in xml
            and "Clip-path hidden image should not appear" not in xml
        ),
        "contains_css_disabled_controls": all(
            marker in xml
            for marker in (
                'blocked="css_disabled"',
                'name="CSS disabled action"',
                'name="CSS disabled link"',
            )
        ) and "Hidden CSS disabled action should not appear" not in xml and "Offscreen CSS disabled action should not appear" not in xml,
        "contains_table_structure": all(
            marker in xml
            for marker in (
                '<tables items="1"',
                '<cell row="0" col="0" tag="th" header="true"',
                '<cell row="1" col="1" tag="td" header="false"',
                '<ref target=',
                'Revenue table sample',
            )
        ),
        "contains_aria_grid_structure": all(
            marker in xml
            for marker in (
                '<table index="0" role="grid" tag="div"',
                'name="Pipeline grid"',
                'role="columnheader"',
                'role="gridcell"',
                'row_index="2"',
                'col_index="3"',
                '<ref target=',
            )
        ),
        "contains_table_sort_state": all(
            marker in xml
            for marker in (
                'name="Sortable deals"',
                'aria_sort="ascending"',
                'aria_sort="descending"',
            )
        ),
        "contains_disabled_select_options": all(
            marker in xml
            for marker in (
                'visible_options="Intake review | Token audit | Browser world | UI cleanup | Release notes"',
                'disabled_options="Token audit"',
            )
        ),
        "contains_input_image_visual": all(
            marker in xml
            for marker in (
                'kind="input_image"',
                'alt="Submit image search"',
                'input_type="image"',
                'name="Submit image search"',
            )
        ),
        "contains_clickable_visual_target": all(
            marker in xml
            for marker in (
                'name="Revenue heatmap canvas"',
                'source="visual"',
                'kind="canvas"',
                'alt="Revenue heatmap canvas"',
            )
        ) and "Hidden visual action should not appear" not in xml and "Offscreen visual action should not appear" not in xml,
        "contains_display_contents_target": all(
            marker in xml
            for marker in (
                "Open contents card",
                'href_host="example.test"',
                'href_path="/contents-card"',
                "Inline contents button",
            )
        ) and "Hidden display contents action should not appear" not in xml and "Offscreen display contents action should not appear" not in xml,
        "contains_form_error_relationship": all(
            marker in xml
            for marker in (
                'aria_invalid="true"',
                'error_message="Use a work email address."',
                'kind="alert"',
                "Use a work email address.",
            )
        ),
        "contains_native_validation_state": all(
            marker in xml
            for marker in (
                'name="Notification email"',
                'invalid="true"',
                'invalid_reason="type_mismatch"',
                'invalid_reason="range_overflow"',
                'invalid_reason="pattern_mismatch"',
                'name="Required ticket owner"',
                'required="true"',
            )
        ) and 'name="Required ticket owner" source="label" tag="input" input_type="text" invalid=' not in xml,
        "contains_indeterminate_checkbox_state": all(
            marker in xml
            for marker in (
                'role="checkbox"',
                'name="Partially selected categories"',
                'checked="false"',
                'indeterminate="true"',
            )
        ),
        "contains_heading_level_structure": all(
            marker in xml
            for marker in (
                'kind="heading"',
                'tag="h1"',
                'level="2"',
                'Semantic section heading',
            )
        ),
        "contains_aria_collection_position": all(
            marker in xml
            for marker in (
                'role="option"',
                'aria_posinset="2"',
                'aria_setsize="4"',
                'role="treeitem"',
                'aria_level="2"',
                'name="Quarterly forecast"',
            )
        ),
        "contains_busy_state": all(
            marker in xml
            for marker in (
                'kind="busy"',
                'aria_busy="true"',
                'name="Results panel"',
                'Updating recommendations',
                'Refresh results',
            )
        ),
        "contains_iframe_structure": all(
            marker in xml
            for marker in (
                '<frames items="1"',
                'name="Visible child frame"',
                'source="title"',
                'url="about:srcdoc"',
                'frame="0"',
                'Frame paragraph inside viewport.',
                'Frame action',
                'Frame data image',
            )
        ),
        "contains_tooltip_state": all(
            marker in xml
            for marker in (
                'kind="tooltip"',
                'describes="0"',
                'description="Click to save this draft."',
                'Click to save this draft.',
            )
        ),
        "contains_loading_placeholder": all(
            marker in xml
            for marker in (
                'kind="loading_placeholder"',
                'variant="spinner"',
                'variant="skeleton"',
                'name="Loading recommendations"',
                'name="loading placeholder"',
            )
        ),
        "symbol_name_leak": any(token in xml for token in ['name="×"', 'name="★"', 'name="⌄"']),
        "symbol_text_leak": any(token in xml for token in [">×</text>", ">★</text>", ">⌄</text>"]),
        "contains_semantic_graphic_noise_filter": all(
            marker in xml
            for marker in (
                'kind="graphic"',
                'name="Payment verified"',
                'name="Database sync delayed"',
                "Symbol-only marker row",
            )
        ) and not any(token in xml for token in ['name="★"', ">★</text>", "Decorative semantic graphic should not appear"]),
        "contains_pseudo_visual_structure": all(
            marker in xml
            for marker in (
                'kind="pseudo_background"',
                'pseudo="before"',
                'alt="Plan preview pseudo visual"',
            )
        ) and "Hidden pseudo visual should not appear" not in xml and "Offscreen pseudo visual should not appear" not in xml,
        "state_attr_targets": {
            "aria_current": _target_attr_count(snapshot, "aria_current"),
            "aria_haspopup": _target_attr_count(snapshot, "aria_haspopup"),
            "aria_controls": _target_attr_count(snapshot, "aria_controls"),
            "aria_autocomplete": _target_attr_count(snapshot, "aria_autocomplete"),
            "aria_level": _target_attr_count(snapshot, "aria_level") + _text_block_attr_count(snapshot, "aria_level"),
            "aria_posinset": _target_attr_count(snapshot, "aria_posinset") + _text_block_attr_count(snapshot, "aria_posinset"),
            "aria_setsize": _target_attr_count(snapshot, "aria_setsize") + _text_block_attr_count(snapshot, "aria_setsize"),
            "aria_busy": (
                _target_attr_count(snapshot, "aria_busy")
                + _text_block_attr_count(snapshot, "aria_busy")
                + _table_cell_attr_count(snapshot, "aria_busy")
            ),
            "deleted": (
                _target_attr_count(snapshot, "deleted")
                + _text_block_attr_count(snapshot, "deleted")
                + _table_cell_attr_count(snapshot, "deleted")
            ),
            "inserted": (
                _target_attr_count(snapshot, "inserted")
                + _text_block_attr_count(snapshot, "inserted")
                + _table_cell_attr_count(snapshot, "inserted")
            ),
            "marked": (
                _target_attr_count(snapshot, "marked")
                + _text_block_attr_count(snapshot, "marked")
                + _table_cell_attr_count(snapshot, "marked")
            ),
            "describes": _text_block_attr_count(snapshot, "describes"),
            "css_disabled": _indicator_attr_count(snapshot, "blocked", "css_disabled"),
            "background_tone": (
                _target_attr_count(snapshot, "background_tone")
                + _text_block_attr_count(snapshot, "background_tone")
                + _table_cell_attr_count(snapshot, "background_tone")
                + _indicator_attr_count(snapshot, "background_tone")
            ),
            "text_tone": (
                _target_attr_count(snapshot, "text_tone")
                + _text_block_attr_count(snapshot, "text_tone")
                + _table_cell_attr_count(snapshot, "text_tone")
                + _indicator_attr_count(snapshot, "text_tone")
            ),
            "tone": _indicator_attr_count(snapshot, "tone"),
            "color": _indicator_attr_count(snapshot, "color"),
            "variant": _indicator_attr_count(snapshot, "variant"),
            "aria_invalid": _target_attr_count(snapshot, "aria_invalid"),
            "invalid": _target_attr_count(snapshot, "invalid"),
            "invalid_reason": _target_attr_count(snapshot, "invalid_reason"),
            "required": _target_attr_count(snapshot, "required"),
            "read_only": _target_attr_count(snapshot, "read_only"),
            "aria_sort": (
                _target_attr_count(snapshot, "aria_sort")
                + _text_block_attr_count(snapshot, "aria_sort")
                + _table_cell_attr_count(snapshot, "aria_sort")
            ),
            "contenteditable": _target_attr_count(snapshot, "contenteditable"),
            "open": _target_attr_count(snapshot, "open"),
            "popover_target": _target_attr_count(snapshot, "popover_target"),
            "popover_open": _target_attr_count(snapshot, "popover_open"),
            "tab_index": _target_attr_count(snapshot, "tab_index"),
            "focused": _target_attr_count(snapshot, "focused"),
            "selection_start": _target_attr_count(snapshot, "selection_start"),
            "selection_end": _target_attr_count(snapshot, "selection_end"),
            "text_selection": _target_attr_count(snapshot, "text_selection"),
            "placeholder": _target_attr_count(snapshot, "placeholder"),
            "disabled_options": _target_attr_count(snapshot, "disabled_options"),
            "indeterminate": _target_attr_count(snapshot, "indeterminate"),
            "error_message": _target_attr_count(snapshot, "error_message"),
        },
        "world_path": str(OUT / f"{slug}.world.xml"),
        "snapshot_path": str(OUT / f"{slug}.snapshot.json"),
    }


def _write_sample(slug: str, snapshot: dict[str, Any]) -> dict[str, Any]:
    rendered = render_browser_world_content(snapshot)
    xml, image_parts = _xml_text_and_image_count(rendered)
    (OUT / f"{slug}.world.xml").write_text(xml, encoding="utf-8")
    (OUT / f"{slug}.snapshot.json").write_text(
        json.dumps(_sanitize(snapshot), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return _summarize(slug, snapshot, xml, image_parts)


def export_samples(selected: set[str] | None = None) -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []

    try:
        for slug, raw_url in SAMPLES:
            if selected is not None and slug not in selected:
                continue
            url = _sample_url(raw_url)
            print(f"OPEN {slug}: {url}", flush=True)
            result = execute(
                action="open",
                url=url,
                wait_until="domcontentloaded",
                wait_ms=700,
                visible_images=1 if slug in VISIBLE_IMAGE_WAIT_SAMPLES else 0,
                timeout_ms=25_000,
            )
            if isinstance(result, dict) and result.get("error"):
                errors.append({"slug": slug, "error": str(result.get("error"))})
                print(f"ERROR {slug}: {result.get('error')}", flush=True)
                continue

            snapshot = browser_world_snapshot()
            if not isinstance(snapshot, dict):
                errors.append({"slug": slug, "error": "no snapshot"})
                print(f"ERROR {slug}: no snapshot", flush=True)
                continue

            summary = _write_sample(slug, snapshot)
            if slug == "scroll_region" and (snapshot.get("scroll_regions") or []):
                before = int((snapshot.get("scroll_regions") or [{}])[0].get("scroll_y") or 0)
                execute(action="scroll_region", index=0, pixels=120, wait_ms=300, timeout_ms=5000)
                after_snapshot = browser_world_snapshot()
                after = int(((after_snapshot or {}).get("scroll_regions") or [{}])[0].get("scroll_y") or 0)
                summary["scroll_region_action_changed"] = after != before
                summary["scroll_region_before_after"] = [before, after]
                (OUT / f"{slug}.summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )

            (OUT / f"{slug}.summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            summaries.append(summary)
            print(
                "WROTE "
                f"{slug}: targets={summary['click_targets']} "
                f"text={summary['text_blocks']} "
                f"images={summary['visible_images']} "
                f"bad_rects={summary['out_of_viewport_rects']}",
                flush=True,
            )
    finally:
        try:
            execute(action="close")
        except Exception as exc:
            errors.append({"slug": "close", "error": repr(exc)})

    result = {"summaries": summaries, "errors": errors}
    (OUT / "_index.summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Export real browser world samples.")
    parser.add_argument(
        "samples",
        nargs="*",
        help="Optional sample slugs to export. Defaults to all samples.",
    )
    args = parser.parse_args()

    known = {slug for slug, _ in SAMPLES}
    selected = set(args.samples) if args.samples else None
    if selected:
        unknown = sorted(selected - known)
        if unknown:
            print(f"Unknown samples: {', '.join(unknown)}", file=sys.stderr)
            return 2

    result = export_samples(selected)
    print(json.dumps({"count": len(result["summaries"]), "errors": result["errors"]}, ensure_ascii=False))
    return 1 if result["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
