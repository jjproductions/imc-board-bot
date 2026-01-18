#!/usr/bin/env python3
import json
from pathlib import Path

DEBUG = Path("data/docling_debug_output_20260118T041958Z.json")
CONV = Path("data/docling_converted_output_20260118T041958Z.json")


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def find_list(d):
    for k in ("texts", "blocks", "items", "documents"):
        if k in d and isinstance(d[k], list):
            return k, d[k]
    # fallback: find first list of dicts
    for k, v in d.items():
        if isinstance(v, list) and v and isinstance(v[0], dict):
            return k, v
    return None, []


def make_id(e, prefix):
    for key in ("id", "self_ref"):
        if key in e and e[key]:
            return e[key]
    # try orig or text
    if "orig" in e and e["orig"]:
        return f"{prefix}::orig::{e['orig'][:80]}"
    if "text" in e and e["text"]:
        return f"{prefix}::text::{e['text'][:80]}"
    return f"{prefix}::idx::"


def field_present(e, field):
    return field in e and e[field] is not None


def compare():
    debug = load(DEBUG)
    conv = load(CONV)

    k_dbg, dbg_list = find_list(debug)
    k_conv, conv_list = find_list(conv)

    print(f"Found debug list key: {k_dbg}, converted list key: {k_conv}")
    print(f"Debug items: {len(dbg_list)}, Converted items: {len(conv_list)}\n")

    dbg_map = {}
    for i, e in enumerate(dbg_list):
        idv = make_id(e, "dbg") or f"dbg_idx_{i}"
        # prefer self_ref if exists
        dbg_map[idv] = e

    conv_map = {}
    for i, e in enumerate(conv_list):
        idv = make_id(e, "conv") or f"conv_idx_{i}"
        conv_map[idv] = e

    # Attempt to match by exact text or orig when ids differ
    text_to_conv = {}
    for idv, e in conv_map.items():
        t = e.get("text") or e.get("orig")
        if t:
            text_to_conv[t.strip()] = idv

    mismatches = {
        "missing_in_conv": [],
        "parent_mismatch": [],
        "prov_missing": [],
        "orig_missing": [],
        "enumeration_mismatch": [],
        "type_label_mismatch": [],
        "level_mismatch": [],
        "table_meta_missing": [],
    }

    for idv, de in dbg_map.items():
        key = None
        # prefer matching by self_ref
        if "self_ref" in de and de["self_ref"] in conv_map:
            key = de["self_ref"]
        # try matching by text/orig
        if not key:
            t = (de.get("text") or de.get("orig") or "").strip()
            if t and t in text_to_conv:
                key = text_to_conv[t]
        # fallback: try by id-like substrings
        if not key:
            # try matching conv entries that contain the same orig start
            for cid, ce in conv_map.items():
                if (
                    "orig" in de
                    and "orig" in ce
                    and de["orig"]
                    and ce["orig"]
                    and de["orig"][:30] == ce["orig"][:30]
                ):
                    key = cid
                    break
        if not key:
            mismatches["missing_in_conv"].append(
                {"dbg_id": idv, "text": de.get("text") or de.get("orig")}
            )
            continue
        ce = conv_map.get(key)
        # parent check
        dbg_parent = de.get("parent", de.get("parent_id"))
        conv_parent = ce.get("parent_id", ce.get("parent"))
        if bool(dbg_parent) != bool(conv_parent):
            mismatches["parent_mismatch"].append(
                {
                    "dbg_id": idv,
                    "conv_id": key,
                    "dbg_parent": dbg_parent,
                    "conv_parent": conv_parent,
                }
            )
        # prov check
        if not de.get("prov") and ce.get("prov"):
            # debug missing but conv has -> odd but record
            mismatches["prov_missing"].append(
                {
                    "dbg_id": idv,
                    "conv_id": key,
                    "note": "dbg missing prov, conv has prov",
                }
            )
        elif de.get("prov") and not ce.get("prov"):
            mismatches["prov_missing"].append(
                {"dbg_id": idv, "conv_id": key, "note": "conv missing prov"}
            )
        # orig
        if de.get("orig") and not ce.get("orig"):
            mismatches["orig_missing"].append({"dbg_id": idv, "conv_id": key})
        # enumerated/marker
        if de.get("enumerated") is not None:
            dbg_enum = {"enumerated": de.get("enumerated"), "marker": de.get("marker")}
            conv_enum = {"enumerated": ce.get("enumerated"), "marker": ce.get("marker")}
            if dbg_enum != conv_enum:
                mismatches["enumeration_mismatch"].append(
                    {"dbg_id": idv, "conv_id": key, "dbg": dbg_enum, "conv": conv_enum}
                )
        # label/type mapping
        dbg_label = de.get("label")
        conv_type = ce.get("type")
        # quick heuristic mapping
        map_expect = None
        if dbg_label == "section_header":
            map_expect = "heading"
        elif dbg_label == "list_item":
            map_expect = "list_item"
        elif dbg_label in ("text", "body"):
            map_expect = "paragraph"
        elif dbg_label in ("page_footer", "page_header"):
            map_expect = "page_break"
        if map_expect and conv_type and map_expect != conv_type:
            mismatches["type_label_mismatch"].append(
                {
                    "dbg_id": idv,
                    "conv_id": key,
                    "dbg_label": dbg_label,
                    "conv_type": conv_type,
                    "expected": map_expect,
                }
            )
        # level
        if de.get("level") is not None:
            if de.get("level") != ce.get("level"):
                mismatches["level_mismatch"].append(
                    {
                        "dbg_id": idv,
                        "conv_id": key,
                        "dbg_level": de.get("level"),
                        "conv_level": ce.get("level"),
                    }
                )
        # tables
        if de.get("label") == "table" and not ce.get("meta"):
            mismatches["table_meta_missing"].append({"dbg_id": idv, "conv_id": key})

    # print concise report
    print("--- Comparison Report ---")
    for k, v in mismatches.items():
        print(f"{k}: {len(v)}")
        if v and len(v) <= 10:
            for item in v:
                print(" ", item)
    # write detailed json
    out = Path("tmp/compare_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(mismatches, indent=2), encoding="utf-8")
    print("\nDetailed report written to", out)


if __name__ == "__main__":
    compare()
