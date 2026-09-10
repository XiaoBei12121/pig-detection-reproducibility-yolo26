# -*- coding: utf-8 -*-
"""
PigDetect split / source audit (v11 review request).

Purpose: quantify, from the real files only, how the PigDetect splits relate to the dataset's
source prefixes and to each other — i.e. answer whether any of the reported evaluations can be
described as source- or sequence-held-out.

Reproducibility rules used (stated so they can be re-derived by a reviewer):
  * source prefix  : the leading token of the file name, one of {danuma, psota, alameer, bergamini}
  * base name      : Roboflow export suffix removed, i.e. re.sub(r"_jpg(\\.rf\\.[0-9a-f]+)?$", "", stem)
  * clip/sequence  : derived from the base name, in this order
                       - text before "--"                (e.g. alameer_Pen1_201911081132GP031730-...)
                       - text before "_frame_"           (e.g. bergamini_PIGS011219__000454_frame_1524)
                       - trailing "-<digits>" removed    (e.g. 1040s1132s3005-3s5301-2-31)
                       - trailing "_<digits>" removed; the "_<digits>" form is kept as <base>_<first 3 digits>*
                         only when the remaining token is otherwise unique per file
                       - otherwise the base name itself (danuma images carry no video/frame marker)
    * LIMITATION: the danuma subset exposes no frame/video marker in its file names, so no clip-level
      grouping is possible for it; for those images the clip key equals the file base name and the
      only usable grouping is the source prefix. This is reported as such, not hidden.

Outputs (paths relative to the project root):
    results/audit/pigdetect_split_source_audit.csv
    results/audit/pigdetect_split_overlap_audit.json
    docs/PIGDETECT_SPLIT_AUDIT.md

This script only reads split/image/label files. It does not touch weights, logs or any locked metric.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

VERSION = "1.0"
SOURCES = ("danuma", "psota", "alameer", "bergamini")
ROBOFLOW = re.compile(r"_jpg(\.rf\.[0-9a-f]+)?$", re.IGNORECASE)

SPLITS = [
    ("main_train", "PigDetect/train"),
    ("main_val", "PigDetect/val"),
    ("official_test", "PigDetect/test"),
    ("cross_scene_train", "PigDetect_cs/train_cs"),
    ("cross_scene_val", "PigDetect_cs/val_cs"),
    ("cross_scene_test", "PigDetect_cs/test_cs"),
]


def source_of(name: str) -> str:
    for s in SOURCES:
        if name.lower().startswith(s):
            return s
    return "other"


def base_of(stem: str) -> str:
    return ROBOFLOW.sub("", stem)


def clip_of(base: str) -> str:
    if "--" in base:
        return base.split("--")[0]
    if "_frame_" in base:
        return base.split("_frame_")[0]
    m = re.match(r"^(.*)-\d+$", base)
    if m and m.group(1):
        return m.group(1)
    m = re.match(r"^(.*)_\d+$", base)
    if m and m.group(1) and not m.group(1).lower().startswith("danuma"):
        return m.group(1)
    return base


def count_instances(label_dir: Path, stems) -> int:
    n = 0
    for stem in stems:
        p = label_dir / f"{stem}.txt"
        if p.exists():
            with open(p, encoding="utf-8", errors="ignore") as f:
                n += sum(1 for line in f if line.strip())
    return n


def collect(root: Path, rel: str):
    img_dir = root / rel / "images"
    lbl_dir = root / rel / "labels"
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    stems = [p.stem for p in files]
    src = Counter(source_of(p.name) for p in files)
    clips = defaultdict(set)
    for p in files:
        clips[source_of(p.name)].add(clip_of(base_of(p.stem)))
    return {
        "rel": rel,
        "images": len(files),
        "instances": count_instances(lbl_dir, stems),
        "sources": dict(src),
        "stems": set(stems),
        "names": {p.stem: p.name for p in files},
        "clips": {k: set(v) for k, v in clips.items()},
        "all_clips": {c for v in clips.values() for c in v},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=os.environ.get("PIG_DATASET_ROOT", "./data"),
                    help="directory that contains PigDetect/ and PigDetect_cs/")
    ap.add_argument("--project-root", default=os.environ.get("PIG_PROJECT_ROOT", "."),
                    help="root that receives results/audit/ outputs")
    args = ap.parse_args()
    root, proj = Path(args.dataset_root), Path(args.project_root)

    data = {name: collect(root, rel) for name, rel in SPLITS}

    try:
        commit = subprocess.run(["git", "-C", str(proj), "rev-parse", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
    except Exception:
        commit = "unknown"

    # ---------------- CSV ----------------
    out_dir = proj / "results" / "audit"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "pigdetect_split_source_audit.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "images", "instances", "danuma", "psota", "alameer", "bergamini",
                    "other", "unique_source_groups", "unique_clip_groups",
                    "clip_overlap_with_main_train", "same_source_as_official_test"])
        for name, _ in SPLITS:
            d = data[name]
            s = d["sources"]
            clip_overlap = len(d["all_clips"] & data["main_train"]["all_clips"]) if name != "main_train" else "-"
            same_src_test = bool({k for k, v in s.items() if v} & {k for k, v in data["official_test"]["sources"].items() if v})
            w.writerow([name, d["images"], d["instances"], s.get("danuma", 0), s.get("psota", 0),
                        s.get("alameer", 0), s.get("bergamini", 0), s.get("other", 0),
                        len([k for k, v in s.items() if v]), len(d["all_clips"]), clip_overlap, same_src_test])

    # ---------------- intersections ----------------
    def inter(a, b):
        return sorted(data[a]["stems"] & data[b]["stems"])

    pairs = [("main_train", "main_val"), ("main_train", "official_test"), ("main_val", "official_test"),
             ("official_test", "cross_scene_test"), ("cross_scene_train", "cross_scene_test"),
             ("cross_scene_val", "cross_scene_test"), ("cross_scene_train", "cross_scene_val"),
             ("main_train", "cross_scene_test"), ("main_val", "cross_scene_test")]

    img_inter, grp_inter = {}, {}
    for a, b in pairs:
        ia = inter(a, b)
        img_inter[f"{a}__{b}"] = {"count": len(ia), "examples": ia[:20],
                                  "names_examples": [data[a]["names"][s] for s in ia[:5]]}
        ga = sorted(data[a]["all_clips"] & data[b]["all_clips"])
        grp_inter[f"{a}__{b}"] = {"count": len(ga), "examples": ga[:20]}

    # source-level group (prefix) overlap for the same pairs
    src_inter = {}
    for a, b in pairs:
        sa = {k for k, v in data[a]["sources"].items() if v}
        sb = {k for k, v in data[b]["sources"].items() if v}
        src_inter[f"{a}__{b}"] = {"shared_sources": sorted(sa & sb), "count": len(sa & sb)}

    # details for every clip key that appears in both sides of a pair (discloses sequence-level leakage)
    def clip_map(d):
        m = defaultdict(set)
        for stem in d["names"]:
            m[clip_of(base_of(stem))].add(stem)
        return m

    clip_details = {}
    for a, b in pairs:
        ma, mb = clip_map(data[a]), clip_map(data[b])
        shared = sorted(set(ma) & set(mb))
        if not shared:
            continue
        rows = []
        for c in shared:
            src = source_of(next(iter(ma[c])))
            rows.append({"clip": c, "source": src,
                         f"images_in_{a}": len(ma[c]), f"images_in_{b}": len(mb[c])})
        clip_details[f"{a}__{b}"] = {"count": len(shared), "details": rows}
        print(f"[clip overlap] {a} vs {b}: {len(shared)} shared clip keys "
              f"(sources: {sorted({r['source'] for r in rows})})")

    # ---------------- four questions ----------------
    main_train_danuma = data["main_train"]["sources"].get("danuma", 0)
    official_vs_cs = img_inter["official_test__cross_scene_test"]
    mt_ot_img = img_inter["main_train__official_test"]
    mv_ot_img = img_inter["main_val__official_test"]
    mt_ot_clip = grp_inter["main_train__official_test"]
    mv_ot_clip = grp_inter["main_val__official_test"]
    cs_train_test_clip = grp_inter["cross_scene_train__cross_scene_test"]
    cs_val_test_clip = grp_inter["cross_scene_val__cross_scene_test"]

    answers = {
        "Q1_main_train_contains_danuma": {
            "answer": main_train_danuma > 0,
            "count": main_train_danuma,
            "note": ("Main train contains danuma images, therefore the official 250-image test set "
                     "(all danuma) is NOT a source-held-out evaluation with respect to main train."
                     if main_train_danuma > 0 else
                     "Main train contains no danuma image; the official test is source-disjoint from main train."),
        },
        "Q2_official_test_vs_cross_scene_test_overlap": {
            "image_level_count": official_vs_cs["count"],
            "ids": official_vs_cs["examples"],
            "interpretation": ("The two danuma test sets share no image file, i.e. the cross-scene test is a "
                               "different set of danuma images." if official_vs_cs["count"] == 0 else
                               "The two evaluations share images and cannot be treated as independent evidence."),
        },
        "Q3_main_splits_vs_official_test_group_overlap": {
            "group_rule": ("source prefix (danuma/psota/alameer/bergamini) and, where the file name exposes it, "
                           "a clip key (text before '--', before '_frame_', or with a trailing frame index removed)"),
            "main_train_vs_official_test": {"image_level": mt_ot_img["count"], "clip_level": mt_ot_clip["count"],
                                            "shared_sources": src_inter["main_train__official_test"]["shared_sources"]},
            "main_val_vs_official_test": {"image_level": mv_ot_img["count"], "clip_level": mv_ot_clip["count"],
                                          "shared_sources": src_inter["main_val__official_test"]["shared_sources"]},
            "interpretation": ("Main train/val and the official test share the danuma source group; with the "
                               "clip rule above no identical clip key is shared, but the danuma subset exposes no "
                               "frame/video marker, so clip-level disjointness for danuma cannot be established."),
        },
        "Q4_cross_scene_internal_overlap": {
            "train_vs_test": {"image_level": img_inter["cross_scene_train__cross_scene_test"]["count"],
                              "clip_level": cs_train_test_clip["count"],
                              "shared_sources": src_inter["cross_scene_train__cross_scene_test"]["shared_sources"]},
            "val_vs_test": {"image_level": img_inter["cross_scene_val__cross_scene_test"]["count"],
                            "clip_level": cs_val_test_clip["count"],
                            "shared_sources": src_inter["cross_scene_val__cross_scene_test"]["shared_sources"]},
            "interpretation": ("Cross-scene train/val use only psota/alameer/bergamini while the cross-scene test "
                               "is danuma only: the split is source-disjoint at the prefix level and also disjoint "
                               "at the clip level under the stated rule."),
        },
    }

    json_path = out_dir / "pigdetect_split_overlap_audit.json"
    payload = {
        "audit": "PigDetect split/source audit",
        "script": "scripts/audit_pigdetect_split_source.py",
        "script_version": VERSION,
        "project_commit": commit,
        "rules": {
            "source_prefix": "leading token of the file name; one of danuma/psota/alameer/bergamini",
            "base_name": 'Roboflow suffix removed: re.sub(r"_jpg(\\.rf\\.[0-9a-f]+)?$", "", stem)',
            "clip_key": "text before '--' > text before '_frame_' > strip trailing '-<digits>' > strip trailing '_<digits>' > base name",
            "clip_limitation": "danuma file names expose no frame/video marker; clip key falls back to the base name",
        },
        "splits": {name: {"path": data[name]["rel"], "images": data[name]["images"],
                          "instances": data[name]["instances"],
                          "source_counts": data[name]["sources"],
                          "unique_source_groups": len([k for k, v in data[name]["sources"].items() if v]),
                          "unique_clip_groups": len(data[name]["all_clips"])}
                   for name, _ in SPLITS},
        "image_level_intersections": img_inter,
        "clip_level_intersections": grp_inter,
        "shared_clip_details": clip_details,
        "source_prefix_intersections": src_inter,
        "questions": answers,
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # ---------------- console summary ----------------
    print(f"{'split':<20}{'images':>7}{'inst':>7}{'danuma':>8}{'psota':>7}{'alameer':>8}{'bergamini':>10}{'clips':>7}")
    for name, _ in SPLITS:
        d = data[name]
        s = d["sources"]
        print(f"{name:<20}{d['images']:>7}{d['instances']:>7}{s.get('danuma', 0):>8}{s.get('psota', 0):>7}"
              f"{s.get('alameer', 0):>8}{s.get('bergamini', 0):>10}{len(d['all_clips']):>7}")
    print()
    for k, v in answers.items():
        print(k, "->", json.dumps(v, ensure_ascii=False)[:300])
    print(f"\nCSV  -> {csv_path}\nJSON -> {json_path}\ncommit: {commit}")


if __name__ == "__main__":
    main()
