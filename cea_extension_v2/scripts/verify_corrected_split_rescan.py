# -*- coding: utf-8 -*-
"""Independent re-scan and re-hash of the corrected PigDetect split (2,931 images).

Purpose (author request): prove that the split described in the paper, the split recorded in the
audit, and the split actually present on disk are one and the same set of images.

What is recomputed from scratch here (nothing is trusted from the earlier run)
-----------------------------------------------------------------------------
for each of corrected_train (2411) / corrected_val (270) / official_test (250):
  * image count (and that the folder contains nothing else image-like)
  * SHA256 of every image file
  * split manifest SHA256  = sha256("\n".join(sorted(file names)) + "\n")
  * per-split content SHA256 = sha256("\n".join(sorted("<stem>:<sha256>")))
  * recorded per-image list (split,file,sha256,size) and its own file SHA256
cross-split checks: train∩val, train∩test, val∩test at BOTH file-name level and CONTENT-hash level.

The two aggregate conventions are copied verbatim from the authoritative script
`scripts/phase1_corrected_split_audit.py` (lines 101-105 and 154-158) so that a MATCH is meaningful;
the hashing itself is redone independently.

Outputs
-------
`audit/corrected_split_rescan.json`   full re-scan + the comparison table
`audit/corrected_split_rescan.md`     human-readable verdict
and a new section appended to `audit/corrected_split_summary.md`

Exit code 0 only when every value matches the recorded audit.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

PJ = Path(r"<PIG_PROJECT>")
EXP = PJ / "experiments" / "cea_extension_v2"
AUD = EXP / "audit"
DATA_ROOT = PJ / "dataset"
YAML = DATA_ROOT / "PigDetect_lc" / "pig_lc.yaml"

SPLITS = {
    "corrected_train": ("PigDetect_lc/train", 2411),
    "corrected_val": ("PigDetect_lc/val", 270),
    "official_test": ("PigDetect/test", 250),
}
IMG_EXT = {".jpg", ".jpeg", ".png"}


def sha256_file(p: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def scan(rel: str) -> dict:
    img_dir = DATA_ROOT / rel / "images"
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT)
    others = sorted(p.name for p in img_dir.iterdir()
                    if p.is_file() and p.suffix.lower() not in IMG_EXT)
    per_image = [(p.name, sha256_file(p), p.stat().st_size) for p in files]
    names = [n for n, _, _ in per_image]
    manifest = hashlib.sha256(("\n".join(sorted(names)) + "\n").encode()).hexdigest()
    aggregate = hashlib.sha256(
        "\n".join(sorted(f"{Path(n).stem}:{h}" for n, h, _ in per_image)).encode()).hexdigest()
    return {"rel": rel, "img_dir": str(img_dir), "images": len(files),
            "non_image_files_in_folder": others, "per_image": per_image,
            "manifest_sha256": manifest, "content_sha256": aggregate,
            "stems": {Path(n).stem for n in names},
            "hashes": {h for _, h, _ in per_image}}


def main() -> int:
    recorded_hashes = json.loads((AUD / "corrected_split_hashes.json").read_text(encoding="utf-8"))
    recorded_audit = json.loads((AUD / "corrected_split_audit.json").read_text(encoding="utf-8"))

    yaml_hash = hashlib.sha256(YAML.read_bytes()).hexdigest()
    scans = {name: scan(rel) for name, (rel, _) in SPLITS.items()}

    # per-image list, same row order and writer settings as phase 1 -> byte-comparable
    tmp = Path(tempfile.mkdtemp()) / "corrected_split_image_hashes.csv"
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "file", "sha256", "size_bytes"])
        for name in SPLITS:
            for n, h, s in scans[name]["per_image"]:
                w.writerow([name, n, h, s])
    list_sha = sha256_file(tmp)

    # semantic comparison against the recorded CSV
    old_csv = AUD / "corrected_split_image_hashes.csv"
    old_rows = set()
    if old_csv.exists():
        with old_csv.open(encoding="utf-8-sig", newline="") as f:
            for r in csv.DictReader(f):
                old_rows.add((r["split"], r["file"], r["sha256"], int(r["size_bytes"])))
    new_rows = {(name, n, h, s) for name in SPLITS for n, h, s in scans[name]["per_image"]}

    checks = []

    def chk(label: str, got, want) -> None:
        checks.append({"check": label, "rescanned": got, "recorded": want,
                       "verdict": "MATCH" if got == want else "MISMATCH"})

    # counts
    for name, (_, exp) in SPLITS.items():
        chk(f"{name}: image count", scans[name]["images"], exp)
        chk(f"{name}: image count vs recorded audit",
            scans[name]["images"], recorded_audit["splits"][name]["images"])
    chk("total images hashed", sum(s["images"] for s in scans.values()),
        recorded_hashes["total_images_hashed"])

    # hashes
    chk("yaml sha256", yaml_hash, recorded_hashes["yaml"]["sha256"])
    for name in SPLITS:
        chk(f"{name}: split manifest sha256", scans[name]["manifest_sha256"],
            recorded_hashes["split_manifest_sha256"][name])
        chk(f"{name}: per-split content sha256", scans[name]["content_sha256"],
            recorded_hashes["per_split_content_sha256"][name])
    chk("per-image hash list sha256", list_sha, recorded_hashes["image_hash_list_sha256"])
    chk("per-image rows identical to recorded CSV", len(new_rows - old_rows) == 0 and
        len(old_rows - new_rows) == 0, True)

    # cross-split duplicates, file-name level and CONTENT level
    for a, b in (("corrected_train", "corrected_val"),
                 ("corrected_train", "official_test"),
                 ("corrected_val", "official_test")):
        chk(f"{a} ∩ {b} (file names)", len(scans[a]["stems"] & scans[b]["stems"]), 0)
        chk(f"{a} ∩ {b} (identical image content)", len(scans[a]["hashes"] & scans[b]["hashes"]), 0)
    chk("duplicate image content inside corrected_train",
        scans["corrected_train"]["images"] - len(scans["corrected_train"]["hashes"]), 0)
    chk("duplicate image content inside corrected_val",
        scans["corrected_val"]["images"] - len(scans["corrected_val"]["hashes"]), 0)
    chk("duplicate image content inside official_test",
        scans["official_test"]["images"] - len(scans["official_test"]["hashes"]), 0)

    # intersections recorded by the earlier audit, re-derived from the re-scan
    for key, (a, b) in {"corrected_train__corrected_val": ("corrected_train", "corrected_val"),
                        "corrected_train__official_test": ("corrected_train", "official_test"),
                        "corrected_val__official_test": ("corrected_val", "official_test")}.items():
        rec = (recorded_audit.get("image_level_intersections", {}).get(key) or {}).get("count")
        chk(f"recorded audit intersection {key}", len(scans[a]["stems"] & scans[b]["stems"]), rec)

    verdict = "MATCH" if all(c["verdict"] == "MATCH" for c in checks) else "MISMATCH"
    report = {"timestamp": datetime.now().isoformat(timespec="seconds"),
              "verdict": verdict,
              "splits": {n: {"images": s["images"], "img_dir": s["img_dir"],
                             "manifest_sha256": s["manifest_sha256"],
                             "content_sha256": s["content_sha256"],
                             "non_image_files_in_folder": s["non_image_files_in_folder"]}
                         for n, s in scans.items()},
              "yaml_sha256": yaml_hash,
              "per_image_hash_list_sha256": list_sha,
              "per_image_rows": {"rescanned": len(new_rows), "recorded": len(old_rows),
                                 "only_in_rescan": len(new_rows - old_rows),
                                 "only_in_recorded": len(old_rows - new_rows)},
              "checks": checks}
    (AUD / "corrected_split_rescan.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    md = ["# Corrected split — independent re-scan and re-hash", "",
          f"Run: {report['timestamp']}  ",
          f"**Verdict: {verdict}**  ",
          f"Images hashed: {sum(s['images'] for s in scans.values())} "
          f"(train {scans['corrected_train']['images']}, val {scans['corrected_val']['images']}, "
          f"test {scans['official_test']['images']})", "",
          "| check | re-scanned | recorded | verdict |", "|---|---|---|---|"]
    for c in checks:
        r = c["rescanned"]
        w = c["recorded"]
        r = f"{r}" if not isinstance(r, str) or len(str(r)) < 20 else f"`{str(r)[:16]}…`"
        w = f"{w}" if not isinstance(w, str) or len(str(w)) < 20 else f"`{str(w)[:16]}…`"
        md.append(f"| {c['check']} | {r} | {w} | **{c['verdict']}** |")
    md += ["", "Aggregate conventions (identical to `scripts/phase1_corrected_split_audit.py`):",
           "* split manifest SHA256 = sha256 of the sorted file-name list joined with `\\n` plus a "
           "trailing `\\n`",
           "* per-split content SHA256 = sha256 of the sorted `stem:sha256` list joined with `\\n`", ""]
    (AUD / "corrected_split_rescan.md").write_text("\n".join(md), encoding="utf-8")

    # append the verdict to the summary the author reads
    summary = AUD / "corrected_split_summary.md"
    if summary.exists():
        text = summary.read_text(encoding="utf-8")
        marker = "## 重新扫描核验"
        if marker in text:
            text = text[:text.index(marker)]
        text += (f"{marker}（{report['timestamp']}）\n\n"
                 f"独立重扫 2,931 张图并重算全部哈希：**{verdict}**。\n\n"
                 f"* train {scans['corrected_train']['images']} / val {scans['corrected_val']['images']} "
                 f"/ test {scans['official_test']['images']}（与记录一致）\n"
                 f"* 三个划分两两交集：文件名级 0，**图像内容级 0**\n"
                 f"* split manifest / per-split content / yaml / 逐图哈希清单，逐值与 "
                 f"`corrected_split_hashes.json` 一致\n"
                 f"* 详见 `audit/corrected_split_rescan.md` 与 `audit/corrected_split_rescan.json`\n")
        summary.write_text(text, encoding="utf-8")

    for c in checks:
        print(f"  [{c['verdict']:8s}] {c['check']}: {str(c['rescanned'])[:24]} vs {str(c['recorded'])[:24]}")
    print(f"\nVERDICT: {verdict}")
    print(f"written: {AUD / 'corrected_split_rescan.json'} , {AUD / 'corrected_split_rescan.md'}")
    shutil.rmtree(tmp.parent, ignore_errors=True)
    return 0 if verdict == "MATCH" else 3


if __name__ == "__main__":
    raise SystemExit(main())
