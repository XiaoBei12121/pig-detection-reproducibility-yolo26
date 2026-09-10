# PigDetect split / source audit (v11 review request)

**Scope:** pure file-level audit of the six PigDetect splits used in the manuscript. No model was
retrained, no weight/log/locked metric was modified, and the GitHub release `v1.0.1` was not touched.

- Audit script: `scripts/audit_pigdetect_split_source.py` (v1.0)
- Raw outputs: `results/audit/pigdetect_split_source_audit.csv`, `results/audit/pigdetect_split_overlap_audit.json`
- Project commit at run time: `0f51028`
- Data roots: `dataset/PigDetect/{train,val,test}`, `dataset/PigDetect_cs/{train_cs,val_cs,test_cs}`

## Rules used (re-derivable by a reviewer)

| Concept | Definition |
|---|---|
| source prefix | leading token of the file name: `danuma`, `psota`, `alameer`, `bergamini` (else `other`) |
| base name | Roboflow export suffix removed: `re.sub(r"_jpg(\.rf\.[0-9a-f]+)?$", "", stem)` |
| clip key | first match of: text before `--` → text before `_frame_` → base name with trailing `-<digits>` removed → base name with trailing `_<digits>` removed |
| instances | number of non-empty label rows in the corresponding `labels/*.txt` |

**Stated limitation.** The `danuma` file names contain no frame or video marker (`danuma_025.jpg`),
so no clip-level grouping is possible for that source: its clip key falls back to the file base name,
and clip-level disjointness for danuma cannot be established from file names. The source prefix is
**not** a physical farm: `psota`, `alameer` and `bergamini` are dataset/source names that may each
contain several pens or recording sessions.

## Split composition (measured)

| Split | Images | Instances | danuma | psota | alameer | bergamini | other | unique clip keys |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| main train | 2411 | 33069 | **1319** | 654 | 271 | 167 | 0 | 1553 |
| main val | 270 | 3672 | 135 | 79 | 37 | 19 | 0 | 176 |
| official test | 250 | 5436 | **250** | 0 | 0 | 0 | 0 | 250 |
| cross-scene train | 1098 | 13721 | 0 | 658 | 274 | 166 | 0 | 235 |
| cross-scene val | 129 | 1659 | 0 | 75 | 34 | 20 | 0 | 35 |
| cross-scene test | 450 | 6549 | **450** | 0 | 0 | 0 | 0 | 450 |

Instance totals: main train 33,069 · main val 3,672 · official test 5,436 · cross-scene train 13,721 ·
cross-scene val 1,659 · cross-scene test 6,549.

## Overlap audit

### Image level

| Pair | Shared images |
|---|---:|
| main train ∩ main val | 0 |
| main train ∩ official test | 0 |
| main val ∩ official test | 0 |
| official test ∩ cross-scene test | **0** |
| cross-scene train ∩ cross-scene test | 0 |
| cross-scene val ∩ cross-scene test | 0 |
| cross-scene train ∩ cross-scene val | 0 |
| **main train ∩ cross-scene test** | **411** |
| **main val ∩ cross-scene test** | **39** |

→ 411 + 39 = **450**, i.e. *every* image of the cross-scene test set also belongs to the main
train/val pool (all of them danuma).

### Clip level (rule above)

| Pair | Shared clip keys | Sources involved |
|---|---:|---|
| main train ∩ main val | **17** | alameer, bergamini, psota |
| cross-scene train ∩ cross-scene val | **12** | alameer, bergamini, psota |
| main train ∩ official test | 0 | (danuma; clip = file name, so this is uninformative) |
| main val ∩ official test | 0 | (danuma; uninformative) |
| official test ∩ cross-scene test | 0 | (danuma; uninformative) |
| cross-scene train ∩ cross-scene test | 0 | — |
| cross-scene val ∩ cross-scene test | 0 | — |
| main train ∩ cross-scene test | 411 | danuma (clip = file name) |
| main val ∩ cross-scene test | 39 | danuma (clip = file name) |

Per-clip details (clip key, source, images on each side) are listed in
`results/audit/pigdetect_split_overlap_audit.json → shared_clip_details`.

### Source-prefix level

| Pair | Shared source prefixes |
|---|---|
| main train ∩ official test | danuma |
| main val ∩ official test | danuma |
| main train ∩ main val | alameer, bergamini, danuma, psota |
| cross-scene train ∩ cross-scene test | **none** |
| cross-scene val ∩ cross-scene test | **none** |

## Answers to the four questions

**Q1 — Does main train contain danuma? How many?**
Yes: **1319** images (plus 135 in main val). Consequently the official 250-image test set — which is
**entirely danuma** — is *not* a source-held-out evaluation with respect to main train, and the
manuscript's wording "exploratory test set" is the correct characterisation.

**Q2 — Do the official test (250) and the cross-scene danuma test (450) overlap?**
**No.** Image-level intersection = **0** (empty ID list, see the JSON). They are two disjoint sets of
danuma images, so the two evaluations are independent *with respect to each other*; they are not
independent with respect to the main training pool (see Q3/Q1).

**Q3 — Do main train/val share a video/sequence group with the official test?**
Group rule: source prefix plus, where the file name exposes it, the clip key defined above.
Image level = **0** for both main train and main val; clip level = **0** for both; the shared
*source prefix* is **danuma** in both cases. Caveat: because danuma file names carry no frame marker,
the clip key degenerates to the file name for that source, so the "0 shared clips" result carries no
information for danuma — the honest statement is "different image files from the same source prefix".

**Q4 — Do cross-scene train/val share a group with the cross-scene test?**
**No.** Image level = 0, clip level = 0, shared source prefixes = none: cross-scene train/val contain
only psota/alameer/bergamini, the cross-scene test only danuma. The cross-scene *evaluation* is
therefore source-disjoint. (Independently of this, cross-scene train and val share 12 clip keys with
each other — see the note below.)

## Additional findings that must be disclosed

1. **The cross-scene test images are not unseen images of the study.** All 450 danuma test images
   also occur in the main train (411) or main val (39) split. The cross-scene result is nevertheless
   valid as a *source-held-out* evaluation **for the cross-scene models**, whose training pool contains
   no danuma image; but it must not be presented as an untouched/external image set, and it cannot be
   used to claim that the main models were evaluated on unseen images.
2. **The main train/val split is image-disjoint but not strictly clip-disjoint.** 17 clip keys
   (≈1 % of the 1729 clip keys in train+val) appear on both sides, all from psota/alameer/bergamini
   naming patterns whose file names expose frame indices. The prefix rule used by the split script
   (`stem.split("--")[0]`) separates camera prefixes for the `--`-style names, but degrades to
   file-level grouping for names without `--`, so a few frames of the same recording can land in both
   train and val. This is a weak (≈1 %) sequence-level leakage into the *validation* split only; the
   test splits remain image-disjoint from training.
3. Cross-scene train/val show the same pattern (12 shared clip keys), for the same naming reason.

## English note for the manuscript (measured facts only)

> **Split/source audit of PigDetect.** The six splits used here were audited at file level. The main
> training split contains 1319 `danuma` images (plus 135 in the main validation split) and 1092 images
> from the `psota`, `alameer` and `bergamini` sources; the official 250-image test split consists of
> `danuma` images only. The official test split and the cross-scene test split (450 `danuma` images)
> share **no** image (intersection = 0), but all 450 cross-scene test images also belong to the main
> train/val pool (411 in train, 39 in val), so the cross-scene test cannot be described as an
> untouched external image set; it is source-held-out only for the cross-scene models, which were
> trained without any `danuma` image. Because the main training split contains `danuma` images, the
> official test split is **not** a source-held-out evaluation with respect to the main models and is
> therefore treated as an exploratory test set in this study. Source prefixes are dataset/file-name
> prefixes and are not equivalent to physical farms or pens; the `danuma` file names expose no
> video/frame marker, so no clip-level grouping is possible for that source. At the clip level derived
> from file names, the main train and validation splits share 17 clip keys (≈1 % of the 1729 clip keys
> in train+val) arising from non-`--` naming patterns, i.e. the validation split carries a small
> sequence-level overlap with training; the main and cross-scene test splits remain image-disjoint from
> their respective training pools, and the cross-scene split is disjoint from the cross-scene test at
> both the image and the source-prefix level.

## Files

| File | Content |
|---|---|
| `results/audit/pigdetect_split_source_audit.csv` | the composition table above, machine-readable |
| `results/audit/pigdetect_split_overlap_audit.json` | image/clip/source intersections, per-clip details, source counts, instance counts, rules, script version and commit |
| `docs/PIGDETECT_SPLIT_AUDIT.md` | this document |
