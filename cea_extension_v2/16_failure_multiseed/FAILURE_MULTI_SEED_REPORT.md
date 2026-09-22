# Failure-condition analysis across matched seeds (Phase 3)

Run: 2026-09-21T21:34:03  
Seeds: [1, 21, 42, 101, 113]  
Checkpoint rules: ['best', 'last']  
Datasets: ['pigdetect', 'oinktrack']  
Inference-only runtime: 49.5 min

Overlap levels use the **geometric bounding-box overlap proxy** and must not be described as true visible occlusion. Lighting buckets are the official OinkTrack scene names (D / DN / N), never regrouped. Matching: confidence >= 0.25, greedy one-to-one. Paired deltas are E5 - E0 per matched seed in **exact rational arithmetic** (recall = hits / instances), so the sign-flip enumeration contains no floating-point step.

Two boundary conventions are reported because the locked text and the extension's scripts disagree (author-acceptance finding #13):

| variant | levels | matching | status |
|---|---|---|---|
| A_locked_text_manuscript | Low r<0.3, Medium 0.3<=r<=0.6, High r>0.6 | IoU >= 0.5 | **PRIMARY** (locked text + `scripts/occlusion_analysis.py`) |
| B_extension_strict | Low r<0.3, Medium 0.3<=r<0.6, High r>=0.6 | IoU > 0.5 | sensitivity |

| dataset | variant | condition | rule | n | mean delta | median | SD | min | max | +/-/0 | 95% CI | exact p2 | low support |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|---|---:|---|
| oinktrack | A | lighting:DN | best | 5 | -0.008406 | -0.016914 | 0.034833 | -0.045117 | +0.044091 | 2/3/0 | [-0.033463, +0.021231] | 5/8 (20/32) | no |
| oinktrack | B | lighting:DN | best | 5 | -0.008406 | -0.016914 | 0.034833 | -0.045117 | +0.044091 | 2/3/0 | [-0.033463, +0.021231] | 5/8 (20/32) | no |
| oinktrack | A | lighting:DN | last | 5 | -0.011563 | -0.028637 | 0.037045 | -0.047674 | +0.043379 | 2/3/0 | [-0.038133, +0.021051] | 9/16 (18/32) | no |
| oinktrack | B | lighting:DN | last | 5 | -0.011563 | -0.028637 | 0.037045 | -0.047674 | +0.043379 | 2/3/0 | [-0.038133, +0.021051] | 9/16 (18/32) | no |
| oinktrack | A | lighting:D | best | 5 | -0.002848 | -0.010630 | 0.039814 | -0.041316 | +0.062525 | 2/3/0 | [-0.029041, +0.032551] | 7/8 (28/32) | no |
| oinktrack | B | lighting:D | best | 5 | -0.002848 | -0.010630 | 0.039814 | -0.041316 | +0.062525 | 2/3/0 | [-0.029041, +0.032551] | 7/8 (28/32) | no |
| oinktrack | A | lighting:D | last | 5 | -0.004222 | -0.003109 | 0.046480 | -0.051294 | +0.064079 | 2/3/0 | [-0.038287, +0.032290] | 15/16 (30/32) | no |
| oinktrack | B | lighting:D | last | 5 | -0.004222 | -0.003109 | 0.046480 | -0.051294 | +0.064079 | 2/3/0 | [-0.038287, +0.032290] | 15/16 (30/32) | no |
| oinktrack | A | lighting:N | best | 5 | -0.006941 | -0.002800 | 0.010425 | -0.023383 | +0.002458 | 1/4/0 | [-0.015786, +0.000320] | 3/16 (6/32) | no |
| oinktrack | B | lighting:N | best | 5 | -0.006941 | -0.002800 | 0.010425 | -0.023383 | +0.002458 | 1/4/0 | [-0.015786, +0.000320] | 3/16 (6/32) | no |
| oinktrack | A | lighting:N | last | 5 | -0.014827 | -0.012770 | 0.014569 | -0.035839 | +0.000724 | 1/4/0 | [-0.026753, -0.003985] | 1/8 (4/32) | no |
| oinktrack | B | lighting:N | last | 5 | -0.014827 | -0.012770 | 0.014569 | -0.035839 | +0.000724 | 1/4/0 | [-0.026753, -0.003985] | 1/8 (4/32) | no |
| oinktrack | A | overlap:High | best | 5 | -0.009181 | -0.021297 | 0.027442 | -0.036539 | +0.030646 | 2/3/0 | [-0.029134, +0.014645] | 9/16 (18/32) | no |
| oinktrack | B | overlap:High | best | 5 | -0.009181 | -0.021297 | 0.027442 | -0.036539 | +0.030646 | 2/3/0 | [-0.029134, +0.014645] | 9/16 (18/32) | no |
| oinktrack | A | overlap:High | last | 5 | -0.013877 | -0.025576 | 0.030014 | -0.040006 | +0.029216 | 2/3/0 | [-0.036268, +0.010926] | 3/8 (12/32) | no |
| oinktrack | B | overlap:High | last | 5 | -0.013877 | -0.025576 | 0.030014 | -0.040006 | +0.029216 | 2/3/0 | [-0.036268, +0.010926] | 3/8 (12/32) | no |
| oinktrack | A | overlap:Low | best | 5 | -0.000524 | -0.000898 | 0.028356 | -0.038380 | +0.041409 | 2/3/0 | [-0.023088, +0.023896] | 7/8 (28/32) | no |
| oinktrack | B | overlap:Low | best | 5 | -0.000524 | -0.000898 | 0.028356 | -0.038380 | +0.041409 | 2/3/0 | [-0.023088, +0.023896] | 7/8 (28/32) | no |
| oinktrack | A | overlap:Low | last | 5 | -0.002978 | -0.000935 | 0.029495 | -0.041110 | +0.039502 | 2/3/0 | [-0.025040, +0.021232] | 13/16 (26/32) | no |
| oinktrack | B | overlap:Low | last | 5 | -0.002978 | -0.000935 | 0.029495 | -0.041110 | +0.039502 | 2/3/0 | [-0.025040, +0.021232] | 13/16 (26/32) | no |
| oinktrack | A | overlap:Medium | best | 5 | -0.005755 | -0.011485 | 0.026421 | -0.027346 | +0.037004 | 1/4/0 | [-0.023885, +0.016813] | 9/16 (18/32) | no |
| oinktrack | B | overlap:Medium | best | 5 | -0.005755 | -0.011485 | 0.026421 | -0.027346 | +0.037004 | 1/4/0 | [-0.023885, +0.016813] | 9/16 (18/32) | no |
| oinktrack | A | overlap:Medium | last | 5 | -0.009868 | -0.020492 | 0.028743 | -0.039970 | +0.035612 | 1/4/0 | [-0.029043, +0.013170] | 7/16 (14/32) | no |
| oinktrack | B | overlap:Medium | last | 5 | -0.009868 | -0.020492 | 0.028743 | -0.039970 | +0.035612 | 1/4/0 | [-0.029043, +0.013170] | 7/16 (14/32) | no |
| pigdetect | A | overlap:High | best | 5 | -0.003253 | -0.002140 | 0.002768 | -0.007705 | -0.000428 | 0/5/0 | [-0.005479, -0.001455] | 1/16 (2/32) | no |
| pigdetect | B | overlap:High | best | 5 | -0.003253 | -0.002140 | 0.002768 | -0.007705 | -0.000428 | 0/5/0 | [-0.005479, -0.001455] | 1/16 (2/32) | no |
| pigdetect | A | overlap:High | last | 5 | -0.001712 | -0.002140 | 0.005639 | -0.007705 | +0.006421 | 2/3/0 | [-0.005908, +0.002825] | 1/2 (16/32) | no |
| pigdetect | B | overlap:High | last | 5 | -0.001712 | -0.002140 | 0.005639 | -0.007705 | +0.006421 | 2/3/0 | [-0.005908, +0.002825] | 1/2 (16/32) | no |
| pigdetect | A | overlap:Low | best | 5 | +0.000350 | -0.001166 | 0.004176 | -0.004665 | +0.005248 | 2/3/0 | [-0.002799, +0.003615] | 15/16 (30/32) | no |
| pigdetect | B | overlap:Low | best | 5 | +0.000350 | -0.001166 | 0.004176 | -0.004665 | +0.005248 | 2/3/0 | [-0.002799, +0.003615] | 15/16 (30/32) | no |
| pigdetect | A | overlap:Low | last | 5 | -0.000466 | -0.000583 | 0.003334 | -0.004665 | +0.004665 | 1/4/0 | [-0.003032, +0.002449] | 5/8 (20/32) | no |
| pigdetect | B | overlap:Low | last | 5 | -0.000466 | -0.000583 | 0.003334 | -0.004665 | +0.004665 | 1/4/0 | [-0.003032, +0.002449] | 5/8 (20/32) | no |
| pigdetect | A | overlap:Medium | best | 5 | -0.000866 | -0.001444 | 0.004463 | -0.005054 | +0.005054 | 2/3/0 | [-0.004332, +0.002599] | 13/16 (26/32) | no |
| pigdetect | B | overlap:Medium | best | 5 | -0.000866 | -0.001444 | 0.004463 | -0.005054 | +0.005054 | 2/3/0 | [-0.004332, +0.002599] | 13/16 (26/32) | no |
| pigdetect | A | overlap:Medium | last | 5 | -0.000578 | +0.001444 | 0.004284 | -0.005776 | +0.004332 | 3/2/0 | [-0.004043, +0.002599] | 15/16 (30/32) | no |
| pigdetect | B | overlap:Medium | last | 5 | -0.000578 | +0.001444 | 0.004284 | -0.005776 | +0.004332 | 3/2/0 | [-0.004043, +0.002599] | 15/16 (30/32) | no |

## Method notes

* GT boxes and both level assignments are model-independent and were computed once per run; all IoU/matching work is vectorised with numpy using the manuscript's exact formula and operation order, so the greedy matching result equals the scalar implementation.
* OinkTrack predictions are exported one sequence at a time, so the frame mapping is unambiguous (anomaly #20); exports are deleted immediately after each checkpoint.
* buckets with fewer than 100 GT instances in any seed are flagged `low support`; no significance interpretation should be attached to them.
* no threshold, split, seed or checkpoint rule was changed for this analysis; no model was trained.