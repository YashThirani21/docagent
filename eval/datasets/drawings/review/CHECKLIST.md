# Drawing ground truth - review checklist (test split)

`nist_ctc_03_asme1_rc.json` (30 values, 1 sheet) and `nist_ftc_11_asme1_rb.json` (12 values + 1 optional, 2 sheets)
were drafted by the AI assistant from the rendered sheets (150/300 dpi) and the text layer - **not** from the
extractor's output, so the test split isn't contaminated. They count as `verified: false` until you check them.

Open each overlay next to the PDF (`eval/datasets/drawings/review/<id>_p<n>.png`: red box + number per value,
legend underneath) and tick:

- [ ] every blue annotation on the sheet has a box (nothing missed; leader-only marks and coordinate triads excluded)
- [ ] printed text per value matches the sheet (`raw_text`: digits, ±, Ø, quantities "2X/4X", modifiers Ⓜ Ⓕ)
- [ ] numbers: `nominal`, `tol_plus`, `tol_minus`, `quantity` (inch drawing: .438 = 0.438)
- [ ] GD&T frames: `characteristic` and `datums` in order
- [ ] `belongs` keyword lists: the first word is the main feature (scored by `belongs_primary`), the rest are
      accepted alternatives for the lenient `belongs_to` metric
- [ ] title block, `units`, `n_notes`

Points where the draft made a judgement call - please confirm or change:

| drawing | value | question |
|---|---|---|
| CTC-03 | #20 `∠ .04 A` | symbol read as **angularity** (slanted line over a base) |
| CTC-03 | #21 `▱ .005 / .25 X .25` | flatness per unit area .25 x .25 - nominal kept as .005 |
| CTC-03 | #27 `4X .82 ±.06` | the circle-with-line on the dimension line read as a dimension-origin symbol, not part of the value |
| CTC-03 | #30 `⌖ .03 D B C` | attached to the dimension between the two vertical extension lines (feature unclear: hole pair / slot) |
| CTC-03 | #5 `(.750)` | reference dimension, kept as a required value |
| CTC-03 | `n_notes: 1` | "Datum Feature F not referenced - justifies placing MCS..." counted as a note (CTC-01 has 0 notes) |
| FTC-11 | #5 `↗ 1.2Ⓕ A B` | arrow symbol read as **circular runout** (one arrow; total runout has two) |
| FTC-11 | #3 `○ 1.2Ⓕ` | **circularity**, free state, no datums |
| FTC-11 | #7 `(197.9)` | reference dimension (circumference?) kept as required |
| FTC-11 | #8 `B` (sheet 1) | black view-direction label, marked optional (neither required nor counted as extra) |
| FTC-11 | #9 `2X R0.1 MAX` | stored as nominal 0.1 (a MAX limit; scorer has no upper-only check) |
| both | `bbox` | approximate (from the 150-dpi render), only for this overlay - not scored |

When done: fix the JSON, re-run `python -m eval.review_drawing_gt nist_ctc_03 nist_ftc_11` to re-draw, then set
`"verified": true` in both files. Until then, report test-split drawing numbers as "unverified labels".
