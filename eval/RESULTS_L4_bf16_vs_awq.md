# Eval results: Qwen2.5-VL-7B, BF16 vs AWQ 4-bit (Colab L4, vLLM)

## 1. Current numbers

Final setup (report-qa-v6, 16k context, a pinned report index per model), every suite on all cases with an empty
model cache. Router and drawing are from 2026-10-19: nothing changed since then reaches them.

| suite | metric | BF16 | AWQ | p50 latency BF16 / AWQ |
|---|---|---|---|---|
| router | accuracy (80 cases) | 1.000 | 1.000 | 3.3 s / 1.3 s |
| text questions | pass (35) | **0.829** | 0.771 | 10.7 s / 4.6 s |
| picture questions | pass (30) | 0.767 | 0.767 | 26.4 s / 9.7 s |
| └ picture only | pass (17) | **0.706** | 0.647 | |
| └ picture + question | pass (13) | 0.846 | **0.923** | |
| drawings, PDF | recall / precision, 2 checked sheets | **0.928 / 0.883** | 0.843 / 0.766 | 218 s / 92 s per drawing |
| drawings, PDF | recall / precision, 2 unchecked sheets | 0.833 / 0.878 | 0.810 / 0.875 | |
| drawings, picture | recall / precision, 2 checked sheets | **0.643 / 0.793** | 0.557 / 0.774 | 55 s / 27 s per drawing |
| end to end | success (21) | 0.714 | **0.762** | 29.4 s / 13.4 s |

- **AWQ is 2.2–2.7× faster** everywhere.
- **BF16 is better where exact values matter.** In text questions AWQ drops exact tokens (a registration, a bolt
  count). On drawings it finds and filters fewer values.
- **Pictures and end to end are a tie** within a case or two: BF16 is better with a picture alone, AWQ with a
  picture and a question.
- Each model is measured on its own index (figure descriptions written by that model), as a single-model
  deployment would run. This doesn't separate answering quality from description quality.

## 2. How the data was made and scored

| suite | cases | how they were made |
|---|---|---|
| router | 80 | generated from the 14 source PDFs: drawing sheets as PDFs and as degraded pictures (blur, rotation, perspective, "photo of a printout"), report pages, figure crops, charts and text questions. The expected route follows from how each case was made |
| text questions | 35 | 20 written early and used for tuning (dev); 15 drafted later by the AI assistant (test). Every expected fact is checked against the PDF text |
| picture questions | 30 | the reports' own figures, cropped without their caption and lightly degraded, with and without a question |
| drawings | 11 drawings | 4 sheets have labels drafted by the AI assistant from the sheets (2 checked, 2 not). All 11 are also compared with an earlier full run (fix4), which shows how much a version changes, not whether it is right |
| end to end | 21 | realistic requests (text, photos, drawings, mixed), scored with the other suites' labels |

- **Scoring is rule-based**, with no AI judge. An answer passes when it contains the expected facts, none of the
  forbidden ones, and cites the right page (text) or report (pictures). A drawing value counts when it matches a
  label. End to end also needs correct routing and no errors.
- **Dev vs test:** dev cases were used while tuning, so they score higher. Test cases are held out.
- **Caveats:** all labels are AI-drafted and mostly unreviewed, there are no real phone photos, the picture cases
  come from the reports themselves, and one case is worth 3–5 points in the QA and end-to-end suites.

Raw results are in `eval/results/` (one JSON per run). Every metric for both models is in
`l4_bf16_vs_awq_comparison.csv` (2026-10-19) and `l4_bf16_vs_awq_v6_comparison.csv` (now).

## 3. How the numbers evolved

| | BF16, 2026-10-19 → now | AWQ, 2026-10-19 → now |
|---|---|---|
| text questions | 0.857 → 0.829 | 0.771 → 0.771 |
| picture questions | 0.633 → 0.767 | 0.600 → 0.767 |
| end to end | 0.667 → 0.714 | 0.667 → 0.762 |
| end-to-end errors | 1 → 0 | 1 → 0 |
| slowest picture question | 479 s → 56 s | 158 s → 18 s |
| slowest text question | 19 s → 17 s | 7 s → 24 s (a loop that reached 279 s in between, now capped) |

**What changed and why:**
- **Too many pictures crashed a request.** A report PDF becomes 3 page images; with 2 report figures that is over
  the server's limit of 4. The answer step now keeps to the limit, and a failed answer no longer takes the rest of
  the request with it. That case now passes on both models.
- **The model sometimes repeated itself until it ran out of room**, taking 2–8 minutes for one case. Both model
  calls in a report answer now have a length limit, and the context is 16k.
- **The report index wasn't reproducible.** Figure descriptions are written by the model when the reports are
  indexed, and every new Colab session wrote new ones, so each built a slightly different index. The descriptions
  are now stored with the code, so every session builds the same index, in 32 s instead of ~10 minutes.

**What the re-runs showed:**
- On the same index, a full re-run in a new session retrieved exactly the same sources and flipped **no** cases.
  The fixes cost no accuracy, and the crash and the stalls are gone.
- The picture-question gains, and the one text case lost, come from the new index, not from the fixes. Rebuilding
  the index moved picture results by several cases: picture search depends heavily on the AI-written descriptions.

## 4. Open issues

1. **Picture-only questions:** the model often mixes up fatigue and overstress, because the paragraph stating the
   report's conclusion isn't retrieved with the figure. Fix: bring in the paragraphs that cite the figure.
2. **Refusals:** 1 of 4 questions the reports can't answer gets an answer instead of a refusal, on both models.
3. **Picking the right number** when a page holds several similar values (two text cases).
4. **Scoring:** word variants fail ("oxidation" vs "oxidized"), negation isn't detected, and citing many sources is
   rewarded.
5. **Labels:** AI-drafted and mostly unreviewed; no real phone photos.
6. **BF16 loops on four specific pictures** every run. The limit holds each to about 50 s, but the loop is still
   there.
7. **Index pinning is a procedure, not enforced:** runs don't yet record which index they used.
8. **A shared-index run** of both models is still needed, to split AWQ's gap into answering vs describing.
