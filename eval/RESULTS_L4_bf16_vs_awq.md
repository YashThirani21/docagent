# Eval results: Qwen2.5-VL-7B BF16 vs AWQ 4-bit on an L4 (2026-10-19)

Covers the twelve results files in `eval/results/` dated 2026-10-19, six per model:

| suite | BF16 (`configs/colab_vllm.yaml`, label `baseline-l4`) | AWQ (`configs/colab_vllm_awq.yaml`, label `awq-l4`) |
|---|---|---|
| router | `20261019-104259_router_baseline-l4.json` | `20261019-132044_router_awq-l4.json` |
| qa_text | `20261019-104621_qa_text_baseline-l4.json` | `20261019-132202_qa_text_awq-l4.json` |
| qa_image | `20261019-105306_qa_image_baseline-l4.json` | `20261019-132509_qa_image_awq-l4.json` |
| drawing, PDF route (11 sheets) | `20261019-120306_drawing_baseline-l4.json` | `20261019-133240_drawing_awq-l4.json` |
| drawing, picture route (`--as-images`, 2 sheets) | `20261019-113530_drawing_baseline-l4.json` | `20261019-134729_drawing_awq-l4.json` |
| e2e | `20261019-113957_e2e_baseline-l4.json` | `20261019-134926_e2e_awq-l4.json` |

Each pair runs on the same hardware (Colab L4, vLLM with `--max-model-len 8192` from the notebook), with `--split all` and `--fresh` (empty VLM cache, so latency
and tokens are real), and on the same dataset hash. Only the `llm` section of the config differs. The AWQ config
reuses the report index that BF16 built, including the figure descriptions, so on the QA side only the
answering and observing model changes.

---

## 1. The datasets

### Dev vs test

- **dev**: cases the agent was tuned against: prompts were written while looking at them and fixes were checked on
  them. Their scores are optimistic.
- **test**: cases written afterwards and not consulted while the agent changed. These are the honest numbers.

The harness defaults to `--split dev`, so a tuning loop can't touch `test` by accident. These runs used `--split
all`, so every row carries its split and both numbers come from the same run (section 3).

### Per suite: what is in it and how it was made

| suite | n (dev / test) | how it was made | split rule | labels |
|---|---|---|---|---|
| **router** | 80 (17 / 63) | `eval/build_router_set.py` builds every case from the source files: the 11 drawing PDFs, single sheets as 1-page PDFs, sheets rendered as pictures (rotation, blur, JPEG artefacts, perspective, crop, "photo of a printout"), the 3 report PDFs, report text and figure pages as PDFs and pictures, figure crops **without caption** (stand-ins for part photos), load charts, and 6 text-only questions | the 14 original PDFs (the routing rules were written against them) and 3 text questions are dev. Everything generated is test | each label follows from how the case was made, so nothing is hand-labelled and there is no label noise. All synthetic, and `real/` (phone photos) is empty |
| **qa_text** | 35 (20 / 15) | q01–q20 were hand-written early and used for the section 6 fix rounds. t01–t15 were drafted later by the AI assistant from the report text | q* = dev, t* = test | `must` / `must_not` facts plus the gold page(s). Every fact is checked programmatically against the PDF text. **The test split has not been reviewed by a human**, so a wrong expectation counts against the agent |
| **qa_image** | 30 (15 / 15) | `eval/build_qa_image_set.py`: 20 report figures cropped without caption and lightly augmented (17 picture-only, 3 with a question), plus 10 picture + question cases (3 diagnosis, 3 lookup, 2 refusal, 2 wrong_aircraft) | alternating dev / test in build order | expected report, figure and page, plus `must` / `must_not`. Agent-drafted, checked against the text, not human-reviewed. **In-distribution**: the pictures are the corpus's own figures, and printed labels such as "Corrosion" can stay visible. Some facts are single vague words ("heat", "grain") |
| **drawing** | 11 NIST PMI drawings | 4 sheets have labels drafted by the AI assistant, not taken from extractor output. CTC-01 and FTC-07 (**dev**, verified) were used to build the geometry method. CTC-03 and FTC-11 (**test**) were drafted from the rendered sheets, not from extractor output, and are still `verified: false` | the split lives in the GT file. Unlabelled drawings count as dev | two scores. **Accuracy** against these labels: the headline uses only the 2 verified dev sheets, and the 2 test sheets are reported separately as `draft_labels`. **Agreement** with the reference run `geo-l4-fix4` (an earlier BF16 run) on all 11: this measures *change*, not correctness |
| **e2e** | 21 (9 / 12) | hand-composed realistic requests: text only, photo only, photo + question, drawing PDF or picture, several drawings, text + drawing, photo + drawing, report PDF + question | per case | the expected outputs reuse the other suites' labels and scorers. A case passes when routing is right, every component passes (drawing recall and precision ≥ 0.7, or agreement ≥ 0.7 when there are no labels) and nothing errors |

### The drawing comparison basis is fix4, a model run, not manual annotation

The baseline that every drawing run is compared against is **`geo-l4-fix4`**, the 2026-10-18 L4 run (Qwen2.5-VL-7B
BF16 on vLLM, geometry method). **Its JSONs are the extractor's own outputs, mistakes included. Nobody annotated
them by hand.**

- **Reference outputs:** `eval/datasets/drawings/reference/geo-l4-fix4/`, with all 11 drawings on the PDF route and
  CTC-01 and FTC-07 in `jpg/` for the picture route. `PROVENANCE.md` there records the source run.
- **Baseline score files** for `--compare`, in `eval/results/`:
  - `20261020-032010_drawing_geo-l4-fix4-pdf.json` (PDF route) and `20261020-032011_drawing_geo-l4-fix4-jpg.json`
    (picture route): **use these.** They are the fix4 JSONs rescored with today's harness (`--pred-dir`, no model
    calls), in the current results format, and they give per-value flips. They reproduce fix4's original scores
    exactly: PDF 0.900 / 0.892 and picture 0.643 / 0.807 recall / precision. Their timestamp and `env` are those of
    the rescoring on the Mac, not of the L4 run.
  - `20261018-122437_drawings_geo-l4-fix4-pdf.json` and `-jpg.json`: the original score files from the fix4 run, kept
    for provenance. They predate the current format, so `--compare` against them shows metric deltas only, with no
    per-value flips.
- **Agreement** (`agree_*`, dropped / new / changed vs reference) measures how much a run *differs from fix4*. It
  does not measure correctness. A run that fixes one of fix4's mistakes scores lower agreement. Correctness is only
  measured by **accuracy** against the label files in `eval/datasets/drawings/*.json`, and those labels have their own
  provenance caveats: all four were drafted by the AI assistant before the geometry method was built. FTC-07 was
  checked against its text layer; CTC-01 is marked verified, but no separate check of it is recorded. CTC-03 and
  FTC-11 are unverified.

```bash
python -m eval drawing --config <new>.yaml --label <new> \
    --compare eval/results/20261020-032010_drawing_geo-l4-fix4-pdf.json
python -m eval drawing --config <new>.yaml --label <new>-jpg --as-images \
    --ids nist_ctc_01_asme1_rd nist_ftc_07_asme1_rd \
    --compare eval/results/20261020-032011_drawing_geo-l4-fix4-jpg.json
```

The **part suites isolate one component with gold inputs**: the drawing suite calls the extractor directly, the QA
suites call `ReportQA.run`, and the router suite only routes. A change in one part can't move another part's
numbers. **e2e** is the only suite where errors compound.

### How to read the sizes

One case is worth 2.9 points in qa_text, 3.3 in qa_image and 4.8 in e2e. Most deltas below are 1–3 cases, so
the **per-case flips** matter more than the averages. The drawing suite scores ~70 labelled values and 515
reference values, so it is the only suite where a few points are statistically meaningful.

---

## 2. Speed: AWQ is 2–3× faster across the board

| suite | p50 latency BF16 → AWQ | p95 latency BF16 → AWQ | wall-clock BF16 → AWQ |
|---|---|---|---|
| router | 3.32 s → 1.27 s (VLM tier 3.9 s → 1.5 s; rule tier 0.05 s in both) | 5.2 s → 2.0 s | 3.3 min → 1.3 min |
| qa_text | 11.1 s → 4.7 s | 14.8 s → 6.2 s | 6.5 min → 2.8 min |
| qa_image | 25.6 s → 9.6 s | **473 s** → 14.7 s | 42 min → 7.2 min |
| drawing (PDF, per drawing) | 218 s → 92 s | 316 s → 196 s | 33.7 min → 14.8 min |
| drawing (picture, per drawing) | 55 s → 27 s | 76 s → 29 s | 4.4 min → 1.9 min |
| e2e | 29.4 s → 14.3 s | 218 s → 92 s | 23 min → 10.3 min |
| **all six** | | | **113 min → 38 min (≈ 2.9×)** |

Prompt tokens are almost the same in both runs (the same pipeline sends the same context), so the speed-up is
per-token decode and prefill speed, not less work. AWQ actually made *more* calls on drawings (640 vs 562
fresh calls, 408k vs 359k prompt tokens) and was still 2.2× faster.

**The BF16 qa_image p95 is an anomaly.** Four cases took ~440–480 s each (`img-16-006-f7`, `img-18-061-f2`,
`img-18-061-f13`, `iq02-18-061-f2`) against ~26 s for the rest. BF16 also produced 42.6k completion tokens on this
suite against AWQ's 17.8k, while the final answers are short (~300–500 chars). Together that suggests a runaway
or repeating generation in an intermediate call (image observation?) that runs until the token cap. Without those
four, BF16 qa_image is still ~2.7× slower than AWQ.

**Confirmed later from the traces** (`store/traces.db`, AWQ request `a25e8710df02`, case `img-18-061-f19`, 158 s):
the image observation call has no `max_tokens`, looped, and stopped only when the context was full: 608 prompt +
7,584 completion = 8,192 tokens, the notebook's `--max-model-len` for these runs (not the 16k in the configs). It
was cut off (`finish=length`), and the retry with a "compact JSON" prompt answered in 94 tokens and 1.8 s. BF16 fits
the same pattern: 64 calls for 30 requests (60 + one retry for each slow case) and ~32k of its 42.6k completion
tokens in those four cases. Normal observations are 102 tokens (median), 347 at most. **Fix (report-qa-v5):** the
observation call is capped at 512 tokens, so a loop ends after ~512 tokens and the same retry takes over. Normal
observations, and therefore the answers built from them, are unchanged. The answer call was left uncapped at first;
on 2026-10-21 it looped once (AWQ qa_text q04, 279 s, to the 16k context), so report-qa-v6 caps it at 1,024 tokens
(answers run to ~413; prompts to ~7.1k, which fits a 16k context with the cap, not an 8k one). The notebook now starts vLLM with
`--max-model-len 16384` as in the configs; runs after this change differ from these in both, so label them.

---

## 3. Quality: headline numbers and per-split

| suite (headline metric) | BF16 all | AWQ all | BF16 dev / test | AWQ dev / test | flips (BF16 → AWQ) |
|---|---|---|---|---|---|
| router (accuracy) | **1.000** | **1.000** | 1.00 / 1.00 | 1.00 / 1.00 | none |
| qa_text (pass) | **0.857** | 0.771 | 0.90 / 0.80 | 0.80 / 0.73 | 3 lost, 0 gained |
| qa_image (pass) | **0.633** | 0.600 | 0.73 / 0.53 | 0.67 / 0.53 | 3 lost, 2 gained |
| drawing PDF, verified dev sheets (recall / precision) | **0.928 / 0.883** | 0.843 / 0.766 | | | 70 labelled values |
| drawing PDF, draft test sheets (recall / precision) | 0.833 / 0.878 | 0.810 / 0.875 | | | 42 labelled values |
| drawing picture route, verified sheets (recall / precision) | 0.643 / 0.793 | 0.557 / 0.774 | | | |
| e2e (success) | 0.667 | 0.667 | 0.89 / 0.50 | 0.78 / 0.58 | 1 lost, 1 gained |

### What the numbers say about the trade-off

1. **The cost of AWQ falls on the parts that depend on fine detail.** Routing loses nothing. Answers that need an
   exact number or registration lose a few cases (qa_text −8.6 pts, all three flips going the same way). Drawing
   reading on the dev sheets loses ~8.5 pts recall and ~12 pts precision, which across 70 values is beyond run-to-run
   noise.
2. **On held-out data the gap mostly closes.** On the two test drawings AWQ is within 2 pts of BF16 (0.81 vs 0.83
   recall, 0.875 vs 0.878 precision). On qa_image test both models score 0.53. A plausible reason is that the
   drawing method, and the QA fixes, were **tuned on BF16 outputs from the dev cases** (NOTES section 8: "tune and
   judge on the deployment model"). Part of BF16's dev lead is therefore fit to BF16, not general quality. The test
   sets are small and unverified, so this is a hypothesis, not a conclusion.
3. **End to end, the two models are tied** (14/21 each). AWQ lost one case and gained another, and the shared failures
   are identical. The losses that matter at e2e come from weaknesses both models share (section 4), not from
   quantisation.
4. **Noise floor.** On the same two verified sheets, this BF16 run scored 0.928 / 0.883. The earlier BF16 reference
   run fix4 scored 0.900 / 0.892, and this run agrees with fix4 at only 0.98. Temperature-0 vLLM is not bit-for-bit
   reproducible (batching), so differences of ~±3 pts on drawings and ±1 case on QA can come from a re-run alone.
5. **Bottom line.** AWQ costs ~3× less GPU time, with a measurable but modest quality loss that is concentrated in
   text-QA fact precision and drawing extraction on the tuned sheets. If AWQ becomes the deployment model, re-tune
   prompts and the geometry assembly on AWQ outputs and re-measure. The test-split numbers suggest much of the gap
   would not survive that.

---

## 4. Each component

### Router: solved on this data

- 80/80 for both models. No drawing went to QA and no report went to the drawing pipeline. 100% on every way a
  case was made, including blurred, rotated, perspective-warped and "photo" renders.
- 40% of cases (32/80) are decided by the rules in ~0.05 s. The VLM tier decides the other 48 at 3.9 s (BF16) or 1.5 s
  (AWQ), with no accuracy difference.
- **Use AWQ here; it costs nothing.** The caveat is that every case is synthetic and the `real/` phone-photo folder
  is empty. An earlier Mac/Ollama 4-bit smoke run did misroute one photo-look figure page, so real photos are the
  next thing to add.

### Text QA: strong retrieval, weak on exact facts and refusal

- BF16 0.857, AWQ 0.771. Retrieval is not the problem: BF16 cites the right page every time (`pages` 1.0,
  `page_recall` 1.0), and AWQ is close (0.969 / 0.938). No filler: neither model adds diagnosis or actions to a
  plain lookup.
- **AWQ's three losses are all lookups where it dropped a specific token** that BF16 kept: the registration N545PZ
  (q02), the bolt count "10" (q05), and a wrong page cited for t11. AWQ also cites fewer sources (1.4 vs 2.0 per
  answer).
- **Five shared failures**, which neither model gets right:
  - q16: the known diagnosis gap ("bent", "roll groove"), accepted in NOTES section 8.
  - t08, t03: **picked the wrong number from a page full of numbers.** t08 expects crack sizes 0.047 / 0.038 in;
    the model answered 0.026 in, another measurement in the same report. t03 expects 25 lb; the model said 23 lb,
    then 43 lb, both real values on that page.
  - q18: **the scorer is too strict.** The answer says "oxidation", but the check looks for the exact word
    "oxidized", so a correct answer fails.
  - t14 (pilot flight hours): **should refuse, doesn't.** It mixes up similar facts and gives the *aircraft's*
    7,690.6 hours as the pilot's. Refusal is 0.75 for both models (1 of 4 fails), and e10 in e2e fails the same way.
- **Takeaway: retrieval is not the bottleneck.** In every failing case the right page was cited, so the fact was in
  the model's context. The failures are about precision when several similar facts sit close together, not about
  missing information, and at least one "failure" (q18) is a word-matching problem in the scorer. Text-QA
  pass rates therefore slightly understate quality.
- Dev 0.90 → test 0.80 for BF16: a 10-pt generalisation gap, as expected from tuning on dev.

### Image QA: the weakest component, and picture-only is the problem

| | BF16 | AWQ |
|---|---|---|
| picture only (n = 17) | 0.529 | **0.353** |
| picture + question (n = 13) | 0.769 | **0.923** |
| `figure` kind (n = 20) | 0.50 | 0.40 |
| diagnosis / lookup / refusal / wrong_aircraft (n = 10) | 9 / 10 | 10 / 10 |
| doc_hit (right report cited) | 1.000 | 0.893 |
| wrong-report citations per answer | 0.11 | 0.50 |
| forbidden claims made | 4 | 6 |

- **When the user asks a question, both models do well**, and AWQ does slightly better: 12/13 against 10/13. Refusal and
  wrong-aircraft traps pass 4/4 in both.
- **When the user sends only a picture, quality drops sharply.** The model has to recognise the part, find the right
  report and give its conclusion. The failure pattern is the same in both models: they confuse **fatigue with
  overstress** (24-013 is overstress, 18-061 is fatigue) and **hedge with lists** ("fatigue, corrosion, and potential
  abrasive wear"). A hedge then trips the `must_not: corrosion` check.
- **Takeaway: a high doc_hit does not mean retrieval is strong here; it only finds the right report.** In BF16,
  retrieval finds the right report every time (doc_hit 1.0). But it pulls the actual figure into context only 71% of
  the time (`fig_in_context` 0.714) and cites the figure's page only 36% of the time (`page_cited` 0.357; AWQ:
  0.75 / 0.25). The report's conclusion ("overstress" or "fatigue") is often on a different page from the figure.
  Without that page, the model falls back on its own reading of the picture, and that is where the
  fatigue/overstress mix-ups and hedged answers come from. When a question is attached, it gives retrieval text to
  match on, which explains why picture + question scores much higher.
- **AWQ's specific regression is retrieval from the picture.** It cited the wrong report for all three 24-013 figures
  (doc_hit 0.893, wrong-report citations 0.11 → 0.50), and it says "corrosion" more often. The index is the same, so
  this comes from AWQ's image observations steering the query to the wrong report.
- Caveats: the cases are in-distribution (the corpus's own figures), so real photos of a new part will score lower.
  Some `must` facts are single vague words ("heat", "grain"). Fix the BF16 runaway generation (section 2) before the
  next timing run.

### Drawing extraction: the PDF route is good, the picture route is not

**PDF route (geometry method), verified dev sheets, 70 required values:**

| field | BF16 | AWQ |
|---|---|---|
| recall / precision | **0.928 / 0.883** | 0.843 / 0.766 |
| numbers | 0.924 | 0.914 |
| type | 0.927 | 0.920 |
| GD&T symbol | 0.586 | **0.652** |
| datums | 0.897 | 0.877 |
| belongs_to (lenient) / belongs_primary | 0.675 / 0.383 | **0.845 / 0.507** |
| title block / notes / units | 1.0 / 1.0 / 1.0 | 1.0 / 1.0 / 1.0 |

- When a value is found, it is read correctly in both models: numbers and type are > 0.9 and nearly equal. **AWQ's
  loss is in finding and filtering**: it misses 6 more of the 70 values and adds more extras (81 predictions against
  77). Precision drops the most.
- AWQ is *better* at GD&T symbols and at matching a value to its feature. Those are the weakest fields for both
  models: GD&T symbol ~0.6, and the strict `belongs_primary` 0.38–0.51.
- On the 2 test sheets with draft labels the models are nearly equal (see section 3).
- **Agreement with fix4** across all 11 drawings: BF16 0.981 / 0.982, AWQ 0.872 / 0.902 (66 values dropped, 49 new).
  fix4 is itself a BF16 run, so this metric favours BF16 by construction. Read it as "AWQ's outputs differ from
  fix4 on ~13% of values", not as "AWQ is 13% worse". Each of the 66 dropped / 49 new values may be a regression,
  an improvement where fix4 was wrong, or an equally valid reading. On the 9 drawings without labels there is no way
  to tell which. On the labelled sheets, accuracy does tell: AWQ is lower on the 2 dev sheets and within ~2 pts on the
  2 test sheets (above).

**Picture route** (each sheet rendered as a 200-dpi JPG): recall 0.643 (BF16) and 0.557 (AWQ), against 0.93 / 0.84
on the PDF. Without the vector paths and the text layer, the model has to read everything from pixels. The **title
block** breaks under AWQ (0.667 → 0.236). This route is what photos of printed drawings take, and at e2e it is the
worst case (e16 below).

### End to end: routing is perfect; losses come from components and one crash

- Success 0.667 for both models. Route accuracy is 1.0, component pass given correct routing is 0.70, and
  route × component = 0.70. **Every e2e loss comes from the components, none from routing**, plus one crash (e18,
  `error_rate` 0.048), which is why success is 0.667 rather than 0.70.
- Dev 0.89 / test 0.50 (BF16): the largest dev–test gap of any suite.
- The flips are e06 (drawing as a picture), lost by AWQ because CTC-01 picture recall 0.588 falls below the 0.7
  bar, and e12 (photo + question), gained by AWQ.
- **Six shared failures**, which map to the weaknesses above:
  - e03, e11: **photo only**. This is the image-QA picture-only weakness.
  - e10: **text that must be refused**. The model answers instead, as in qa_text t14.
  - e16: **photo of a printed drawing sheet**. Agreement with the reference is 0.10, so the picture route has almost
    failed.
  - e17: **photo + drawing in one message**. The drawing part passes (~0.9 agreement) but the photo answer fails.
  - e18: **crash in both models**. vLLM returns `400: At most 4 image(s) may be provided in one prompt`. This is a
    deterministic bug: the report PDF is rendered as 3 page images and retrieval adds 2 figure crops, so the call
    carries 5 images against the server's limit of 4. The exception also escaped `Agent.handle`, so the request
    raised instead of returning its error. **Fixed after these runs (report-qa-v4):** the answer call now keeps to
    an image budget (`llm.max_images`, the same number `start_vllm.sh` passes to vLLM), and a failed answer is
    recorded in `resp.errors` like any failed attachment. Every request that stayed within 4 images sends exactly
    the same prompt as before, so only e18 needs a re-run; the other 20 cases cannot change.
- e08 passes but records that a question sent with a drawing goes unanswered (`unanswered_text_with_drawing` = 1).
  That is the current design, not a scored failure.

---

## 5. Next investigations and steps with more time

1. **Re-run e18** on both models (`--ids e18`) now that the image budget is in (report-qa-v4); success is 14/21
   plus e18's result. A further step: send a born-digital report PDF as its text, not as page images.
2. **Re-run qa_image and e2e on both models** with report-qa-v5 (observation cap) and `--max-model-len 16384`.
   Expected: pass rates within noise, BF16 qa_image p95 from 473 s to ~30 s. qa_text, router and drawing make no
   observation calls and need no re-run.
3. **Picture-only image QA**: after finding the figure, also retrieve the page that states the report's conclusion.
   Then tell the model to use that conclusion rather than its own reading of the picture, and to commit to one
   failure mode (no hedged lists). It is the largest source of e2e losses (e03, e11, e17).
   **Text QA precision**: make the answer quote the exact sentence it got each number from. Also make the scorer
   accept word variants (oxidized / oxidation) so correct answers stop counting as failures.
4. **Refusal on out-of-corpus facts** (t14, e10): with 1 of 4 refusals failing, this is a safety issue for a
   maintenance assistant.
5. **Human-verify the test labels** (qa_text t01–t15, qa_image, drawing CTC-03 / FTC-11), then freeze them. Most of
   the AWQ-vs-BF16 conclusions rest on these unverified test splits.
6. **Measure noise**: re-run BF16 twice with `--fresh` on the same splits to get a real noise band before reading
   1–3 case deltas.
7. **If AWQ is the deployment target**: re-tune drawing assembly and QA prompts on AWQ outputs. Add real phone photos
   to `router/real/` and `qa_image/out_of_scope/`.
