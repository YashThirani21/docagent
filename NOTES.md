## What I'm building

One chat agent that works out which of two jobs a request needs:

- **Report QA**: a text question or a photo of a damaged part: what is wrong and what to do,
  answered from the 3 pdf reports with citations.
- **Drawing extraction**: a technical drawing: a JSON entry for every value on it, with what it belongs to and what it means.

Around it: automatic routing, bulk upload, fully offline running, a design that works at scale, a trace of every request, and an eval harness that compares versions.

## Starting plan

1. **Get basic routing, retrieval and generation working** Bulk, UI, Harness can be handled later
2. **Contracts and logging first.** Everything later plugs into them, adding them at the end means rewriting
3. **I think I will need Colab here** since qwen2.5vl:7b is too clunky to work with on Ollama on local - better to use VLLM and GPU on Colab
4. **We need 2 independent systems** one focused on QA, one on JSON from drawings, one called based on input routing

## Data inspection (before any design)

Profiled every page of the 14 PDFs with PyMuPDF (text chars, raster images, vector paths, page size).

**Reports (3 NTSB Materials Laboratory Factual Reports, 14/40/6 pages)**
- Born-digital (Word -> PDF): full text layer on every page. **No OCR needed for the corpus.**
- Figure pages hold one or two figures, each captioned `Figure N. ...` below the image.
- A figure is not one embedded image: it is many small raster tiles plus vector arrows and text
  labels (e.g. 18-061 p.13 has 19 image placements). Extracting embedded images individually is useless.
  -> crop the page region above each caption and render it, as a reader sees it.

**Drawings (11 NIST PMI test cases, 1-5 sheets each, 30 sheets total, all landscape 792x612 pt)**
- 10 of 11 draw dimensions and GD&T as vector outlines, the text layer
  only has the title block and notes (CTC-01: 153 chars on a sheet full of annotations).
- Only FTC-07 (AutoCAD export) has the annotations as real text.
- The VLM must read annotations from pixels; the text layer is used as exact truth where it exists.
- Annotations are blue; part geometry is black/purple.


## Model and infrastructure choices

- **VLM: qwen2.5vl:7b via Ollama** (runs on the 16 GB M4, ~6 GB). One model for figure description,
  photo observation, drawing reading, routing fallback and answer generation. Called through the
  OpenAI-compatible API so vLLM on a GPU is a `base_url` change.
- **Embeddings: bge-m3** - dense + sparse (lexical) from one model, so hybrid search without a separate BM25 index.
- **Vector DB: Qdrant**, embedded mode for now (no Docker), same API as a server. Named `dense` + `sparse`
  vectors, server-side RRF fusion. Collection name carries the embedder (`reports_bge-m3_v1`).
- **No OCR engine in v1**: text layer covers the corpus; VLM reads drawings and photos.

### Phase 1: Contracts and skeleton
- `schemas.py`: Pydantic models for Attachment, Request, Route, RouteDecision, ReportAnswer, DrawingExtraction, Response etc. These are the agreement between every part.
- `config.py`: Skeleton Pydantic models for config of components (LLM, Embedding, Retrieval, 
Router, Trace etc)
- `llm.py`: Utility functions available to a VLM building around configs
- `pdf.py`: Skeleton for pdfs
- `embeddings.py`: Interface to work with any embedder - serving default model from config
- `store.py`: Interface to work with any Vector Store - serving default Qdrant
- `ingest/reports.py`: Ingesting pdfs from data/ with pymupdf, chunking and populating Qdrant


## Takeaways from early expts (around 45 mins - 1 hr):

- **Batched bge-m3 on Apple MPS returns corrupted vectors.** After the first ingestion only 19 of 94
  stored dense vectors were unique; a batch of 16 chunks produced 2 distinct vectors; dense scores were
  identical for unrelated chunks, and a report vanished from results. Short texts were fine,
  long padded batches were not. CPU: all 94 unique, retrieval correct. -> `device: cpu` + a self-check at
  load time (batched vs single cosine must be > 0.99, else refuse to start). CPU embedding of the whole
  corpus takes ~30 s, CUDA at scale.
- **VLM figure descriptions overrode the reports.** Q: "what caused the throttle arm to break?" The
  report says bending overstress, no pre-existing cracks; the answer said fatigue because the
  auto-generated figure description mentioned beach marks. -> descriptions are now used for
  **retrieval only** (`Chunk.description`, part of `embed_text`); the generator sees caption, printed
  labels and the report sentences citing the figure. Prompt: report conclusions are authoritative.
  After the fix the answer is correct.
- **Failure-mode guess on user photos biased retrieval.** Earlier the observation step asked for a "likely failure
  mode" and its guess ("fatigue") steered both search and answer even when irrelevant. Removed it and brought it only when explicit text query nudges it there

## Routing rules (summary)

Source: `src/docagent/router.py` (version `rules+vlm-v1`), called per attachment from `Agent.handle`.
Principle: cheap deterministic rules first, the VLM only when the rules can't decide. Every decision
records `route`, `tier` (rule | vlm), `confidence` and a `reason` in the trace.

**Two possible routes:** `report_qa` (answer from the NTSB reports) or `drawing` (extract values to JSON).

**1. Request level**
- No attachment (text only) -> `report_qa`, confidence 1.0.
- Otherwise each attachment is routed on its own (one task each, so one bad file fails only itself).
- Same file bytes + same router/drawing/model versions + same config -> the earlier result is reused,
  no routing (idempotency key).
- Chat only: a zip, or >= 2 files with no question, becomes a background batch (`batch.chat_min_files`);
  each file is then still routed individually.

**2. PDFs: rules on page 1** (`pdf.page_stats`)
- Drawing score, 0-4, one point each:
  - >= 2 distinct drawing keywords in the text (ASME Y14, UNLESS OTHERWISE SPECIFIED, TOLERANC, UNIT(S):,
    SCALE, SHEET, DWG, REV X, Ø, ±, PMI, DATUM)
  - landscape page
  - > 300 vector paths
  - < 10% of the page covered by raster images
- Prose = > 1500 characters of text on a portrait page.
- score >= 3 and not prose -> `drawing`, confidence 0.6 + 0.1 x score.
- prose, or (score <= 1 and > 30% image coverage) -> `report_qa`, confidence 0.8.
- Anything else -> undecided -> VLM on page 1 rendered at 80 dpi.

**3. Images (PNG/JPG, ...)** -> always the VLM (no rules for pixels).

**4. VLM classifier** (image downscaled to `router.classify_max_side` = 768 px)
- Classes: technical_drawing, photo_of_part, document_page, chart_or_plot, other.
- `technical_drawing` (including a photo or screenshot of one) -> `drawing`; every other class -> `report_qa`.
- Confidence and reason come from the model.

**5. Other file types** -> `report_qa`, confidence 0.3, treated as context for a question.

**After routing**
- Each `drawing` attachment -> its own JSON file in `outputs/requests/<request_id>/`.
- All `report_qa` attachments are pooled (up to 3 pages rendered per PDF) with the text into one answer.

**Known limits / measurements**
- Only page 1 is inspected: report 24-013 has a near-empty cover, so it falls through to the VLM tier.
- Router eval (Mac, Ollama 4-bit, test n=15): accuracy 0.93, 0 drawings sent to QA, 1 report figure page
  (looks like a photo) sent to the drawing pipeline; rules decided 2/15, VLM 13/15 at ~11 s each.

## Chunking and retrieval of the reports (summary)

Source: `src/docagent/ingest/reports.py` (ingestion), `retriever.py` (search), `pipelines/report_qa.py`
(how chunks reach the prompt). Run once: `docagent ingest data/reports`.

### Ingestion overview
- Per PDF: sha256 -> skip if already in the index (`has_document`)
- Document metadata from the first 3 pages by regex: report number, vehicle (-> registration), location, date.
  Becomes one **header line** prepended to every chunk's `embed_text`, e.g.
  `NTSB Materials Laboratory Factual Report 18-061 | <vehicle> | <place> | <date>`.
- Every chunk has two texts: `text` (shown to the model / user, the report's own words) and `embed_text`
  (header + section + subject + text [+ description], what gets embedded). Both chunk kinds go into the same
  Qdrant collection with the same bge-m3 dense + sparse vectors.

### Text chunks
1. **Blocks**: PyMuPDF text blocks per page, whitespace normalised, sorted top to bottom. Page headers/footers
   dropped ("Page No. X", "Pg X of Y", "NATIONAL TRANSPORTATION SAFETY BOARD").
2. **Structure tracking** (carried across pages):
   - Section: lettered headings `A. ...` -> `section` ("A. Background").
   - Subject: numbered subsections `1. Accident Airplane Main Wing Spars` -> `subject`. If the heading says
     "accident" but names no aircraft, the report's registration is appended. This is the fix
     for a case where answers mixed up N104ER / N106ER
3. **Filtering**: skip captions, skip short blocks (< 200 chars) on figure / chart pages (those are labels
   printed on the figure, not body text), skip blocks < 40 chars and "intentionally left blank".
4. **Runs**: consecutive blocks with the same (section, subject) form a run. A chunk never mixes two subjects.
5. **Size**: greedy merge of paragraphs within a run up to `embedding.max_chunk_tokens` = 350 bge-m3 tokens;
   a paragraph longer than that is split on sentence ends. No overlap between chunks - the
   "next paragraph" step at retrieval time covers findings that continue in the next chunk.
6. **Figure references**: `figure_refs` = figure numbers the text cites ("Figure 5", "Figures 3 and 4",
   "Figures 2 through 6").
7. `embed_text` = header + `Section: ...` + `About: <subject>` + text.

### Figure chunks
1. **Why not extract images**: a figure is many small raster tiles plus vector arrows and labels, so the
   embedded images are useless one by one. Instead the page region is **rendered as a reader sees it**.
2. **Region**: one per `Figure N.` caption, from the previous caption (or below the page header, y = 95 pt)
   down to this caption, full page width. Rendered at 150 dpi, saved to `store/images/<doc>/figN_pP.png`.
3. **Chart pages**: a page with no caption, > 40 vector paths and < 600 chars (18-061 pp.37-40, load-extension
   plots) becomes one whole-page "chart" chunk (`figure_no = None`).
4. **Labels**: short text blocks inside the region (text printed on the figure: arrows, "RC-1", ...).
5. **VLM description** (`fig-v1` prompt): Factual sentences on component, view and visible conditions.
   Cached by (image hash, model, prompt version); on re-ingestion descriptions already in the index are reused,
   so a chunking change costs no VLM calls.
6. **Linking back to the text**: body chunks citing figure N are attached to it:
   - their most common `subject` becomes the figure's subject (a figure is about what cites it);
   - up to 1500 chars of the citing text are appended ("Report text citing this figure: ...").
7. `text` = caption + labels (+ citing text). `description` is kept separately. `embed_text` = header + section +
   caption + labels + `Visual description: ...` + subject + citing text.
8. **Rule: the VLM description is for finding the figure only, never evidence.** It once said "fatigue" (beach
   marks) where the report concluded bending overstress, and the answer followed it. So it is embedded, but
   never shown to the answer model.

### Retrieval (`HybridRetriever.search`)
1. **Query**: user text + for each photo, the VLM observation (component + visible features, no failure-mode guess). If no picture then "What failed and why?"
2. **Two hybrid searches** in the one collection, each dense (top 30) + sparse (top 30) fused with RRF in Qdrant:
   one over all chunks, one filtered to `kind = text`. Pool = both, k x 2 each (top_k = 8), deduplicated,
   sorted by score. No separate re-ranker.
3. **Filter dropping irrelevant subjects**: if the question names a registration (N106ER), chunks whose `subject` names a
   different one are dropped.
4. **Report focus**: if a term in the question (place, registration, report no., type) matches the header of
   exactly one report (whole word), only that report is kept.
5. **Text first**: at least `min_text` = 4 text chunks; figures fill the rest but at most half of the k slots;
   text comes before figures. (Figure captions used to outrank the paragraphs that hold the findings.)
6. **Next paragraph**: each of the top 3 text hits pulls in the chunk after it (next on the page, else first
   on the next page), inserted right after it (`via = linked`).
7. **Linked figures**: figures cited by the top 3 text hits (up to 2 each) are added from the store
   (`via = linked`, score 0), same other-aircraft filter.

### Current limits
- Regexes for headers, captions, sections and subsections are tuned to the NTSB layout; a new document family
  needs its own parser (or a layout model).
- Tables are flattened into text blocks; table structure is lost.
- 350-token chunks with no overlap rely on the next-paragraph step for findings that span chunks.
- Report focus and the aircraft filter are string matches on headers/registrations; at 10^6 similar documents
  they become metadata filters (doc type, vehicle, part) in Qdrant rather than post-filtering.

### Generation (`ReportQA.generate`)
- Sources numbered `[n] report <doc>, page P, <text | Figure N>, section ..., about: <subject>` + evidence text.
- For figure chunks the evidence is **caption + labels only**: the citing paragraphs are cut off
  (`evidence_text`), since those paragraphs are separate sources and showing them twice steered answers and let
  a figure "back" a number that is really on a text page. The VLM description is not shown.
- Up to `max_context_figures` = 2 figure crops are sent as images (after the user's own pictures), each tied to
  its source number.
- After generation, `verify_citations` drops citations of other aircraft, adds the source that contains each
  distinctive number in the answer, and adds the paragraph the answer is nearly quoting.

## Drawing extraction iterations (around 2 hrs - lot of deduplication of extracted vals needed)

2 hrs of experiments before I decided manual tuning was too difficult for the constraints at hand and I have to let AI find a lot of things by itself rather than handholding here (possible overfitting too).

- 10 of the 11 PDFs draw their annotations as vector shapes, not text. The text layer only holds the title block and notes. So the dimensions and GD&T have to be read from the picture.
- FTC-07 (an AutoCAD export) is the exception: it has real text, which later made its labels checkable.
- Whole-sheet reading doesn't work. One VLM call on a full page at 110 dpi missed or invented about half the values. A 200-dpi tile read them correctly.

## First drafts
- First design: tile reading (v1/v2, the baseline)
- Split each sheet into 2×2 overlapping tiles, VLM transcribes each tile, code parses the numbers and writes the "meaning" from templates, and one more call names the feature each value belongs to.
- high recall, but about half the entries were junk: duplicates from the overlaps, frames split into letters, one value copied from a prompt example
- GD&T symbols wrong: 0% correct
- The root cause: the model was finding, positioning and reading values all at once, on overlapping crops. Cleaning up afterwards couldn't fix that. (addressing this later)

## Report QA spot checks (no eval set yet)

| question | result |
|---|---|
| throttle arm alloy / cause | correct after fixes (bending overstress, no pre-existing cracks; alloy not stated), cites 24-013 Figs. 2-3 |
| photo of 24-013 Fig. 3 (picture only) | routed via VLM classifier -> right report + figure; answer adds "corrosion/scale" from VLM observation (hallucination) |
| why did N106ER's left spar fail | fatigue - but attributes it to N104ER: report 18-061 covers spars from 3 aircraft; registration in every chunk header removes its discriminative power |

Latency: ~40-60 s per answer on the M4 (generation over ~4.5-6k prompt tokens with 2-3 images).

## Next steps

1. Eval sets + harness (routing, retrieval recall@k, answer faithfulness, drawing value P/R) before more tuning.
2. Drawings: need a proper relook
3. QA: per-chunk aircraft/part metadata (not just the doc header) so N104ER vs N106ER is separable;
   restrict context to the best-matching report when one clearly dominates.
4. Tracing (per-request spans) and bulk run of all 11 drawings.

## Next commit:
1. Added some configs, LLM integration for vLLM
2. Added vLLM scripts
3. Drawing pipeline modification

## Next commit:
1. Added a test set for text QA to run on Colab and see results

### Drawing extraction approach: find annotations geometrically, then read each one once

- Grouping the blue vector paths gave essentially one box per annotation on CTC-01 (19 boxes vs 32 junk-filled entries), and whole callouts on the denser sheets.
- Two front ends feed one shared back end (PDFs based on vector paths/text layer and pictures based on OCR text boses)
- One short VLM read per callout
- datum letters read by OCR
- "belongs to" from a crop at the leader's end
- confidence from the checks that pass
- Reference labels (not hand-annotated) for 2 drawings (CTC-01 and FTC-07, 70 values), drafted by the AI assistant before the geometry method was built, not taken from extractor output. FTC-07 was checked against its text layer. The scorer reports per-value missed and extra lists, so every later fix could be diagnosed precisely.

## Fixes to run on Colab/L4 (each diagnosed from the output files and the cached model replies)

- Fix 1:
  - repetition loops removed (one callout read had looped on the same value 15 times)
  - token limit lowered
  - frame rows rebuilt from single letters
  - picture title block and notes read from the image text
  Precision 0.47 → 0.88, but recall dipped: 4 datum symbols were glued onto neighbouring rows.
- Fix 2: only a run of letters joins a frame row; a lone letter stays a datum. No regressions, datums 0.86.
- Fix 3: word annotations (4 COPLANAR SURFACES, SIM REQT, …) rebuilt from the text layer instead of being dropped as "notes". Recall 0.81 → 0.87. Two side effects:
  - two datums lost on PDFs
  - a picture-route rule that regressed pictures (precision 0.81 → 0.59)
- Fix 4: picture rule reverted, phrase order preserved, frame rows with datums take no extra letter. D and E recovered, nothing newly missed. - *This is the version we decide to freeze*

## Next steps
- Tracing - separating request level, task level, span level
- Plan: For now, SQLite based. Postgres can happen later
- Eval Harness - components: Router, qa_text, qa_image, drawing pdf, drawing image, e2e
- Dataset generation? Router should be easy, qa_text we already have some, qa_image and drawings will take work

## Tracing and request persistence (design)

**What I need.** For every request, someone technical should be able to see what the agent did and what it called. That record has to be stored, queryable, and survive a crash. I'll also need it later for resuming bulk runs and for cost numbers in the evals.

**Where it goes: SQLite.** One file at `store/traces.db` (path set by `trace.db_path`). It's in the stdlib and the data never leaves the machine. I turned on WAL mode so bulk worker threads can keep writing while I read, e.g. running `docagent trace` in the middle of a batch. It's plain SQL, so if this ever runs on many machines the same tables can move to Postgres.

**Three tables, from coarse to fine.**
- `requests`: one row per user request. Its status (`queued`, `running`, `succeeded`, `partial`, `failed`, `interrupted`), timings, which agent version and config hash answered it, and totals for tokens, model calls and cache hits. `batch_id` groups the requests of one bulk run or eval run.
- `tasks`: one row per attachment. Route, tier, confidence, reason, stage, attempts, output path, and the error + traceback if it failed.
  Each attachment is routed and processed inside its own task, so one bad file only fails its own task. A corrupt PDF, an unreadable image or a failed classify call gets recorded in `resp.errors` and the other attachments carry on. A request is `partial` if some task failed but something still came out (a drawing JSON or an answer), and `failed` only if nothing did. The joint report answer is isolated the same way (fixed after e2e case e18 showed a failed answer raising out of `Agent.handle` and losing the drawings of the same request).
  In bulk, every file is its own request. The worker threads share one Agent (it keeps no state between calls), but each thread opens its own SQLite connection and starts with a clean trace context. Tested with corrupt files mixed in with good ones.
- `spans`: one row per step (`router`, `drawing.find_annotations`, `qa.retrieve`, `llm.call`, ...). Spans form a parent/child tree and store a duration plus a free-form `attrs` JSON (counts, chunk ids, tokens). This is the "what did it do, what did it call" part.

**Write on every event, not on a timer.** Every state change is one small transaction. A span row is inserted when a step starts and updated when it finishes. If the process crashes, there's a span stuck at `running` on exactly the step that died, and only that step is lost. Saving on a timer would lose up to a whole interval and need an extra thread.

**Model usage is tracked in one place.** Every model call goes through `VLM._complete`, so that's the only place I record usage. No duplicated tracking code, and the numbers stay consistent. Each call gets an `llm.call` span with the model, number of images, prompt hash, cache status, token usage, latency and finish reason. The prompt hash also points to the full response saved on disk, so I don't store the prompt over and over.

A reply that comes from the cache cost no model work, so it doesn't add to the request's `prompt_tokens`, `completion_tokens` or `llm_calls`. It only bumps `cache_hits`.

**Passing the trace context around.** The current request, task and span live in `contextvars`, so the pipelines don't need extra arguments. One catch: `ThreadPoolExecutor` doesn't copy contextvars into its worker threads, so spans from the parallel callout reads would end up with no parent. To avoid that, every traced job is submitted through `trace.submit`, which wraps it in `copy_context().run`.

**Logs are for reading, the database is for numbers.** Python logging also writes JSON lines to `store/logs/agent.jsonl`, each stamped with the request/task/span ids so a log line can be matched to its trace. I never parse metrics out of the logs; SQL is the source of truth.

**Recovering from failures.**
- Inside a task: the VLM runs at temperature 0 and its responses are cached, so the cache already works as a checkpoint. A rerun replays the calls that finished, for free (rerunning a bulk batch took 7 min instead of 18). So I didn't write any per-stage checkpoint code.
- Across tasks: on startup, any row still `running` with a stale heartbeat is marked `interrupted` (`mark_interrupted`). Otherwise it would show "in progress" forever.
- Errors: temporary ones (connection refused, timeouts while vLLM or Ollama restarts) are retried by the OpenAI client with backoff (`max_retries`). Errors that would just happen again (output that can't be parsed) aren't retried. They're recorded and the next attachment goes ahead.

### Retry, bulk resume, lookup by file

A drawing takes minutes, so losing finished work to one crash or one bad file is expensive. And with a million files, starting over is not an option. What I want is that a failure only costs the part that wasn't done yet.

- **A task only counts as done once its result file exists.** Results are written atomically, so a crash never leaves a half-written file that looks finished.
- **If the input hasn't changed, the result hasn't either.** Each task is keyed on the file hash + model + versions + config. Same key, reuse the old result. Change the config or a version and nothing gets reused.
- **Retry is a new attempt linked to the old one.** Finished drawings get reused and the rest re-runs. Even the re-run is mostly free, because the temperature-0 cache replays the model calls that already finished. So the cache doubles as my checkpoint and I don't need per-stage checkpoint code.
- **Report answers always re-run.** They're one answer over the question and all its pictures together, so there's nothing per-file to reuse.
- **Bulk writes down the whole plan before it starts.** Every file goes in as `queued` first. After a crash I can see what finished, what died midway and what never started, and `--resume` only picks up the unfinished ones.
- **Reuse is on by default everywhere, `--force` turns it off.** One flag that means the same thing in every command. The chat says when it reused a result, so a demo doesn't look like it ran something it didn't.
- **Drawing uploads have no text to search by**, so requests can be looked up by file name (`docagent requests --file`), and `trace` shows every attempt.
- **All of this depends on prompts being identical between runs.** A cache miss on retry would quietly redo the work. So I checked that nothing random gets into a prompt (timestamps, temp paths, set ordering). Two runs with different hash seeds: 74 fresh calls, then 0.

Checked with a stub model: a crashed two-drawing request retried with 0 fresh calls. A bulk run killed partway through resumed only the unfinished files. Running the same bulk again made no model calls at all.

### Bulk from the chat, uploads, zips

A big chat upload can run for an hour. The browser tab can't be the thing keeping it alive, and a retry needs the original bytes even after Gradio cleans up its temp files or the user moves the files.

- **Keep my own copy of every upload**, named by content hash. Retry and resume read from there, never from wherever the file came from.
- **One batch engine for both the CLI and the chat.** A chat message becomes a batch when it has several files and no question, or a zip. With a question, all the pictures get one answer together, so splitting them up would change the answer. Without one the files are independent, which is just bulk.
- **The work runs in the app, the chat only watches.** Workers run in the background and the chat reply polls SQLite for progress. Closing the tab stops the watching, not the work. On restart the app resumes unfinished chat batches but not failed files, otherwise a file that always fails would be retried forever.
- **Treat zips as hostile.** Files are stored by hash, never under a name from the archive, so path tricks like `../evil.pdf` don't work. Size and count limits stop zip bombs.

Checked: the chat replies right away with a batch id, and its results match the CLI bulk byte for byte. Killing the app mid-batch and restarting it resumes only the unfinished files. A zip full of bad paths only processes the one safe file.

**What I didn't fix.**
- Embedded Qdrant can only be open in one process, so the chat app and a bulk run that needs report QA can't run at the same time. Drawing-only bulk is fine. The real fix is running Qdrant as a server.
- Two processes resuming the same batch could grab the same file. Not an issue on one machine. With many workers they'd need to claim work through the database (Postgres `SKIP LOCKED` or a queue).
- No retention policy yet. Traces and outputs hold user data and grow forever, but how long to keep them is the operator's call, not something I should hard-code.

**Why I built my own tracing instead of using Opik or Langfuse.** Their cloud versions send data off the machine, and self-hosting means 4-6 extra containers to replace ~300 lines I fully understand. They make sense later if a team needs a UI, and my spans map onto theirs.

**Colab.** Experiment runs copy the trace database and logs into the experiment folder before the VM goes away.

**Scale.** ~100 spans per drawing means ~10^8 rows for a million drawings. At that point spans move to Postgres partitioned by day or to a columnar store, with retention. Requests and tasks stay small.

## Evaluation design

**Goal.** Req: code that measures each important part and the whole system, incrementally - plug in a
new version of the agent or of one part, run it on a chosen number of cases, see better or worse. No interface.

**Reality** Creating a golden dataset by ourselves is not practical especially with drawings. We are just going to have to get AI to generate QA sets, routing decisions, and use imperfect stand-ins for drawing ground truth: a frozen best run (fix4) as the regression baseline, and AI-drafted labels on 4 sheets as the correctness estimate. Some are already there for refining. This is not at all good practice for production but only for demo purposes - because especially for drawings we would ideally manually annotate them but I simply do not have the skill for it under these constraints.

| brief says | mechanism |
|---|---|
| "each important part" | suites `router`, `qa_text`, `qa_image`, `drawing` - one per decision the agent makes |
| "and for the whole system" | suite `e2e`: `Agent.handle` on mixed requests, no gold inputs anywhere |
| "plug in a new version of the agent, or of one part" | config = version (`--config`), or one knob changed with `--set drawing.method=tiles`; the record stores config hash, overrides and `Agent.versions` |
| "a chosen number of test cases" | `--n N --seed S` (deterministic sample), `--ids`, `--split dev\|test\|all` |
| "whether it got better or worse" | `--compare <earlier results file>`: metric deltas, **per-case flips**, latency/token deltas, warnings when datasets/splits differ |
| "no interface needed" | one CLI: `python -m eval <suite> ...`, results as JSON in `eval/results/` |

**Each part suite isolates its part with gold inputs.** The drawing suite feeds known drawings straight to the
extractor (router bypassed), the QA suites call `ReportQA.run` directly, the router suite only routes. A new
router version can't make the drawing numbers move, so a change is attributed to the part that changed. `e2e`
is the only place errors compound; comparing `router accuracy x component pass-rate` with the e2e success rate
shows where e2e losses come from.

**Dev / test split.** The 20 QA questions were used to tune and the two
labelled drawings to build the geometry method - their scores are optimistic. They are `dev`. New cases written
afterwards are `test`: not looked at while changing the agent, reported separately. Default `--split dev`, so
tuning loops don't touch `test` by accident. Router: cases derived from the 14 original PDFs (which the rules
were written against) and 3 of the 6 text questions are dev, all generated pictures / single pages are test.

**Flips over averages.** With 10-35 cases one case is 3-10 points; an average can stay flat while two cases swap.
`--compare` lists newly failing / newly passing cases (for drawings: per ground-truth value), which is what a
reviewer acts on.

**Real-world e2e latency needs `--fresh`.** Temperature-0 replies replay from the VLM cache, so a re-run is fast and spends no
tokens - fine for accuracy comparisons (same prompts, same replies), meaningless for latency and cost. `--fresh`
points `llm.cache_dir` at an empty temp dir for the run. Cost and latency come from the trace database (every
eval case is a traced request under batch `eval-<suite>-<label>-<ts>`): p50/p95 latency, fresh tokens, cached calls.

**How the labels were made, and where they're weak.**
- **Router:** generated from the source PDFs by a script: drawing sheets rendered with rotation, blur, JPEG artefacts, crops and photo-like effects, plus report pages, figure crops, charts and text questions. The label comes from how each case was made, so there's no hand labelling and no label noise. The catch is that it's all synthetic. Real phone photos would go in `real/`.
- **Report QA (text and picture questions):** drafted by the AI assistant from the report text, including the picture + question cases added later. A script checks every required fact against the page it cites, but **no human has reviewed them**. I decided to use them as drafted, so a wrong expected answer counts against the agent. Some checks are weak too, e.g. bare numbers like "23" that can match by accident, or one-word facts like "heat".
- **The picture questions are easy mode.** They're the reports' own figures with the caption cropped off, so a real photo of a new part will be harder. I couldn't make out-of-scope photos offline.
- **"Wrong aircraft" questions** check that the agent answers for the aircraft the question names, not the one in the picture (a picture can't show a registration).
- **Drawing test labels (CTC-03, FTC-11):** drafted from the rendered sheets and the text layer, on purpose not from the extractor's output, so the test isn't contaminated. Marked unverified until I check them against the overlays.
- **`belongs_to` is lenient:** "hole" passes for any hole. I added a stricter version that needs the main feature word. Checking that the leader line points at the right feature isn't possible, because the output only stores where the annotation is.
- The datasets are frozen and every run records their hash, so an edit to one shows up in `--compare`.

**Drawings: compare against a reference run, not just the label files.** Only 2 drawings have verified labels, which isn't enough to judge a change. So I took my best full run so far (`geo-l4-fix4`) as a reference, and the drawing suite reports two things:
- **accuracy** against the label files in `eval/datasets/drawings/` (the drafted test ones shown separately until verified)
- **agreement** with the reference on all 11: which values were dropped, which are new, which readings changed

Agreement only says what changed, not whether it's right: a drop in agreement can be a regression, an improvement or an equally valid reading. The reference has mistakes, and fixing one lowers agreement, so I never look at it without accuracy next to it. End-to-end drawings without labels pass if agreement is at least 0.7. Sanity checks: the reference scored against itself gives 1.0, a copy with 2 values removed and 1 changed shows exactly those 3 differences, and its accuracy matches its original results.

**Sizes.** Router 80 cases (mostly test), report QA 35 text + 30 picture, 4 labelled drawings plus the reference on all 11, end-to-end 21.

**We also compare BF16 weights with AWQ version to benchmark gains** 

**Status.** The harness and all five suites work end to end (tested with a stub model, including `--compare` and `--fresh`). Still to do: check the drawing test labels, add real phone photos, and get GPU numbers for every suite.

## Chat front end (summary)

Source: `app/chat.py`

**Why Gradio**
- `gr.ChatInterface(multimodal=True)` gives a chat window with text + multi-file upload in a few lines:
  the brief needs text, pictures, PDFs and many files at once, and asks for an agent, not a web product.
- Pure Python, same process as the agent: no separate frontend build, no API layer, no JS. The UI is a thin
  shell; all logic stays in `Agent`, `BatchRunner` and the trace DB, shared with the CLI and the eval.
- Generators stream: `respond` can `yield` several times, so a long batch shows live progress.
- `gr.File` returns the drawing JSON / results zip as downloads directly in the reply.
- Runs fully local: bound to `127.0.0.1`, `share=False` (no public Gradio tunnel), `GRADIO_ANALYTICS_ENABLED=False`
  (no telemetry calls). Fits requirement 6 (private).
- Alternatives considered: Streamlit (reruns the whole script on each interaction, awkward for chat + background
  jobs), FastAPI + custom frontend (more work, nothing gained for a demo). Gradio is a demo UI: for production,
  the same `Agent` sits behind an API with auth.

**What else the chat uses**
- `docagent.agent.Agent` - one shared instance (router, both pipelines, tracer, VLM client).
- `docagent.batch.BatchRunner` - bounded thread pool (`batch.workers`) for background batches.
- `docagent.render.render_markdown` - turns a `Response` into the reply text (answer, diagnosis, what to do,
  sources, drawing summary, request id). Same renderer as the CLI.
- `docagent.trace.install_log_file` - JSON-lines log with request/task/span ids.

**How a message is handled** (`respond`)
1. `status <batch_id>` with no files -> stream that batch's progress again (from the DB).
2. A zip, or >= `batch.chat_min_files` (2) files with no question -> **background batch**:
   files copied to `store/uploads/<sha256><ext>` (Gradio's temp files can disappear), `runner.start` queues one
   request per file, the reply shows the batch id at once, then `watch` polls the trace DB every 1 s and updates
   the message; when finished it attaches a results zip. A question sent with a batch is ignored (said in the reply).
3. Anything else (text only, one file, or files + a question) -> `agent.ask(...)` in the request, one combined
   reply, drawing JSONs offered as downloads.
- The chat never decides the route itself: inside `Agent`, `Router` decides per attachment (rules, then VLM;
  see "Routing rules"). The user never says what they want (requirement 4).
- "Force re-run" checkbox: ignore earlier results for the same file + version + config (otherwise reused;
  model calls still replay from the VLM cache).

**Bulk without depending on the browser**
- The batch runs in the app process, not in the browser: closing the tab only stops *watching*.
  `watch` only reads the DB, so stopping it touches nothing.
- Every file is enqueued as a `queued` request before processing, so the DB always has the full picture.
- On app start: requests left `running` by a crashed process are marked `interrupted`, and unfinished `chat-`
  batches resume (queued / interrupted files only; failed files are not retried automatically -> `docagent bulk --resume`).
- Same batch code as `docagent bulk` on the CLI: one implementation, two entry points.

**Tracking / logging behind the chat**
- Every chat message is a traced request in `store/traces.db` (SQLite, WAL): `requests` -> `tasks` (one per
  attachment: route, tier, confidence, reason, output) -> `spans` (router, qa.retrieve, llm.call, drawing steps,
  with tokens and latency). Chat batches get a `chat-...` batch id.
- The request id is shown under every reply; a technical person looks it up with `docagent trace <request_id>`,
  `docagent requests`, or `docagent batch <batch_id>` - the chat user never sees the internals.
- Log lines go to `store/logs/agent.jsonl` with the same ids, so a log line joins to its trace.

**Limits**
- One app process: embedded Qdrant takes a file lock, so the chat and a CLI `bulk` with report-QA files can't
  run at the same time (fix: Qdrant server via `store.qdrant_url`).
- No authentication or per-user separation: fine on localhost, not for a shared deployment.