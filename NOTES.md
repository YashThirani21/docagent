## What I'm building

One chat agent that works out which of two jobs a request needs:

- **Report QA**: a text question or a photo of a damaged part: what is wrong and what to do,
  answered from the 3 pdf reports with citations.
- **Drawing extraction**: a technical drawing: a JSON entry for every value on it, with what it belongs to and what it means.

Around it: automatic routing, bulk upload, fully offline running, a design that works at scale, a trace of every request, and an eval harness that compares versions.

## How I'll work

1. **Every part working badly before any part working well.** With limited time, a complete weak system beats one perfect component.
2. **Contracts and tracing first.** Everything later plugs into them; adding them at the end means rewriting.
3. **I think I will need Colab here** since qwen2.5vl:7b is too clunky to work with on Ollama on local - better to use VLLM and GPU on Colab

## 1. Data inspection (before any design)

Profiled every page of the 14 PDFs with PyMuPDF (text chars, raster images, vector paths, page size).

**Reports (3 NTSB Materials Laboratory Factual Reports, 14/40/6 pages)**
- Born-digital (Word -> PDF): full text layer on every page. **No OCR needed for the corpus.**
- Figure pages hold one or two figures, each captioned `Figure N. ...` below the image.
- A figure is *not* one embedded image: it is many small raster tiles plus vector arrows and text
  labels (e.g. 18-061 p.13 has 19 image placements). Extracting embedded images individually is useless.
  -> crop the page region above each caption and render it, as a reader sees it.
- 18-061 pp.37-40 are vector load-extension charts (no caption) -> treated as "chart" figures.
- 24-013 has a near-empty cover page and a blank page.

**Drawings (11 NIST PMI test cases, 1-5 sheets each, 30 sheets total, all landscape 792x612 pt)**
- 10 of 11 (Acrobat Distiller / PScript) draw dimensions and GD&T as **vector outlines**: the text layer
  only has the title block and notes (CTC-01: 153 chars on a sheet full of annotations).
- Only FTC-07 (AutoCAD export) has the annotations as real text.
- -> the VLM must read annotations from pixels; the text layer is used as exact truth where it exists.
- Annotations are blue; part geometry is black/purple. Not yet exploited (see Next steps).


## Model and infrastructure choices

- **VLM: qwen2.5vl:7b via Ollama** (runs on the 16 GB M4, ~6 GB). One model for figure description,
  photo observation, drawing reading, routing fallback and answer generation. Called through the
  OpenAI-compatible API so vLLM on a GPU is a `base_url` change.
- **Ollama context bug:** `ollama ps` showed the model loaded with **4096 tokens** although every request
  passed `num_ctx=16384`: the OpenAI-compatible endpoint ignores per-request options and truncates
  silently. Fixed with a model variant `qwen2.5vl-16k` (Modelfile, `scripts/setup_ollama.sh`).
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


## Bugs found and fixed (measured)

- **Batched bge-m3 on Apple MPS returns corrupted vectors.** After the first ingestion only 19 of 94
  stored dense vectors were unique; a batch of 16 chunks produced 2 distinct vectors; dense scores were
  identical (0.285) for unrelated chunks, and report 24-013 vanished from results. Short texts were fine,
  long padded batches were not. CPU: all 94 unique, retrieval correct. -> `device: cpu` + a self-check at
  load time (batched vs single cosine must be > 0.99, else refuse to start). CPU embedding of the whole
  corpus takes ~30 s, so no cost here; at scale use CUDA.
- **VLM figure descriptions overrode the reports.** Q: "what caused the throttle arm to break?" The
  report says *bending overstress, no pre-existing cracks*; the answer said *fatigue* because the
  auto-generated figure description mentioned beach marks. -> descriptions are now used for
  **retrieval only** (`Chunk.description`, part of `embed_text`); the generator sees caption, printed
  labels and the report sentences citing the figure. Prompt: report conclusions are authoritative.
  After the fix the answer is correct.
- **Failure-mode guess on user photos biased retrieval.** Observation step asked for a "likely failure
  mode"; its guess ("fatigue") steered both search and answer. Removed: the VLM now reports visible
  features only and the reports do the diagnosing.

## Drawing extraction iterations

| version | approach | CTC-01 (1 sheet) | observation |
|---|---|---|---|
| probe | whole sheet @110 dpi, free text | 23 s | ~half the values missed or invented |
| probe | one 200 dpi tile, free text | 15 s | values read correctly |
| v1 | overview + 4 tiles, every field from VLM | not completed | estimated 4-8 min/sheet (output tokens dominate at ~16 tok/s) |
| v2 | READ tiles (text+type+box) -> code parses numbers -> LOCATE with numbered marks on the sheet -> code writes `meaning` | **182 s**, 43 values | readable values mostly right; boxes only roughly placed; one frame split into invented letter boxes; title-block text leaked; frames mistyped as diameters |
| v2.1 | + rules: retype by printed text (value + datum letters = frame), drop title/notes text, angle units | - | fixes the systematic typing errors |

Timing estimate for all 30 sheets on the laptop: ~90 min. On a GPU with vLLM batching: minutes.

## Report QA spot checks (no eval set yet)

| question | result |
|---|---|
| throttle arm alloy / cause | correct after fixes (bending overstress, no pre-existing cracks; alloy not stated), cites 24-013 Figs. 2-3 |
| photo of 24-013 Fig. 3 (picture only) | routed via VLM classifier -> right report + figure; answer adds "corrosion/scale" from VLM observation (hallucination) |
| why did N106ER's left spar fail | fatigue - but attributes it to N104ER: report 18-061 covers spars from 3 aircraft; registration in every chunk header removes its discriminative power |

Latency: ~40-60 s per answer on the M4 (generation over ~4.5-6k prompt tokens with 2-3 images).

## Next steps

1. Eval sets + harness (routing, retrieval recall@k, answer faithfulness, drawing value P/R) before more tuning.
2. Drawings: exploit the vector layer - annotations are blue paths. Cluster blue paths into annotation
   groups -> exact boxes from geometry, leader-line endpoints -> the feature they point to; VLM reads one
   small crop per group (short outputs). Expected: better recall, exact boxes, real `belongs_to`.
3. QA: per-chunk aircraft/part metadata (not just the doc header) so N104ER vs N106ER is separable;
   restrict context to the best-matching report when one clearly dominates.
4. Tracing (per-request spans) and bulk run of all 11 drawings.

## Next commit:
1. Added some configs, LLM integration for vLLM
2. Added vLLM scripts
3. Drawing pipeline modification 

## Next commit:
1. Added a test set for text QA to run on Colab and see results

## Drawing extraction: geometry first (PDF + pictures)

**Why.** The tile reader let a 7B VLM find, place and read annotations on overlapping tiles: high
recall but ~half the entries junk (duplicates from overlaps, frames split into letters, prompt examples
copied), GD&T symbols 0% right, positions approximate.

**Ground truth + eval** (`eval/datasets/drawings/`, `eval/run_drawings.py`): CTC-01 (1 sheet, isometric,
metric, 17 values) and FTC-07 (4 sheets, AutoCAD export with a real text layer, 53 values), labelled from
the rendered sheets (FTC-07 strings checked against its text layer). One-to-one matching by normalised
text / identical numbers per sheet. `belongs_to` is scored by keyword lists per value - lenient (any
"hole" answer passes for a hole); treat it as a coarse signal.

**Pipeline** (`drawing.method: geometry`):
1. *Find annotations* - `drawing_annotations.py`. Vector route: annotation layer chosen automatically
   (OCG layer name, else the colour whose paths look most like glyphs + frames); text from glyph outlines
   or the text layer (per-character boxes); frames = rectangles/quads; leaders = segments (or thin filled
   polygons) that start at a callout and end away from it, followed through polylines. Grouping:
   glyphs -> words -> frames absorb words inside their real (rotated) polygon -> nearby groups merge into a
   callout unless both already have their own leader. Picture route: RapidOCR detector boxes on 3x3
   tiles, OpenCV contours for frames and Hough segments for leaders, same grouping.
2. *Read each callout once* - crop rotated to horizontal text, one short VLM call (no literal examples in
   the prompt: the 7B model copied them into every answer - precision 0.22 in the first try).
3. *GD&T symbol without the VLM* - `gdt_symbols.py`: the VLM could not recognise symbols even on clean
   horizontal crops. Frame cells = straight-sided contour holes, first cell un-sheared (isometric frames are
   parallelograms), symbol classified by topology (4 enclosed regions = position, 4-corner outline =
   flatness, closed half-circle = profile of a surface, perpendicular lines = perpendicularity, ...).
4. *Datum boxes* - box interior un-sheared, letter recognised by RapidOCR at 4 rotations (VLM fallback).
5. *Code*: frame rows rebuilt from fragments ('0.75' + 'A B C', projected zones, MAX), stacked tolerances
   and limit dimensions merged (lower/upper limits added to the schema), text-layer digits override the
   reading, unread text-layer word annotations added, implausible readings dropped.
6. *belongs_to*: a crop around the leader end (red circle) named by the VLM; where that gives nothing
   useful, one sheet-level numbered-marks call fills in. Confidence = checks passed (text layer agrees,
   numbers parsed, symbol from geometry, leader found, datums defined on the drawing), listed per value.

**Findings on the way**
- All 11 PDFs are portrait pages with /Rotate 270: PyMuPDF returns paths AND text-layer boxes unrotated.
  The tile pipeline used unrotated text-layer boxes against rotated tiles (wrong hints on FTC-07).
- AutoCAD (FTC-07) draws wide lines as thin filled polygons, arrowheads as small fills, frames as lines
  (not quads), and stores text as real text on diagonal lines whose axis-aligned boxes overlap - each
  needed its own handling (thin-polygon segments, per-character boxes, quad-true containment).
- A revision letter "D" in the known title-block text hid datum "D"; part shading fills looked like
  coordinate-triad marks. Both fixed (min length for known text; triads = small saturated strokes only).

**Results on the ground-truth sheets** (70 required values; local Ollama 4-bit for geometry, L4 bf16 for
the baseline - the geometry method on the L4 should do at least as well):

| method | recall | precision | numbers | type | GD&T symbol | datums | belongs_to | title/notes/units |
|---|---|---|---|---|---|---|---|---|
| tiles v2 (baseline, L4) | 0.73 | 0.45 | 0.80 | 0.73 | 0.00 | 0.81 | 0.58 | 1.0 / 1.0 / 1.0 |
| geometry, PDF route | 0.89 | 0.87 | 1.00 | 0.91 | 0.71 | 0.89 | 0.75 | 1.0 / 1.0 / 1.0 |
| geometry, picture route (JPG) | see below | | | | | | | |

Iterations (geo-v1..v8 in eval/results/): example-copying fixed (precision 0.22 -> 0.78), deskewed crops,
symbol classifier (0 -> 1.0 on CTC-01), frame-part joining + empty-read fallback (FTC-07 recall 0.72 ->
0.87), datum boxes via un-shear + OCR, sheet-level belongs_to fallback (0.37 -> 0.75).

**Remaining risks / dataset-specific assumptions**
- Tuned on 2 labelled drawings; the other 9 were only checked by overlay (`tmp/annot_overlays/`).
- Vector route assumes annotations are separable by layer or colour (true for all 11, typical for CAD
  exports); otherwise the picture route is used. Title-block regexes are NIST-specific (generic VLM
  fallback when they find nothing).
- GD&T symbols on AutoCAD frames: 0.61 (CTC-01: 1.0). Rarely used symbols (cylindricity, runout,
  concentricity) have no examples here and are unvalidated.
- belongs_to is the weakest output and the metric is lenient.

## Next steps
- Tracing - separating request level, task level, span level
- Plan: For now, SQLite based. Postgres can happen later
- Eval Harness - components: Router, qa_text, qa_image, drawing pdf, drawing image, e2e
- Dataset generation? Router should be easy, qa_text we already have some, qa_image and drawings will take work

## Tracing and request persistence (design)

**Objective.** Requirement: a technical person sees, for every request, what the agent did and what it called -
stored, queryable, surviving crashes. Also the groundwork for resumable bulk runs and for eval cost numbers.

**Store: SQLite** (`store/traces.db`, `trace.db_path`). Stdlib, a file inside the system (nothing leaves the
box), WAL mode so bulk worker threads write concurrently with readers (`docagent trace` while a batch runs).
Plain SQL: the same three tables move to Postgres when there are many machines.

**Three tables, three granularities.**
- `requests` - one per user request. Lifecycle (`queued/running/succeeded/partial/failed/interrupted`),
  timings, agent versions + config hash (which agent version answered), token / call / cache-hit totals.
  `batch_id` groups a bulk run or an eval run.
- `tasks` - one per attachment: route, tier, confidence, reason, stage, attempts, output path, error +
  traceback. The attachment is the unit of failure isolation and of retry/resume: it is routed *and*
  processed inside its own task, so a corrupt PDF, an unreadable image or a failed classify call fails that
  task only (error in `resp.errors`) and the other attachments still run. A request is `partial` when some
  task failed but something came out (a drawing JSON or an answer), `failed` only when nothing did. In bulk
  every file is its own request; worker threads share the Agent (stateless per call), but each thread has
  its own SQLite connection and starts with an empty trace context - tested with corrupt files among good ones.
- `spans` - one per step (`router`, `drawing.find_annotations`, `qa.retrieve`, `llm.call`, ...), parent/child
  tree, duration, free-form `attrs` JSON (counts, chunk ids, tokens). What it did and what it called.

**Write on events, not on a timer.** Every state transition is one small transaction. A span row is inserted
when the step *opens* and updated when it closes, so a crash leaves a visible `running` span at the exact step
that died, and loses at most that step. Periodic saves would lose up to one interval and add a thread.

**Tokens and latency captured once**, in `VLM._complete` - the single point every model call passes through.
Each call (cached or not) is an `llm.call` span: model, images, prompt hash (= cache key, i.e. a pointer to
the full reply on disk without storing prompts twice), cache hit, tokens, latency, finish reason.
**Spend = fresh calls only.** A reply replayed from the cache cost no model work, so it adds nothing to the
request's `prompt_tokens` / `completion_tokens` / `llm_calls`; it increments `cache_hits`. The VLM cache stores
`{text, usage, latency_s, finish}`, and a replayed call's span keeps that original usage under separate names
(`orig_prompt_tokens`, `orig_completion_tokens`, `orig_latency_s`) that nothing sums as spend - `docagent trace`
shows them as "replayed from cache, originally N tokens". A stable cost-per-case for evals, if needed, must be
derived explicitly from fresh + orig_* and labelled as such. Old `.txt` cache entries are still read
(original usage unknown = null).

**Context propagation.** The current request/task/span live in `contextvars`, so pipelines need no extra
arguments. Gotcha: `ThreadPoolExecutor` does not copy contextvars into worker threads - spans from parallel
callout reads would lose their parent. Every traced submit goes through `trace.submit` (`copy_context().run`).

**Logging stays narrative.** Python logging additionally writes JSON lines (`store/logs/agent.jsonl`) stamped
with request/task/span ids, so a log line can be joined to its trace. Metrics are never parsed from logs; SQL
is the source of truth.

**Failure recovery.**
- Inside a task: the temperature-0 VLM cache is already a checkpoint - a rerun replays finished calls for free
  (fix2 bulk: 7 min instead of 18). No per-stage checkpoint code.
- Across tasks: on startup rows still `running` whose heartbeat is stale are marked `interrupted`
  (`mark_interrupted`) - visible instead of silently "in progress" forever. *(implemented)*
- Transient errors (connection refused, timeouts while vLLM/Ollama restarts) are retried by the OpenAI client
  (`max_retries`, backoff); deterministic errors (unparseable output) are not retried - recorded and the next
  attachment continues. *(implemented)*
- *Planned with bulk:* outputs written atomically (tmp + `os.replace`), task marked `succeeded` only after its
  output exists, idempotency key `sha256 + extractor version + config hash` to skip done work and exact
  duplicates, `bulk --resume <batch_id>`. The `tasks` columns for this (status, attempts, output_path,
  heartbeat_at) exist now.

### Retry, bulk resume, lookup by file

**Need.** A failure must not lose finished work: a drawing that failed (or a process killed mid-batch) should
cost only what was not done yet, and a 10^6-file bulk upload has to be restartable without redoing the done part.

**Design.**
- Results are files written by the agent, atomically (tmp + `os.replace`), at `<store.output_dir>/<request_id>/`
  - one place for chat, ask and bulk, keyed by the trace id. `tasks.output_path` is set before the task closes,
  so `succeeded` always means "its file exists". Bulk `--out/drawings/` is a convenience copy.
- `tasks.idem_key` = sha256(file sha256 + router/drawing version + model + config hash). Same key = same result,
  so it can be reused. The config hash leaves out the `trace` section (where traces go changes no output).
- `docagent retry <id>`: new request, `retry_of` = latest attempt of the chain, `attempt` + 1, same text and files
  (refused if a file is gone or its sha256 changed - that is a new request, not a retry). Drawing tasks with a
  finished result for the same idem_key become `reused` tasks pointing at the existing file (no copy); the rest
  re-runs and its finished model calls replay from the VLM cache - the cache is the intra-task checkpoint. Config
  or version changed: nothing reused, said in the output. A succeeded request needs `--force` (reuses nothing).
  Report answers are one joint answer over the question and all pictures: re-run (cheap via the cache), never reused.
- Bulk enqueues every file as a `queued` request before processing, so a crash leaves a complete picture: done,
  interrupted (`running` from a dead pid on this host is marked at once - pid + host stored per request; other
  hosts after 15 min), and never started. `bulk --resume <batch>`: per file the latest attempt; queued -> run,
  failed / partial / interrupted -> retry, succeeded -> skip. Plain `bulk` skips files whose idem_key already has a
  finished result (`--force` to redo) - since 9.2 recorded as instant `reused` requests so the batch lists them.
- Migration: new columns are added with `ALTER TABLE ADD COLUMN` when an older `traces.db` is opened.
- Lookup: `docagent requests --file <substring>` (drawing uploads have no text); `trace` shows the attempt chain.
- Reuse is the default everywhere, not only on retry: `ask` and the chat app also take a finished drawing result
  for an identical file (task `reused`, `reused_from`; the reply says "reused result from request X" so a demo
  viewer is not misled). One switch with one meaning on ask / bulk / retry / chat: `--force` = reuse no earlier
  result. It does not bypass the VLM cache - that is a config change (`llm.cache_dir`), i.e. a new agent version.

**Prompt determinism** (a cache miss on retry would silently redo work). Checked what goes into every VLM call:
text-layer strings, geometry hints, mark lists, retrieved sources, rendered crops - no timestamps, ids, paths or
temp names; sets are only used for membership, never iterated into a prompt. Measured: CTC-01 + FTC-07 in two
processes with different `PYTHONHASHSEED`: 74 fresh calls, then 0.

**Measured** (stub model, so counts not seconds are the point):
| case | before | after |
|---|---|---|
| CTC-01 + CTC-03, CTC-03 crashes after reading -> retry | attempt 1: 55 fresh calls | attempt 2: 0 fresh; CTC-01 reused, CTC-03's 29 calls replayed |
| bulk of 4, process killed 5 calls into file 3 -> `--resume` | files 1-2 done | only files 3-4 run; file 3 replays its 5 calls, 19 fresh |
| same bulk again | - | 4 x "skipped (done)", 0 model calls (since 9.2: recorded as 4 instant `reused` requests) |
Real run (Ollama, laptop): CTC-01 + a corrupt PDF -> `partial` (CTC-01's 26 calls were already in the dev cache,
0.7 s); `docagent retry` -> CTC-01 `reused` (0 model calls, not even a replay; 0.0 s), the corrupt PDF fails again
with the same error. On a cold cache CTC-01 is ~26 fresh VLM calls (minutes on the laptop) that the retry skips.

### Bulk from the chat, uploads, zips

**Need.** Bulk upload must not depend on the browser or on where files came from: a chat upload of many drawings
can take an hour (minutes per drawing on the laptop), tabs get closed, Gradio deletes its temp files, users move
files - and a retry needs the exact bytes again.

**Design.**
- *Upload store*: every submitted file is copied to `store/uploads/<sha256><ext>` (atomic, once per content) and
  the request records that path plus the original name. Retry / resume read the store, never the original; the
  original name drives display, the drawing id and `--file`. Old rows with original paths still work (the path is
  just checked for existence and sha256).
- *One batch implementation* (`batch.py`) for `docagent bulk`, `--resume`, `docagent batch` and the chat. A chat
  message becomes a batch with >= 2 files and no question, or any zip. Threshold reasoning: a question applies to
  all its pictures jointly (one answer), so splitting it into per-file requests would change the answer; without
  a question the files are independent, exactly what bulk does. One file stays synchronous as before.
- The chat batch runs in a bounded thread pool in the app process (`batch.workers`); the reply is a generator that
  only *reads* the batch's state from SQLite every second. Closing the tab stops the reading, not the work; `status
  <batch_id>` reads it again. On start the app marks dead requests interrupted and resumes chat batches with files
  queued / interrupted - not failed ones (a file that always fails would be retried at every start).
- A file already done is recorded in a new batch as an instant `reused` request rather than left out, so batch
  status and its results zip are complete. Identical content twice in one batch runs once.
- *Zips*: members are streamed into the upload store by content, never written under an archive-supplied name, so
  traversal is impossible by construction; `..`, absolute and drive paths are still reported (a crafted archive),
  hidden / `__MACOSX` / unsupported members skipped; member-count and total-size limits refuse zip bombs, and a
  per-member cap counts actual bytes (headers can lie).
- The Agent is shared by the chat's request threads and batch workers: the report index is opened under a lock
  (two embedded Qdrant clients in one process would collide).

**Measured** (stub model): a chat message with 3 drawings answers with the batch id in < 0.01 s and streams 5
updates; its results are byte-identical to `docagent bulk --force` on the same files; dropping the generator after
the first update still finishes all 3; a process killed 5 calls into file 3 of a 4-file chat batch -> the next app
start resumes only files 3-4 (attempts 1, 1, 2, 1). Deleting the originals after submission: retry and `--resume`
still succeed. Malicious zip (`../evil.pdf`, `/abs.pdf`, `C:/win.pdf`, `ok.pdf`): only `ok.pdf` processed.

**Known limits (documented, not built).**
- *Embedded Qdrant is single-process* (`store.py`): the chat app and a `bulk` containing photos / report pages /
  text questions cannot run at the same time - the second process fails on those files. Drawing-only batches are
  unaffected: the index is opened only by `Agent.qa`, i.e. when an attachment routes to report QA or there is a
  question (checked in code: router, drawing pipeline and batch code never open it). Fix: a Qdrant server
  (`store.qdrant_url`, Docker).
- *No task claiming across concurrent workers.* Two processes resuming the same batch at once could both pick up
  a queued file. On one host this is avoided in practice (a running request belongs to a live pid; one resumer at
  a time). *For the scale page:* with many workers, claim work in the database - Postgres
  `SELECT ... FOR UPDATE SKIP LOCKED` on queued rows (or a queue) - instead of each worker planning from a snapshot.

**Why not Opik / Langfuse.** Their cloud versions send prompts and documents out - breaks requirement 6.
Self-hosting means 4-6 extra containers (ClickHouse, Redis, ...) for what is ~300 lines here, each line
explainable. They are the production option once there is a UI team to serve; the span model maps onto theirs.

**Experiments.** `scripts/colab_experiment.py` ends by copying the trace database (SQLite online backup API: a
raw copy of a WAL database can miss committed rows still in the `-wal` file) and the JSON-lines log into the
experiment folder, so a Colab run's traces survive the VM.

**Retention - not implemented, on purpose.** `outputs/requests/` (result files) and `store/traces.db` (request
texts, attachment paths, answers in the summaries, error messages) hold user data and grow without bound. They
need a retention policy - e.g. delete results and request rows after N days, keep spans summarised or sampled
beyond that - set by whoever operates the system; it is a policy decision, not code to guess here.
*For the privacy & scale page:* both live inside the system boundary (no external tracing service), but they are
copies of user data with their own lifetime and need the same retention / access rules as the uploads.

**Scale.** One drawing = ~50-150 `llm.call` spans; a million drawings = ~10^8 span rows. Then: Postgres with
spans partitioned by day, retention on spans (keep requests/tasks), or spans to a columnar store. Requests and
tasks stay small enough for one table.

## Evaluation design

**Goal.** Req: code that measures each important part and the whole system, incrementally - plug in a
new version of the agent or of one part, run it on a chosen number of cases, see better or worse. No interface.

**Reality** Creating a golden dataset by ourselves is not practical especially with drawings. We are just going to have to get AI to generate QA sets, routing decisions, and use imperfect drawings as ground truth to compare with. Some are already there for refining

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

**Dev / test split.** The 20 QA questions were used to tune (the q10 / q14 / q19 fixes in section 6), and the two
labelled drawings to build the geometry method - their scores are optimistic. They are `dev`. New cases written
afterwards are `test`: not looked at while changing the agent, reported separately. Default `--split dev`, so
tuning loops don't touch `test` by accident. Router: cases derived from the 14 original PDFs (which the rules
were written against) and 3 of the 6 text questions are dev, all generated pictures / single pages are test.

**Flips over averages.** With 10-35 cases one case is 3-10 points; an average can stay flat while two cases swap.
`--compare` lists newly failing / newly passing cases (for drawings: per ground-truth value), which is what a
reviewer acts on.

**Latency needs `--fresh`.** Temperature-0 replies replay from the VLM cache, so a re-run is fast and spends no
tokens - fine for accuracy comparisons (same prompts, same replies), meaningless for latency and cost. `--fresh`
points `llm.cache_dir` at an empty temp dir for the run. Cost and latency come from the trace database (every
eval case is a traced request under batch `eval-<suite>-<label>-<ts>`): p50/p95 latency, fresh tokens, cached calls.

**How labels were made, and their limits.**
- router: generated from the source files by `eval/build_router_set.py` (rendered sheets with rotation / blur /
  JPEG / perspective / crop / photo-like, report pages, figure crops, charts, text questions). The label comes
  from where the case was made, so no hand labelling and no label noise. Limit: all synthetic; `real/` takes phone
  photos.
- qa_text test (15) and qa_image (20): drafted by the AI assistant from the report text; every `must` fact and page
  checked programmatically against the PDF text. **Not reviewed by a human** (user decision, 2026-10-18): the
  qa_text test split (t01-t15) is used as drafted, so a wrong expected answer would count against the agent;
  the qa_image cases were not reviewed either. The 10 picture + question cases added later (iq01-iq10: diagnosis,
  lookup, refusal, wrong_aircraft; 5 dev / 5 test) are agent-generated and not human-reviewed either (user decision,
  2026-10-18); their `must` facts were checked against the stated pages, and wrong_aircraft `must_not` values were
  checked to be the *other* aircraft's values in the report. Expected wrong_aircraft behaviour: answer for the aircraft
  the question names (a picture can't show a registration), never the pictured one's value. Every qa_image case has
  `input: image | image+text`; the suite reports all metrics per input type and per kind. Frozen as of 2026-10-18 -
  sha256 `qa_text.jsonl` ebac121c9489..., `qa_image/cases.jsonl` 1fda89c8d252... (every run records the dataset
  hash, so a later edit shows in `--compare`).
  Weak spots seen on a read-through: t03 / t10 check bare numbers ("23", "80.0") that can match by accident;
  some qa_image facts are single vague words ("heat", "grain"). Limit: qa_image cases
  are the reports' own figures (caption cropped off, augmented) - in-distribution; a real user's photo of a new
  part is harder. No out-of-scope photos could be made offline (folder + README for real ones).
- drawing test (CTC-03, FTC-11): drafted from the rendered sheets and text layer - not from the extractor's output,
  so the test isn't contaminated - and `verified: false` until the user checks the overlays.
- `belongs_to`: the lenient metric accepts any keyword from a list ("hole" passes for any hole). Added
  `belongs_primary` (the GT's main feature word must appear). A location-based check (does the leader point at the
  right feature?) isn't possible: the output JSON stores where the annotation is, not where its leader ends.

**Drawings: a reference run next to the hand labels** (user decision). Only 2 drawings have verified hand labels,
so the L4 run `geo-l4-fix4` (11 drawing JSONs, vLLM bf16, geometry method; copied to
`eval/datasets/drawings/reference/geo-l4-fix4/` with PROVENANCE.md, since experiments/ is gitignored) is the baseline
for all 11. `python -m eval drawing` reports both: **accuracy** on the verified sheets (CTC-01, FTC-07; the drafted
CTC-03 / FTC-11 labels shown separately as draft labels, not in the headline, until verified) and **agreement** with
fix4 on every drawing, with the same matcher: agree_recall / agree_precision / numbers_agree / type_agree / gdt_agree /
datums_agree / belongs_to_agree, per value dropped vs reference / new vs reference / changed reading; `--compare` flips
include reference values reproduced or dropped. Agreement measures *change relative to fix4, not correctness*: fix4
has mistakes, and a version that fixes one shows lower agreement - so it is never read without the accuracy next to
it. The picture route is compared with fix4's own picture-route outputs where they exist (CTC-01, FTC-07), else with
its PDF outputs (flagged). e2e drawings without hand labels (CTC-02, a photo of FTC-09 sheet 1, CTC-05) pass on
agreement >= 0.7. Checks: fix4 scored against itself = 1.0 on every agreement metric; a copy with 2 values dropped and
1 number changed shows exactly 2 "dropped vs reference" (2 flips) and 1 "changed reading (numbers)"; fix4's accuracy
on CTC-01 + FTC-07 reproduces its original results files exactly (PDF 0.900 / 0.892 recall / precision, JPG 0.643 / 0.807).

**Sizes.** router 80 (17 dev / 63 test, 5.4 MB of pictures), qa_text 35 (20 / 15), qa_image 30 (15 / 15: 17 picture only, 13 picture + question),
drawing 4 labelled drawings (2 / 2; 30 + 12 required values in the new ones) + the fix4 reference on all 11
(515 reference values excluding title block / notes), e2e 21 (9 / 12).

**First real numbers** (Mac, Ollama qwen2.5vl 4-bit - not comparable to the L4 runs; tiny samples, a smoke check of
the harness, not a result): router test n=15: accuracy 0.93, 0 drawings sent to QA, 1 report figure page (photo
look) sent to the drawing pipeline; rules decided 2/15, the VLM 13/15 at ~11 s each. qa_text test n=3: 3/3 pass.
Seen on the way: report 24-013's PDF goes to the VLM tier - its first page is a near-empty cover, the rules only
look at page 1.

**Status.** Done (verified with a stubbed model: every suite end to end, `--compare` deltas + flips incl. a forced
flip, `--set` changes the config hash, `--fresh` replays nothing, old results files compare where metrics match):
harness, five suites, router / qa_image generators, qa_text test split, drawing test GT (unverified).
Planned: user check of drawing GT and the QA test set (then frozen), real phone photos, L4 numbers per suite.

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

## Chat front end (summary)

Source: `app/chat.py` (~110 lines). Run: `.venv/bin/python app/chat.py` -> http://127.0.0.1:7860.

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

## Chunking and retrieval of the reports (summary)

Source: `src/docagent/ingest/reports.py` (ingestion), `retriever.py` (search), `pipelines/report_qa.py`
(how chunks reach the prompt). Run once: `docagent ingest data/reports`.

### Ingestion overview
- Per PDF: sha256 -> skip if already in the index (`has_document`), else delete the old version of that doc and
  re-ingest. Chunk ids are deterministic (`<doc>:p<page>:t<n>`, `<doc>:p<page>:fig<N>`), so re-ingesting overwrites.
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
     "accident" but names no aircraft, the report's registration is appended ("... (N106ER)"). This is the fix
     for multi-aircraft report 18-061 (answers mixed up N104ER / N106ER).
3. **Filtering**: skip captions, skip short blocks (< 200 chars) on figure / chart pages (those are labels
   printed on the figure, not body text), skip blocks < 40 chars and "intentionally left blank".
4. **Runs**: consecutive blocks with the same (section, subject) form a run. A chunk never mixes two subjects.
5. **Size**: greedy merge of paragraphs within a run up to `embedding.max_chunk_tokens` = 350 bge-m3 tokens;
   a paragraph longer than that is split on sentence ends (`.` / `;`). No overlap between chunks - the
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
5. **VLM description** (`fig-v1` prompt): 3-6 factual sentences on component, view and visible conditions.
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

### Retrieval (`HybridRetriever.search`, version `hybrid-rrf-v3-textfirst-reg-focus-next`)
1. **Query**: user text + per user picture the VLM's *observations* (component + visible features, no
   failure-mode guess - the guess biased retrieval). No text and no picture -> "What failed and why?".
2. **Two hybrid searches** in the one collection, each dense (top 30) + sparse (top 30) fused with RRF in Qdrant:
   one over all chunks, one filtered to `kind = text`. Pool = both, k x 2 each (top_k = 8), deduplicated,
   sorted by score.
3. **Other-aircraft filter**: if the question names a registration (N106ER), chunks whose `subject` names a
   different one are dropped.
4. **Report focus**: if a term in the question (place, registration, report no., type) matches the header of
   exactly one report (whole word), only that report is kept.
5. **Text first**: at least `min_text` = 4 text chunks; figures fill the rest but at most half of the k slots;
   text comes before figures. (Figure captions used to outrank the paragraphs that hold the findings.)
6. **Next paragraph**: each of the top 3 text hits pulls in the chunk after it (next on the page, else first
   on the next page), inserted right after it (`via = linked`).
7. **Linked figures**: figures cited by the top 3 text hits (up to 2 each) are added from the store
   (`via = linked`, score 0), same other-aircraft filter.

### In the answer prompt (`ReportQA.generate`)
- Sources numbered `[n] report <doc>, page P, <text | Figure N>, section ..., about: <subject>` + evidence text.
- For figure chunks the evidence is **caption + labels only**: the citing paragraphs are cut off
  (`evidence_text`), since those paragraphs are separate sources and showing them twice steered answers and let
  a figure "back" a number that is really on a text page. The VLM description is not shown.
- Up to `max_context_figures` = 2 figure crops are sent as images (after the user's own pictures), each tied to
  its source number.
- After generation, `verify_citations` drops citations of other aircraft, adds the source that contains each
  distinctive number in the answer, and adds the paragraph the answer is nearly quoting.

### Known limits
- Regexes for headers, captions, sections and subsections are tuned to the NTSB layout; a new document family
  needs its own parser (or a layout model).
- Tables are flattened into text blocks; table structure is lost.
- 350-token chunks with no overlap rely on the next-paragraph step for findings that span chunks.
- Report focus and the aircraft filter are string matches on headers/registrations; at 10^6 similar documents
  they become metadata filters (doc type, vehicle, part) in Qdrant rather than post-filtering.
