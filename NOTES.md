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


## 2. Model and infrastructure choices

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


## 3. Bugs found and fixed (measured)

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

## 4. Drawing extraction iterations

| version | approach | CTC-01 (1 sheet) | observation |
|---|---|---|---|
| probe | whole sheet @110 dpi, free text | 23 s | ~half the values missed or invented |
| probe | one 200 dpi tile, free text | 15 s | values read correctly |
| v1 | overview + 4 tiles, every field from VLM | not completed | estimated 4-8 min/sheet (output tokens dominate at ~16 tok/s) |
| v2 | READ tiles (text+type+box) -> code parses numbers -> LOCATE with numbered marks on the sheet -> code writes `meaning` | **182 s**, 43 values | readable values mostly right; boxes only roughly placed; one frame split into invented letter boxes; title-block text leaked; frames mistyped as diameters |
| v2.1 | + rules: retype by printed text (value + datum letters = frame), drop title/notes text, angle units | - | fixes the systematic typing errors |

Timing estimate for all 30 sheets on the laptop: ~90 min. On a GPU with vLLM batching: minutes.

## 5. Report QA spot checks (no eval set yet)

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
