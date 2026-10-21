# docagent: a private document agent

A chat agent for industrial technical documents. It runs entirely on your own machine, with no
internet access after setup. You give it text, a photo, a PDF or several files at once, and it works
out what to do with them on its own:

| You send | The agent does |
|---|---|
| A question, or a photo of a damaged part | Says **what is wrong and what to do**, answered from the NTSB Materials Laboratory reports in `data/reports/`, with citations |
| A technical drawing (PDF or picture) | Returns a **JSON file listing every value on the drawing**, with what each value belongs to and what it means |
| Many drawings at once (or a `.zip`) | Processes them as a **bulk batch** in the background, one result per file |

Every request is **traced**: a technical person can later look up what the agent did, which models
it called, how long each step took and what it produced. An **evaluation harness** measures each
part (routing, report answers, drawing reading) and the whole system, and compares a new agent
version against an old one.

---

## How it works (one minute)

```
            ┌──────────────┐
 request ──▶│    router    │  decides per attachment: report question / photo, or drawing
            └──────┬───────┘
         ┌─────────┴──────────┐
         ▼                    ▼
 ┌───────────────┐    ┌──────────────────┐
 │  report QA    │    │ drawing → JSON   │
 │ hybrid search │    │ find annotations,│
 │ (bge-m3 +     │    │ read each one    │
 │  Qdrant) +VLM │    │ with the VLM     │
 └───────┬───────┘    └────────┬─────────┘
         └──────────┬──────────┘
                    ▼
        response + trace in SQLite (store/traces.db)
```

- **VLM:** Qwen2.5-VL 7B, served locally by **Ollama** (laptop) or **vLLM** (GPU server / Colab).
- **Embeddings:** BAAI/bge-m3 (dense + sparse), on CPU on a laptop and on CUDA on a GPU server. Not on
  Apple's MPS: batched bge-m3 returns corrupted vectors there (see `NOTES.md`).
- **Vector store:** Qdrant, embedded in `store/` (or a Qdrant server at scale).
- **UI:** Gradio, bound to `127.0.0.1` only. Gradio analytics are off.

---

## Running from a notebook

`notebooks/colab_notebook_git.ipynb` runs the whole flow (setup, model server, ingest, chat, questions,
trace lookup, evaluation) from top to bottom on a Google Colab GPU runtime:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YashThirani21/docagent/blob/main/notebooks/colab_notebook_git.ipynb)

It clones the project from GitHub, starts vLLM, and runs with no setup or tokens. By default nothing is
kept between sessions and the run's results download as a zip; set `PERSIST` to keep `store/` and results
in your Google Drive (`"drive"`) or a private Hugging Face dataset (`"hf"`, needs an `HF_TOKEN` Colab
secret). On a local machine, use the commands below instead.

---

## Deliverables

| What | Where |
|---|---|
| JSON output for the 11 drawings | `DrawingJSONs/` (one `<drawing_id>.json` per drawing) |
| Evaluation data | `eval/datasets/` (router, qa_text, qa_image, drawings, e2e; dev and test splits) |
| Evaluation results | `eval/results/` (one JSON per run); bf16 vs AWQ on the L4: `eval/RESULTS_L4_bf16_vs_awq.md` |
| Working notes | `NOTES.md` |
| AI assistant session logs | `ai_logs/` |
| Privacy and scale | `PRIVACY_SCALE.md` |

---

## 1. Setup (once)

**You need:** Python 3.11, [Ollama](https://ollama.com) installed and running, and about 10 GB of free disk.
macOS or Linux.

```bash
# 1. Python environment and the docagent package
python3.11 -m venv .venv
.venv/bin/pip install -r requirements.txt     # pinned, tested versions
.venv/bin/pip install -e .                    # installs the `docagent` command

# 2. Models. This is the only step that uses the internet.
.venv/bin/python scripts/download_models.py   # bge-m3 embedding model (~2.3 GB)
scripts/setup_ollama.sh                       # Qwen2.5-VL 7B + a 16k-context variant "qwen2.5vl-16k"

# 3. Index the reports (drawings are never indexed; they are processed when uploaded)
.venv/bin/docagent ingest data/reports
.venv/bin/docagent status                     # check what is in the index
```

The first ingest takes about 15 minutes on a laptop, mostly because the VLM writes a description of
each of the ~50 figures in the reports. Later runs skip files that have not changed.

From here on the system runs offline: the config sets `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE`,
and every model is served from this machine.

> **Tip:** to avoid typing `.venv/bin/` each time, run `source .venv/bin/activate` first.

> **On an NVIDIA GPU machine (vLLM),** don't use `requirements.txt`. It pins torch for the Mac, and vLLM needs
> its own version. Install with the GPU constraints, i.e. the exact versions of a working Colab run (Python 3.13+):
>
> ```bash
> pip install vllm==0.31.0 -e . -c requirements-gpu.txt
> ```
>
> Then follow [Running on a GPU](#running-on-a-gpu-optional) for the model server. The two environments differ
> in a few packages (for example pymupdf 1.28 on the Mac vs 2.0 on the GPU), so each eval run records its
> package versions and `--compare` warns when they differ.

---

## 2. Chat

```bash
.venv/bin/python app/chat.py          # then open http://127.0.0.1:7860
```

Type a question, attach files, or both. You never have to tell it what you want. Try:

- **Text:** *"Why did the left wing main spar fail?"* gives an answer with citations to report pages.
- **Photo:** attach a picture of a fractured part, with or without a question. It explains what is
  wrong and what to do.
- **Drawing:** attach `data/drawings/nist_ctc_01_asme1_rd.pdf`. You get a JSON file to download, also
  saved under `outputs/requests/<request_id>/`.
- **Several drawings with no question, or a `.zip`:** these start a background batch. The reply shows
  a batch id and its progress, and offers a zip of all results when the batch is done. Closing the tab
  does not stop the batch. Send `status <batch_id>` to check on it again.

Every reply ends with its **request id**, which you need to look the request up later (section 4).

The same thing works without the UI:

```bash
.venv/bin/docagent ask "Why did the left wing main spar fail?"
.venv/bin/docagent ask -f data/drawings/nist_ctc_01_asme1_rd.pdf
.venv/bin/docagent ask "what is wrong here?" -f photo.jpg --json   # full response object
```

---

## 3. Bulk upload

```bash
.venv/bin/docagent bulk data/drawings --out outputs   # a folder: each file routed and processed
.venv/bin/docagent bulk uploads.zip                   # a zip, unpacked safely
```

- Each file becomes **its own request**, so one corrupt file fails only that file.
- The run prints a **batch id** (`bulk-YYYYMMDD-...`). Use it to check the batch or collect its results:

```bash
.venv/bin/docagent batch bulk-20261018-...                # per file: status, route, output, first error
.venv/bin/docagent batch bulk-20261018-... --zip out.zip  # every result + batch_summary.json
.venv/bin/docagent bulk --resume bulk-20261018-...        # after a crash or failures: runs only what is not done
```

- A drawing already processed with the same file, agent version and config is **reused**, not run
  again. Add `--force` to process it again.
- Uploaded files are stored by content in `store/uploads/`, so moving or deleting the originals does not
  break a retry or a resume.

> **One process at a time:** the embedded vector store allows a single process. If the chat app is
> running, a bulk run that contains photos or questions cannot open the store, but drawing-only bulk
> runs work fine. To run both at once, start a Qdrant server and set `store.qdrant_url` in the config.

---

## 4. Looking up a request

Every request (from chat, `ask`, `bulk` or an eval run) is stored in **`store/traces.db`** (SQLite).
For each request it holds the status, the route chosen for each file, every step and model call with
timings and token counts, the output files and any error with its traceback.

```bash
# find requests
.venv/bin/docagent requests                               # latest requests: status, time, LLM calls, tokens
.venv/bin/docagent requests --status failed --since 2h    # also: partial, interrupted, running, queued
.venv/bin/docagent requests --batch bulk-20261018-...     # everything in one bulk or eval run
.venv/bin/docagent requests --file nist_ctc_01            # every request that included this file

# inspect one
.venv/bin/docagent trace d59f9bbac916                     # what it did and called, step by step
.venv/bin/docagent trace last --llm                       # every model call of the most recent request

# re-run
.venv/bin/docagent retry d59f9bbac916                     # new attempt; finished parts are reused
```

Results are plain files in `outputs/requests/<request_id>/`: one JSON per drawing, or `answer.json` for
a report answer. Log lines in `store/logs/agent.jsonl` carry the same ids.

To browse the database in a web page (local, offline):

```bash
pip install datasette && datasette store/traces.db
```

---

## 5. Validating a new agent version

**An agent version is a config file.** `configs/default.yaml` holds every setting: models, context
length, retrieval top-k, drawing method and so on. To make a new version, copy it, change what you want
to test, and run the evaluation suites on both versions. Instead of `--config`, you can also set
`DOCAGENT_CONFIG=configs/v2.yaml`, which the CLI, the chat app and the notebook all read.

A change to one part's **code** (router rules, extractor, prompts) is validated the same way: run the suite on
the new code and `--compare` it with the earlier run. Every run records the component versions
(`Agent.versions`) and package versions, and `--compare` warns when they differ.

### The suites

| Suite | What it measures |
|---|---|
| `router` | Is each attachment sent to the right pipeline? |
| `qa_text` | Answers to text questions about the reports |
| `qa_image` | Answers to a photo, with or without a question |
| `drawing` | Values read from drawings: accuracy against 4 AI-drafted label files (2 verified dev, 2 unverified test), plus agreement with the fix4 baseline run (measures change, not correctness) |
| `e2e` | The whole agent on mixed requests |

Each suite has a **dev** split (cases the agent was tuned on) and a **test** split (held out). The
**default is `--split dev`**, so a tuning loop can't touch the test cases by accident; use `--split test` or
`--split all` for the held-out numbers. The datasets are in `eval/datasets/`.

An e2e case passes when every attachment is routed correctly, every component passes (for drawings: recall
and precision ≥ 0.7 against the labels, or agreement ≥ 0.7 with fix4 when a drawing has no labels) and
nothing errors.

### Compare a new version with the current one

```bash
# 1. score the current version
python -m eval qa_text --label baseline

# 2. make a new version
cp configs/default.yaml configs/v2.yaml     # then edit configs/v2.yaml

# 3. score the new version and compare it with the baseline
python -m eval qa_text --config configs/v2.yaml --label v2 \
    --compare eval/results/<timestamp>_qa_text_baseline.json

# the whole system, the same way
python -m eval e2e --label baseline
python -m eval e2e --config configs/v2.yaml --label v2 \
    --compare eval/results/<timestamp>_e2e_baseline.json
```

`--compare` prints the change in each metric, the cases that **broke** and the cases that were
**fixed**, and the change in cost and latency. It warns if the two runs used different datasets or
cases.

To change a single setting without making a new file:

```bash
python -m eval drawing --set drawing.method=tiles --label tiles \
    --compare eval/results/<timestamp>_drawing_geo.json
```

**Drawings: the comparison basis is the fix4 run, not manual annotation.** `geo-l4-fix4` is an L4 run
(Qwen2.5-VL-7B in BF16, geometry method), and its 11 drawing JSONs are model outputs with their mistakes.
Agreement with fix4 measures how much a version changed, not whether it is correct. fix4's own score
(0.900 / 0.892 recall / precision) covers only the 2 verified label files, CTC-01 and FTC-07. To compare a
new version with fix4:

```bash
python -m eval drawing --config configs/v2.yaml --label v2 \
    --compare eval/results/20261020-032010_drawing_geo-l4-fix4-pdf.json
```

Details: `eval/datasets/drawings/reference/geo-l4-fix4/PROVENANCE.md`. The BF16 vs AWQ results are in
`eval/RESULTS_L4_bf16_vs_awq.md`.

### Useful options

```bash
python -m eval router --split test --n 20 --seed 1    # 20 held-out cases, reproducible sample
python -m eval qa_text --ids q10 t05                  # specific cases only
python -m eval e2e --n 5 --fresh                      # --fresh: bypass the model cache for real latency and tokens
python -m eval <suite> --help                         # all options

# drawing suite only
python -m eval drawing --as-images --label geo-jpg    # picture route: each sheet as a JPG
python -m eval drawing --pred-dir outputs/drawings    # score drawing JSONs that already exist, no model calls
python -m eval drawing --reference <run>/drawings     # agreement with another reference run
python -m eval drawing --no-reference                 # accuracy on the labelled sheets only
```

Each run writes `eval/results/<timestamp>_<suite>_<label>.json`, containing the config hash, overrides,
component versions, model, dataset hash, case ids, seed, the scores, the cost and every case. Drawing and
e2e runs also keep their output files under `eval/results/runs/`. The run prints a
`docagent requests --batch eval-...` command so you can open the trace of each case.

**Datasets.** The generated sets are rebuilt deterministically with `python -m eval.build_router_set` and
`python -m eval.build_qa_image_set` (the latter checks every expected fact against the report text). To review
drawing labels, `python -m eval.review_drawing_gt` draws the unverified label files onto their sheets
(`eval/datasets/drawings/review/`, checklist in `CHECKLIST.md`); set `"verified": true` in a label file once
it has been checked.

**On Colab,** notebook step 17 runs any suite against the model vLLM is serving, with `--split all --fresh`,
and compares it with the latest saved run of the same model in `eval/results/`.

> **Caching:** model replies are cached on disk for each prompt, so re-running an accuracy comparison
> is fast and free. If you compare latency or tokens, use `--fresh` on both runs. If you put a new model
> behind the same `llm.model` name, the old cached replies would be replayed, so change the name or use
> `--fresh`.

---

## Running on a GPU (optional)

The same code runs with vLLM instead of Ollama. You only switch the config. **The config's `llm.model`
must equal the name vLLM serves (`SERVED`)**, or every model call fails with a 404 "model does not
exist" error:

| Config | Serves | Weights |
|---|---|---|
| `configs/colab_vllm_awq.yaml` | `qwen2.5-vl-7b-awq` | 4-bit AWQ on any GPU (fits a 16 GB T4) |
| `configs/colab_vllm.yaml` | `qwen2.5-vl-7b` | bf16 on GPUs with ≥ 20 GB, AWQ below |

Set the config once, and the start script and the agent both read it:

```bash
export DOCAGENT_CONFIG=configs/colab_vllm_awq.yaml   # the CLI, chat and eval read it (or pass --config)
bash scripts/start_vllm.sh                           # serves the config's llm.model, with the matching weights
```

The script refuses to start if an explicit `SERVED=` disagrees with the config. The agent also checks at
startup that the server offers the configured model and stops with a clear message if it doesn't. If the
server isn't up yet, it only logs a warning, because replies already in the cache still work. The
notebook reads the served name from the config as well. If you start vLLM
again with a different config, stop the old server first (`pkill -f vllm.entrypoints`): it keeps port
8000. In production, the same vLLM server would run on-premises with no outbound network.

---

## Project layout

```
app/chat.py              Gradio chat UI
src/docagent/
  agent.py router.py     request → route each attachment → response
  pipelines/             report_qa (answers from the reports) · drawing (drawing → JSON)
  ingest/                report PDFs → text chunks + figure crops → Qdrant
  retriever.py store.py embeddings.py   hybrid search (bge-m3 dense + sparse, RRF)
  llm.py                 OpenAI-compatible VLM client (Ollama / vLLM) with a disk cache
  trace.py               requests / tasks / spans in SQLite
  batch.py uploads.py    bulk batches, resume, safe zip handling
  cli.py                 the `docagent` command
eval/                    `python -m eval <suite>`: suites, datasets, results
configs/                 agent versions (default = Ollama on a laptop, colab_vllm / colab_vllm_awq = GPU)
notebooks/               colab_notebook_git (Colab, cloned from GitHub)
scripts/                 model download, Ollama setup, vLLM start
data/                    11 NIST drawings, 3 NTSB reports
outputs/                 results per request
DrawingJSONs/            the JSON output for the 11 drawings (deliverable)
store/                   index, uploads, cache, traces (not in git)
```

## More detail

- `PRIVACY_SCALE.md`: how the design keeps data in-house and handles a million documents.
- `NOTES.md`: what was tried, what was measured, and why each choice was made.