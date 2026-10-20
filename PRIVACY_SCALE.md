# Privacy and scale

## Privacy: nothing leaves the system

**Every component runs inside the system boundary.** Internet is used once, at setup (packages and model
weights). After that:

| Component | Runs where | Network |
|---|---|---|
| VLM: Qwen2.5-VL 7B | Ollama (laptop) or vLLM (GPU server), self-hosted | Bound to `127.0.0.1` (`scripts/start_vllm.sh --host 127.0.0.1`; Ollama's default) |
| Embeddings: bge-m3 | In-process (CPU / CUDA) | `privacy.offline: true` sets `HF_HUB_OFFLINE` and `TRANSFORMERS_OFFLINE`, so no model downloads or update checks |
| Vector store: Qdrant | Embedded in `store/` (or an in-house Qdrant server) | None |
| Traces, results, uploads | SQLite `store/traces.db`, `outputs/`, `store/uploads/` | None: no external tracing service. Opik or Langfuse cloud were rejected because they send prompts and documents out. |
| Chat UI: Gradio | Local process | Bound to `127.0.0.1`, `share=False`, `GRADIO_ANALYTICS_ENABLED=False` |

**The agent has no tool that reaches outside.** The VLM only returns text or JSON to our code. It can't browse,
fetch or send anything. A document containing injected instructions can at worst change an answer; it has no
way to send data out.

**Untrusted input is contained.** Each upload is its own task, so a malformed file fails only itself. Zips
are unpacked by content: members with `..` or absolute paths are skipped, as are hidden files and
unsupported types. Archives over 5,000 members or 4 GB are refused.

**What is not built (operator decisions, listed so they aren't mistaken for done):**
- **Network enforcement.** The offline settings are application-level. The real guarantee is a host or
  network policy that denies all egress from the model, app and store hosts. That is also how to verify the
  claim: run the eval suites with egress blocked.
- **Retention.** `store/traces.db` (request texts and answers), `outputs/`, `store/uploads/` and the model reply
  cache (`store/cache/`) are copies of user data that grow without limit. They need the same retention and
  access rules as the source documents, for example deleting results and request rows after N days.
- **Access control and encryption at rest.** The chat has no login, and `store/` isn't encrypted. In
  production: SSO in front of the UI, an encrypted volume, and tracing data readable by the technical team only.
- **Development exceptions.** Colab and AI assistants saw the 14 public documents, as the brief allows. The
  Colab notebook uses a public Gradio share link for convenience; the local notebook and the production
  setup do not.

## Scale: a million similar documents

**Drawings don't touch the corpus.** They aren't indexed; each one is processed when uploaded. Cost is linear
in the number of drawings and doesn't depend on corpus size. Each file is independent, so the work splits
across as many workers and GPUs as there are.

| Measured on one L4 GPU, vLLM bf16, cache off | Per drawing | Per report question |
|---|---|---|
| Latency (p50 / p95) | 218 s / 316 s | 11 s / 15 s |
| VLM calls | ~52 | 1 |
| Tokens (prompt + completion) | ~38,000 | ~3,600 |

Source: `eval/results/20261019-120306_drawing_baseline-l4.json` (all 11 drawings, 30 sheets, PDF route, one
drawing at a time) and `20261019-104621_qa_text_baseline-l4.json` (35 questions). The same drawings with AWQ
4-bit weights: 92 s / 196 s per drawing, at lower accuracy (`20261019-133240_drawing_awq-l4.json`).

At one drawing at a time, a million drawings would take about 2,500 GPU-days (bf16). That's an upper bound: each
drawing's reads run in parallel (`llm.max_concurrency`), several drawings run at once (`batch.workers`), and
vLLM batches all of them on the GPU. Throughput with many concurrent drawings per GPU hasn't been measured yet;
it's the next number to get.

**Work is only done once.** A file identical byte for byte (sha256), with the same agent version and config,
reuses its earlier result. Bulk runs queue every file in the database before processing, so a crash resumes
only the unfinished files (`bulk --resume`). Ingest skips unchanged reports.

**Routing is cheap.** Deterministic rules on page 1 (keywords, orientation, vector-path count, image coverage)
decide most PDFs without a model call. The VLM is called only when the rules can't decide, and always for
photos.

**Retrieval, where "all very alike" matters most:**
- *Size.* The 3 reports produced ~94 chunks. At that rate, a million reports would be ~3×10⁷ dense vectors:
  ~120 GB in fp32 at 1,024 dimensions, ~30 GB with int8 quantization.
- *Today's store has no index.* Embedded ("local") Qdrant is a pure-Python stand-in for the server: every query
  scores **every** stored vector, dense and sparse, with numpy and sorts all the scores, and all vectors live in
  the process's memory. HNSW settings are accepted but ignored. At 94 chunks that is exact and instant
  (`qa.retrieve` ≈ 0.09 s in the traces); at 3×10⁷ vectors, query time and RAM grow linearly and it stops
  working. It is a development store, not the scale design.
- *At a million documents: the Qdrant server.* Same client API, collections and RRF fusion; the only change is
  `store.qdrant_url` instead of `store.qdrant_path`, and the server does the indexing:
  - **HNSW** for the dense vectors: a graph index that finds approximate nearest neighbours in roughly
    logarithmic time instead of scanning everything (tunable recall vs speed with `m` / `ef`).
  - **An inverted index** for the sparse (lexical) vectors, so exact identifiers stay fast to match.
  - **int8 scalar quantization** kept in RAM for the search (~30 GB instead of ~120 GB), with the full-precision
    vectors **on disk** for re-scoring the top hits.
  - **Sharding and replicas** across nodes when one machine is not enough: shards split the vectors, replicas
    add query throughput and fail-over.
  - **Payload indexes** on document type, aircraft, part and date, so filters narrow the search inside the
    index instead of after it.
  - Ingestion uploads in batches; the HNSW graph is built as segments fill, so a million documents can be
    indexed incrementally without downtime.
- *Near-duplicates.* With very similar documents, the top results fill up with copies of the same finding.
  Hybrid search already helps: the sparse vectors match exact identifiers such as registrations and part
  numbers. The aircraft and report filters are string matches today and should become Qdrant payload filters
  (document type, vehicle, part, date) applied *before* search, plus de-duplication of near-identical chunks at
  ingest.
- *Ingest cost.* It's dominated by one VLM description per figure (~50 figures, ~15 minutes on a laptop). It's
  a one-time batch job per document and parallelizes on GPUs like drawings do.

**The single-host parts and what replaces them:**

| Today | At scale |
|---|---|
| Embedded Qdrant: exact scan of every vector, no index, all vectors in memory, one process at a time | Qdrant server cluster: HNSW + sparse inverted index, int8 quantization, vectors on disk, sharded and replicated |
| SQLite trace database. ~50–150 spans per drawing means ~10⁸ rows per million drawings | Postgres with spans partitioned by day, a retention period on spans (requests and tasks kept), or a columnar store |
| Each worker plans its own batch from a snapshot | Workers claim queued rows (`SELECT … FOR UPDATE SKIP LOCKED`) or read from a queue, so stateless workers scale horizontally |
| One vLLM server | Several vLLM replicas behind a load balancer. The agent only needs `llm.base_url` |

**Evaluation scales independently of the corpus.** Each suite runs a chosen, seeded sample (`--n`, `--seed`),
so checking a new version costs the same whether the corpus has 14 documents or a million.
