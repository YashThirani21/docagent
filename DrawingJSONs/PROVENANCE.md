# Reference run: geo-l4-fix4 (model outputs, NOT verified labels)

Baseline that every new drawing extractor version is compared to (user decision, 2026-10-18: "even if not perfect").
These are the extractor's own outputs - they contain its mistakes. Agreement with them measures **change relative to
fix4, not correctness**: a new version that fixes a fix4 mistake shows *lower* agreement. That is why
`python -m eval drawing` reports accuracy on the verified ground-truth sheets next to agreement.

fix4's own score (PDF 0.900 / 0.892 recall / precision) rests on **2 of the 11 drawings**: CTC-01 (1 sheet, isometric,
metric, 17 values) and FTC-07 (4 sheets, inch, real text layer, 53 values) - 70 values, the only labelled sheets at
the time. The other 9 JSONs were not scored, only spot-checked visually on the annotation overlays.

- source run: `experiments/20261018-121642_geo-l4-fix4/` (gitignored, so copied here)
- PDF route (this folder, all 11 drawings): `bulk/drawings/*.json` - `docagent bulk data/drawings` on Colab L4
- picture route (`jpg/`, CTC-01 and FTC-07 only - the only sheets fix4 ran as pictures):
  `eval/drawing_runs/20261018-122437_geo-l4-fix4-jpg_jpg/*.json` (each sheet as a 200-dpi JPG)
- config `configs/colab_vllm.yaml`, model `qwen2.5-vl-7b` (vLLM, bf16 on the L4), drawing method `geometry`,
  extractor_version `drawing-geometry-v1`, Python 3.13, run started 2026-10-18 12:16
- its scores against the label files (CTC-01 + FTC-07) are in `eval/results/20261018-122437_drawings_geo-l4-fix4-pdf.json`
  and `...-jpg.json` (copied from the run's `eval/` folder, original names; copied 2026-10-20). Old format:
  `--compare` against them shows metric deltas only, no per-value flips
- **`--compare` baselines (use these):** `eval/results/20261020-032010_drawing_geo-l4-fix4-pdf.json` and
  `eval/results/20261020-032011_drawing_geo-l4-fix4-jpg.json` - these JSONs rescored with the current harness
  (`python -m eval drawing --config configs/colab_vllm.yaml --pred-dir <this folder>[/jpg]`, no model calls). They
  reproduce the original scores exactly (PDF 0.900 / 0.892, JPG 0.643 / 0.807) and give per-value flips. Timestamp
  and `env` are those of the rescoring (Mac), not of the L4 run

PDF route

| drawing | sheets | values | extractor_version | units |
|---|---|---|---|---|
| nist_ctc_01_asme1_rd | 1 | 25 | drawing-geometry-v1 | millimeters |
| nist_ctc_02_asme1_rc | 3 | 67 | drawing-geometry-v1 | millimeters |
| nist_ctc_03_asme1_rc | 1 | 38 | drawing-geometry-v1 | inches |
| nist_ctc_04_asme1_rd | 1 | 27 | drawing-geometry-v1 | millimeters |
| nist_ctc_05_asme1_rd | 2 | 33 | drawing-geometry-v1 | inches |
| nist_ftc_06_asme1_rd | 3 | 90 | drawing-geometry-v1 | inches |
| nist_ftc_07_asme1_rd | 4 | 67 | drawing-geometry-v1 | inches |
| nist_ftc_08_asme1_rc | 4 | 69 | drawing-geometry-v1 | inches |
| nist_ftc_09_asme1_rd | 4 | 82 | drawing-geometry-v1 | inches |
| nist_ftc_10_asme1_rb | 5 | 102 | drawing-geometry-v1 | millimeters |
| nist_ftc_11_asme1_rb | 2 | 25 | drawing-geometry-v1 | millimeters |

Picture route (`jpg/`)

| drawing | values | coord_units |
|---|---|---|
| nist_ctc_01_asme1_rd | 20 | px |
| nist_ftc_07_asme1_rd | 72 | px |

Source manifest:

```json
{
 "name": "geo-l4-fix4",
 "started": "20261018-121642",
 "config": "configs/colab_vllm.yaml",
 "model": "qwen2.5-vl-7b",
 "drawing_method": "geometry",
 "python": "3.13.15",
 "steps": {
  "check": {
   "ok": true,
   "returncode": 0,
   "seconds": 4,
   "vllm_up": true,
   "models_downloaded": true
  },
  "bulk": {
   "ok": true,
   "returncode": 0,
   "seconds": 467,
   "drawing_jsons": 11
  },
  "eval_pdf": {
   "ok": true,
   "returncode": 0,
   "seconds": 0,
   "results": [
    "20261018-122437_drawings_geo-l4-fix4-pdf.json"
   ]
  },
  "eval_jpg": {
   "ok": true,
   "returncode": 0,
   "seconds": 67,
   "results": [
    "20261018-122437_drawings_geo-l4-fix4-jpg.json"
   ]
  }
 }
}
```
