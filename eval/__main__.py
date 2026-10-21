"""python -m eval <suite> [options]   suites: router | qa_text | qa_image | drawing | e2e

  python -m eval router --split test --n 20
  python -m eval qa_text --config configs/colab_vllm.yaml --label v4 --compare eval/results/<earlier>.json
  python -m eval drawing --set drawing.method=tiles --label tiles
  python -m eval e2e --n 5 --fresh

Common options (every suite): --config --set --n --seed --ids --split dev|test|all --label --compare --fresh.
`python -m eval <suite> --help` for suite-specific options. Design: NOTES.md
"""
import importlib
import sys

from eval.suites import SUITES


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in SUITES:
        print(__doc__)
        raise SystemExit(0 if argv and argv[0] in ("-h", "--help") else 2)
    suite = importlib.import_module(f"eval.suites.{argv[0]}")
    from eval.core import run_suite
    run_suite(suite, argv[1:], getattr(suite, "add_args", None))


if __name__ == "__main__":
    main()
