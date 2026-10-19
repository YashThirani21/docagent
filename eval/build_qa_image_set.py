"""Generate the picture-QA eval set from the reports' own figures: python -m eval.build_qa_image_set

Each case = one report figure cropped WITHOUT its caption, lightly augmented (rotation, JPEG, blur, photo look),
optionally with a user question. Expected: the report and figure it comes from, facts the answer must contain
and claims it must not make - written from the caption and the report sentences citing the figure (drafted by the
AI assistant, see NOTES.md section 10). The script checks that every `must` fact occurs in that report's text and
every `must_not` claim does not, so a typo or an invented fact fails the build.
Limit: in-distribution - the pictures are the corpus's own figures (printed labels like "Corrosion" stay visible).
Split: alternating dev / test in the order below (figure cases and picture + question cases separately).
Output: eval/datasets/qa_image/cases.jsonl + images/.
"""
from __future__ import annotations

import json
import random
import re

import pymupdf

from docagent.config import PROJECT_ROOT
from eval.build_router_set import augment, figure_rects, render
from eval.run_qa import _norm

OUT = PROJECT_ROOT / "eval" / "datasets" / "qa_image"
# (doc, figure, problem shown?, must, must_not, question or None)
SPEC = [
    ("ntsb_24-013", 2, True, ["overstress"], ["corrosion"], "This part broke in flight. Why?"),
    ("ntsb_24-013", 3, True, ["overstress"], ["corrosion"], None),
    ("ntsb_24-013", 4, True, ["overstress"], ["corrosion"], None),
    ("ntsb_16-006", 1, True, ["heat"], [], None),
    ("ntsb_16-006", 3, False, ["pawl"], [], None),
    ("ntsb_16-006", 7, False, ["inspection hole"], [], None),
    ("ntsb_16-006", 9, False, ["release rod"], [], None),
    ("ntsb_16-006", 12, True, ["corrosion"], [], "The release rod on this control will not move. What is wrong and what should we check?"),
    ("ntsb_18-061", 2, True, ["fatigue"], ["corrosion"], None),
    ("ntsb_18-061", 5, True, ["fatigue", "LC-1"], ["corrosion"], None),
    ("ntsb_18-061", 6, True, ["fatigue"], ["corrosion"], None),
    ("ntsb_18-061", 8, True, ["fatigue"], ["corrosion"], None),
    ("ntsb_18-061", 10, True, ["fatigue"], ["corrosion"], None),
    ("ntsb_18-061", 13, False, ["grain"], ["corrosion"], None),
    ("ntsb_18-061", 15, True, ["overstress"], ["corrosion"], "What happened to this spar cap?"),
    ("ntsb_18-061", 16, True, ["fatigue"], ["corrosion"], None),
    ("ntsb_18-061", 17, True, ["fatigue"], ["corrosion"], None),
    ("ntsb_18-061", 19, True, ["LC-1"], ["corrosion"], None),
    ("ntsb_18-061", 20, True, ["fatigue"], ["corrosion"], None),
    ("ntsb_18-061", 23, False, ["N3986M"], ["corrosion"], None),
]
AUGS = ["clean", "rotate", "jpeg", "photo", "blur"]
# Picture + question cases (agent-generated, not human-reviewed - user decision 2026-10-18). Same figures, another
# augmentation. kind: diagnosis (problem shown, must = the report's conclusion) | lookup (a fact about the pictured
# part: no diagnosis/actions expected - scored for filler) | refusal (not in the reports) | wrong_aircraft (the
# question names a different aircraft than the figure shows). Expected behaviour for wrong_aircraft: answer for the
# aircraft the user NAMES - a picture of a spar cap can't show its registration, the text can - and never give the
# pictured aircraft's value; `must_not` holds the other aircraft's values, which must exist in the report (a trap,
# not an invented claim). `pages`: where the expected fact is stated (checked below, like qa_text).
QA = [
    dict(doc="ntsb_24-013", fig=4, kind="diagnosis", q="Why did this throttle arm break?",
         must=["overstress"], must_not=["corrosion"], pages=[3]),
    dict(doc="ntsb_18-061", fig=2, kind="diagnosis", q="What went wrong with this wing spar?",
         must=["fatigue"], must_not=["corrosion"], pages=[2]),
    dict(doc="ntsb_16-006", fig=12, kind="diagnosis", q="Why would this control be stuck?",
         must=["corrosion"], must_not=[], pages=[4]),
    dict(doc="ntsb_18-061", fig=13, kind="lookup", q="What alloy is this spar made of?",
         must=["2024-T3511"], must_not=[], pages=[1]),
    dict(doc="ntsb_18-061", fig=6, kind="lookup", q="At which bolt hole did this fatigue crack start?",
         must=["LC-1"], must_not=[], pages=[2]),
    dict(doc="ntsb_16-006", fig=3, kind="lookup", q="How far did the release rod extend from the nut in the locked position?",
         must=["0.733"], must_not=[], pages=[2]),
    dict(doc="ntsb_24-013", fig=2, kind="refusal", q="What torque was this throttle arm's nut installed with?",
         must=[], must_not=[], pages=[]),
    dict(doc="ntsb_18-061", fig=15, kind="refusal", q="How many flight hours did the pilot of this airplane have?",
         must=[], must_not=[], pages=[]),
    dict(doc="ntsb_18-061", fig=5, kind="wrong_aircraft", q="What hardness was measured on this N104ER spar cap?",
         must=["80.8"], must_not=["81.0", "79.9"], pages=[8]),
    dict(doc="ntsb_18-061", fig=19, kind="wrong_aircraft", q="What conductivity was measured on the N106ER spar shown here?",
         must=["29.3"], must_not=["29.8", "29.7"], pages=[4]),
]


def build() -> list[dict]:
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    for old in (OUT / "images").glob("*"):
        old.unlink()
    rng = random.Random(11)
    figs, text, page_text = {}, {}, {}
    for doc_id in {s[0] for s in SPEC} | {q["doc"] for q in QA}:
        with pymupdf.open(PROJECT_ROOT / "data" / "reports" / f"{doc_id}.pdf") as doc:
            text[doc_id] = _norm(" ".join(" ".join(p.get_text().split()) for p in doc))
            for pno, page in enumerate(doc, 1):
                page_text[(doc_id, pno)] = _norm(" ".join(page.get_text().split()))
                for fig, rect in figure_rects(page):
                    figs[(doc_id, fig)] = (pno, render(page, 150, clip=rect))
    cases, problems = [], []
    for i, (doc_id, fig, problem, must, must_not, question) in enumerate(SPEC):
        if (doc_id, fig) not in figs:
            problems.append(f"{doc_id} figure {fig}: not found")
            continue
        problems += [f"{doc_id} fig {fig}: must {m!r} not in the report" for m in must if _norm(m) not in text[doc_id]]
        problems += [f"{doc_id} fig {fig}: must_not {m!r} IS in the report" for m in must_not
                     if _norm(m) in text[doc_id]]
        pno, img = figs[(doc_id, fig)]
        aug = AUGS[i % len(AUGS)]
        img = augment(img, aug, rng).convert("RGB")
        s = min(1.0, 1200 / max(img.size))
        if s < 1:
            img = img.resize((int(img.width * s), int(img.height * s)))
        name = f"{doc_id[5:]}_fig{fig}_{aug}.jpg"
        img.save(OUT / "images" / name, "JPEG", quality=88)
        cases.append({"id": f"img-{doc_id[5:]}-f{fig}", "image": f"eval/datasets/qa_image/images/{name}",
                      "question": question, "doc_id": doc_id, "figure_no": fig, "page": pno, "problem": problem,
                      "must": must, "must_not": must_not, "augmentation": aug,
                      "split": "dev" if i % 2 == 0 else "test", "kind": "figure",
                      "input": "image+text" if question else "image"})
    for i, q in enumerate(QA):             # after the 20 above: their random draws (and files) stay unchanged
        doc_id, fig = q["doc"], q["fig"]
        if (doc_id, fig) not in figs:
            problems.append(f"{doc_id} figure {fig}: not found")
            continue
        for m in q["must"]:
            if not any(_norm(m) in page_text[(doc_id, pg)] for pg in q["pages"]):
                problems.append(f"{doc_id} fig {fig} ({q['kind']}): must {m!r} not on pages {q['pages']}")
        for m in q["must_not"]:
            present = _norm(m) in text[doc_id]
            if q["kind"] == "wrong_aircraft" and not present:
                problems.append(f"{doc_id} fig {fig}: must_not {m!r} (other aircraft's value) not in the report")
            if q["kind"] != "wrong_aircraft" and present:
                problems.append(f"{doc_id} fig {fig}: must_not {m!r} IS in the report")
        pno, img = figs[(doc_id, fig)]
        aug = AUGS[(i + 2) % len(AUGS)]
        img = augment(img, aug, rng).convert("RGB")
        s = min(1.0, 1200 / max(img.size))
        if s < 1:
            img = img.resize((int(img.width * s), int(img.height * s)))
        name = f"{doc_id[5:]}_fig{fig}_q{i + 1}_{aug}.jpg"
        img.save(OUT / "images" / name, "JPEG", quality=88)
        cases.append({"id": f"iq{i + 1:02d}-{doc_id[5:]}-f{fig}", "image": f"eval/datasets/qa_image/images/{name}",
                      "question": q["q"], "doc_id": doc_id, "figure_no": fig, "page": pno,
                      "problem": q["kind"] == "diagnosis", "must": q["must"], "must_not": q["must_not"],
                      "pages": [[doc_id, pg] for pg in q["pages"]], "augmentation": aug,
                      "split": "dev" if i % 2 == 0 else "test", "kind": q["kind"], "input": "image+text"})
    if problems:
        raise SystemExit("dataset problems:\n  " + "\n  ".join(problems))
    (OUT / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    return cases


if __name__ == "__main__":
    cs = build()
    size = sum(p.stat().st_size for p in (OUT / "images").glob("*")) / 1e6
    print(f"{len(cs)} cases ({sum(c['split'] == 'dev' for c in cs)} dev / {sum(c['split'] == 'test' for c in cs)} test), "
          f"images {size:.1f} MB; every must fact found in its report / on its pages, must_not checked")
