"""Human-readable rendering of a Response (chat UI, CLI)."""
from __future__ import annotations

from collections import Counter

from .schemas import Response


def render_markdown(resp: Response) -> str:
    parts: list[str] = []
    a = resp.report_answer
    if a:
        if not a.grounded:
            parts.append("_The reports do not clearly cover this; treat the answer with caution._")
        parts.append(a.answer)
        if a.diagnosis:
            parts.append(f"**What is wrong:** {a.diagnosis}")
        if a.recommended_actions:
            parts.append("**What to do** (inferred from the report findings):\n"
                         + "\n".join(f"- {x}" for x in a.recommended_actions))
        if a.citations:
            parts.append("**Sources:** " + "; ".join(
                f"{c.doc_id} p.{c.page}" + (f" (Fig. {c.figure_no})" if c.figure_no else "")
                for c in a.citations))
    for d in resp.drawings:
        types = Counter(v.type for v in d.values)
        reused = f" _Reused result from request {resp.reused[d.drawing_id]} (same file, same agent version)._" \
            if d.drawing_id in resp.reused else ""
        parts.append(f"**Drawing `{d.drawing_id}`**: {len(d.values)} values extracted "
                     f"({', '.join(f'{n} {t}' for t, n in types.most_common())}); units: {d.units or 'unknown'}."
                     + reused)
    if resp.errors:
        parts.append("**Errors:** " + "; ".join(resp.errors))
    if resp.notes:
        parts.append("_Notes: " + "; ".join(resp.notes) + "_")
    parts.append(f"<sub>request {resp.request_id} · routes: "
                 + ", ".join(f"{r.route.value} ({r.tier})" for r in resp.routes) + "</sub>")
    return "\n\n".join(parts)
