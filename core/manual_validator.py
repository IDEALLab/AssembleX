"""Validate generated assembly instruction manuals via a single VLM call.

For each step in an assembly:
  checklist = deterministic list of facts built from the Step's metadata
      (part name, motion direction/axis, tool, parts to hold steady).
  manual = the page produced by `assembly.manual.make_manual(step_idx)`,
      which dispatches to whichever backend is configured in
      `settings.manual_method`.
  verdict = ONE VLM call that looks at the manual image + the checklist
      and reports, per fact, whether the manual conveys it correctly.

Each step's record + an aggregate summary land under
    <assembly.storage_dir>/manual_validation/
        manual_validation.json   — machine-readable, full per-step detail
        manual_validation.txt    — human digest

A batch run also writes a cross-assembly digest via `write_batch_summary()`.

Token accounting honours `evaluation.tokens_exhausted(...)`/`tokens_used` the
same way the rest of the pipeline does; judge calls are content-hash cached
on disk under `manual_validation/responses/` so re-running is free.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import json
import time
from pathlib import Path

import settings

_VALIDATION_SUBDIR = "manual_validation"
_RESPONSE_CACHE_SUBDIR = "responses"

# Ablation variants run by validate_step.  Order matters: "full" must come
# first because it produces the baseline manual file the downstream summaries
# alias as `record["verification"]`.
#   full              — baseline: all manual features on, judge sees no extra hints
#   no_angle_ranking  — manual rendered with default angle (iso1)
#   no_text           — bottom instruction text blanked
#   no_motion         — base render omits red ghost + purple path trail
#   informed_judge    — SAME manual as "full"; the judge prompt additionally
#                       receives a paragraph describing the colour code and
#                       the manual's general structure.  Lets us isolate "did
#                       the judge fail because the manual was unclear, or
#                       because the judge didn't know the conventions".
ABLATIONS = ("full", "no_angle_ranking", "no_text", "no_motion", "informed_judge")


# ----------------------------------------------------------------------
# Section A: deterministic canonical instruction built from step metadata


def _signed_axis_components(d):
    """Return [('+X', val), ('-X', val), ...] sorted by magnitude descending.

    Each value is the projection of d onto that signed axis (clamped at 0).
    Only the three larger signs (one per axis) end up non-zero.
    """
    out = []
    for label, val in (("+X", float(d[0])), ("+Y", float(d[1])), ("+Z", float(d[2]))):
        if val >= 0:
            out.append((label, val))
            out.append(("-" + label[1], 0.0))
        else:
            out.append((label, 0.0))
            out.append(("-" + label[1], -val))
    out.sort(key=lambda kv: -kv[1])
    return out


def _rotation_axis_angle_deg(R):
    """Return (unit-axis, angle_deg) for a 3x3 rotation matrix.

    Robust against numerical jitter; collapses near-identity to (None, 0)."""
    import math

    import numpy as _np

    R = _np.asarray(R, dtype=float)
    cos_a = max(-1.0, min(1.0, (_np.trace(R) - 1.0) / 2.0))
    angle = math.acos(cos_a)
    if abs(angle) < 1e-4:
        return None, 0.0
    if abs(angle - math.pi) < 1e-4:
        diag = _np.diag(R)
        i = int(_np.argmax(diag))
        axis = _np.zeros(3)
        axis[i] = math.sqrt(max(0.0, (R[i, i] + 1.0) / 2.0))
        for j in range(3):
            if j != i:
                axis[j] = (R[i, j] + R[j, i]) / (4.0 * axis[i] + 1e-12)
        return axis / (_np.linalg.norm(axis) + 1e-12), math.degrees(angle)
    axis = _np.array(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ]
    ) / (2.0 * math.sin(angle))
    return axis / (_np.linalg.norm(axis) + 1e-12), math.degrees(angle)


def _motion_summary(step) -> str:
    """Multi-sentence description of how the part moves into its final pose.

    Uses step.matrices (per-frame transforms) to extract:
      * translation: dominant signed axis, or up to two axis components if
        the motion is diagonal (no single axis carries ≥70% of the magnitude)
      * rotation: axis-aligned check (rounded to nearest world axis) + angle
      * trajectory shape: straight vs bent (detected via the angle between
        the first-half and second-half displacement vectors)
    Falls back to pose-aware / rotated cues when matrices are unavailable.

    All directions are GLOBAL world-frame axes (the same frame the camera
    renders) — see `assembly._to_canonical` for the convention.
    """
    matrices = getattr(step, "matrices", None) or []
    sentences = []
    if len(matrices) >= 2:
        try:
            import numpy as _np

            T0 = _np.asarray(matrices[0], dtype=float)
            T1 = _np.asarray(matrices[-1], dtype=float)
            t_start, t_end = T0[:3, 3], T1[:3, 3]
            d = t_end - t_start
            mag = float(_np.linalg.norm(d))

            # --- translation ---
            if mag > 1e-6:
                comps = _signed_axis_components(d)
                top = comps[0]
                second = comps[1]
                top_frac = top[1] / mag
                second_frac = second[1] / mag
                if second_frac >= 0.30:
                    sentences.append(
                        f"Translate the part along a diagonal direction in the "
                        f"global frame: ~{top_frac * 100:.0f}% along {top[0]} and "
                        f"~{second_frac * 100:.0f}% along {second[0]}, ending in "
                        f"its final position."
                    )
                else:
                    sentences.append(
                        f"Translate the part roughly along the global {top[0]} axis "
                        f"into its final position."
                    )
            else:
                sentences.append(
                    "Lower the part directly into its final position with "
                    "negligible translation."
                )

            # --- trajectory shape (bend detection) ---
            if len(matrices) >= 3 and mag > 1e-6:
                mid_idx = len(matrices) // 2
                t_mid = _np.asarray(matrices[mid_idx], dtype=float)[:3, 3]
                d1 = t_mid - t_start
                d2 = t_end - t_mid
                if _np.linalg.norm(d1) > 1e-6 and _np.linalg.norm(d2) > 1e-6:
                    cos_bend = float(
                        _np.dot(d1, d2) / (_np.linalg.norm(d1) * _np.linalg.norm(d2))
                    )
                    cos_bend = max(-1.0, min(1.0, cos_bend))
                    import math

                    bend_deg = math.degrees(math.acos(cos_bend))
                    if bend_deg > 25.0:
                        d1_top = _signed_axis_components(d1)[:2]
                        d2_top = _signed_axis_components(d2)[:2]

                        def _phase(top_pair):
                            mag_ = (top_pair[0][1] ** 2 + top_pair[1][1] ** 2) ** 0.5
                            if mag_ < 1e-9 or top_pair[1][1] / mag_ < 0.20:
                                return top_pair[0][0]
                            return f"{top_pair[0][0]} & {top_pair[1][0]}"

                        sentences.append(
                            f"The path is not a straight line: the part first "
                            f"moves along {_phase(d1_top)}, then changes direction "
                            f"by about {bend_deg:.0f}° and finishes along "
                            f"{_phase(d2_top)}."
                        )

            # --- rotation ---
            R = T1[:3, :3] @ T0[:3, :3].T
            axis, angle_deg = _rotation_axis_angle_deg(R)
            if axis is not None and angle_deg >= 5.0:
                # Round axis to nearest signed world axis for readability.
                signed = _signed_axis_components(axis)
                rot_axis = signed[0][0]
                sentences.append(
                    f"While translating, rotate the part by about "
                    f"{angle_deg:.0f}° around the global {rot_axis} axis."
                )
        except Exception:
            pass

    if sentences:
        return " ".join(sentences)

    if getattr(step, "rotated", False):
        return (
            "After the assembly is rotated to a stable orientation, "
            "slide the part into its final position."
        )
    if getattr(step, "pose", None) is not None:
        return "Slide the part into its final position along its insertion direction."
    return "Place the part into its final position."


def build_instruction_a(step_idx, step, assembly) -> str:
    """Construct the canonical instruction text from `step` + `assembly`.

    The line is intentionally terse but covers every fact the manual should
    visually convey: part identity, motion, tool, parts to hold steady.
    """
    obj_name = assembly.objects[step.obj_id].name
    tool = getattr(step, "tool", None)
    parts_fix = list(getattr(step, "parts_fix", []) or [])

    pieces = [
        f"Step {step_idx + 1}: install the {obj_name}.",
        f"Motion: {_motion_summary(step)}.",
    ]
    if tool and str(tool).lower() not in ("none", "null"):
        pieces.append(f"Tool required: {tool}.")
    else:
        pieces.append("Tool required: none.")
    if parts_fix:
        fix_names = [assembly.objects[p].name for p in parts_fix]
        pieces.append(
            "Hold these parts steady while assembling: " + ", ".join(fix_names) + "."
        )
    else:
        pieces.append("No other parts need to be held in place.")
    return " ".join(pieces)


# ----------------------------------------------------------------------
# Manual path resolution + (optional) generation


def _candidate_manual_paths(assembly, step_idx, step):
    """All locations where the various manual-method backends save their PNG."""
    out = []
    obj_id = getattr(step, "obj_id", "?")
    output_dir = Path(getattr(assembly, "output_dir", assembly.storage_dir))
    manual_dir = output_dir / "manual"
    # Per backend:
    #   geometric → <output_dir>/manual/{idx}_{obj}_manual.png
    #   offline   → <save_dir>/{idx}_{obj}_manual_offline.png
    method = getattr(settings, "manual_method", "offline")
    suffix_for = {
        "geometric": "manual",
        "offline": "manual_offline",
    }
    suffix = suffix_for.get(method, "manual")

    out.append(manual_dir / f"{step_idx}_{obj_id}_{suffix}.png")
    # Fallbacks: try every other backend's naming too in case the user changed
    # methods between runs.
    for other_suffix in set(suffix_for.values()):
        if other_suffix != suffix:
            out.append(manual_dir / f"{step_idx}_{obj_id}_{other_suffix}.png")
    # Last-resort glob via the caller.
    return out


def resolve_manual_path(assembly, step_idx, step, ablation="full"):
    """Find the most-likely manual PNG for this step. Returns Path or None.

    ablation: when not "full", look for the per-ablation file produced by
        the offline backend (e.g. `<idx>_<obj>_manual_offline_no_text.png`).
        Falls back to a glob over `<idx>_<obj>_manual_offline_<ablation>.*`.
    """
    obj_id = getattr(step, "obj_id", "?")
    manual_dir = Path(getattr(assembly, "output_dir", assembly.storage_dir)) / "manual"

    if ablation != "full":
        for ext in ("png", "jpg", "jpeg"):
            cand = manual_dir / f"{step_idx}_{obj_id}_manual_offline_{ablation}.{ext}"
            if cand.exists():
                return cand
        if manual_dir.exists():
            for p in sorted(
                manual_dir.glob(f"{step_idx}_{obj_id}_manual_offline_{ablation}.*")
            ):
                return p
        return None

    for cand in _candidate_manual_paths(assembly, step_idx, step):
        if cand.exists():
            return cand
    # Glob fallback in <output_dir>/manual/ for anything matching this step.
    if manual_dir.exists():
        for p in sorted(manual_dir.glob(f"{step_idx}_{obj_id}_*.png")):
            # Skip per-ablation files when looking for the baseline.
            if any(
                f"_{a}" in p.stem for a in ("no_angle_ranking", "no_text", "no_motion")
            ):
                continue
            return p
    return None


# ----------------------------------------------------------------------
# Single-call judge: compare manual image directly against the checklist


def _to_data_url(image_path) -> str:
    with open(image_path, "rb") as f:
        return "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")


def _hash_call(*parts) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(b"\x00")
        h.update(p if isinstance(p, bytes) else str(p).encode("utf-8"))
    return h.hexdigest()[:24]


def _read_cache(cache_dir, key):
    if cache_dir is None:
        return None
    cache_path = Path(cache_dir) / f"{key}.json"
    if not cache_path.exists():
        return None
    try:
        return json.loads(cache_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _write_cache(cache_dir, key, payload):
    if cache_dir is None or payload is None:
        return
    cache_path = Path(cache_dir) / f"{key}.json"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(OSError):
        cache_path.write_text(json.dumps(payload))


def judge_manual_against_checklist(
    manual_path,
    checklist,
    evaluation=None,
    model=None,
    cache_dir=None,
    assembly_step_nr=None,
    informed=False,
):
    """One VLM call: looks at the manual image and the checklist text, returns
    a per-fact verdict. Output dict has keys:
        verdict   ∈ {"yes", "no", "error", "skipped"}
        facts     list of {"fact", "covered_correctly", "note"}
        reasoning one-sentence summary
        response  raw model output
        status    ∈ {"ok", "cached", "error", "skipped"}

    informed: when True, prepend an extra paragraph telling the judge how the
        manual is structured (colour code, panel layout, bottom text region).
        Defaults to False — the regular judge gets no hints, exactly like a
        first-time human reader.
    """
    model = model or getattr(settings, "LLM_model", "gpt-4o")
    try:
        with open(manual_path, "rb") as f:
            img_bytes = f.read()
    except OSError as e:
        return {
            "verdict": "error",
            "facts": [],
            "reasoning": f"could not read manual image: {e}",
            "response": "",
            "status": "error",
        }

    cache_key = _hash_call(
        "judge",
        model,
        checklist,
        str(assembly_step_nr),
        str(informed),
        hashlib.sha256(img_bytes).digest(),
    )
    cached = _read_cache(cache_dir, cache_key)
    if cached is not None:
        cached["status"] = "cached"
        return cached

    if evaluation is not None and evaluation.tokens_exhausted(
        "manual_validation_judge"
    ):
        return {
            "verdict": "skipped",
            "facts": [],
            "reasoning": "token budget exhausted",
            "response": "SKIPPED",
            "status": "skipped",
        }

    structure_note = ""
    if informed:
        structure_note = (
            "MANUAL STRUCTURE (read this first):\n"
            "  • The page is a single step of a multi-part assembly manual.\n"
            "  • The MAIN RENDER (large area, top ~3/4 of the page) shows the "
            "assembly from a 3-D isometric viewpoint with the moving part "
            "depicted in TWO copies:\n"
            "      - BLUE = the part in its FINAL ASSEMBLED position (where "
            "it should end up).\n"
            "      - RED = the SAME part in its STARTING (pre-insertion) "
            "position, before it has been moved into place. The motion is "
            "from RED → BLUE.\n"
            "      - PURPLE small spheres (if present) trace the centre-of-"
            "mass path the part travels along, sampled at intermediate "
            "positions between RED and BLUE.\n"
            "      - Every other part of the assembly is drawn in LIGHT GREY "
            "in its installed position for context.\n"
            "  • TOP-LEFT corner panel (if present): rotation hint. Shows "
            "the pose of the previous assembly step, indicating the assembly "
            "must be reoriented before this step.\n"
            "  • TOP-RIGHT corner panel (if present): tool icon + name. "
            "Only appears when a tool is required.\n"
            "  • BOTTOM REGION: the page title ('Step N') and 1–3 short "
            "sentences of natural-language instruction.\n"
            "Use this knowledge when judging each checklist fact: e.g. the "
            "presence of a RED ghost adjacent to a BLUE final-position copy "
            "is the manual's standard way of showing direction of "
            "insertion; the absence of a top-right panel means no tool is "
            "shown; etc.\n\n"
        )

    step_note = ""
    if assembly_step_nr is not None:
        step_note = (
            f"NOTE ON STEP NUMBERING: the checklist labels the step using its "
            f"disassembly index (the underlying planner enumerates from the "
            f"finished assembly backwards). The manual page is an ASSEMBLY "
            f"manual and labels this same step as 'Step {assembly_step_nr}' "
            f"instead. Do NOT mark the step-number fact as failed just because "
            f"the two labels differ — match on the part being installed, not "
            f"on the numeric label.\n\n"
        )
    intro = (
        "You are a verifier for an automatically generated assembly manual. "
        "You see ONE page of the manual (image, below) and a CHECKLIST of "
        "facts the page must convey to the user.\n\n"
        + structure_note
        + step_note
        + "For EVERY discrete fact in the checklist, look at the manual page "
        "and decide whether the page correctly conveys that fact (not "
        "missing, not contradicted, not mangled). Names in the checklist "
        "may appear paraphrased on the page — treat them as matched if "
        "the referent is clearly the same.\n\n"
        "PART IDENTITY does NOT require the name in writing. The fact is "
        "satisfied as long as a reader can clearly tell WHICH part is being "
        "installed — either because the page text names it, OR because the "
        "depicted part is visually distinct (highlighted in colour, shown "
        "offset from the rest of the assembly, marked by an arrow, or simply "
        "the only obvious thing to insert at this stage). Treat the textual "
        "name and the visual depiction as INTERCHANGEABLE evidence for this "
        "fact — either alone is sufficient.\n\n"
        "The motion fact is QUALITATIVE and should be judged LENIENTLY: the "
        "manual must show where the part ultimately sits in the assembly AND "
        "give the reader a reasonable sense of the insertion direction / "
        "path / rotation (e.g. via an arrow, a ghosted before-position, a "
        "rotation indicator, OR simply showing the part offset from its "
        "final location along the right general direction). Do NOT require a "
        "numeric displacement, exact distance, or explicit axis label. Minor "
        "directional ambiguity is fine as long as the depicted motion is "
        "broadly plausible for the described insertion.\n\n"
        "NOTE ON AXES: when the checklist names an axis (e.g. '+X axis', "
        "'-Z axis'), those refer to the GLOBAL world-coordinate axes of the "
        "render itself — the same frame the camera is positioned in. The "
        "manual page does NOT print axis labels on the image; the page "
        "satisfies the motion fact by visually depicting movement that is "
        "ROUGHLY consistent with that axis. Accept matches that are in the "
        "approximate direction even without a labelled axis, and do not "
        "penalise small angular discrepancies between the depicted arrow / "
        "offset and the exact world axis.\n\n"
        "Reply with ONLY a JSON object of the shape:\n"
        "{\n"
        '  "facts": [\n'
        '    {"fact": "<short summary of one checklist fact>",\n'
        '     "covered_correctly": true/false,\n'
        '     "note": "<one-sentence justification grounded in the image>"},\n'
        "    ...\n"
        "  ],\n"
        '  "verdict": "yes" | "no",\n'
        '  "reasoning": "<one-sentence summary>"\n'
        "}\n\n"
        '"verdict" is "yes" iff EVERY fact has covered_correctly=true; "no" '
        "otherwise.\n\n"
        "CHECKLIST:\n"
        f"{checklist}\n"
    )
    content = [
        {"type": "text", "text": intro},
        {"type": "image_url", "image_url": {"url": _to_data_url(manual_path)}},
    ]

    text = None
    parsed = None
    last_err = None
    for attempt in range(3):
        try:
            from openai import OpenAI

            client = OpenAI()
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": content}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            usage = getattr(resp, "usage", None)
            if evaluation is not None and usage is not None:
                evaluation.tokens_used += int(getattr(usage, "total_tokens", 0) or 0)
            text = resp.choices[0].message.content or ""
            parsed = json.loads(text)
            break
        except Exception as e:
            last_err = e
            if attempt < 2:
                time.sleep(2**attempt)
            continue

    if parsed is None:
        return {
            "verdict": "error",
            "facts": [],
            "reasoning": f"judge call failed: {last_err}",
            "response": text or "",
            "status": "error",
        }

    verdict = str(parsed.get("verdict", "")).strip().lower()
    if verdict not in ("yes", "no"):
        verdict = "no"
    payload = {
        "verdict": verdict,
        "facts": parsed.get("facts") or [],
        "reasoning": parsed.get("reasoning", ""),
        "response": text,
        "status": "ok",
    }
    _write_cache(cache_dir, cache_key, payload)
    return payload


# ----------------------------------------------------------------------
# Per-step + per-assembly orchestration


def _base_page_offset(assembly) -> int:
    """1 when the manual opens with a base page, else 0.

    Mirrors ManualGenerator.generate_manual_base_offline, which gives the parts
    the disassembly left behind an assembly page of their own, shifting every
    sequence step one number up."""
    manual = getattr(assembly, "manual", None)
    if manual is None or not hasattr(manual, "_base_ids"):
        return 0
    return 1 if manual._base_ids() else 0


def _assembly_step_nr(step_idx, assembly) -> int:
    """Convert a disassembly index into the assembly step number the manual
    displays.  sequence[0] is the LAST assembly step, so step 0 of disassembly
    becomes Step N (where N = len(sequence)), offset by the base page."""
    n_steps = len(getattr(assembly, "sequence", []) or []) or (step_idx + 1)
    return n_steps - step_idx + _base_page_offset(assembly)


def _checklist_filename(step_idx, step) -> str:
    obj_id = getattr(step, "obj_id", "?")
    return f"step_{step_idx:03d}_{obj_id}_checklist.txt"


def _explanation_filename(step_idx, step) -> str:
    obj_id = getattr(step, "obj_id", "?")
    return f"step_{step_idx:03d}_{obj_id}_explanation.txt"


def _build_step_explanation(assembly_step_nr, record) -> str:
    """Human-readable rationale for one step's decision across all ablations."""
    lines = []
    lines.append(f"Step {assembly_step_nr}  (obj_id={record.get('obj_id')})")
    lines.append("=" * 72)
    lines.append(f"status:     {record.get('status')}")
    lines.append("")

    verifications = record.get("verifications") or {}
    if not verifications and record.get("verification") is not None:
        verifications = {"full": record["verification"]}

    # Overview table of per-ablation verdicts.
    lines.append("Per-ablation verdicts:")
    for ablation in ABLATIONS:
        ver = verifications.get(ablation) or {}
        v = ver.get("verdict", ver.get("status", "n/a"))
        lines.append(f"  {ablation:<18} {v}")
    lines.append("")

    # Detailed breakdown per ablation.
    for ablation in ABLATIONS:
        ver = verifications.get(ablation) or {}
        lines.append("-" * 72)
        lines.append(f"Ablation: {ablation}")
        if ver.get("status") == "no_manual":
            lines.append("  No manual image was found for this variant.")
            continue
        verdict = ver.get("verdict", "n/a")
        lines.append(f"  verdict:   {verdict}")
        if ver.get("reasoning"):
            lines.append(f"  reasoning: {ver['reasoning']}")
        facts = ver.get("facts") or []
        for f in facts:
            mark = "PASS" if f.get("covered_correctly") else "FAIL"
            lines.append(f"  [{mark}] {f.get('fact', '')}")
            if f.get("note"):
                lines.append(f"         note: {f['note']}")

    lines.append("")
    lines.append("-" * 72)
    lines.append("Checklist (facts the manual must convey):")
    lines.append(record.get("checklist", "") or "(none)")
    lines.append("")
    return "\n".join(lines)


def validate_step(
    step_idx,
    step,
    assembly,
    evaluation,
    cache_dir,
    run_manual_generation=False,
    judge_model=None,
    out_dir=None,
):
    """Build the checklist, optionally generate the manual, then ask ONE VLM
    judge to verify each checklist fact against the manual image.

    If `out_dir` is given, the checklist is also written to
    `<out_dir>/step_<idx>_<obj_id>_checklist.txt`, and a per-step explanation
    (verdict + reasoning + per-fact breakdown) is written to
    `<out_dir>/step_<idx>_<obj_id>_explanation.txt`.
    """
    assembly_step_nr = _assembly_step_nr(step_idx, assembly)

    def _flush_explanation(rec):
        if out_dir is None:
            return
        try:
            e_path = Path(out_dir) / _explanation_filename(step_idx, step)
            e_path.parent.mkdir(parents=True, exist_ok=True)
            e_path.write_text(_build_step_explanation(assembly_step_nr, rec))
            rec["explanation_path"] = str(e_path)
        except OSError as e:
            print(
                f"[manual-val] {getattr(assembly, 'id', '?')} step {step_idx}: "
                f"failed to write explanation: {e}"
            )

    record = {
        "step_idx": step_idx,
        "assembly_step_nr": assembly_step_nr,
        "obj_id": getattr(step, "obj_id", None),
        "tool": getattr(step, "tool", None),
        "parts_fix": list(getattr(step, "parts_fix", []) or []),
        "rotated": bool(getattr(step, "rotated", False)),
    }
    record["checklist"] = build_instruction_a(step_idx, step, assembly)

    if out_dir is not None:
        try:
            c_path = Path(out_dir) / _checklist_filename(step_idx, step)
            c_path.parent.mkdir(parents=True, exist_ok=True)
            c_path.write_text(record["checklist"] + "\n")
            record["checklist_path"] = str(c_path)
        except OSError as e:
            print(
                f"[manual-val] {getattr(assembly, 'id', '?')} step {step_idx}: "
                f"failed to write checklist: {e}"
            )

    # Run the same checklist judge against four manual variants:
    #   full              — baseline (all features on)
    #   no_angle_ranking  — manual rendered with default angle (iso1) instead
    #                       of the SSIM-best per-step angle
    #   no_text           — bottom instruction text blanked
    #   no_motion         — base render omits the red disassembled-position
    #                       ghost and the purple path-trail dots
    record["verifications"] = {}
    record["manual_paths"] = {}
    for ablation in ABLATIONS:
        # informed_judge is a JUDGE-side variant — it reuses the "full" manual
        # but passes the judge an extra paragraph describing the manual's
        # structure (colour code, panel layout). Don't regenerate the page.
        is_informed = ablation == "informed_judge"
        image_ablation = "full" if is_informed else ablation

        if run_manual_generation and hasattr(assembly, "manual") and not is_informed:
            try:
                assembly.manual.make_manual(step_idx, ablation=ablation)
            except Exception as e:
                print(
                    f"[manual-val] {getattr(assembly, 'id', '?')} step {step_idx} "
                    f"({ablation}): make_manual failed: {e}"
                )

        manual_path = resolve_manual_path(
            assembly, step_idx, step, ablation=image_ablation
        )
        record["manual_paths"][ablation] = str(manual_path) if manual_path else None
        if manual_path is None:
            record["verifications"][ablation] = {"status": "no_manual"}
            continue

        verify = judge_manual_against_checklist(
            manual_path=manual_path,
            checklist=record["checklist"],
            evaluation=evaluation,
            model=judge_model,
            cache_dir=cache_dir,
            assembly_step_nr=assembly_step_nr,
            informed=is_informed,
        )
        record["verifications"][ablation] = verify

    # Back-compat aliases for the baseline so the explanation/summary writers
    # still work without ablation-awareness everywhere.
    record["manual_path"] = record["manual_paths"].get("full")
    record["verification"] = record["verifications"].get("full")
    if record["manual_path"] is None:
        record["status"] = "no_manual"
    else:
        record["status"] = "ok"
    _flush_explanation(record)
    return record


def validate_assembly_manual(
    assembly, evaluation, run_manual_generation=False, judge_model=None
):
    """Validate every step's manual for one assembly. Writes outputs under
    `<assembly.output_dir>/manual_validation/`. Returns the per-step records."""
    out_dir = (
        Path(getattr(assembly, "output_dir", None) or assembly.storage_dir)
        / _VALIDATION_SUBDIR
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / _RESPONSE_CACHE_SUBDIR

    sequence = list(getattr(assembly, "sequence", []) or [])
    if not sequence:
        print(f"[manual-val] {assembly.id}: empty sequence; nothing to validate")
        return None

    records = []
    last_idx = len(sequence) - 1
    # Without a base page the last disassembly index is the first assembly step
    # ("place starting part on work surface"): no real motion / tool / fixturing
    # to verify, so skip it entirely. With one, that page carries the caption
    # instead and every sequence step is a real installation worth validating.
    initial_idx = None if _base_page_offset(assembly) else last_idx
    for step_idx, step in enumerate(sequence):
        if step_idx == initial_idx:
            print(
                f"[manual-val] {assembly.id}  step {step_idx}/{last_idx}  "
                f"obj_id={getattr(step, 'obj_id', '?')}  SKIPPED (initial-state step)",
                flush=True,
            )
            continue
        print(
            f"[manual-val] {assembly.id}  step {step_idx}/{last_idx}  "
            f"obj_id={getattr(step, 'obj_id', '?')}  …",
            flush=True,
        )
        rec = validate_step(
            step_idx,
            step,
            assembly,
            evaluation,
            cache_dir=str(cache_dir),
            run_manual_generation=run_manual_generation,
            judge_model=judge_model,
            out_dir=str(out_dir),
        )
        records.append(rec)
        v = (rec.get("verification") or {}).get("verdict")
        print(
            f"[manual-val] {assembly.id}  step {step_idx}: "
            f"status={rec.get('status')}  verdict={v}"
        )

    summary = _summarize(records)
    _write_per_assembly_outputs(out_dir, assembly, records, summary)
    return records


def _summarize(records):
    total = len(records)
    by_status = {}
    for r in records:
        s = r.get("status", "?")
        by_status[s] = by_status.get(s, 0) + 1

    # Per-ablation aggregation: yes/no counts and verdict rate.
    per_ablation = {}
    for ablation in ABLATIONS:
        yes = 0
        no = 0
        for r in records:
            ver = (r.get("verifications") or {}).get(ablation) or {}
            if ver.get("verdict") == "yes":
                yes += 1
            elif ver.get("verdict") == "no":
                no += 1
        per_ablation[ablation] = {
            "yes": yes,
            "no": no,
            "verdict_rate": (yes / (yes + no)) if (yes + no) else None,
        }

    # Back-compat top-level baseline numbers (mirror the "full" ablation).
    base = per_ablation.get("full", {"yes": 0, "no": 0, "verdict_rate": None})
    return {
        "total_steps": total,
        "status_counts": by_status,
        "yes": base["yes"],
        "no": base["no"],
        "verdict_rate": base["verdict_rate"],
        "per_ablation": per_ablation,
    }


def _write_per_assembly_outputs(out_dir, assembly, records, summary):
    json_path = Path(out_dir) / "manual_validation.json"
    txt_path = Path(out_dir) / "manual_validation.txt"

    json_path.write_text(
        json.dumps(
            {
                "assembly_id": assembly.id,
                "n_steps": len(records),
                "summary": summary,
                "steps": records,
            },
            indent=2,
            default=str,
        )
    )

    lines = []
    lines.append(f"Manual Validation — assembly {assembly.id}")
    lines.append("=" * 72)
    lines.append(f"Total steps:        {summary['total_steps']}")
    for s, c in summary["status_counts"].items():
        lines.append(f"  status={s:<14} {c}")
    lines.append("")
    lines.append("Per-ablation verdict rate (yes / yes+no):")
    per_abl = summary.get("per_ablation") or {}
    for ablation in ABLATIONS:
        agg = per_abl.get(ablation, {})
        y, n = agg.get("yes", 0), agg.get("no", 0)
        rate = agg.get("verdict_rate")
        rate_s = f"{100 * rate:.1f}%" if rate is not None else "n/a"
        lines.append(f"  {ablation:<18} {y}/{y + n} = {rate_s}")
    lines.append("")
    for rec in records:
        a_nr = rec.get("assembly_step_nr", rec.get("step_idx"))
        lines.append(f"-- Step {a_nr}  (status={rec.get('status')}) --")
        lines.append(f"  obj_id:    {rec.get('obj_id')}")
        lines.append(f"  tool:      {rec.get('tool')}")
        lines.append(f"  parts_fix: {rec.get('parts_fix')}")
        lines.append(f"  Checklist: {rec.get('checklist', '')}")
        vers = rec.get("verifications") or {}
        for ablation in ABLATIONS:
            ver = vers.get(ablation) or {}
            v = ver.get("verdict", ver.get("status", "n/a"))
            lines.append(f"  [{ablation:<18}] {v}")
            if ver.get("reasoning"):
                lines.append(f"     reasoning: {ver['reasoning']}")
        lines.append("")
    txt_path.write_text("\n".join(lines))
    print(f"[manual-val] {assembly.id}: wrote {json_path}")


def write_batch_summary(assemblies, output_folder):
    """Aggregate every assembly's `manual_validation.json` into a batch digest
    under `<output_folder>/manual_validation_batch.{json,txt}`."""
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    rows = []
    for ass in assemblies:
        p = (
            Path(getattr(ass, "output_dir", None) or ass.storage_dir)
            / _VALIDATION_SUBDIR
            / "manual_validation.json"
        )
        rec = {"id": ass.id}
        if not p.exists():
            rec["status"] = "missing"
            rows.append(rec)
            continue
        try:
            data = json.loads(p.read_text())
        except Exception as e:
            rec["status"] = f"unreadable: {e}"
            rows.append(rec)
            continue
        s = data.get("summary") or {}
        rec.update(
            {
                "status": "ok",
                "n_steps": data.get("n_steps", 0),
                "yes": s.get("yes", 0),
                "no": s.get("no", 0),
                "verdict_rate": s.get("verdict_rate"),
                "status_counts": s.get("status_counts", {}),
                "per_ablation": s.get("per_ablation", {}),
            }
        )
        rows.append(rec)

    total_yes = sum(r.get("yes", 0) for r in rows if r["status"] == "ok")
    total_no = sum(r.get("no", 0) for r in rows if r["status"] == "ok")
    n_steps = sum(r.get("n_steps", 0) for r in rows if r["status"] == "ok")

    # Per-ablation aggregation across assemblies.
    batch_per_ablation = {}
    for ablation in ABLATIONS:
        y = sum(
            (r.get("per_ablation", {}).get(ablation, {}).get("yes", 0))
            for r in rows
            if r["status"] == "ok"
        )
        n = sum(
            (r.get("per_ablation", {}).get(ablation, {}).get("no", 0))
            for r in rows
            if r["status"] == "ok"
        )
        batch_per_ablation[ablation] = {
            "yes": y,
            "no": n,
            "verdict_rate": (y / (y + n)) if (y + n) else None,
        }

    batch = {
        "n_assemblies": len(rows),
        "n_ok": sum(1 for r in rows if r["status"] == "ok"),
        "n_missing": sum(1 for r in rows if r["status"] == "missing"),
        "n_steps_total": n_steps,
        "yes": total_yes,
        "no": total_no,
        "verdict_rate": (total_yes / (total_yes + total_no))
        if (total_yes + total_no)
        else None,
        "per_ablation": batch_per_ablation,
        "per_assembly": rows,
    }

    json_path = output_folder / "manual_validation_batch.json"
    txt_path = output_folder / "manual_validation_batch.txt"
    json_path.write_text(json.dumps(batch, indent=2, default=str))

    lines = []
    lines.append("Manual Validation — batch summary")
    lines.append("=" * 72)
    lines.append(f"Assemblies in batch:  {batch['n_assemblies']}")
    lines.append(f"  with results:       {batch['n_ok']}")
    lines.append(f"  missing summary:    {batch['n_missing']}")
    lines.append(f"Total steps:          {n_steps}")
    if batch["verdict_rate"] is not None:
        lines.append(
            f"Verdict yes/total:    {total_yes}/{total_yes + total_no}"
            f" = {100 * batch['verdict_rate']:.1f}%"
        )
    else:
        lines.append("Verdict:              (no data)")
    lines.append("")
    lines.append("Per-ablation across batch:")
    for ablation in ABLATIONS:
        agg = batch_per_ablation[ablation]
        y, n = agg["yes"], agg["no"]
        rate = agg["verdict_rate"]
        rate_s = f"{100 * rate:.1f}%" if rate is not None else "n/a"
        lines.append(f"  {ablation:<18} {y}/{y + n} = {rate_s}")
    lines.append("")
    lines.append(
        f"  {'id':<10}  {'status':<13}  {'steps':>5}  "
        + "  ".join(f"{a[:10]:<10}" for a in ABLATIONS)
    )
    for r in rows:
        if r["status"] != "ok":
            lines.append(f"  {r['id']:<10}  {r['status']:<13}")
            continue
        cells = []
        for ablation in ABLATIONS:
            agg = r.get("per_ablation", {}).get(ablation, {})
            y, n = agg.get("yes", 0), agg.get("no", 0)
            rate = (
                (100 * agg["verdict_rate"])
                if isinstance(agg.get("verdict_rate"), (int, float))
                else 0.0
            )
            cells.append(f"{y:>2}/{y + n:<3} {rate:>4.0f}%")
        lines.append(
            f"  {r['id']:<10}  ok             {r['n_steps']:>5}  " + "  ".join(cells)
        )
    txt_path.write_text("\n".join(lines) + "\n")
    print(f"[manual-val] batch summary written to {json_path} and {txt_path}")

    plot_path = _write_batch_plot(output_folder, batch_per_ablation, rows)
    if plot_path is not None:
        print(f"[manual-val] batch plot written to {plot_path}")


def _write_batch_plot(output_folder, batch_per_ablation, rows):
    """Render a two-panel ablation summary plot.

    Top panel:  batch-wide verdict-rate bars per ablation (with yes/total labels).
    Bottom panel: per-assembly grouped bars (one cluster per assembly, one bar
        per ablation) so per-assembly variance is visible at a glance.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[manual-val] matplotlib unavailable, skipping plot: {e}")
        return None

    ok_rows = [r for r in rows if r.get("status") == "ok"]
    if not ok_rows and not any(
        batch_per_ablation[a].get("verdict_rate") is not None for a in ABLATIONS
    ):
        print("[manual-val] no validation data to plot; skipping")
        return None

    ablations = list(ABLATIONS)
    # full, no-angle, no-text, no-motion, informed_judge
    colors = ["#2e7d32", "#ef6c00", "#1565c0", "#6a1b9a", "#00838f"]

    fig_h = 4 + 0.18 * max(1, len(ok_rows))
    fig, (ax_top, ax_bot) = plt.subplots(
        2,
        1,
        figsize=(max(8, 0.55 * len(ok_rows) * len(ablations) + 4), fig_h),
        gridspec_kw={"height_ratios": [1, max(1.0, 0.4 + 0.18 * len(ok_rows))]},
    )

    # --- top: batch-wide bars ---
    rates = []
    labels = []
    for ablation in ablations:
        agg = batch_per_ablation.get(ablation, {})
        rate = agg.get("verdict_rate")
        rates.append(100 * rate if rate is not None else 0.0)
        labels.append(f"{agg.get('yes', 0)}/{agg.get('yes', 0) + agg.get('no', 0)}")

    bars = ax_top.bar(ablations, rates, color=colors)
    ax_top.set_ylim(0, 105)
    ax_top.set_ylabel("Verdict rate  (% yes)")
    ax_top.set_title("Manual validation — ablation study (batch totals)")
    ax_top.grid(axis="y", linestyle=":", alpha=0.4)
    for bar, rate, label in zip(bars, rates, labels, strict=False):
        ax_top.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 2,
            f"{rate:.1f}%\n({label})",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # --- bottom: grouped bars per assembly ---
    if ok_rows:
        import numpy as np

        x = np.arange(len(ok_rows))
        width = 0.8 / len(ablations)
        for i, ablation in enumerate(ablations):
            heights = []
            for r in ok_rows:
                agg = r.get("per_ablation", {}).get(ablation, {})
                rate = agg.get("verdict_rate")
                heights.append(100 * rate if rate is not None else 0.0)
            ax_bot.bar(
                x + (i - (len(ablations) - 1) / 2) * width,
                heights,
                width=width,
                color=colors[i],
                label=ablation,
            )
        ax_bot.set_xticks(x)
        ax_bot.set_xticklabels([str(r["id"]) for r in ok_rows], rotation=45, ha="right")
        ax_bot.set_ylim(0, 105)
        ax_bot.set_ylabel("Verdict rate  (% yes)")
        ax_bot.set_xlabel("Assembly id")
        ax_bot.set_title("Per-assembly ablation rates")
        ax_bot.grid(axis="y", linestyle=":", alpha=0.4)
        ax_bot.legend(loc="lower right", ncol=len(ablations), fontsize=8)
    else:
        ax_bot.axis("off")
        ax_bot.text(
            0.5,
            0.5,
            "(no per-assembly data)",
            ha="center",
            va="center",
            transform=ax_bot.transAxes,
        )

    plt.tight_layout()
    plot_path = Path(output_folder) / "manual_validation_batch.png"
    try:
        fig.savefig(plot_path, dpi=130)
    except Exception as e:
        print(f"[manual-val] failed to write plot: {e}")
        plt.close(fig)
        return None
    plt.close(fig)
    return plot_path
