"""Shared CLI infrastructure for main.py: ID resolution, output-dir
creation, and batch-summary aggregation reused across run modules."""

import json
import os
from datetime import datetime
from pathlib import Path


def create_output_directory():
    time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = Path("assets/output") / time_stamp
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def resolve_ids(id_arg, dir_arg, max_parts=None, min_parts=None):
    """Return a list of assembly IDs to process.

    Accepts either a single ID ("00042") or an inclusive range ("00010-00050").
    For ranges, only IDs that exist as subdirectories of dir_arg are returned.
    Optional `min_parts` / `max_parts` filters drop IDs whose `.obj` count
    falls outside [min_parts, max_parts]. Filters only apply in the range case.
    """
    if "-" in id_arg:
        start, end = id_arg.split("-", 1)
        width = max(len(start), len(end))
        in_range = {str(i).zfill(width) for i in range(int(start), int(end) + 1)}
        existing = {
            d for d in os.listdir(dir_arg) if os.path.isdir(os.path.join(dir_arg, d))
        }
        if max_parts is not None or min_parts is not None:

            def _n_parts(d):
                full = os.path.join(dir_arg, d)
                return sum(1 for f in os.listdir(full) if f.endswith(".obj"))

            def _ok(d):
                n = _n_parts(d)
                if max_parts is not None and n > max_parts:
                    return False
                return not (min_parts is not None and n < min_parts)

            existing = {d for d in existing if _ok(d)}
        return sorted(in_range & existing)
    return [id_arg]


def _write_comparison_batch_summary(assemblies, output_folder):
    """Aggregate every assembly's per-run `log/comparison_summary.json` (written
    by `ComparisonDFASequencePlanner._write_summary`) into a batch-level digest.

    Writes <output_folder>/comparison_batch_summary.{json,txt}. Safe to call
    mid-run: assemblies whose summary file is missing are reported as 'missing'.
    """
    from collections import Counter as _Counter
    from itertools import combinations as _combos

    batch = {
        "n_assemblies": len(assemblies),
        "assemblies_with_decisions": 0,
        "assemblies_no_decisions": 0,
        "assemblies_missing_summary": 0,
        "total_decisions": 0,
        "selectors": [],
        "per_selector_responded": _Counter(),
        "per_selector_match_random": _Counter(),
        "pairwise_agree": _Counter(),
        "pairwise_both_responded": _Counter(),
        "vlm_meta_responded": 0,
        "vlm_meta_match_random": 0,
        "vlm_meta_match_constituent": _Counter(),
        "per_assembly": [],
    }

    all_selectors = set()
    for ass in assemblies:
        summary_path = ass.storage_dir / "log" / "comparison_summary.json"
        rec = {"id": ass.id, "summary_path": str(summary_path)}
        if not summary_path.exists():
            batch["assemblies_missing_summary"] += 1
            rec["status"] = "missing"
            batch["per_assembly"].append(rec)
            continue
        try:
            with open(summary_path) as _f:
                s = json.load(_f)
        except Exception as _exc:
            rec["status"] = f"unreadable: {_exc}"
            batch["per_assembly"].append(rec)
            continue

        total = int(s.get("total_decisions", 0) or 0)
        if total == 0:
            batch["assemblies_no_decisions"] += 1
        else:
            batch["assemblies_with_decisions"] += 1
        batch["total_decisions"] += total

        names = list(s.get("selectors") or [])
        all_selectors.update(names)
        for n in names:
            batch["per_selector_responded"][n] += int(
                (s.get("per_selector_responded") or {}).get(n, 0) or 0
            )
            batch["per_selector_match_random"][n] += int(
                (s.get("per_selector_match_random_count") or {}).get(n, 0) or 0
            )
        for k, v in (s.get("pairwise_agreement_count") or {}).items():
            batch["pairwise_agree"][k] += int(v or 0)
        for k, v in (s.get("pairwise_both_responded") or {}).items():
            batch["pairwise_both_responded"][k] += int(v or 0)

        batch["vlm_meta_responded"] += int(s.get("vlm_meta_responded", 0) or 0)
        batch["vlm_meta_match_random"] += int(
            s.get("vlm_meta_match_random_count", 0) or 0
        )
        for n, c in (s.get("vlm_meta_match_constituent_count") or {}).items():
            batch["vlm_meta_match_constituent"][n] += int(c or 0)

        rec.update(
            {
                "status": "ok" if total > 0 else "no_decisions",
                "total_decisions": total,
                "selectors": names,
                "avg_sample_size": s.get("avg_sample_size"),
                "per_selector_match_random_rate": s.get(
                    "per_selector_match_random_rate"
                ),
                "pairwise_agreement_rate": s.get("pairwise_agreement_rate"),
                "vlm_meta_responded": s.get("vlm_meta_responded"),
                "vlm_meta_match_random_rate": s.get("vlm_meta_match_random_rate"),
            }
        )
        batch["per_assembly"].append(rec)

    batch["selectors"] = sorted(all_selectors)
    # Convert Counters to dicts for JSON.
    batch["per_selector_responded"] = dict(batch["per_selector_responded"])
    batch["per_selector_match_random"] = dict(batch["per_selector_match_random"])
    batch["pairwise_agree"] = dict(batch["pairwise_agree"])
    batch["pairwise_both_responded"] = dict(batch["pairwise_both_responded"])
    batch["vlm_meta_match_constituent"] = dict(batch["vlm_meta_match_constituent"])

    # Aggregate rates.
    batch["per_selector_match_random_rate"] = {
        n: (batch["per_selector_match_random"][n] / batch["per_selector_responded"][n])
        if batch["per_selector_responded"].get(n)
        else None
        for n in batch["selectors"]
    }
    batch["pairwise_agreement_rate"] = {
        k: (batch["pairwise_agree"][k] / batch["pairwise_both_responded"][k])
        if batch["pairwise_both_responded"].get(k)
        else None
        for k in batch["pairwise_both_responded"]
    }
    batch["vlm_meta_match_random_rate"] = (
        batch["vlm_meta_match_random"] / batch["vlm_meta_responded"]
        if batch["vlm_meta_responded"]
        else None
    )
    batch["vlm_meta_match_constituent_rate"] = {
        n: (batch["vlm_meta_match_constituent"].get(n, 0) / batch["vlm_meta_responded"])
        if batch["vlm_meta_responded"]
        else None
        for n in batch["selectors"]
    }

    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "comparison_batch_summary.json"
    txt_path = out_dir / "comparison_batch_summary.txt"

    try:
        with open(json_path, "w") as _f:
            json.dump(batch, _f, indent=2, default=str)
    except OSError as _e:
        print(f"[comparison-batch] could not write {json_path}: {_e}")

    def _pct(v):
        return f"{100 * v:.1f}%" if isinstance(v, (int, float)) else "—"

    lines = []
    lines.append("Comparison Planner — Batch Summary")
    lines.append("(forward exploration uniform-random; all selectors logged only)")
    lines.append("=" * 80)
    lines.append(f"Assemblies in batch:                {batch['n_assemblies']}")
    lines.append(
        f"  with decisions logged:            {batch['assemblies_with_decisions']}"
    )
    lines.append(
        f"  ran but no decisions (trivial):   {batch['assemblies_no_decisions']}"
    )
    lines.append(
        f"  missing summary file:             {batch['assemblies_missing_summary']}"
    )
    lines.append("")
    lines.append(f"Total decisions across batch:       {batch['total_decisions']}")
    lines.append(f"Selectors observed:                 {batch['selectors']}")
    lines.append("")
    lines.append("Per-selector match with random (forward) pick:")
    lines.append(f"  {'selector':<14}  {'matched':>8}  {'responded':>9}  rate")
    for n in batch["selectors"]:
        r = batch["per_selector_match_random"].get(n, 0)
        resp = batch["per_selector_responded"].get(n, 0)
        rate = batch["per_selector_match_random_rate"].get(n)
        lines.append(f"  {n:<14}  {r:>8d}  {resp:>9d}  {_pct(rate):>6}")
    lines.append("")

    if len(batch["selectors"]) > 1 and batch["pairwise_both_responded"]:
        lines.append("Pairwise agreement (both selectors responded):")
        lines.append(f"  {'pair':<32}  {'agree':>6}  {'both':>5}  rate")
        for a, b in _combos(batch["selectors"], 2):
            key = f"{a}|{b}"
            cnt = batch["pairwise_agree"].get(key, 0)
            base = batch["pairwise_both_responded"].get(key, 0)
            rate = batch["pairwise_agreement_rate"].get(key)
            lines.append(
                f"  {a + ' vs ' + b:<32}  {cnt:>6d}  {base:>5d}  {_pct(rate):>6}"
            )
        lines.append("")

    if batch["vlm_meta_responded"] > 0:
        lines.append("VLM meta-selector (picked among constituents' choices):")
        lines.append(f"  responded:                {batch['vlm_meta_responded']}")
        lines.append(
            f"  matched random pick:      "
            f"{batch['vlm_meta_match_random']}/{batch['vlm_meta_responded']} = "
            f"{_pct(batch['vlm_meta_match_random_rate'])}"
        )
        lines.append("  matched constituent:")
        for n in batch["selectors"]:
            c = batch["vlm_meta_match_constituent"].get(n, 0)
            rate = batch["vlm_meta_match_constituent_rate"].get(n)
            lines.append(
                f"    {n:<14}  {c:>4d}/{batch['vlm_meta_responded']}  = {_pct(rate):>6}"
            )
        lines.append("")

    lines.append("Per-assembly:")
    lines.append(
        f"  {'id':<10}  {'status':<13}  {'n_dec':>5}  {'avg_l':>5}  {'vlm_resp':>8}"
    )
    for rec in batch["per_assembly"]:
        if rec["status"] in ("missing", "no_decisions") or rec["status"].startswith(
            "unreadable"
        ):
            lines.append(f"  {rec['id']:<10}  {rec['status']:<13}")
            continue
        avg_l = rec.get("avg_sample_size")
        avg_l_str = f"{avg_l:.1f}" if isinstance(avg_l, (int, float)) else "—"
        lines.append(
            f"  {rec['id']:<10}  {rec['status']:<13}  "
            f"{rec.get('total_decisions', 0):>5}  "
            f"{avg_l_str:>5}  "
            f"{(rec.get('vlm_meta_responded') or 0):>8}"
        )

    try:
        with open(txt_path, "w") as _f:
            _f.write("\n".join(lines) + "\n")
    except OSError as _e:
        print(f"[comparison-batch] could not write {txt_path}: {_e}")

    print(f"\n[comparison-batch] summary written to {json_path} and {txt_path}")


def _write_llm_batch_summary(assemblies, output_folder):
    """Aggregate every assembly's per-run `log/llm_summary.json` (written by
    `LLMDFASequencePlanner._write_llm_summary`) into a batch-level digest.

    Writes <output_folder>/llm_batch_summary.{json,txt}. Safe to call mid-run:
    assemblies whose summary file is missing are reported as 'missing'.
    """
    from collections import Counter as _Counter

    batch = {
        "n_assemblies": len(assemblies),
        "assemblies_with_decisions": 0,
        "assemblies_no_decisions": 0,
        "assemblies_missing_summary": 0,
        "total_decisions": 0,
        "total_llm_responded": 0,
        "total_llm_vs_heuristic_agreements": 0,
        "total_llm_matches_random": 0,
        "total_heuristic_matches_random": 0,
        "status_counts": _Counter(),
        "per_assembly": [],
    }

    for ass in assemblies:
        summary_path = ass.storage_dir / "log" / "llm_summary.json"
        rec = {"id": ass.id, "summary_path": str(summary_path)}
        if not summary_path.exists():
            batch["assemblies_missing_summary"] += 1
            rec["status"] = "missing"
            batch["per_assembly"].append(rec)
            continue
        try:
            with open(summary_path) as _f:
                s = json.load(_f)
        except Exception as _exc:
            rec["status"] = f"unreadable: {_exc}"
            batch["per_assembly"].append(rec)
            continue

        total = int(s.get("total_decisions", 0) or 0)
        if total == 0:
            batch["assemblies_no_decisions"] += 1
        else:
            batch["assemblies_with_decisions"] += 1
        batch["total_decisions"] += total
        batch["total_llm_responded"] += int(s.get("llm_responded", 0) or 0)
        batch["total_llm_vs_heuristic_agreements"] += int(
            s.get("llm_vs_heuristic_agreements", 0) or 0
        )
        batch["total_llm_matches_random"] += int(
            s.get("llm_matches_random_count", 0) or 0
        )
        batch["total_heuristic_matches_random"] += int(
            s.get("heuristic_matches_random_count", 0) or 0
        )
        for k, v in (s.get("status_counts") or {}).items():
            batch["status_counts"][k] += int(v)

        rec.update(
            {
                "status": "ok" if total > 0 else "no_decisions",
                "total_decisions": total,
                "llm_responded": s.get("llm_responded"),
                "llm_vs_heuristic_rate": s.get("llm_vs_heuristic_rate"),
                "llm_matches_random_rate": s.get("llm_matches_random_rate"),
                "heuristic_matches_random_rate": s.get("heuristic_matches_random_rate"),
                "avg_sample_size": s.get("avg_sample_size"),
            }
        )
        batch["per_assembly"].append(rec)

    batch["status_counts"] = dict(batch["status_counts"])

    if batch["total_llm_responded"] > 0:
        batch["llm_vs_heuristic_rate"] = (
            batch["total_llm_vs_heuristic_agreements"] / batch["total_llm_responded"]
        )
        batch["llm_matches_random_rate"] = (
            batch["total_llm_matches_random"] / batch["total_llm_responded"]
        )
    else:
        batch["llm_vs_heuristic_rate"] = None
        batch["llm_matches_random_rate"] = None
    if batch["total_decisions"] > 0:
        batch["heuristic_matches_random_rate"] = (
            batch["total_heuristic_matches_random"] / batch["total_decisions"]
        )
    else:
        batch["heuristic_matches_random_rate"] = None

    out_dir = Path(output_folder)
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "llm_batch_summary.json"
    txt_path = out_dir / "llm_batch_summary.txt"

    try:
        with open(json_path, "w") as _f:
            json.dump(batch, _f, indent=2, default=str)
    except OSError as _e:
        print(f"[llm-batch] could not write {json_path}: {_e}")

    def _pct(v):
        return f"{100 * v:.1f}%" if isinstance(v, (int, float)) else "—"

    lines = []
    lines.append("LLM vs Heuristic — Batch Summary")
    lines.append(
        "(forward exploration is uniform-random; LLM/heuristic picks logged only)"
    )
    lines.append("=" * 80)
    lines.append(f"Assemblies in batch:                {batch['n_assemblies']}")
    lines.append(
        f"  with decisions logged:            {batch['assemblies_with_decisions']}"
    )
    lines.append(
        f"  ran but no decisions (trivial):   {batch['assemblies_no_decisions']}"
    )
    lines.append(
        f"  missing summary file:             {batch['assemblies_missing_summary']}"
    )
    lines.append("")
    lines.append(f"Total decisions across batch:       {batch['total_decisions']}")
    lines.append(f"  LLM responded:                    {batch['total_llm_responded']}")
    lines.append(f"  status breakdown:                 {batch['status_counts']}")
    lines.append("")
    lines.append(
        f"LLM vs heuristic agreement:         "
        f"{batch['total_llm_vs_heuristic_agreements']}/{batch['total_llm_responded']}"
        f" = {_pct(batch['llm_vs_heuristic_rate'])}"
    )
    lines.append(
        f"LLM picked random's choice:         "
        f"{batch['total_llm_matches_random']}/{batch['total_llm_responded']}"
        f" = {_pct(batch['llm_matches_random_rate'])}"
    )
    lines.append(
        f"Heuristic picked random's choice:   "
        f"{batch['total_heuristic_matches_random']}/{batch['total_decisions']}"
        f" = {_pct(batch['heuristic_matches_random_rate'])}"
    )
    lines.append("")
    lines.append("Per-assembly:")
    lines.append(
        f"  {'id':<10}  {'status':<13}  {'n_dec':>5}  {'llm_resp':>8}  "
        f"{'llm/heu':>8}  {'llm/rand':>9}  {'heu/rand':>9}"
    )
    for rec in batch["per_assembly"]:
        if rec["status"] in ("missing", "no_decisions") or rec["status"].startswith(
            "unreadable"
        ):
            lines.append(f"  {rec['id']:<10}  {rec['status']:<13}")
            continue
        lines.append(
            f"  {rec['id']:<10}  {rec['status']:<13}  "
            f"{rec.get('total_decisions', 0):>5}  "
            f"{(rec.get('llm_responded') or 0):>8}  "
            f"{_pct(rec.get('llm_vs_heuristic_rate')):>8}  "
            f"{_pct(rec.get('llm_matches_random_rate')):>9}  "
            f"{_pct(rec.get('heuristic_matches_random_rate')):>9}"
        )

    try:
        with open(txt_path, "w") as _f:
            _f.write("\n".join(lines) + "\n")
    except OSError as _e:
        print(f"[llm-batch] could not write {txt_path}: {_e}")

    print(f"\n[llm-batch] summary written to {json_path} and {txt_path}")
