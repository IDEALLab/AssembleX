"""Standalone visualization for ComparisonDFASequencePlanner outputs.

Walks a target directory, finds every `comparison_summary.json` (per-assembly)
AND `comparison_batch_summary.json` (cross-assembly aggregate) it can, and
emits a clean composite matplotlib figure next to each one.

Per-summary figure layout (2 × 2):

    ┌────────────────────────────────┬─────────────────────────────────┐
    │ Selector vs random baseline    │ Pairwise agreement heatmap      │
    │ (bar + baseline line at 1/l)   │                                 │
    ├────────────────────────────────┼─────────────────────────────────┤
    │ VLM meta agreement (if used)   │ Forward-source breakdown +      │
    │                                │ key totals  (per-assembly)      │
    │                                │   — OR —                        │
    │                                │ Assembly-status breakdown +     │
    │                                │ key totals  (batch summary)     │
    └────────────────────────────────┴─────────────────────────────────┘

Usage:
    python core/plot_comparison.py <target_dir>
    python core/plot_comparison.py <target_dir> --out <save_dir>
    python core/plot_comparison.py <target_dir> --no-recursive --show

No dependencies beyond matplotlib + numpy (both already in the project env).
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from itertools import combinations
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------------
# data loading + normalization


_KNOWN_STEMS = (
    "comparison_summary",  # ComparisonDFASequencePlanner per-assembly
    "comparison_batch_summary",  # cross-assembly aggregate from test_pipeline_batch
    "validation_summary",  # test_heuristic_validation k-greedy validation
    "assembly_time_summary",  # data_assembly_time multi-generator timing
    "manual_validation_batch",  # data_manual_validation ablation batch digest
    "convex_decomp_results",  # test_convex_decomp per-assembly AI/validator results
    "human_labels",  # collect_tool_data per-assembly tool ground truth
    #   (paired with sibling ai_labels*.json)
)

# Mirror of `ABLATIONS` in manual_validator.py — kept here so this script
# stays standalone (no manual_validator import). If the ablation list is
# ever changed there, update this tuple to match or the per-ablation
# panels will fall out of sync.
_MANUAL_ABLATIONS = (
    "full",
    "no_angle_ranking",
    "no_text",
    "no_motion",
    "informed_judge",
)
_MANUAL_ABLATION_COLORS = ("#2e7d32", "#ef6c00", "#1565c0", "#6a1b9a", "#00838f")
# Kept for back-compat / docstrings; matches `<stem>.json` exactly.
_KNOWN_NAMES = tuple(f"{s}.json" for s in _KNOWN_STEMS)


def _stem_of(path: Path) -> str | None:
    """If `path.name` matches `<stem>.json` or `<stem>_<suffix>.json` for one
    of the recognised stems, return that stem. Else None.

    We pick the longest matching stem so `comparison_batch_summary_2.json` is
    classified as `comparison_batch_summary`, not `comparison_summary`.
    """
    name = path.name
    if not name.endswith(".json"):
        return None
    # Backward-compat for the pre-rename convex-decomp writer that emitted
    # a generic `results.json` inside `<out>/convex_decomp/<id>/`. Detect
    # by ancestor directory so unrelated `results.json` files elsewhere in
    # the tree don't get misclassified.
    if (
        name == "results.json"
        and len(path.parents) >= 2
        and path.parent.parent.name == "convex_decomp"
    ):
        return "convex_decomp_results"
    base = name[: -len(".json")]
    best = None
    for stem in _KNOWN_STEMS:
        if base == stem or base.startswith(stem + "_"):
            if best is None or len(stem) > len(best):
                best = stem
    return best


def find_summaries(target_dir: Path, recursive: bool = True) -> list[Path]:
    """Return every recognised summary JSON under `target_dir`. Accepts both
    bare names (e.g. `comparison_summary.json`) and numbered variants
    (`comparison_summary_2.json`, `validation_summary_run3.json`, …) so
    parallel runs that write into the same directory all get picked up."""
    paths: set[Path] = set()
    for stem in _KNOWN_STEMS:
        for pat in (f"{stem}.json", f"{stem}_*.json"):
            pattern = f"**/{pat}" if recursive else pat
            paths.update(target_dir.glob(pattern))
    # Legacy convex-decomp output predates the rename — pre-rename results
    # live at `<out>/convex_decomp/<id>/results.json`. Glob those explicitly
    # so the old data still gets discovered; `_stem_of` then classifies
    # them as `convex_decomp_results`.
    legacy_cd_pat = (
        "**/convex_decomp/*/results.json"
        if recursive
        else "convex_decomp/*/results.json"
    )
    paths.update(target_dir.glob(legacy_cd_pat))
    # Filter to only those whose stem classifier returns a known stem — this
    # guards against accidental collisions if a stem prefix appears in
    # unrelated filenames.
    return sorted(p for p in paths if _stem_of(p) is not None)


def load_summary(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _is_batch(summary: dict) -> bool:
    """A batch summary always carries a `per_assembly` list."""
    return isinstance(summary.get("per_assembly"), list)


def _normalize_summary(s: dict) -> dict:
    """Return a copy of `s` with per-assembly-style field names so the plot
    helpers see a uniform shape regardless of whether the input is the
    per-assembly or batch summary."""
    out = dict(s)

    # Batch summary uses shorter count names; alias to per-assembly style.
    if (
        "per_selector_match_random" in out
        and "per_selector_match_random_count" not in out
    ):
        out["per_selector_match_random_count"] = out["per_selector_match_random"]
    if "pairwise_agree" in out and "pairwise_agreement_count" not in out:
        out["pairwise_agreement_count"] = out["pairwise_agree"]
    if "vlm_meta_match_random" in out and "vlm_meta_match_random_count" not in out:
        out["vlm_meta_match_random_count"] = out["vlm_meta_match_random"]
    if (
        "vlm_meta_match_constituent" in out
        and "vlm_meta_match_constituent_count" not in out
    ):
        out["vlm_meta_match_constituent_count"] = out["vlm_meta_match_constituent"]

    # `vlm_meta_used` isn't recorded in batch; infer from response counts.
    if "vlm_meta_used" not in out:
        out["vlm_meta_used"] = bool(out.get("vlm_meta_responded"))

    # If avg_sample_size is missing (batch), derive a decision-weighted mean
    # from per_assembly records that carry it.
    if not out.get("avg_sample_size") and isinstance(out.get("per_assembly"), list):
        num = 0.0
        den = 0
        for r in out["per_assembly"]:
            ts = r.get("avg_sample_size")
            nd = r.get("total_decisions") or 0
            if ts is not None and nd > 0:
                num += float(ts) * nd
                den += nd
        if den > 0:
            out["avg_sample_size"] = num / den

    # random_baseline_pct from avg_sample_size.
    if not out.get("random_baseline_pct") and out.get("avg_sample_size"):
        out["random_baseline_pct"] = 100.0 / out["avg_sample_size"]

    return out


# ----------------------------------------------------------------------
# plotting helpers (operate on normalized dicts)


def _bar_match_baseline(ax, summary: dict) -> None:
    names = list(summary.get("selectors", []))
    rates = summary.get("per_selector_match_random_rate", {}) or {}
    resp = summary.get("per_selector_responded", {}) or {}
    baseline = (summary.get("random_baseline_pct", 0.0) or 0.0) / 100.0

    if not names:
        ax.text(0.5, 0.5, "(no selectors)", ha="center", va="center")
        ax.set_axis_off()
        return

    pcts = [(rates.get(n) or 0.0) * 100.0 for n in names]
    counts = [(resp.get(n) or 0) for n in names]

    x = np.arange(len(names))
    bars = ax.bar(x, pcts, color="#4C78A8", edgecolor="black")
    ax.axhline(
        baseline * 100.0,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label=f"baseline 1/l = {baseline * 100:.1f}%",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("% matched random baseline")
    ax.set_title("Selector vs random baseline")
    ax.set_ylim(0, max(100, (max([*pcts, baseline * 100]) if pcts else 0) * 1.1))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right")
    for b, n in zip(bars, counts, strict=False):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"n={n}",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def _heatmap_pairwise(ax, summary: dict) -> None:
    names = sorted(summary.get("selectors", []))
    n = len(names)
    if n < 2:
        ax.text(0.5, 0.5, "(needs ≥2 selectors)", ha="center", va="center")
        ax.set_axis_off()
        return

    rates = summary.get("pairwise_agreement_rate", {}) or {}
    counts = summary.get("pairwise_both_responded", {}) or {}

    grid = np.full((n, n), np.nan)
    cnt_grid = np.zeros((n, n), dtype=int)
    for i, _ in enumerate(names):
        grid[i, i] = 1.0
    for a, b in combinations(names, 2):
        key = f"{a}|{b}"
        r = rates.get(key)
        c = counts.get(key, 0) or 0
        i, j = names.index(a), names.index(b)
        if r is not None:
            grid[i, j] = grid[j, i] = float(r)
        cnt_grid[i, j] = cnt_grid[j, i] = int(c)

    im = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap="viridis", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_title("Pairwise agreement rate")
    for i in range(n):
        for j in range(n):
            if np.isnan(grid[i, j]):
                txt = "—"
            elif i == j:
                txt = "·"
            else:
                txt = f"{grid[i, j] * 100:.0f}%\n(n={cnt_grid[i, j]})"
            color = (
                "white" if (not np.isnan(grid[i, j]) and grid[i, j] < 0.55) else "black"
            )
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=7)
    plt.colorbar(im, ax=ax, shrink=0.75, label="agreement rate")


def _bar_vlm_match_constituents(ax, summary: dict) -> None:
    if not summary.get("vlm_meta_used"):
        ax.text(
            0.5,
            0.5,
            "VLM meta-selector disabled",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_axis_off()
        return

    vlm_resp = summary.get("vlm_meta_responded", 0) or 0
    names = list(summary.get("selectors", []))
    rates = summary.get("vlm_meta_match_constituent_rate", {}) or {}
    counts = summary.get("vlm_meta_match_constituent_count", {}) or {}

    if vlm_resp == 0:
        ax.text(
            0.5,
            0.5,
            "VLM meta-selector never responded",
            ha="center",
            va="center",
            fontsize=10,
        )
        ax.set_axis_off()
        return

    pcts = [(rates.get(n) or 0.0) * 100.0 for n in names]
    cnts = [counts.get(n) or 0 for n in names]
    x = np.arange(len(names))
    bars = ax.bar(x, pcts, color="#E45756", edgecolor="black")
    baseline = summary.get("random_baseline_pct", 0.0) or 0.0
    ax.axhline(
        baseline,
        color="dimgray",
        linestyle="--",
        linewidth=1.0,
        label=f"chance ({baseline:.1f}%)",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel(f"% VLM agreed with constituent (n={vlm_resp})")
    ax.set_title("VLM meta-selector ↔ constituent agreement")
    ax.set_ylim(0, max(100, (max([*pcts, baseline]) if pcts else 0) * 1.1))
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="upper right")
    for b, c in zip(bars, cnts, strict=False):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height(),
            f"{c}",
            ha="center",
            va="bottom",
            fontsize=7,
        )


def _forward_source_block(ax, summary: dict, source_label: str) -> None:
    """Per-assembly bottom-right: forward-source bar + key totals."""
    fs = summary.get("forward_source_counts", {}) or {}
    use_vlm_progress = summary.get("use_vlm_for_progress", False)
    total = summary.get("total_decisions", 0)
    vlm_used = summary.get("vlm_meta_used", False)
    vlm_resp = summary.get("vlm_meta_responded", 0) or 0
    vlm_match_random = summary.get("vlm_meta_match_random_rate")
    avg_l = summary.get("avg_sample_size", 0.0) or 0.0

    if fs:
        keys = list(fs.keys())
        vals = [fs[k] for k in keys]
        colors = ["#54A24B" if k == "vlm_meta" else "#9D9D9D" for k in keys]
        ax.bar(keys, vals, color=colors, edgecolor="black")
        for i, v in enumerate(vals):
            ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("decisions")
        ax.set_title("Forward-progress source")
        ax.grid(axis="y", linestyle=":", alpha=0.4)
    else:
        ax.set_axis_off()

    rate = (
        (vlm_match_random * 100.0)
        if isinstance(vlm_match_random, (int, float))
        else None
    )
    txt = (
        f"source: {source_label}\n"
        f"total decisions:        {total}\n"
        f"avg sample size (l):    {avg_l:.2f}\n"
        f"random baseline (1/l):  {summary.get('random_baseline_pct', 0):.1f}%\n"
        f"use_vlm_for_progress:   {use_vlm_progress}\n"
    )
    if vlm_used:
        txt += (
            f"VLM meta responded:     {vlm_resp}/{total}\n"
            f"VLM vs random:          "
            f"{f'{rate:.1f}%' if rate is not None else '—'}\n"
        )
    ax.text(
        1.04,
        0.5,
        txt,
        transform=ax.transAxes,
        va="center",
        ha="left",
        family="monospace",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "#F7F7F7",
            "edgecolor": "#999999",
        },
    )


def _assembly_status_block(ax, summary: dict, source_label: str) -> None:
    """Batch bottom-right: assembly status bar + batch-level totals."""
    n_total = summary.get("n_assemblies", 0)
    n_dec = summary.get("assemblies_with_decisions", 0)
    n_no = summary.get("assemblies_no_decisions", 0)
    n_miss = summary.get("assemblies_missing_summary", 0)
    keys = ["with decisions", "no decisions", "missing"]
    vals = [n_dec, n_no, n_miss]
    colors = ["#54A24B", "#F58518", "#9D9D9D"]
    ax.bar(keys, vals, color=colors, edgecolor="black")
    for i, v in enumerate(vals):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("assemblies")
    ax.set_title(f"Per-assembly status  (n={n_total})")
    ax.grid(axis="y", linestyle=":", alpha=0.4)

    total = summary.get("total_decisions", 0)
    avg_l = summary.get("avg_sample_size", 0.0) or 0.0
    baseline_pct = summary.get("random_baseline_pct", 0.0) or 0.0
    vlm_resp = summary.get("vlm_meta_responded", 0) or 0
    vlm_match_random = summary.get("vlm_meta_match_random_rate")
    vlm_used = summary.get("vlm_meta_used", False)

    rate = (
        (vlm_match_random * 100.0)
        if isinstance(vlm_match_random, (int, float))
        else None
    )
    txt = (
        f"source: {source_label}\n"
        f"assemblies:             {n_total}\n"
        f"  with decisions:       {n_dec}\n"
        f"  no decisions:         {n_no}\n"
        f"  missing summary:      {n_miss}\n"
        f"total decisions:        {total}\n"
        f"avg sample size (l):    {avg_l:.2f}\n"
        f"random baseline (1/l):  {baseline_pct:.1f}%\n"
    )
    if vlm_used:
        txt += (
            f"VLM meta responded:     {vlm_resp}\n"
            f"VLM vs random:          "
            f"{f'{rate:.1f}%' if rate is not None else '—'}\n"
        )
    ax.text(
        1.04,
        0.5,
        txt,
        transform=ax.transAxes,
        va="center",
        ha="left",
        family="monospace",
        fontsize=8,
        bbox={
            "boxstyle": "round,pad=0.6",
            "facecolor": "#F7F7F7",
            "edgecolor": "#999999",
        },
    )


# ----------------------------------------------------------------------
# plotting helpers — k-greedy validation (test_heuristic_validation)


def _validation_finished_rows(summary: dict) -> list[dict]:
    """Per-assembly rows that count as finished: `status=='ok'` AND a finite
    `global_best_score` (the exhaustive-BFS reference that every per-k % is
    normalised against). Anything without a usable reference would just sit
    out of both panels anyway — keeping the filter unified means the title's
    `n=` matches what's actually plotted."""
    out = []
    for rec in summary.get("per_assembly") or []:
        if rec.get("status") != "ok":
            continue
        gb = rec.get("global_best_score")
        if gb is None:
            gb = rec.get("global_best_cost")
        try:
            gb_f = float(gb)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(gb_f):
            continue
        out.append(rec)
    return out


def _validation_effective_max_k(
    summary: dict,
    finished_rows: list[dict] | None = None,
    override: int | None = None,
) -> int:
    """Effective k cap for the validation plot.

    Default (no override): `min(configured max_k_tested, max(n_parts) across
    finished rows)`. The `max(n_parts)` cap turned out to be misleading
    though — for beam search over the subassembly state DAG, k=N isn't
    exhaustive (true exhaustive is closer to C(N, N/2) ≈ 2^N/√N), so the
    `--max-k` CLI flag (-> `override`) skips the data-derived cap and uses
    the user's value directly, still clamped to `max_k_tested` since the
    summary has no k_results past that.

    Returns 0 only when no usable rows AND no configured max_k exist."""
    configured = int(summary.get("max_k_tested", 0) or 0)
    if override is not None and override > 0:
        return min(override, configured) if configured > 0 else override
    rows = (
        finished_rows
        if finished_rows is not None
        else _validation_finished_rows(summary)
    )
    n_parts_list = []
    for r in rows:
        n = r.get("n_parts")
        if n is None:
            n = r.get("n_tree_nodes")
        if n is None:
            continue
        try:
            n_parts_list.append(int(n))
        except (TypeError, ValueError):
            continue
    if not n_parts_list:
        return configured
    cap = max(n_parts_list)
    if configured <= 0:
        return cap
    return min(configured, cap)


def _scatter_size_vs_lowest_k(ax, summary: dict, eff_max_k: int | None = None) -> None:
    """Per-assembly scatter: n_parts (or fall back to n_tree_nodes) on x,
    lowest_k_for_global_optimum on y. Only finished rows contribute; y-axis
    is capped at `eff_max_k` (defaults to the data-derived cap)."""
    finished = _validation_finished_rows(summary)
    if eff_max_k is None:
        eff_max_k = _validation_effective_max_k(summary, finished)

    xs_found, ys_found = [], []
    for rec in finished:
        nx_ = rec.get("n_parts") or rec.get("n_tree_nodes")
        if nx_ is None:
            continue
        lk = rec.get("lowest_k_for_global_optimum")
        if lk is None:
            continue
        xs_found.append(nx_)
        ys_found.append(int(lk))

    if not xs_found:
        ax.text(0.5, 0.5, "(no points to plot)", ha="center", va="center")
        ax.set_axis_off()
        return

    ax.scatter(
        xs_found,
        ys_found,
        color="#4C78A8",
        s=42,
        edgecolor="black",
        linewidth=0.6,
        alpha=0.85,
        label=f"recovered ({len(xs_found)})",
    )
    ax.set_xlabel("# parts in assembly")
    ax.set_ylabel("lowest k for global optimum")
    ax.set_title("Assembly size vs lowest-k needed")
    ax.grid(linestyle=":", alpha=0.5)
    if eff_max_k:
        ax.set_ylim(0.5, eff_max_k + 0.5)
        ax.set_yticks(list(range(1, eff_max_k + 1)))
        ax.set_yticklabels([str(k) for k in range(1, eff_max_k + 1)], fontsize=8)
    ax.legend(fontsize=8, loc="upper left")


def _line_avg_best_cost(ax, summary: dict, eff_max_k: int | None = None) -> None:
    """For each beam width k, plot the mean across assemblies of
    100 * (best score at k) / (global optimum score), where the global
    optimum score is `global_best_score` from exhaustive BFS. Normalising
    per assembly makes assemblies of different absolute score comparable;
    100% is the global optimum.

    One line per distinct `n_parts` value — bigger assemblies have a much
    larger state DAG (≈C(N, N/2) states at the middle), so plotting the
    grand-average across sizes hides where the heuristic actually breaks
    down. Per-size lines make the size-dependence explicit. Only finished
    rows with finite `global_best_score` and at least one finite
    `k_results[k]['best_score']` contribute. `eff_max_k` defaults to the
    data-derived cap.
    """
    finished = _validation_finished_rows(summary)
    if eff_max_k is None:
        eff_max_k = _validation_effective_max_k(summary, finished)
    if not finished or not eff_max_k:
        ax.text(0.5, 0.5, "(no per_assembly data)", ha="center", va="center")
        ax.set_axis_off()
        return

    def _num(d: dict, *keys):
        """First finite float among `keys`, else None."""
        for key in keys:
            v = d.get(key)
            if v is None:
                continue
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(vf):
                return vf
        return None

    ks = list(range(1, eff_max_k + 1))

    # Group finished rows by their integer n_parts (fall back to
    # n_tree_nodes when missing). Rows with no size metadata are skipped
    # because we can't bucket them — they'd just blur the per-size signal
    # we're trying to surface.
    by_size: dict[int, list[dict]] = {}
    for rec in finished:
        np_ = rec.get("n_parts") or rec.get("n_tree_nodes")
        if np_ is None:
            continue
        try:
            np_i = int(np_)
        except (TypeError, ValueError):
            continue
        by_size.setdefault(np_i, []).append(rec)

    if not by_size:
        ax.text(0.5, 0.5, "(no n_parts metadata)", ha="center", va="center")
        ax.set_axis_off()
        return

    sizes = sorted(by_size.keys())
    # Viridis stepped per size — bigger assemblies are visually "later" in
    # the colormap, matching the intuition that bigger N is harder.
    cmap = plt.get_cmap("viridis")
    if len(sizes) > 1:
        colors = [cmap(i / (len(sizes) - 1)) for i in range(len(sizes))]
    else:
        colors = [cmap(0.5)]

    x = np.arange(len(ks))
    any_data = False
    # Per-k buckets pooled across ALL sizes — used for the grand-average
    # line drawn over the per-size lines.
    all_sizes_per_k: dict[int, list[float]] = {k: [] for k in ks}
    n_total = 0
    for color, size in zip(colors, sizes, strict=False):
        rows = by_size[size]
        per_k_pct: dict[int, list[float]] = {k: [] for k in ks}
        for rec in rows:
            gb = _num(rec, "global_best_score", "global_best_cost")
            if gb is None or gb == 0:
                continue
            kres = rec.get("k_results") or {}
            for k in ks:
                entry = kres.get(k) or kres.get(str(k))
                if not isinstance(entry, dict):
                    continue
                vf = _num(entry, "best_score", "best_cost")
                if vf is None:
                    continue
                per_k_pct[k].append(100.0 * vf / gb)
                all_sizes_per_k[k].append(100.0 * vf / gb)
        if not any(per_k_pct[k] for k in ks):
            continue
        means = [float(np.mean(per_k_pct[k])) if per_k_pct[k] else np.nan for k in ks]
        ax.plot(
            x,
            means,
            marker="o",
            markersize=4,
            color=color,
            linewidth=1.4,
            label=f"n_parts={size}  (m={len(rows)})",
        )
        any_data = True
        n_total += len(rows)

    if not any_data:
        ax.text(0.5, 0.5, "(no best_score data per k)", ha="center", va="center")
        ax.set_axis_off()
        return

    # Grand average across all sizes, drawn on top in black so it reads as
    # the headline summary rather than blending with the viridis per-size
    # lines. Same per-assembly weighting as the per-size lines: each row
    # contributes one datapoint per k to whichever buckets it has data for.
    if any(all_sizes_per_k[k] for k in ks):
        overall_means = [
            float(np.mean(all_sizes_per_k[k])) if all_sizes_per_k[k] else np.nan
            for k in ks
        ]
        ax.plot(
            x,
            overall_means,
            marker="o",
            markersize=5,
            color="black",
            linewidth=2.0,
            zorder=5,
            label=f"all sizes  (n={n_total})",
        )

    ax.axhline(
        100.0,
        color="#54A24B",
        linestyle="--",
        linewidth=1.0,
        label="global optimum (100%)",
    )

    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks], fontsize=9)
    ax.set_xlabel("beam width k")
    ax.set_ylabel("mean best score (% of global optimum)")
    ax.set_title("Average best-score recovered per k, by assembly size")
    ax.grid(linestyle=":", alpha=0.5)
    ax.legend(fontsize=8, loc="lower right")


# ----------------------------------------------------------------------
# plotting helpers — assembly-time generator comparison (data_assembly_time)


_TIME_COMPONENT_COLORS = {
    "step_disassembly_s": "#4C72B0",  # blue
    "transitions_s": "#55A868",  # green
    "base_travel_s": "#C44E52",  # red
    "reorientation_s": "#8172B2",  # purple
    "hold_s": "#CCB974",  # tan
}
_TIME_COMPONENT_LABELS = {
    "step_disassembly_s": "disassembly",
    "transitions_s": "transitions",
    "base_travel_s": "base travel",
    "reorientation_s": "reorientation",
    "hold_s": "hold penalty",
}


def _run_labels_of(summary: dict) -> list[str]:
    runs = summary.get("runs") or []
    out = []
    for r in runs:
        if isinstance(r, dict) and r.get("label"):
            out.append(str(r["label"]))
        elif isinstance(r, str):
            out.append(r)
    return out


def _bar_assembly_time_means(ax, summary: dict) -> None:
    """Stacked bar per run, height = mean total assembly time across OK runs,
    stacking = mean per-component contribution. Matches the cross-assembly
    chart from `data_assembly_time` in main.py but without std error bars."""
    run_labels = _run_labels_of(summary)
    components = list(summary.get("components") or _TIME_COMPONENT_COLORS.keys())
    per_assembly = summary.get("per_assembly") or {}

    if not run_labels:
        ax.text(0.5, 0.5, "(no runs in summary)", ha="center", va="center")
        ax.set_axis_off()
        return

    # Per-run per-component sample lists (only OK assemblies contribute).
    samples: dict[str, dict[str, list[float]]] = {
        label: {c: [] for c in components} for label in run_labels
    }
    n_ok = dict.fromkeys(run_labels, 0)
    for runs_data in per_assembly.values():
        if not isinstance(runs_data, dict):
            continue
        for label in run_labels:
            entry = runs_data.get(label) or {}
            if entry.get("status") != "ok":
                continue
            totals = entry.get("totals") or {}
            n_ok[label] += 1
            for c in components:
                samples[label][c].append(float(totals.get(c, 0.0) or 0.0))

    x = np.arange(len(run_labels))
    bottom = np.zeros(len(run_labels))
    for c in components:
        vals = [
            (float(np.mean(samples[label][c])) if samples[label][c] else 0.0)
            for label in run_labels
        ]
        ax.bar(
            x,
            vals,
            bottom=bottom,
            label=_TIME_COMPONENT_LABELS.get(c, c),
            color=_TIME_COMPONENT_COLORS.get(c, "#9D9D9D"),
            edgecolor="white",
            linewidth=0.3,
        )
        bottom += np.array(vals)

    # Mark zero-data runs with a hatched empty bar so they read as missing.
    for i, label in enumerate(run_labels):
        if n_ok[label] == 0:
            ax.bar([x[i]], [0.0], color="lightgray")
            ax.text(
                x[i],
                0.0,
                "no data",
                ha="center",
                va="bottom",
                fontsize=8,
                color="dimgray",
            )

    for xi, label in enumerate(run_labels):
        ax.text(
            xi, bottom[xi], f"n={n_ok[label]}", ha="center", va="bottom", fontsize=8
        )

    ax.set_xticks(x)
    ax.set_xticklabels(run_labels, rotation=30, ha="right")
    ax.set_ylabel("mean estimated assembly time (s)")
    ax.set_title("Mean total time per generator  (OK runs only)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)


DEFAULT_ASSEMBLY_TIME_BASELINE = "heur-out"


def _resolve_baseline_label(summary: dict, baseline_key: str) -> str | None:
    """Match `baseline_key` against the runs metadata. Tries the `generator`
    field first (so the ASAPx generator name `heur-out` resolves to whatever
    label is paired with it, e.g. `gen:heur-out`), then the `label`. Returns
    None if no match exists in this summary."""
    runs = summary.get("runs") or []
    for r in runs:
        if not isinstance(r, dict):
            continue
        if r.get("generator") == baseline_key or r.get("label") == baseline_key:
            return r.get("label")
    if baseline_key in _run_labels_of(summary):
        return baseline_key
    return None


def _gather_baseline_ratios(
    summary: dict, baseline_label: str
) -> dict[str, list[tuple[str, float]]]:
    """For each non-baseline run, collect `(assembly_id, total_s / baseline_total_s)`
    pairs. Drops assemblies where the baseline or the run failed, or where the
    baseline total is non-positive (avoids divide-by-zero blowups)."""
    run_labels = _run_labels_of(summary)
    per_assembly = summary.get("per_assembly") or {}
    out: dict[str, list[tuple[str, float]]] = {
        label: [] for label in run_labels if label != baseline_label
    }
    for aid, runs_data in per_assembly.items():
        if not isinstance(runs_data, dict):
            continue
        base_entry = runs_data.get(baseline_label) or {}
        if base_entry.get("status") != "ok":
            continue
        base_total = float((base_entry.get("totals") or {}).get("total_s", 0.0) or 0.0)
        if base_total <= 0:
            continue
        for label in out:
            entry = runs_data.get(label) or {}
            if entry.get("status") != "ok":
                continue
            t = float((entry.get("totals") or {}).get("total_s", 0.0) or 0.0)
            if t <= 0:
                continue
            out[label].append((aid, t / base_total))
    return out


def _gm_se_ci(
    vals: list[float],
    n_bootstrap: int = 2000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> tuple[float, tuple[float, float], tuple[float, float]]:
    """For a list of strictly positive ratios, return:
      gm           — geometric mean = exp(mean(ln(x)))
      (se_lo, se_hi) — ±1 geometric SE bounds = gm * exp(±SE(ln x))
      (ci_lo, ci_hi) — `(1-alpha)` percentile-bootstrap CI on the GM

    Geometric stats are the right home for ratio data: log-transforming makes
    the distribution roughly symmetric, so mean and SE in log-space are
    meaningful even when the raw ratios are heavy-tailed. The bootstrap CI
    makes no parametric assumption — robust against the wild swings between
    small and large assemblies.

    n < 2 makes the SE undefined and the CI a point; we return (gm, gm) for
    both bands rather than NaN so the caller can blindly use the numbers.
    """
    if not vals:
        raise ValueError("empty input")
    log_arr = np.log(np.asarray(vals, dtype=float))
    n = len(log_arr)
    mu = float(np.mean(log_arr))
    gm = float(np.exp(mu))
    if n < 2:
        return gm, (gm, gm), (gm, gm)
    se_log = float(np.std(log_arr, ddof=1) / np.sqrt(n))
    se_lo = float(np.exp(mu - se_log))
    se_hi = float(np.exp(mu + se_log))
    if rng is None:
        rng = np.random.default_rng(0)
    idx = rng.integers(0, n, size=(n_bootstrap, n))
    boot_log_means = log_arr[idx].mean(axis=1)
    boot_gms = np.exp(boot_log_means)
    ci_lo = float(np.quantile(boot_gms, alpha / 2))
    ci_hi = float(np.quantile(boot_gms, 1 - alpha / 2))
    return gm, (se_lo, se_hi), (ci_lo, ci_hi)


def _bar_assembly_time_ratios(ax, summary: dict, baseline_label: str) -> None:
    """Per-assembly strip plot of `total_s / baseline_total_s` for each
    non-baseline run. Each blue dot is one assembly (jittered horizontally);
    overlaid to the right of each column is the geometric mean (diamond)
    with a ±1 geometric SE span. Each assembly is its own control, so
    part-count variance cancels and the residual signal is relative-time."""
    ratios_per_run = _gather_baseline_ratios(summary, baseline_label)
    if not ratios_per_run:
        ax.text(
            0.5,
            0.5,
            f"(no non-baseline runs paired with {baseline_label})",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        return

    labels = list(ratios_per_run.keys())
    values_per_run = [[r for _, r in ratios_per_run[l]] for l in labels]

    positions = np.arange(len(labels))

    # Boxplot of each run's ratio distribution (box = IQR, line = median,
    # whiskers = 1.5·IQR), drawn behind the scatter + GM/SE overlay so the
    # per-assembly dots stay readable on top. Fliers are suppressed because
    # the scatter already shows every individual assembly. Only runs that
    # actually have paired assemblies get a box.
    box_positions = [positions[i] for i, v in enumerate(values_per_run) if v]
    box_values = [v for v in values_per_run if v]
    if box_values:
        ax.boxplot(
            box_values,
            positions=box_positions,
            widths=0.5,
            showfliers=False,
            patch_artist=True,
            zorder=1,
            boxprops={
                "facecolor": "#CBD9EC",
                "edgecolor": "#2A4F77",
                "linewidth": 1.0,
                "alpha": 0.9,
            },
            medianprops={"color": "black", "linewidth": 1.6},
            whiskerprops={"color": "#2A4F77", "linewidth": 1.0},
            capprops={"color": "#2A4F77", "linewidth": 1.0},
        )

    rng = np.random.default_rng(0)
    for i, v in enumerate(values_per_run):
        if not v:
            continue
        jitter = (rng.random(len(v)) - 0.5) * 0.25
        ax.scatter(
            np.full(len(v), positions[i]) + jitter,
            v,
            s=18,
            color="#2A4F77",
            alpha=0.7,
            edgecolor="white",
            linewidth=0.4,
            zorder=3,
        )

    # GM + SE per run, drawn to the right of the scatter cluster so the
    # raw points and the inferential overlay stay legible side-by-side
    # instead of stacking on top of each other.
    SE_COLOR = "#B83C5C"  # rosé
    GM_COLOR = "#B83C5C"
    se_offset = 0.30
    rng_boot = np.random.default_rng(42)
    gm_se: list[tuple[float, float, float] | None] = []  # for annotation below
    for i, v in enumerate(values_per_run):
        if not v:
            gm_se.append(None)
            continue
        gm, (se_lo, se_hi), _ = _gm_se_ci(
            v,
            n_bootstrap=2000,
            alpha=0.05,
            rng=rng_boot,
        )
        gm_se.append((gm, se_lo, se_hi))
        x_err = positions[i] + se_offset
        # Thick span: ±1 geometric SE.
        ax.plot(
            [x_err, x_err],
            [se_lo, se_hi],
            color=SE_COLOR,
            linewidth=4.0,
            solid_capstyle="butt",
            alpha=0.85,
            zorder=5,
        )
        # GM marker.
        ax.scatter(
            [x_err],
            [gm],
            marker="D",
            s=36,
            color=GM_COLOR,
            edgecolor="white",
            linewidth=0.8,
            zorder=6,
        )

    # Decide log y-scale before computing annotation positions so labels
    # stay above the scatter cluster in the right coordinate system.
    flat = [x for v in values_per_run for x in v]
    use_log = bool(flat) and (max(flat) / max(min(flat), 1e-9) > 10.0)
    if use_log:
        from matplotlib import ticker

        ax.set_yscale("log")
        # Ticks at {1, 2, 5} × 10^n per decade — canonical log spacing that
        # stays readable, and every tick gets its own plain-number label
        # ("1" instead of "10^0", "0.5" instead of "5×10⁻¹") via %g.
        ax.yaxis.set_major_locator(
            ticker.LogLocator(base=10.0, subs=(1.0, 2.0, 5.0), numticks=12)
        )
        ax.yaxis.set_minor_locator(ticker.NullLocator())
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _pos: f"{y:g}"))

    # Mark empty columns (no paired assemblies for that run) so they don't
    # silently look like a 1.0 ratio. Single-line note centred on the
    # column; everything else is the scatter + overlay above.
    for i, info in enumerate(gm_se):
        if info is None:
            ax.text(
                positions[i],
                1.0,
                "no data",
                ha="center",
                va="bottom",
                fontsize=8,
                color="dimgray",
            )

    ax.axhline(
        1.0,
        color="crimson",
        linestyle="--",
        linewidth=1.2,
        label=f"baseline = {baseline_label}",
    )

    # Custom legend entries for the boxplot + GM / SE overlay.
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    legend_handles = [
        Line2D(
            [0],
            [0],
            color="crimson",
            linestyle="--",
            linewidth=1.2,
            label=f"baseline = {baseline_label}",
        ),
        Patch(
            facecolor="#CBD9EC",
            edgecolor="#2A4F77",
            label="ratio distribution (box: IQR, whiskers: 1.5·IQR)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color=GM_COLOR,
            linestyle="",
            markersize=7,
            markeredgecolor="white",
            label="geometric mean",
        ),
        Line2D([0], [0], color=SE_COLOR, linewidth=4.0, label="±1 geometric SE"),
    ]
    ax.legend(handles=legend_handles, fontsize=8, loc="upper right")

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel(f"total_s / total_s ({baseline_label})")
    ax.set_title(f"Assembly time relative to {baseline_label}  (per-assembly ratio)")
    ax.grid(axis="y", linestyle=":", alpha=0.5)


def _default_assembly_time_save_path(json_path: Path) -> Path:
    suffix = _variant_suffix(json_path)
    return (
        json_path.parent
        / f"assembly_time_comparison_{_run_tag_from(json_path)}{suffix}.png"
    )


def plot_assembly_time_summary(
    json_path: Path,
    save_path: Path | None = None,
    show: bool = False,
    baseline: str = DEFAULT_ASSEMBLY_TIME_BASELINE,
) -> Path | None:
    """Plot an `assembly_time_summary.json` produced by `data_assembly_time`.

    Single-panel per-assembly ratio chart: for each non-baseline run, plot the
    distribution of `total_s / total_s(baseline)` across assemblies. Each
    assembly serves as its own control so part-count variance cancels and the
    chart reads as "relative speedup vs the baseline generator"."""
    try:
        summary = load_summary(json_path)
    except Exception as e:
        print(f"[plot] {json_path}: failed to load ({e})")
        return None

    if not summary.get("per_assembly"):
        print(f"[plot] {json_path}: empty assembly_time summary, skipping")
        return None

    source_label = f"…/{json_path.parent.name}/{json_path.name}"
    primary = _render_assembly_time_figure(
        summary,
        source_label,
        save_path=save_path or _default_assembly_time_save_path(json_path),
        show=show,
        baseline=baseline,
    )

    # Side overview: the per-run GM/SE/n table we used to inline above each
    # column. Written next to the chart so headline visual stays clean.
    overview_path = _default_assembly_time_overview_path(
        save_path if save_path is not None else json_path,
    )
    baseline_label = _resolve_baseline_label(summary, baseline)
    if baseline_label is not None:
        overview = _render_assembly_time_overview(
            summary,
            baseline_label,
            save_path=overview_path,
            show=show,
        )
        if overview is not None:
            print(f"  ✓ overview  →  {overview}")
    return primary


def _render_assembly_time_figure(
    summary: dict,
    source_label: str,
    save_path: Path,
    show: bool = False,
    title_suffix: str = "",
    baseline: str = DEFAULT_ASSEMBLY_TIME_BASELINE,
) -> Path | None:
    run_labels = _run_labels_of(summary)
    baseline_label = _resolve_baseline_label(summary, baseline)
    fig, ax = plt.subplots(figsize=(max(6, 1.2 * max(len(run_labels), 1)), 5.0))
    if baseline_label is None:
        ax.text(
            0.5,
            0.5,
            f"(baseline '{baseline}' not found in runs metadata)",
            ha="center",
            va="center",
        )
        ax.set_axis_off()
    else:
        _bar_assembly_time_ratios(ax, summary, baseline_label)
    n_assemblies = len(summary.get("per_assembly") or {})
    fig.suptitle(
        f"Assembly-Time Generator Comparison  "
        f"({n_assemblies} assemblies){title_suffix}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def _default_assembly_time_overview_path(reference_path: Path) -> Path:
    """Derive the overview-stats PNG path from the main-plot path: same
    directory, same suffix, but `comparison` → `stats` (or appended
    `_stats` if no `comparison` token is present). Falls back to deriving
    from the source JSON tag when given a JSON path."""
    if reference_path.suffix == ".png":
        stem = reference_path.stem
        if "comparison" in stem:
            new_stem = stem.replace("comparison", "stats")
        else:
            new_stem = f"{stem}_stats"
        return reference_path.with_name(f"{new_stem}.png")
    # JSON path → mirror the main-plot naming.
    suffix = _variant_suffix(reference_path)
    return (
        reference_path.parent
        / f"assembly_time_stats_{_run_tag_from(reference_path)}{suffix}.png"
    )


def _render_assembly_time_overview(
    summary: dict, baseline_label: str, save_path: Path, show: bool = False
) -> Path | None:
    """Side companion PNG: per-run table with n, raw min/max, geometric mean
    (GM), and ±1 geometric SE bounds. Same statistics that used to live
    above each column in the main chart; broken out so the chart stays
    clean and the numbers stay easy to read."""
    ratios_per_run = _gather_baseline_ratios(summary, baseline_label)
    if not ratios_per_run:
        return None

    labels = list(ratios_per_run.keys())
    rng_boot = np.random.default_rng(42)
    cell_text = []
    for label in labels:
        vals = [r for _, r in ratios_per_run[label]]
        if not vals:
            cell_text.append([label, "0", "—", "—", "—", "—"])
            continue
        gm, (se_lo, se_hi), _ = _gm_se_ci(
            vals,
            n_bootstrap=2000,
            alpha=0.05,
            rng=rng_boot,
        )
        cell_text.append(
            [
                label,
                str(len(vals)),
                f"{min(vals):.2f}",
                f"{max(vals):.2f}",
                f"{gm:.2f}",
                f"[{se_lo:.2f}, {se_hi:.2f}]",
            ]
        )

    headers = ["run", "n", "min", "max", "GM", "±1 SE"]
    # Run column is wider than each numeric column so longer labels
    # ("heuristic_trained", "heuristic+optimizer") aren't truncated.
    col_widths = [1.43, 1.0, 1.0, 1.0, 1.0, 1.0]
    total_weight = sum(col_widths)
    col_widths_norm = [w / total_weight for w in col_widths]
    n_rows = len(cell_text) + 1
    # Bump the figure width proportionally so the numeric columns keep
    # roughly their previous absolute size instead of getting squeezed.
    fig_width = 8.0 * (total_weight / 6.0)
    fig, ax = plt.subplots(figsize=(fig_width, max(2.2, 0.55 * n_rows)))
    ax.set_axis_off()
    ax.set_title(
        f"Assembly time vs {baseline_label}  —  per-run statistics",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    tbl = ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=col_widths_norm,
        colColours=["#E8EEF7"] * len(headers),
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.55)
    for (row, _col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_text_props(fontweight="bold")
        cell.set_edgecolor("#9AAAC2")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def _merge_assembly_time_summaries(summaries: list[dict]) -> dict:
    """Merge a list of assembly_time summaries by concatenating per_assembly
    entries. Runs metadata is taken from the first non-empty summary; we
    assume the same RUNS list across the inputs (the typical case)."""
    out: dict = {
        "runs": [],
        "components": [],
        "per_assembly": {},
    }
    for s in summaries:
        if not out["runs"] and s.get("runs"):
            out["runs"] = s["runs"]
        if not out["components"] and s.get("components"):
            out["components"] = list(s["components"])
        for aid, runs_data in (s.get("per_assembly") or {}).items():
            # If the same assembly id shows up twice, the later file wins —
            # callers are responsible for not double-counting.
            out["per_assembly"][aid] = runs_data
    return out


# ----------------------------------------------------------------------
# plotting helpers — manual-validation ablation batch (data_manual_validation)
#
# Replicates the bars from `manual_validator._write_batch_plot` but emits
# the two panels as separate files: a clean summary chart (batch totals per
# ablation) and a per-assembly overview chart (grouped bars). Same colors
# and labels as the in-pipeline plot so the two are visually interchangeable.


def _manual_validation_complete_rows(batch: dict) -> list[dict]:
    """Filter per-assembly rows to those that were fully evaluated.

    An assembly contributes to either plot only if every disassembly step
    produced a verdict — i.e. `yes + no == n_steps` and `n_steps > 0`. Rows
    where the run aborted partway (timeout, error, missing summary, …) are
    silently dropped so they neither move the batch percentages nor clutter
    the per-assembly chart."""
    out = []
    for r in batch.get("per_assembly") or []:
        if r.get("status") != "ok":
            continue
        n_steps = int(r.get("n_steps", 0) or 0)
        yes = int(r.get("yes", 0) or 0)
        no = int(r.get("no", 0) or 0)
        if n_steps <= 0 or yes + no != n_steps:
            continue
        out.append(r)
    return out


class _AblationAgg(TypedDict):
    yes: int
    no: int
    verdict_rate: float | None


def _aggregate_per_ablation(rows: list[dict]) -> dict:
    """Sum yes/no per ablation across the given rows; verdict_rate recomputed
    from the sums. Mirrors `manual_validator._write_batch_summary` but works
    off a filtered row list so it can express "totals over complete runs"."""
    out: dict[str, _AblationAgg] = {
        a: _AblationAgg(yes=0, no=0, verdict_rate=None) for a in _MANUAL_ABLATIONS
    }
    for r in rows:
        per_abl = r.get("per_ablation") or {}
        for a in _MANUAL_ABLATIONS:
            agg = per_abl.get(a) or {}
            out[a]["yes"] += int(agg.get("yes") or 0)
            out[a]["no"] += int(agg.get("no") or 0)
    for a in _MANUAL_ABLATIONS:
        s = out[a]["yes"] + out[a]["no"]
        out[a]["verdict_rate"] = (out[a]["yes"] / s) if s else None
    return out


def _bar_manual_validation_summary(ax, batch: dict) -> None:
    """Top-panel equivalent: one bar per ablation, height = batch-wide
    verdict-rate (% yes) computed from complete-assembly contributions only,
    labels = yes/total raw counts over the same complete set."""
    rows = _manual_validation_complete_rows(batch)
    per_abl = _aggregate_per_ablation(rows)
    rates = []
    labels = []
    for ablation in _MANUAL_ABLATIONS:
        agg = per_abl.get(ablation) or {}
        rate = agg.get("verdict_rate")
        rates.append(100.0 * rate if rate is not None else 0.0)
        labels.append(f"{agg.get('yes', 0)}/{agg.get('yes', 0) + agg.get('no', 0)}")

    bars = ax.bar(list(_MANUAL_ABLATIONS), rates, color=list(_MANUAL_ABLATION_COLORS))
    ax.set_ylim(0, 105)
    ax.set_ylabel("Verdict rate  (% yes)")
    ax.set_title("Manual validation — ablation study (batch totals)")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for bar, rate, label in zip(bars, rates, labels, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            rate + 2,
            f"{rate:.1f}%\n({label})",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def _bar_manual_validation_per_assembly(ax, batch: dict) -> None:
    """Bottom-panel equivalent: one cluster per complete assembly, one
    coloured bar per ablation. Shows per-assembly variance that the summary
    bars hide. Incomplete runs are filtered out."""
    rows = _manual_validation_complete_rows(batch)
    if not rows:
        ax.axis("off")
        ax.text(
            0.5,
            0.5,
            "(no per-assembly data)",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        return

    x = np.arange(len(rows))
    width = 0.8 / len(_MANUAL_ABLATIONS)
    for i, ablation in enumerate(_MANUAL_ABLATIONS):
        heights = []
        for r in rows:
            agg = (r.get("per_ablation") or {}).get(ablation) or {}
            rate = agg.get("verdict_rate")
            heights.append(100.0 * rate if rate is not None else 0.0)
        ax.bar(
            x + (i - (len(_MANUAL_ABLATIONS) - 1) / 2) * width,
            heights,
            width=width,
            color=_MANUAL_ABLATION_COLORS[i],
            label=ablation,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([str(r.get("id", "?")) for r in rows], rotation=45, ha="right")
    ax.set_ylim(0, 105)
    ax.set_ylabel("Verdict rate  (% yes)")
    ax.set_xlabel("Assembly id")
    ax.set_title("Per-assembly ablation rates")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    ax.legend(loc="lower right", ncol=len(_MANUAL_ABLATIONS), fontsize=8)


def _default_manual_validation_save_paths(json_path: Path) -> tuple[Path, Path]:
    """Return (summary_png, per_assembly_png) next to the JSON."""
    suffix = _variant_suffix(json_path)
    tag = _run_tag_from(json_path)
    return (
        json_path.parent / f"manual_validation_summary_{tag}{suffix}.png",
        json_path.parent / f"manual_validation_per_assembly_{tag}{suffix}.png",
    )


def _render_manual_validation_summary(
    batch: dict, save_path: Path, show: bool = False, title_suffix: str = ""
) -> Path | None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    _bar_manual_validation_summary(ax, batch)
    n_complete = len(_manual_validation_complete_rows(batch))
    fig.suptitle(
        f"Manual Validation — batch summary  (n={n_complete}){title_suffix}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def _render_manual_validation_per_assembly(
    batch: dict, save_path: Path, show: bool = False, title_suffix: str = ""
) -> Path | None:
    rows = _manual_validation_complete_rows(batch)
    width = max(8.0, 0.55 * len(rows) * len(_MANUAL_ABLATIONS) + 4.0)
    height = max(4.0, 0.4 + 0.18 * len(rows) + 3.0)
    fig, ax = plt.subplots(figsize=(width, height))
    _bar_manual_validation_per_assembly(ax, batch)
    fig.suptitle(
        f"Manual Validation — per-assembly overview  (n={len(rows)}){title_suffix}",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def plot_manual_validation_batch(
    json_path: Path, save_path: Path | None = None, show: bool = False
) -> Path | None:
    """Render `manual_validation_batch.json` as two side-by-side files:
    a clean ablation summary and a per-assembly overview. Returns the
    summary file's path (the per-assembly file is written as a side-effect
    and printed separately) so the caller can keep using the same Path|None
    contract as the other plot kinds."""
    try:
        batch = load_summary(json_path)
    except Exception as e:
        print(f"[plot] {json_path}: failed to load ({e})")
        return None

    if not batch.get("per_assembly") and not batch.get("per_ablation"):
        print(f"[plot] {json_path}: empty manual_validation batch, skipping")
        return None
    if not _manual_validation_complete_rows(batch):
        print(
            f"[plot] {json_path}: no fully-evaluated assemblies "
            "(yes+no != n_steps everywhere), skipping"
        )
        return None

    if save_path is not None:
        # If the caller picked one explicit path (e.g. via --out), put the
        # summary there and derive the per-assembly path next to it.
        summary_path = save_path
        per_assembly_path = save_path.with_name(
            save_path.stem.replace(
                "manual_validation_summary", "manual_validation_per_assembly"
            )
            + save_path.suffix
        )
        # Fallback when the caller's filename didn't contain the summary tag.
        if per_assembly_path == save_path:
            per_assembly_path = save_path.with_name(
                f"{save_path.stem}_per_assembly{save_path.suffix}"
            )
    else:
        summary_path, per_assembly_path = _default_manual_validation_save_paths(
            json_path
        )

    summary_out = _render_manual_validation_summary(batch, summary_path, show=show)
    per_assembly_out = _render_manual_validation_per_assembly(
        batch,
        per_assembly_path,
        show=show,
    )
    if per_assembly_out is not None:
        print(f"  ✓ per-assembly  →  {per_assembly_out}")
    return summary_out


def _merge_manual_validation_batches(batches: list[dict]) -> dict:
    """Combine multiple `manual_validation_batch.json` payloads. Counts add,
    rates recompute from the merged counts, per_assembly rows concatenate."""
    out: dict = {
        "n_assemblies": 0,
        "n_ok": 0,
        "n_missing": 0,
        "n_steps_total": 0,
        "yes": 0,
        "no": 0,
        "verdict_rate": None,
        "per_ablation": {
            a: {"yes": 0, "no": 0, "verdict_rate": None} for a in _MANUAL_ABLATIONS
        },
        "per_assembly": [],
    }
    for b in batches:
        out["n_assemblies"] += int(b.get("n_assemblies") or 0)
        out["n_ok"] += int(b.get("n_ok") or 0)
        out["n_missing"] += int(b.get("n_missing") or 0)
        out["n_steps_total"] += int(b.get("n_steps_total") or 0)
        out["yes"] += int(b.get("yes") or 0)
        out["no"] += int(b.get("no") or 0)
        for ablation in _MANUAL_ABLATIONS:
            agg = (b.get("per_ablation") or {}).get(ablation) or {}
            out["per_ablation"][ablation]["yes"] += int(agg.get("yes") or 0)
            out["per_ablation"][ablation]["no"] += int(agg.get("no") or 0)
        if isinstance(b.get("per_assembly"), list):
            out["per_assembly"].extend(b["per_assembly"])

    # Recompute rates from merged counts.
    yn = out["yes"] + out["no"]
    out["verdict_rate"] = (out["yes"] / yn) if yn else None
    for ablation in _MANUAL_ABLATIONS:
        agg = out["per_ablation"][ablation]
        s = agg["yes"] + agg["no"]
        agg["verdict_rate"] = (agg["yes"] / s) if s else None
    return out


# ----------------------------------------------------------------------
# plotting helpers — convex-decomp AI axis-picker validation
# (test_convex_decomp data-collection mode)
#
# Each per-assembly `convex_decomp_results.json` (legacy: `results.json`
# inside a `convex_decomp/<id>/` directory) holds:
#   { assembly_id, n_samples,
#     per_tool: { tool_id: [ { sample, direction, contact, chosen_img,
#                              correct: bool }, ... ] } }
# We collapse this per-sample structure into per-tool accuracy = correct/total
# and plot a bar chart with binomial SE error bars, mirroring the chart
# that main.py emits at the end of the test_convex_decomp run.


def _convex_decomp_tally(payload: dict) -> tuple[dict[str, int], dict[str, int]]:
    """For one per-assembly payload, return `(totals, correct)` keyed by
    `tool_id`. Entries with `correct=None` are ignored — they represent
    samples the user hasn't validated yet."""
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    for tool_id, entries in (payload.get("per_tool") or {}).items():
        if not isinstance(entries, list):
            continue
        for e in entries:
            v = e.get("correct")
            if v is None:
                continue
            totals[tool_id] = totals.get(tool_id, 0) + 1
            if v:
                correct[tool_id] = correct.get(tool_id, 0) + 1
    return totals, correct


def _merge_convex_decomp_tallies(
    tallies: list[tuple[dict[str, int], dict[str, int]]],
) -> tuple[dict[str, int], dict[str, int]]:
    """Sum tallies across multiple per-assembly payloads, keyed by tool id
    so the same tool used by multiple assemblies aggregates into one bar."""
    totals: dict[str, int] = {}
    correct: dict[str, int] = {}
    for t, c in tallies:
        for k, v in t.items():
            totals[k] = totals.get(k, 0) + v
        for k, v in c.items():
            correct[k] = correct.get(k, 0) + v
    return totals, correct


def _default_convex_decomp_save_path(json_path: Path) -> Path:
    suffix = _variant_suffix(json_path)
    return (
        json_path.parent
        / f"convex_decomp_accuracy_{_assembly_tag_from(json_path)}{suffix}.png"
    )


def _render_convex_decomp_accuracy(
    tool_totals: dict[str, int],
    tool_correct: dict[str, int],
    save_path: Path,
    show: bool = False,
    n_samples: int | None = None,
    title_suffix: str = "",
) -> Path | None:
    """Bar chart of per-tool AI accuracy with binomial SE error bars.
    Same colours / sizing / label rule as main.py's in-pipeline plot so
    re-plotted figures look like the originals."""
    if not tool_totals:
        return None
    tools_sorted = sorted(tool_totals.keys())
    ps, ses = [], []
    for tid in tools_sorted:
        n = tool_totals[tid]
        c = tool_correct.get(tid, 0)
        p = c / n if n else 0.0
        se = (p * (1 - p) / n) ** 0.5 if n > 0 else 0.0
        ps.append(p * 100.0)
        ses.append(se * 100.0)

    fig, ax = plt.subplots(figsize=(max(7, 1.0 * len(tools_sorted) + 4), 5.0))
    xpos = np.arange(len(tools_sorted))
    ax.bar(
        xpos,
        ps,
        yerr=ses,
        capsize=4,
        color="#4C78A8",
        edgecolor="black",
        linewidth=0.8,
        alpha=0.9,
    )
    for xi, (p, se) in enumerate(zip(ps, ses, strict=False)):
        ax.text(xi, p + se + 1.5, f"{p:.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_xticks(xpos)
    ax.set_xticklabels(tools_sorted, rotation=30, ha="right")
    ax.set_ylabel("AI accuracy  (% accepted by validator)")
    title = "AI axis-picker accuracy by tool"
    if n_samples:
        title += f"  (n_samples={n_samples})"
    title += "; SE bars"
    if title_suffix:
        title += title_suffix
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def _resolve_chosen_img(rel_or_abs: str, json_path: Path) -> Path | None:
    """Map a `chosen_img` string in `results.json` to a real file on disk.

    Stored paths are normally relative to the AssembleX working
    directory (e.g. `assets/output/.../sample_00/allenkey/allenkey_chosen.png`).
    We try the literal path first (relative to cwd) so the common case
    works without IO, then walk the JSON's parent ancestors so a moved or
    re-rooted results file still resolves cleanly."""
    if not rel_or_abs:
        return None
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p if p.exists() else None
    if p.exists():
        return p
    for ancestor in [json_path.parent, *json_path.parents]:
        cand = ancestor / p
        if cand.exists():
            return cand
    return None


def _render_convex_decomp_all_tools(
    payload: dict, save_path: Path, json_path: Path, show: bool = False
) -> Path | None:
    """Regenerate the `all_tools.png` grid that the in-pipeline plot in
    main.py produces: one cell per tool, sample-0's chosen-direction
    preview, 5 cells per row wrapping to new rows. Same sizing rules as
    main.py so the re-plot looks like the original."""
    per_tool = payload.get("per_tool") or {}
    if not per_tool:
        return None
    tool_ids = list(per_tool.keys())
    n_cols = 5
    n_rows = (len(tool_ids) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.5 * n_cols, 3.0 * n_rows),
        squeeze=False,
    )
    for idx, tool_id in enumerate(tool_ids):
        ax = axes[idx // n_cols][idx % n_cols]
        entries = per_tool.get(tool_id) or []
        if not entries:
            ax.axis("off")
            continue
        first = entries[0]
        img_str = first.get("chosen_img")
        img_path = _resolve_chosen_img(img_str, json_path) if img_str else None
        if img_path is not None:
            try:
                ax.imshow(plt.imread(img_path))
            except Exception as _e:
                ax.text(
                    0.5,
                    0.5,
                    f"(read error: {_e})",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
        else:
            ax.text(
                0.5,
                0.5,
                "(missing image)",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
        # Per-cell headline: tool name (line 1, default colour) + validated
        # ratio (line 2, coloured by score). None-verdicts (mid-validation
        # state) don't contribute to either side of the ratio. Green
        # iff `correct >= ceil(total / 2)` so an exactly-half score on an
        # even sample budget reads as a (just-barely) pass; below that
        # threshold is red.
        verdicted = [e for e in entries if e.get("correct") is not None]
        correct = sum(1 for e in verdicted if e.get("correct"))
        total = len(verdicted)
        if total > 0:
            threshold = -(-total // 2)  # ceil(total / 2) without importing math
            score_color = "#2e7d32" if correct >= threshold else "#c62828"
            score_str = f"{correct}/{total}"
        else:
            score_color = "#666666"
            score_str = "—"
        # Two ax.text() layers instead of a multi-line set_title, since
        # matplotlib titles only accept one colour for the whole string.
        ax.text(
            0.5,
            1.18,
            tool_id,
            fontsize=10,
            ha="center",
            va="bottom",
            transform=ax.transAxes,
        )
        ax.text(
            0.5,
            1.02,
            score_str,
            fontsize=10,
            color=score_color,
            fontweight="bold",
            ha="center",
            va="bottom",
            transform=ax.transAxes,
        )
        ax.axis("off")
    # Blank trailing cells when len(tool_ids) % n_cols != 0.
    for empty_idx in range(len(tool_ids), n_rows * n_cols):
        axes[empty_idx // n_cols][empty_idx % n_cols].axis("off")
    fig.tight_layout()

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def plot_convex_decomp_results(
    json_path: Path, save_path: Path | None = None, show: bool = False
) -> Path | None:
    """Plot a single-assembly `convex_decomp_results.json` (or the legacy
    `results.json` under `convex_decomp/<id>/`). Renders BOTH the accuracy
    bar chart and the `all_tools.png` grid so re-plotting fully
    reproduces the in-pipeline output. Skips silently when no validated
    entries are present yet."""
    try:
        payload = load_summary(json_path)
    except Exception as e:
        print(f"[plot] {json_path}: failed to load ({e})")
        return None
    if not payload.get("per_tool"):
        print(f"[plot] {json_path}: empty convex_decomp results, skipping")
        return None
    totals, correct = _convex_decomp_tally(payload)
    if not totals:
        print(f"[plot] {json_path}: no validated convex_decomp entries, skipping")
        return None
    primary = _render_convex_decomp_accuracy(
        totals,
        correct,
        save_path=save_path or _default_convex_decomp_save_path(json_path),
        show=show,
        n_samples=payload.get("n_samples"),
    )
    # Side artifact: the all-tools grid next to the JSON, named the same
    # as the in-pipeline file so it overwrites the original cleanly.
    grid_path = json_path.parent / "all_tools.png"
    grid_result = _render_convex_decomp_all_tools(
        payload,
        save_path=grid_path,
        json_path=json_path,
        show=show,
    )
    if grid_result is not None:
        print(f"  ✓ all_tools  →  {grid_result}")
    return primary


# ----------------------------------------------------------------------
# plotting helpers — tool-selection accuracy (collect_tool_data)
#
# Tool-decision data lives as PAIRED per-assembly files, not a single
# summary: each assembly directory holds a `human_labels.json` (ground
# truth: part_id -> tool) plus one or more `ai_labels*.json` (the VLM's
# part_id -> tool, optionally a list of repeated samples). We discover the
# `human_labels.json` files, pair each with its sibling `ai_labels*.json`,
# and render a confusion-matrix view of the tool-need decision: the 3-class
# confusion matrix plus the two headline numbers (accuracy and the
# false-negative rate — a needed tool was missed), each with a part-level
# bootstrap CI and per-assembly spread dots. `--combine` aggregates every
# assembly into one figure; per-file renders just that assembly.


_TOOL_HUMAN_NON_TOOLS = {"none", "error", "aborted", "missing"}


def _tool_selection_pairs(
    human_paths: list[Path],
) -> list[tuple[str, Path, list[Path]]]:
    """Pair each `human_labels.json` with the sibling `ai_labels*.json` in
    its directory. Returns `[(assembly_id, human_path, [ai_paths]), ...]`;
    assemblies with no AI file are dropped."""
    pairs = []
    for hpath in human_paths:
        ai_paths = sorted(hpath.parent.glob("ai_labels*.json"))
        if not ai_paths:
            print(f"  ! skip {hpath} (no sibling ai_labels*.json)")
            continue
        try:
            with open(hpath) as f:
                hdata = json.load(f)
        except Exception as e:
            print(f"  ! skip {hpath} ({e})")
            continue
        ass_id = hdata.get("assembly_id") or hpath.parent.name
        pairs.append((ass_id, hpath, ai_paths))
    return pairs


def _load_tool_selection_samples(
    pairs: list[tuple[str, Path, list[Path]]],
) -> tuple[list, dict]:
    """Build the flat sample list + per-part collapse from paired files.

    Returns `(samples, part_data)` where
      samples   = [(ass_id, part_id, sample_idx, human, ai), ...]
      part_data = {(ass_id, part_id): (human_label, [ai_run, ...])}
    Multiple AI files for one assembly concatenate their samples; the first
    human label seen for a part wins (later conflicting ones are ignored).
    Mirrors the aggregation in `run_collect_tool_data`."""
    samples = []
    human_first: dict = {}  # (ass, part) -> first human label
    sample_counters: dict = {}  # (ass, part) -> next free sample idx
    for ass_id, hpath, ai_paths in pairs:
        try:
            with open(hpath) as f:
                hlabels = (json.load(f) or {}).get("labels", {})
        except Exception as e:
            print(f"  ! skip {hpath} ({e})")
            continue
        for apath in ai_paths:
            try:
                with open(apath) as f:
                    alabels = (json.load(f) or {}).get("labels", {})
            except Exception as e:
                print(f"  ! skip {apath} ({e})")
                continue
            for part_id, hlabel in hlabels.items():
                key = (ass_id, part_id)
                if key not in human_first:
                    human_first[key] = hlabel
                hlabel_used = human_first[key]
                # Backward compat: old format had a single string label;
                # new format has a list of repeated samples.
                if part_id in alabels:
                    raw = alabels[part_id]
                    ai_sample_list = raw if isinstance(raw, list) else [raw]
                else:
                    ai_sample_list = ["missing"]
                for ai_lbl in ai_sample_list:
                    s_idx = sample_counters.get(key, 0)
                    sample_counters[key] = s_idx + 1
                    samples.append((ass_id, part_id, s_idx, hlabel_used, ai_lbl))

    part_data: dict = {}
    for ass_id, part_id, _s_idx, h, a in samples:
        key = (ass_id, part_id)
        if key not in part_data:
            part_data[key] = (h, [])
        part_data[key][1].append(a)
    return samples, part_data


def _write_tool_selection_csvs(samples: list, part_data: dict, png_path: Path) -> None:
    """Write the run-level and part-level CSVs next to the PNG, mirroring
    the two CSVs `run_collect_tool_data` emits."""
    runs_csv = png_path.with_name(png_path.stem + "_runs.csv")
    with open(runs_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["assembly_id", "part_id", "sample_idx", "human", "ai", "match"])
        for ass_id, part_id, s_idx, h, a in sorted(samples):
            w.writerow([ass_id, part_id, s_idx, h, a, "YES" if h == a else "NO"])
    parts_csv = png_path.with_name(png_path.stem + "_parts.csv")
    with open(parts_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "assembly_id",
                "part_id",
                "human",
                "n_runs",
                "n_correct",
                "p_correct",
                "all_agreed",
            ]
        )
        for (ass_id, part_id), (h, al) in sorted(part_data.items()):
            n_c = sum(1 for a in al if a == h)
            w.writerow(
                [
                    ass_id,
                    part_id,
                    h,
                    len(al),
                    n_c,
                    f"{n_c / len(al):.4f}" if al else "",
                    "YES" if al and len(set(al)) == 1 else "NO",
                ]
            )


def _short_tool(name: str) -> str:
    s = str(name)
    return s if len(s) <= 16 else s[:15] + "…"


def _tool_selection_metrics(part_data: dict) -> dict:
    """Everything the confusion-matrix figure needs, all PART-weighted so
    the headline numbers read as "for a given part" and stay consistent
    with the matrix, the per-part bootstrap CI, and the per-assembly dots.

    Each part contributes weight 1, split across its repeated-run
    predictions. A part is a "false negative" when its true label is a real
    tool but the algorithm predicts a no-tool decision (none/error/missing)
    — i.e. a needed tool was not found."""
    no_tool = _TOOL_HUMAN_NON_TOOLS  # none/error/aborted/missing
    human_set = {h for h, _ in part_data.values()}
    pred_set = {a for _, runs in part_data.values() for a in runs}
    tool_classes = sorted(human_set - no_tool)
    canonical = (
        ["none"] if ("none" in human_set or "none" in pred_set) else []
    ) + tool_classes
    true_classes = [c for c in canonical if c in human_set]
    true_classes += sorted(human_set - set(true_classes))
    pred_classes = [c for c in canonical if c in pred_set]
    pred_classes += sorted(pred_set - set(pred_classes))

    # Part-weighted confusion C[true][pred] (in part units) + per-part recs.
    C = {t: dict.fromkeys(pred_classes, 0.0) for t in true_classes}
    row_parts = dict.fromkeys(true_classes, 0)
    part_recs = []  # (assembly_id, acc_part, is_tool, missed_frac|None)
    for (ass_id, _pid), (h, runs) in part_data.items():
        if not runs or h not in C:
            continue
        row_parts[h] += 1
        w = 1.0 / len(runs)
        for a in runs:
            if a in C[h]:
                C[h][a] += w
        acc_p = sum(1 for a in runs if a == h) / len(runs)
        is_tool = h not in no_tool
        missed = (sum(1 for a in runs if a in no_tool) / len(runs)) if is_tool else None
        part_recs.append((ass_id, acc_p, is_tool, missed))

    total_parts = sum(row_parts.values())
    tool_parts = sum(row_parts[t] for t in tool_classes)
    acc = (
        (sum(C[t].get(t, 0.0) for t in true_classes) / total_parts)
        if total_parts
        else 0.0
    )
    fn_mass = sum(C[t][p] for t in tool_classes for p in pred_classes if p in no_tool)
    fnr = (fn_mass / tool_parts) if tool_parts else float("nan")
    # Chance = always-predict-the-majority-true-class accuracy.
    chance = (max(row_parts.values()) / total_parts) if total_parts else 0.0
    # Balanced accuracy = mean per-true-class recall (imbalance-robust).
    recalls = [C[t].get(t, 0.0) / row_parts[t] for t in true_classes if row_parts[t]]
    bal_acc = (sum(recalls) / len(recalls)) if recalls else 0.0
    # Run consistency: parts with >1 run whose runs all agree.
    multi = [runs for _, runs in part_data.values() if len(runs) > 1]
    consistency = (
        (sum(1 for r in multi if len(set(r)) == 1) / len(multi))
        if multi
        else float("nan")
    )

    return {
        "true_classes": true_classes,
        "pred_classes": pred_classes,
        "tool_classes": tool_classes,
        "no_tool": no_tool,
        "C": C,
        "row_parts": row_parts,
        "total_parts": total_parts,
        "tool_parts": tool_parts,
        "acc": acc,
        "fnr": fnr,
        "fn_mass": fn_mass,
        "chance": chance,
        "bal_acc": bal_acc,
        "consistency": consistency,
        "part_recs": part_recs,
    }


def _tool_metric_strip(
    ax,
    value,
    lo,
    hi,
    color,
    title,
    baseline=None,
    baseline_label="",
    dots=None,
    note="",
) -> None:
    """A horizontal 0–100% number line highlighting one scalar metric: the
    point estimate (big dot + value), its 95% CI (band), an optional
    reference line (chance / ideal), and the per-assembly values as a rug of
    hollow dots so the between-assembly spread is visible without pretending
    to a smooth distribution."""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks(range(0, 101, 25))
    ax.set_xticklabels([f"{t}%" for t in range(0, 101, 25)], fontsize=7)
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    ax.set_axisbelow(True)
    if baseline is not None:
        bx = baseline * 100
        ax.axvline(bx, color="gray", linestyle="--", linewidth=1.0)
        ha = "left" if bx < 12 else ("right" if bx > 88 else "center")
        ax.text(
            min(max(bx, 1), 99),
            0.9,
            baseline_label,
            color="gray",
            fontsize=7,
            ha=ha,
            va="top",
        )
    ax.plot(
        [lo * 100, hi * 100],
        [0.58, 0.58],
        color=color,
        linewidth=7,
        solid_capstyle="round",
        alpha=0.30,
        zorder=2,
    )
    if dots:
        ax.scatter(
            [d * 100 for d in dots],
            [0.36] * len(dots),
            s=24,
            facecolor="white",
            edgecolor=color,
            linewidth=1.0,
            zorder=3,
            alpha=0.95,
        )
    ax.plot(
        [value * 100],
        [0.58],
        marker="o",
        markersize=13,
        color=color,
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=4,
    )
    ax.text(
        min(max(value * 100, 8), 92),
        0.78,
        f"{value * 100:.1f}%",
        ha="center",
        va="bottom",
        fontsize=15,
        fontweight="bold",
        color=color,
    )
    ax.set_title(title, fontsize=10, fontweight="bold")
    if note:
        ax.text(50, 0.10, note, ha="center", va="center", fontsize=7.5, color="#333333")


def _render_tool_selection_figure(
    samples: list,
    part_data: dict,
    save_path: Path,
    show: bool = False,
    write_csv: bool = True,
) -> Path | None:
    """Confusion-matrix view of the tool-need decision. Left: the 3-class
    confusion matrix (ground truth × algorithm decision), row-normalised and
    part-weighted, with false-negative cells outlined. Right: the two
    headline numbers — Accuracy and the False-Negative rate (a needed tool
    was missed) — each with a part-level bootstrap 95% CI, a reference line,
    and the per-assembly values as dots."""
    if not part_data:
        return None
    m = _tool_selection_metrics(part_data)
    true_classes, pred_classes = m["true_classes"], m["pred_classes"]
    tool_classes, no_tool = m["tool_classes"], m["no_tool"]
    C, row_parts = m["C"], m["row_parts"]
    part_recs = m["part_recs"]
    n_parts = m["total_parts"]
    n_ass = len({a for a, *_ in part_recs})
    runs_seen = [len(runs) for _, runs in part_data.values() if runs]
    rlo, rhi = (min(runs_seen), max(runs_seen)) if runs_seen else (0, 0)
    runs_str = f"{rlo}" if rlo == rhi else f"{rlo}–{rhi}"

    # Part-level bootstrap CIs — resample parts (plenty of them) rather than
    # the 4 assemblies (too few clusters to bootstrap honestly).
    rng = np.random.default_rng(42)
    B = 10_000

    def _ci(vals):
        a = np.asarray(vals, dtype=float)
        if a.size < 2:
            v = float(a.mean()) if a.size else 0.0
            return v, v
        boot = a[rng.integers(0, a.size, size=(B, a.size))].mean(axis=1)
        return float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))

    acc_lo, acc_hi = _ci([r[1] for r in part_recs])
    tool_missed = [r[3] for r in part_recs if r[2]]
    fnr_lo, fnr_hi = _ci(tool_missed) if tool_missed else (0.0, 0.0)

    # Per-assembly point values for the spread dots.
    by_ass_acc: dict[str, list[float]] = {}
    by_ass_miss: dict[str, list[float]] = {}
    for ass_id, acc_p, is_tool, missed in part_recs:
        by_ass_acc.setdefault(ass_id, []).append(acc_p)
        if is_tool:
            by_ass_miss.setdefault(ass_id, []).append(missed)
    ass_acc = [float(np.mean(v)) for v in by_ass_acc.values()]
    ass_fnr = [float(np.mean(v)) for v in by_ass_miss.values() if v]

    fig = plt.figure(figsize=(13.5, 5.8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0])
    ax_cm = fig.add_subplot(gs[:, 0])
    ax_acc = fig.add_subplot(gs[0, 1])
    ax_fnr = fig.add_subplot(gs[1, 1])

    # --- confusion matrix (row-normalised, part-weighted) ---
    M = np.array(
        [
            [(C[t][p] / row_parts[t] if row_parts[t] else 0.0) for p in pred_classes]
            for t in true_classes
        ]
    )
    im = ax_cm.imshow(M, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    ax_cm.set_xticks(range(len(pred_classes)))
    ax_cm.set_xticklabels(
        [_short_tool(c) for c in pred_classes], rotation=30, ha="right", fontsize=8
    )
    ax_cm.set_yticks(range(len(true_classes)))
    ax_cm.set_yticklabels(
        [f"{_short_tool(t)}\n(n={row_parts[t]})" for t in true_classes], fontsize=8
    )
    ax_cm.set_xlabel("algorithm decision")
    ax_cm.set_ylabel("ground truth (human)")
    ax_cm.set_title("Decision confusion  (row-normalised, per-part)", fontsize=10)
    for i, t in enumerate(true_classes):
        for j in range(len(pred_classes)):
            if not row_parts[t]:
                continue
            v = M[i, j]
            ax_cm.text(
                j,
                i,
                f"{v * 100:.0f}%",
                ha="center",
                va="center",
                color=("white" if v > 0.55 else "black"),
                fontsize=9,
            )
    # Outline the false-negative cells: true tool, predicted no-tool.
    fn_cols = [j for j, p in enumerate(pred_classes) if p in no_tool]
    for i, t in enumerate(true_classes):
        if t in tool_classes:
            for j in fn_cols:
                ax_cm.add_patch(
                    plt.Rectangle(
                        (j - 0.5, i - 0.5),
                        1,
                        1,
                        fill=False,
                        edgecolor="#c0392b",
                        lw=2.5,
                        zorder=5,
                    )
                )
    fig.colorbar(im, ax=ax_cm, shrink=0.72, label="P(decision | truth)")

    # --- headline numbers ---
    _tool_metric_strip(
        ax_acc,
        value=m["acc"],
        lo=acc_lo,
        hi=acc_hi,
        color="#1f3a93",
        title="Accuracy  (correct decision per part)",
        baseline=m["chance"],
        baseline_label=f"chance {m['chance'] * 100:.0f}%",
        dots=ass_acc,
        note=(
            f"95% CI [{acc_lo * 100:.0f}, {acc_hi * 100:.0f}]   ·   "
            f"balanced {m['bal_acc'] * 100:.0f}%   ·   n={n_parts} parts"
        ),
    )
    if m["tool_parts"]:
        _tool_metric_strip(
            ax_fnr,
            value=m["fnr"],
            lo=fnr_lo,
            hi=fnr_hi,
            color="#c0392b",
            title="False negatives  (needed tool missed — lower is better)",
            baseline=None,
            dots=ass_fnr,
            note=(
                f"95% CI [{fnr_lo * 100:.0f}, {fnr_hi * 100:.0f}]   ·   "
                f"≈{m['fn_mass']:.1f}/{m['tool_parts']} tool-parts missed"
            ),
        )
    else:
        ax_fnr.set_axis_off()
        ax_fnr.text(
            0.5, 0.5, "(no tool-parts to score)", ha="center", va="center", fontsize=10
        )

    cons = m["consistency"]
    cons_str = (
        f"{cons * 100:.0f}% of parts unanimous across runs"
        if cons == cons
        else "single run per part"
    )
    fig.suptitle(
        f"Tool-need decision validation  —  {n_ass} assemblies · "
        f"{n_parts} parts · {runs_str} runs/part   ({cons_str})",
        fontsize=12,
        fontweight="bold",
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)

    if write_csv:
        _write_tool_selection_csvs(samples, part_data, save_path)
    return save_path


def _default_tool_selection_save_path(json_path: Path) -> Path:
    suffix = _variant_suffix(json_path)
    return (
        json_path.parent
        / f"tool_selection_accuracy_{_assembly_tag_from(json_path)}{suffix}.png"
    )


def plot_tool_selection(
    json_path: Path, save_path: Path | None = None, show: bool = False
) -> Path | None:
    """Plot a single assembly's tool-detection accuracy from its
    `human_labels.json` + sibling `ai_labels*.json`. The cross-assembly
    clustered-bootstrap chart is produced by `find_and_plot(..., combine=True)`."""
    pairs = _tool_selection_pairs([json_path])
    if not pairs:
        return None
    samples, part_data = _load_tool_selection_samples(pairs)
    if not part_data:
        print(f"[plot] {json_path}: no labeled parts, skipping")
        return None
    return _render_tool_selection_figure(
        samples,
        part_data,
        save_path=save_path or _default_tool_selection_save_path(json_path),
        show=show,
    )


# ----------------------------------------------------------------------
# main plotting function


# Output filenames are derived from what each plot actually MEASURES, so the
# PNGs are self-describing when they sit next to a stack of other artifacts:
#
#   comparison_summary.json        → selector_agreement_<assembly>.png
#   comparison_batch_summary.json  → selector_agreement_batch_<run>.png
#   validation_summary.json        → beam_width_to_optimum_<run>.png
#   assembly_time_summary.json     → assembly_time_comparison_<run>.png
#   manual_validation_batch.json   → manual_validation_summary_<run>.png
#                                  + manual_validation_per_assembly_<run>.png
#   human_labels.json (+ai_labels) → tool_selection_accuracy_<assembly>.png
#
# "selector_agreement"        — comparison-planner plot measures per-decision
#                               agreement between the N constituent selectors,
#                               with random/VLM baselines.
# "beam_width_to_optimum"     — k-greedy validation plot measures the lowest
#                               beam width k at which the global-optimal
#                               sequence is recovered.
# "assembly_time_comparison"  — data_assembly_time plot measures relative
#                               assembly time per generator: each assembly's
#                               per-run total_s is divided by the same
#                               assembly's baseline run (default `heur-out`),
#                               and the distribution of ratios is shown as a
#                               box-and-strip per non-baseline run.


def _assembly_tag_from(json_path: Path) -> str:
    """Pull the assembly id from a path like .../<id>/log/comparison_summary.json."""
    if json_path.parent.name == "log":
        return json_path.parent.parent.name
    return json_path.parent.name


def _run_tag_from(json_path: Path) -> str:
    """Tag a batch/aggregate file by the directory it lives in."""
    return json_path.parent.name or "run"


def _variant_suffix(json_path: Path) -> str:
    """Return e.g. `_2` for `comparison_summary_2.json`, `""` for the bare
    `comparison_summary.json`. Preserves numbering so two variants in the
    same directory don't overwrite each other's PNG."""
    stem = _stem_of(json_path)
    if not stem:
        return ""
    return json_path.stem[len(stem) :]


def _default_save_path(json_path: Path, is_batch: bool) -> Path:
    suffix = _variant_suffix(json_path)
    if is_batch:
        return (
            json_path.parent
            / f"selector_agreement_batch_{_run_tag_from(json_path)}{suffix}.png"
        )
    return (
        json_path.parent
        / f"selector_agreement_{_assembly_tag_from(json_path)}{suffix}.png"
    )


def _default_validation_save_path(json_path: Path) -> Path:
    suffix = _variant_suffix(json_path)
    return (
        json_path.parent
        / f"beam_width_to_optimum_{_run_tag_from(json_path)}{suffix}.png"
    )


def _default_validation_overview_path(reference_path: Path) -> Path:
    """Side stats-table PNG path, derived from the main-plot path. PNG input
    → swap `beam_width_to_optimum` → `beam_width_to_optimum_stats` (or
    append `_stats`); JSON input → mirror the main-plot naming."""
    if reference_path.suffix == ".png":
        stem = reference_path.stem
        if "beam_width_to_optimum" in stem and "stats" not in stem:
            new_stem = stem.replace(
                "beam_width_to_optimum", "beam_width_to_optimum_stats"
            )
        else:
            new_stem = f"{stem}_stats"
        return reference_path.with_name(f"{new_stem}.png")
    suffix = _variant_suffix(reference_path)
    return (
        reference_path.parent
        / f"beam_width_to_optimum_stats_{_run_tag_from(reference_path)}{suffix}.png"
    )


def _validation_overview_rows(summary: dict, eff_max_k: int) -> list[dict]:
    """Build per-size + all-sizes rows for the validation overview table.

    Per row, computes:
      m                — assemblies in this size bucket
      pct_k1           — mean (best score % of global optimum) at k=1
      pct_kmax         — mean at k=eff_max_k
      recovered_np     — "<X>/<Y>": rows whose lowest_k_for_global_optimum
                         is <= n_parts, over rows with valid n_parts
      recovered_any    — "<X>/<m>": rows that recovered the optimum at SOME
                         k ≤ eff_max_k (i.e. lowest_k_for_global_optimum
                         is set)
      median_lk        — median lowest_k_for_global_optimum among recovered
                         rows; None if no row recovered

    Last row is the "all sizes" pool. The grand row is computed off the
    same per-row buckets so its numbers are exactly the headline numbers
    the grand-average line on the chart represents.
    """
    finished = _validation_finished_rows(summary)
    if not finished or not eff_max_k:
        return []

    def _num(d, *keys):
        for key in keys:
            v = d.get(key)
            if v is None:
                continue
            try:
                vf = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(vf):
                return vf
        return None

    by_size: dict[int, list[dict]] = {}
    for rec in finished:
        np_ = rec.get("n_parts") or rec.get("n_tree_nodes")
        if np_ is None:
            continue
        try:
            np_i = int(np_)
        except (TypeError, ValueError):
            continue
        by_size.setdefault(np_i, []).append(rec)

    def _bucket_stats(rows: list[dict], label: str) -> dict:
        m = len(rows)
        pct_at_k: dict[int, float | None] = {}
        for k in (1, eff_max_k):
            vals = []
            for rec in rows:
                gb = _num(rec, "global_best_score", "global_best_cost")
                if gb is None or gb == 0:
                    continue
                kres = rec.get("k_results") or {}
                entry = kres.get(k) or kres.get(str(k))
                if not isinstance(entry, dict):
                    continue
                vf = _num(entry, "best_score", "best_cost")
                if vf is None:
                    continue
                vals.append(100.0 * vf / gb)
            pct_at_k[k] = float(np.mean(vals)) if vals else None

        rec_within_np = 0
        rec_within_np_denom = 0
        rec_any = 0
        lks = []
        for rec in rows:
            np_ = rec.get("n_parts") or rec.get("n_tree_nodes")
            try:
                np_i = int(np_) if np_ is not None else None
            except (TypeError, ValueError):
                np_i = None
            lk = rec.get("lowest_k_for_global_optimum")
            try:
                lk_i = int(lk) if lk is not None else None
            except (TypeError, ValueError):
                lk_i = None
            if lk_i is not None:
                rec_any += 1
                lks.append(lk_i)
            if np_i is not None:
                rec_within_np_denom += 1
                if lk_i is not None and lk_i <= np_i:
                    rec_within_np += 1

        return {
            "label": label,
            "m": m,
            "pct_k1": pct_at_k.get(1),
            "pct_kmax": pct_at_k.get(eff_max_k),
            "recovered_np": (
                f"{rec_within_np}/{rec_within_np_denom}" if rec_within_np_denom else "—"
            ),
            "recovered_any": f"{rec_any}/{m}",
            "median_lk": float(np.median(lks)) if lks else None,
        }

    out = [
        _bucket_stats(by_size[size], f"n_parts={size}")
        for size in sorted(by_size.keys())
    ]
    out.append(_bucket_stats(finished, "all sizes"))
    return out


def _render_validation_overview(
    summary: dict, save_path: Path, eff_max_k: int, show: bool = False
) -> Path | None:
    """Side companion PNG: per-size statistics table for the k-greedy
    validation plot. Mirrors the layout we used for the timing overview —
    same colours, same row/column conventions — so the two plot kinds
    look like a family."""
    rows = _validation_overview_rows(summary, eff_max_k)
    if not rows:
        return None

    # 2-line headers for the wordy columns so they fit their cells without
    # spilling out. matplotlib's Table doesn't auto-wrap; the newlines are
    # the only handle we have.
    headers = [
        "bucket",
        "m",
        "% at k=1",
        f"% at k={eff_max_k}",
        "recovered at\nk≤n_parts",
        "recovered at\nk≤k_max",
        "median\nlowest-k",
    ]
    cell_text = []
    for r in rows:
        cell_text.append(
            [
                r["label"],
                str(r["m"]),
                f"{r['pct_k1']:.1f}" if r["pct_k1"] is not None else "—",
                f"{r['pct_kmax']:.1f}" if r["pct_kmax"] is not None else "—",
                r["recovered_np"],
                r["recovered_any"],
                (f"{r['median_lk']:.1f}" if r["median_lk"] is not None else "—"),
            ]
        )

    # Wider bucket + recovery columns; numerics stay compact.
    col_widths = [1.55, 0.6, 0.95, 1.05, 1.5, 1.4, 1.25]
    total_weight = sum(col_widths)
    col_widths_norm = [w / total_weight for w in col_widths]
    n_rows = len(cell_text) + 1
    fig_width = 8.0 * (total_weight / 6.0)
    # +0.4 inch on the height to absorb the taller two-line header below
    # without compressing the data rows.
    fig_height = max(2.4, 0.55 * n_rows) + 0.4
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_axis_off()
    ax.set_title(
        f"Beam search optimality — per-size statistics  (k_max={eff_max_k})",
        fontsize=12,
        fontweight="bold",
        pad=10,
    )
    tbl = ax.table(
        cellText=cell_text,
        colLabels=headers,
        loc="center",
        cellLoc="center",
        colWidths=col_widths_norm,
        colColours=["#E8EEF7"] * len(headers),
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.55)
    last_row_idx = n_rows - 1
    # Two-line headers need extra vertical room; bump header-row height
    # to ~1.9× a normal row so the second line doesn't clip into the data.
    header_height_scale = 1.9
    for (row, _col), cell in tbl.get_celld().items():
        if row in (0, last_row_idx):
            cell.set_text_props(fontweight="bold")
        if row == 0:
            cell.set_height(cell.get_height() * header_height_scale)
        cell.set_edgecolor("#9AAAC2")

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=140, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def _title_for(json_path: Path, summary: dict, is_batch: bool) -> str:
    if is_batch:
        n = summary.get("n_assemblies", "?")
        return f"Comparison Planner — batch summary  (n={n} assemblies)"
    # per-assembly: directory two levels up usually carries the id (… / <id> / log)
    if json_path.parent.name == "log":
        return f"Comparison Planner — assembly {json_path.parent.parent.name}"
    return f"Comparison Planner — {json_path.parent.name}"


def plot_validation_summary(
    json_path: Path,
    save_path: Path | None = None,
    show: bool = False,
    max_k_override: int | None = None,
    line_only: bool = False,
) -> Path | None:
    """Plot a `validation_summary.json` produced by `test_heuristic_validation`.

    2-panel (1 × 2) layout by default:
        (0) per-k mean best-score across assemblies, as a percentage of each
            assembly's global optimum score (no confidence band)
        (1) scatter of n_parts vs lowest-k

    `line_only=True` (= CLI `--line-only`) drops panel (1) and renders only
    panel (0). `max_k_override` (= CLI `--max-k`) replaces the data-derived
    k cap; see `_validation_effective_max_k` for the rationale.
    """
    try:
        summary = load_summary(json_path)
    except Exception as e:
        print(f"[plot] {json_path}: failed to load ({e})")
        return None

    if not summary.get("per_assembly") and not summary.get("aggregate_found_at_k"):
        print(f"[plot] {json_path}: empty validation summary, skipping")
        return None

    source_label = f"…/{json_path.parent.name}/{json_path.name}"
    primary = _render_validation_figure(
        summary,
        source_label,
        save_path=save_path or _default_validation_save_path(json_path),
        show=show,
        max_k_override=max_k_override,
        line_only=line_only,
    )

    # Side stats-overview PNG. eff_max_k recomputed here so the table's
    # "% at k=k_max" column matches whatever the chart's k cap ended up
    # being (--max-k override and data-derived cap both honoured).
    finished = _validation_finished_rows(summary)
    eff_max_k = _validation_effective_max_k(summary, finished, override=max_k_override)
    overview_path = _default_validation_overview_path(
        save_path if save_path is not None else json_path,
    )
    overview = _render_validation_overview(
        summary,
        save_path=overview_path,
        eff_max_k=eff_max_k,
        show=show,
    )
    if overview is not None:
        print(f"  ✓ overview  →  {overview}")
    return primary


def _render_validation_figure(
    summary: dict,
    source_label: str,
    save_path: Path,
    show: bool = False,
    title_suffix: str = "",
    max_k_override: int | None = None,
    line_only: bool = False,
) -> Path | None:
    # Compute the k cap once (honors --max-k) and pass it to both panels so
    # the axes and the title agree.
    finished = _validation_finished_rows(summary)
    eff_max_k = _validation_effective_max_k(summary, finished, override=max_k_override)
    n_finished = len(finished)

    # Console fact: how many finished assemblies recovered the global optimum
    # at beam width <= n_parts (i.e. the "k = number of parts" rule of thumb
    # we previously argued isn't actually exhaustive). Counts the rows where
    # `lowest_k_for_global_optimum` is finite and <= the assembly's part count;
    # rows missing either field (never recovered up to max_k_tested, or no
    # n_parts metadata) count as not-recovered-within-n_parts.
    n_within_np = 0
    n_with_np = 0
    for rec in finished:
        np_ = rec.get("n_parts") or rec.get("n_tree_nodes")
        if np_ is None:
            continue
        try:
            np_i = int(np_)
        except (TypeError, ValueError):
            continue
        n_with_np += 1
        lk = rec.get("lowest_k_for_global_optimum")
        if lk is None:
            continue
        try:
            lk_i = int(lk)
        except (TypeError, ValueError):
            continue
        if lk_i <= np_i:
            n_within_np += 1
    pct = (100.0 * n_within_np / n_with_np) if n_with_np else 0.0
    print(
        f"[plot] beam-search validation: {n_within_np}/{n_with_np} "
        f"finished assemblies recovered the global optimum at k <= n_parts "
        f"({pct:.1f}%)"
    )

    if line_only:
        fig, ax_line = plt.subplots(1, 1, figsize=(7.5, 5.5))
        _line_avg_best_cost(ax_line, summary, eff_max_k=eff_max_k)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
        _line_avg_best_cost(axes[0], summary, eff_max_k=eff_max_k)
        _scatter_size_vs_lowest_k(axes[1], summary, eff_max_k=eff_max_k)

    fig.suptitle(
        f"Beam search optimality validation  (n={n_finished} assemblies, "
        f"k tested up to {eff_max_k}){title_suffix}",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


def plot_summary(
    json_path: Path,
    save_path: Path | None = None,
    show: bool = False,
    baseline: str = DEFAULT_ASSEMBLY_TIME_BASELINE,
    max_k_override: int | None = None,
    line_only: bool = False,
) -> Path | None:
    """Top-level dispatcher: pick the right plotter based on the file's stem
    (handles numbered variants like comparison_summary_2.json)."""
    stem = _stem_of(json_path)
    if stem == "validation_summary":
        return plot_validation_summary(
            json_path,
            save_path=save_path,
            show=show,
            max_k_override=max_k_override,
            line_only=line_only,
        )
    if stem == "assembly_time_summary":
        return plot_assembly_time_summary(
            json_path, save_path=save_path, show=show, baseline=baseline
        )
    if stem == "manual_validation_batch":
        return plot_manual_validation_batch(json_path, save_path=save_path, show=show)
    if stem == "convex_decomp_results":
        return plot_convex_decomp_results(json_path, save_path=save_path, show=show)
    if stem == "human_labels":
        return plot_tool_selection(json_path, save_path=save_path, show=show)
    # comparison_summary / comparison_batch_summary land here; the per-assembly
    # vs batch shape is detected inside plot_comparison_summary.
    return plot_comparison_summary(json_path, save_path=save_path, show=show)


def plot_comparison_summary(
    json_path: Path, save_path: Path | None = None, show: bool = False
) -> Path | None:
    """Plot a comparison summary (per-assembly OR batch) to PNG.

    Returns the output path, or None on failure / empty data.
    """
    try:
        raw = load_summary(json_path)
    except Exception as e:
        print(f"[plot] {json_path}: failed to load ({e})")
        return None

    summary = _normalize_summary(raw)
    is_batch = _is_batch(summary)

    if summary.get("total_decisions", 0) == 0:
        print(f"[plot] {json_path}: total_decisions=0, skipping")
        return None

    source_label = f"…/{json_path.parent.name}/{json_path.name}"
    return _render_comparison_figure(
        summary,
        source_label,
        is_batch=is_batch,
        title=_title_for(json_path, summary, is_batch),
        save_path=save_path or _default_save_path(json_path, is_batch),
        show=show,
    )


def _render_comparison_figure(
    summary: dict,
    source_label: str,
    is_batch: bool,
    title: str,
    save_path: Path,
    show: bool = False,
) -> Path | None:
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _bar_match_baseline(axes[0, 0], summary)
    _heatmap_pairwise(axes[0, 1], summary)
    _bar_vlm_match_constituents(axes[1, 0], summary)
    if is_batch:
        _assembly_status_block(axes[1, 1], summary, source_label)
    else:
        _forward_source_block(axes[1, 1], summary, source_label)

    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 0.93, 0.96))

    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)
    return save_path


# ----------------------------------------------------------------------
# combine: merge multiple summaries into a single dict to plot once
#
# Per kind:
#   comparison_summary.json / comparison_batch_summary.json
#       → both normalize to the same shape; merge counts across all inputs,
#         recompute the rate-style fields from the merged counts, and treat
#         the merged result as a "batch" so _assembly_status_block runs in
#         the bottom-right (it carries the cross-file totals).
#   validation_summary.json
#       → concatenate per_assembly, sum status counts and aggregate_found_at_k,
#         take max of max_k_tested.


def _merge_comparison_summaries(summaries: list[dict]) -> dict:
    """Merge a list of normalized comparison summaries into one. Counts add,
    rates are recomputed from the merged counts, selectors take the union."""
    out: dict = {
        "selectors": [],
        "total_decisions": 0,
        "per_selector_responded": {},
        "per_selector_match_random_count": {},
        "per_selector_match_random_rate": {},
        "pairwise_both_responded": {},
        "pairwise_agreement_count": {},
        "pairwise_agreement_rate": {},
        "vlm_meta_used": False,
        "vlm_meta_responded": 0,
        "vlm_meta_match_random_count": 0,
        "vlm_meta_match_random_rate": None,
        "vlm_meta_match_constituent_count": {},
        "vlm_meta_match_constituent_rate": {},
        "forward_source_counts": {},
        "use_vlm_for_progress": False,
        # batch-style fields, accumulated below
        "n_assemblies": 0,
        "assemblies_with_decisions": 0,
        "assemblies_no_decisions": 0,
        "assemblies_missing_summary": 0,
        "per_assembly": [],
    }

    sel_set: set[str] = set()
    avg_l_num = 0.0
    avg_l_den = 0

    def _add_dict(target: dict, src: dict | None) -> None:
        if not src:
            return
        for k, v in src.items():
            if isinstance(v, (int, float)):
                target[k] = (target.get(k) or 0) + v

    for s in summaries:
        sel_set.update(s.get("selectors") or [])
        total = int(s.get("total_decisions") or 0)
        out["total_decisions"] += total

        _add_dict(out["per_selector_responded"], s.get("per_selector_responded"))
        _add_dict(
            out["per_selector_match_random_count"],
            s.get("per_selector_match_random_count"),
        )
        _add_dict(out["pairwise_both_responded"], s.get("pairwise_both_responded"))
        _add_dict(out["pairwise_agreement_count"], s.get("pairwise_agreement_count"))
        _add_dict(
            out["vlm_meta_match_constituent_count"],
            s.get("vlm_meta_match_constituent_count"),
        )
        _add_dict(out["forward_source_counts"], s.get("forward_source_counts"))

        out["vlm_meta_used"] = out["vlm_meta_used"] or bool(s.get("vlm_meta_used"))
        out["use_vlm_for_progress"] = out["use_vlm_for_progress"] or bool(
            s.get("use_vlm_for_progress")
        )
        out["vlm_meta_responded"] += int(s.get("vlm_meta_responded") or 0)
        out["vlm_meta_match_random_count"] += int(
            s.get("vlm_meta_match_random_count") or 0
        )

        avg_l = s.get("avg_sample_size")
        if isinstance(avg_l, (int, float)) and total > 0:
            avg_l_num += float(avg_l) * total
            avg_l_den += total

        # Batch-flavored summary contributes its per-assembly aggregates;
        # per-assembly summary contributes itself as a single record.
        if isinstance(s.get("per_assembly"), list):
            out["n_assemblies"] += int(s.get("n_assemblies") or len(s["per_assembly"]))
            out["assemblies_with_decisions"] += int(
                s.get("assemblies_with_decisions") or 0
            )
            out["assemblies_no_decisions"] += int(s.get("assemblies_no_decisions") or 0)
            out["assemblies_missing_summary"] += int(
                s.get("assemblies_missing_summary") or 0
            )
            out["per_assembly"].extend(s["per_assembly"])
        else:
            out["n_assemblies"] += 1
            if total > 0:
                out["assemblies_with_decisions"] += 1
            else:
                out["assemblies_no_decisions"] += 1
            out["per_assembly"].append(
                {"total_decisions": total, "avg_sample_size": s.get("avg_sample_size")}
            )

    out["selectors"] = sorted(sel_set)

    # Recompute rates from merged counts.
    psr = out["per_selector_responded"]
    for n in out["selectors"]:
        resp = psr.get(n) or 0
        cnt = out["per_selector_match_random_count"].get(n) or 0
        out["per_selector_match_random_rate"][n] = (cnt / resp) if resp else 0.0

    for key, both in out["pairwise_both_responded"].items():
        cnt = out["pairwise_agreement_count"].get(key) or 0
        out["pairwise_agreement_rate"][key] = (cnt / both) if both else 0.0

    vlm_resp = out["vlm_meta_responded"]
    if vlm_resp:
        out["vlm_meta_match_random_rate"] = (
            out["vlm_meta_match_random_count"] / vlm_resp
        )
        for n in out["selectors"]:
            cnt = out["vlm_meta_match_constituent_count"].get(n) or 0
            out["vlm_meta_match_constituent_rate"][n] = cnt / vlm_resp

    if avg_l_den:
        out["avg_sample_size"] = avg_l_num / avg_l_den
        out["random_baseline_pct"] = 100.0 / out["avg_sample_size"]
    else:
        out["avg_sample_size"] = 0.0
        out["random_baseline_pct"] = 0.0

    return out


def _merge_validation_summaries(summaries: list[dict]) -> dict:
    """Concatenate per_assembly, sum status counts and aggregate_found_at_k."""
    out: dict = {
        "n_assemblies": 0,
        "ok": 0,
        "errors": 0,
        "no_sequence": 0,
        "n_never_found_at_max_k": 0,
        "max_k_tested": 0,
        "bfs_max_frontier": None,
        "budget_per_run": None,
        "aggregate_found_at_k": {},
        "per_assembly": [],
    }
    for s in summaries:
        out["n_assemblies"] += int(s.get("n_assemblies") or 0)
        out["ok"] += int(s.get("ok") or 0)
        out["errors"] += int(s.get("errors") or 0)
        out["no_sequence"] += int(s.get("no_sequence") or 0)
        out["n_never_found_at_max_k"] += int(s.get("n_never_found_at_max_k") or 0)
        out["max_k_tested"] = max(out["max_k_tested"], int(s.get("max_k_tested") or 0))
        if out["bfs_max_frontier"] is None and s.get("bfs_max_frontier") is not None:
            out["bfs_max_frontier"] = s["bfs_max_frontier"]
        if out["budget_per_run"] is None and s.get("budget_per_run") is not None:
            out["budget_per_run"] = s["budget_per_run"]
        for k, v in (s.get("aggregate_found_at_k") or {}).items():
            kk = str(k)
            out["aggregate_found_at_k"][kk] = (
                out["aggregate_found_at_k"].get(kk) or 0
            ) + int(v)
        if isinstance(s.get("per_assembly"), list):
            out["per_assembly"].extend(s["per_assembly"])
    return out


def find_and_plot(
    target_dir: Path,
    save_dir: Path | None = None,
    recursive: bool = True,
    show: bool = False,
    combine: bool = False,
    baseline: str = DEFAULT_ASSEMBLY_TIME_BASELINE,
    max_k_override: int | None = None,
    line_only: bool = False,
) -> list[Path]:
    """Walk target_dir, plot every recognised summary found.

    Dispatches by filename:
      comparison_summary.json        → `plot_comparison_summary` (per-assembly)
      comparison_batch_summary.json  → `plot_comparison_summary` (batch view)
      validation_summary.json        → `plot_validation_summary` (k-greedy)

    If `save_dir` is given, all plots land there with descriptive filenames.
    Otherwise each plot is written next to its source JSON.
    """
    paths = find_summaries(target_dir, recursive=recursive)
    if not paths:
        print(
            f"[plot] no recognised summary files ({', '.join(_KNOWN_NAMES)}) "
            f"found under {target_dir}"
        )
        return []

    print(f"[plot] {len(paths)} summary file(s) under {target_dir}")
    written = []

    if combine:
        # Group by kind via stem (also matches numbered variants).
        # comparison_summary + comparison_batch_summary both go into the
        # comparison bucket — both share the normalized shape.
        comp_paths = [
            p
            for p in paths
            if _stem_of(p) in ("comparison_summary", "comparison_batch_summary")
        ]
        val_paths = [p for p in paths if _stem_of(p) == "validation_summary"]
        time_paths = [p for p in paths if _stem_of(p) == "assembly_time_summary"]
        manual_paths = [p for p in paths if _stem_of(p) == "manual_validation_batch"]
        cd_paths = [p for p in paths if _stem_of(p) == "convex_decomp_results"]
        tool_paths = [p for p in paths if _stem_of(p) == "human_labels"]

        out_root = save_dir if save_dir is not None else target_dir
        out_root.mkdir(parents=True, exist_ok=True)
        dir_tag = target_dir.resolve().name or "run"

        if comp_paths:
            normalized = []
            for p in comp_paths:
                try:
                    normalized.append(_normalize_summary(load_summary(p)))
                except Exception as e:
                    print(f"  ! skip {p} ({e})")
            if normalized:
                merged = _merge_comparison_summaries(normalized)
                if merged.get("total_decisions", 0) == 0:
                    print("[plot] combined comparison: total_decisions=0, skipping")
                else:
                    label = f"combined ({len(normalized)} files under {target_dir})"
                    title = (
                        f"Comparison Planner — combined "
                        f"({merged.get('n_assemblies', '?')} assemblies, "
                        f"{len(normalized)} files)"
                    )
                    out = out_root / f"combined_selector_agreement_{dir_tag}.png"
                    result = _render_comparison_figure(
                        merged,
                        source_label=label,
                        is_batch=True,
                        title=title,
                        save_path=out,
                        show=show,
                    )
                    if result is not None:
                        written.append(result)
                        print(f"  ✓ comparison ×{len(normalized)}  →  {result}")

        if val_paths:
            loaded = []
            for p in val_paths:
                try:
                    loaded.append(load_summary(p))
                except Exception as e:
                    print(f"  ! skip {p} ({e})")
            if loaded:
                merged = _merge_validation_summaries(loaded)
                if not merged.get("per_assembly") and not merged.get(
                    "aggregate_found_at_k"
                ):
                    print("[plot] combined validation: empty, skipping")
                else:
                    label = f"combined ({len(loaded)} files under {target_dir})"
                    out = out_root / f"combined_beam_width_to_optimum_{dir_tag}.png"
                    result = _render_validation_figure(
                        merged,
                        source_label=label,
                        save_path=out,
                        show=show,
                        max_k_override=max_k_override,
                        line_only=line_only,
                    )
                    if result is not None:
                        written.append(result)
                        print(f"  ✓ validation ×{len(loaded)}  →  {result}")
                    # Side stats table for the merged batch.
                    merged_finished = _validation_finished_rows(merged)
                    merged_eff_max_k = _validation_effective_max_k(
                        merged,
                        merged_finished,
                        override=max_k_override,
                    )
                    overview_out = (
                        out_root / f"combined_beam_width_to_optimum_stats_{dir_tag}.png"
                    )
                    ov_res = _render_validation_overview(
                        merged,
                        save_path=overview_out,
                        eff_max_k=merged_eff_max_k,
                        show=show,
                    )
                    if ov_res is not None:
                        written.append(ov_res)
                        print(f"  ✓ validation ×{len(loaded)} (stats)  →  {ov_res}")

        if time_paths:
            loaded = []
            for p in time_paths:
                try:
                    loaded.append(load_summary(p))
                except Exception as e:
                    print(f"  ! skip {p} ({e})")
            if loaded:
                merged = _merge_assembly_time_summaries(loaded)
                if not merged.get("per_assembly"):
                    print("[plot] combined assembly_time: empty, skipping")
                else:
                    label = f"combined ({len(loaded)} files under {target_dir})"
                    out = out_root / f"combined_assembly_time_comparison_{dir_tag}.png"
                    result = _render_assembly_time_figure(
                        merged,
                        source_label=label,
                        save_path=out,
                        show=show,
                        baseline=baseline,
                    )
                    if result is not None:
                        written.append(result)
                        print(f"  ✓ assembly_time ×{len(loaded)}  →  {result}")
                    # Side overview with the per-run statistics table.
                    baseline_label = _resolve_baseline_label(merged, baseline)
                    if baseline_label is not None:
                        overview_out = (
                            out_root / f"combined_assembly_time_stats_{dir_tag}.png"
                        )
                        ov_res = _render_assembly_time_overview(
                            merged,
                            baseline_label,
                            save_path=overview_out,
                            show=show,
                        )
                        if ov_res is not None:
                            written.append(ov_res)
                            print(
                                f"  ✓ assembly_time ×{len(loaded)} (stats)  →  {ov_res}"
                            )

        if manual_paths:
            loaded = []
            for p in manual_paths:
                try:
                    loaded.append(load_summary(p))
                except Exception as e:
                    print(f"  ! skip {p} ({e})")
            if loaded:
                merged = _merge_manual_validation_batches(loaded)
                if not _manual_validation_complete_rows(merged):
                    print(
                        "[plot] combined manual_validation: no fully-evaluated "
                        "assemblies, skipping"
                    )
                else:
                    summary_out = (
                        out_root / f"combined_manual_validation_summary_{dir_tag}.png"
                    )
                    per_ass_out = (
                        out_root
                        / f"combined_manual_validation_per_assembly_{dir_tag}.png"
                    )
                    s_res = _render_manual_validation_summary(
                        merged,
                        save_path=summary_out,
                        show=show,
                    )
                    a_res = _render_manual_validation_per_assembly(
                        merged,
                        save_path=per_ass_out,
                        show=show,
                    )
                    if s_res is not None:
                        written.append(s_res)
                        print(
                            f"  ✓ manual_validation ×{len(loaded)} (summary)       →  {s_res}"
                        )
                    if a_res is not None:
                        written.append(a_res)
                        print(
                            f"  ✓ manual_validation ×{len(loaded)} (per-assembly)  →  {a_res}"
                        )

        if cd_paths:
            # Aggregate per-tool totals/correct across every convex_decomp
            # results file under target_dir, then render one accuracy bar
            # chart keyed by tool id. Side artifact: regenerate the per-
            # assembly all_tools.png grid for each input file (grid is
            # inherently per-assembly, so no "combined" version of it).
            tallies = []
            n_samples_seen = None
            for p in cd_paths:
                try:
                    payload = load_summary(p)
                except Exception as e:
                    print(f"  ! skip {p} ({e})")
                    continue
                tallies.append(_convex_decomp_tally(payload))
                if n_samples_seen is None:
                    n_samples_seen = payload.get("n_samples")
                grid_path = p.parent / "all_tools.png"
                grid_result = _render_convex_decomp_all_tools(
                    payload,
                    save_path=grid_path,
                    json_path=p,
                    show=show,
                )
                if grid_result is not None:
                    written.append(grid_result)
                    print(f"  ✓ convex_decomp all_tools  →  {grid_result}")
            if tallies:
                totals, correct = _merge_convex_decomp_tallies(tallies)
                if not totals:
                    print(
                        "[plot] combined convex_decomp: no validated entries, skipping"
                    )
                else:
                    out = out_root / f"combined_convex_decomp_accuracy_{dir_tag}.png"
                    cd_res = _render_convex_decomp_accuracy(
                        totals,
                        correct,
                        save_path=out,
                        show=show,
                        n_samples=n_samples_seen,
                    )
                    if cd_res is not None:
                        written.append(cd_res)
                        print(f"  ✓ convex_decomp ×{len(tallies)}  →  {cd_res}")

        if tool_paths:
            # Pair every human_labels.json with its sibling ai_labels*.json
            # and aggregate all assemblies into one clustered-bootstrap
            # tool-detection accuracy chart (the run_collect_tool_data port).
            pairs = _tool_selection_pairs(tool_paths)
            if not pairs:
                print(
                    "[plot] combined tool_selection: no ai_labels*.json "
                    "siblings, skipping"
                )
            else:
                samples, part_data = _load_tool_selection_samples(pairs)
                if not part_data:
                    print("[plot] combined tool_selection: no labeled parts, skipping")
                else:
                    out = out_root / f"combined_tool_selection_accuracy_{dir_tag}.png"
                    result = _render_tool_selection_figure(
                        samples,
                        part_data,
                        save_path=out,
                        show=show,
                    )
                    if result is not None:
                        written.append(result)
                        print(f"  ✓ tool_selection ×{len(pairs)}  →  {result}")

        print(f"[plot] wrote {len(written)} combined plot(s)")
        return written

    for p in paths:
        if save_dir is not None:
            stem = _stem_of(p)
            # Keep any numbered suffix in the output name so files don't collide.
            variant = p.stem[len(stem) :] if stem else ""
            if stem == "comparison_batch_summary":
                out_path = (
                    save_dir
                    / f"selector_agreement_batch_{_run_tag_from(p)}{variant}.png"
                )
            elif stem == "validation_summary":
                out_path = (
                    save_dir / f"beam_width_to_optimum_{_run_tag_from(p)}{variant}.png"
                )
            elif stem == "assembly_time_summary":
                out_path = (
                    save_dir
                    / f"assembly_time_comparison_{_run_tag_from(p)}{variant}.png"
                )
            elif stem == "manual_validation_batch":
                # plot_manual_validation_batch derives the per-assembly path
                # from the summary path's filename, so passing the summary
                # path here is enough.
                out_path = (
                    save_dir
                    / f"manual_validation_summary_{_run_tag_from(p)}{variant}.png"
                )
            elif stem == "convex_decomp_results":
                out_path = (
                    save_dir
                    / f"convex_decomp_accuracy_{_assembly_tag_from(p)}{variant}.png"
                )
            elif stem == "human_labels":
                out_path = (
                    save_dir
                    / f"tool_selection_accuracy_{_assembly_tag_from(p)}{variant}.png"
                )
            else:
                # comparison_summary: … / <assembly-id> / log / file
                out_path = (
                    save_dir
                    / f"selector_agreement_{_assembly_tag_from(p)}{variant}.png"
                )
        else:
            out_path = None
        result = plot_summary(
            p,
            save_path=out_path,
            show=show,
            baseline=baseline,
            max_k_override=max_k_override,
            line_only=line_only,
        )
        if result is not None:
            written.append(result)
            print(f"  ✓ {p}  →  {result}")
    print(f"[plot] wrote {len(written)} plot(s)")
    return written


# ----------------------------------------------------------------------
# CLI


def _cli():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "target_dir",
        type=Path,
        help="Directory to walk for summary JSON files. Picks up "
        "comparison_summary.json, comparison_batch_summary.json, "
        "validation_summary.json (k-greedy validation), "
        "assembly_time_summary.json (data_assembly_time), "
        "manual_validation_batch.json (data_manual_validation), "
        "convex_decomp_results.json (test_convex_decomp; "
        "legacy results.json under convex_decomp/<id>/ is also picked up), "
        "and human_labels.json (collect_tool_data tool decisions, "
        "paired with sibling ai_labels*.json). "
        "Typically an `assets/output/<timestamp>/` run dir.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to writing each "
        "plot next to its source JSON.",
    )
    ap.add_argument(
        "--no-recursive",
        action="store_true",
        help="Don't recurse — only look at target_dir itself.",
    )
    ap.add_argument(
        "--show",
        action="store_true",
        help="Also pop up an interactive window for each plot.",
    )
    ap.add_argument(
        "--combine",
        action="store_true",
        help="Combine all datapoints from the discovered files "
        "into a single plot per kind (comparison vs validation), "
        "instead of one plot per source file.",
    )
    ap.add_argument(
        "--baseline",
        type=str,
        default=DEFAULT_ASSEMBLY_TIME_BASELINE,
        help=f"Baseline run for the assembly-time ratio chart "
        f"(default: {DEFAULT_ASSEMBLY_TIME_BASELINE!r}). Matched "
        f"against the run's `generator` field first, then `label`.",
    )
    ap.add_argument(
        "--max-k",
        type=int,
        default=None,
        help="Override the k-axis cap on the beam-search "
        "optimality validation plot. Default is "
        "min(max_k_tested, max(n_parts) across finished "
        "assemblies); use this when you want to show the "
        "full k range tested (since k=N isn't actually "
        "exhaustive for the beam over the state DAG). "
        "Value is clamped to max_k_tested either way.",
    )
    ap.add_argument(
        "--line-only",
        action="store_true",
        help="Render only the left panel of the beam-search "
        "validation plot (per-k mean best-score line); "
        "drops the size-vs-lowest-k scatter on the right.",
    )
    args = ap.parse_args()

    if not args.target_dir.exists():
        print(f"error: {args.target_dir} does not exist", file=sys.stderr)
        sys.exit(2)

    find_and_plot(
        target_dir=args.target_dir,
        save_dir=args.out,
        recursive=not args.no_recursive,
        show=args.show,
        combine=args.combine,
        baseline=args.baseline,
        max_k_override=args.max_k,
        line_only=args.line_only,
    )


if __name__ == "__main__":
    _cli()
