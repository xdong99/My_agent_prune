import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select a conservative graph repair from generated disagreement tasks: "
            "do not lose generated accuracy relative to AL, then minimize structural edges."
        )
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--baseline", default="al_pruned")
    parser.add_argument("--output_name", default="generated_repair_selection_report.md")
    return parser.parse_args()


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.exists() else pd.DataFrame()


def summarize_variant_rows(rows: pd.DataFrame) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    rows = rows.copy()
    rows["correct"] = rows["correct"].map(parse_bool)
    return (
        rows.groupby("variant")
        .agg(
            accuracy=("correct", "mean"),
            mean_tokens=("tokens", "mean"),
            num_tasks=("record_id", "nunique"),
            num_spatial_edges=("num_spatial_edges", "first"),
        )
        .reset_index()
    )


def parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes"}


def conservative_select(summary: pd.DataFrame, baseline: str) -> pd.DataFrame:
    if summary.empty:
        return pd.DataFrame()
    if baseline not in set(summary["variant"]):
        raise ValueError(f"baseline variant {baseline!r} missing from summary")
    base_acc = float(summary.loc[summary["variant"] == baseline, "accuracy"].iloc[0])
    candidates = summary[summary["accuracy"] >= base_acc - 1e-9].copy()
    candidates["accuracy_margin_vs_baseline"] = candidates["accuracy"] - base_acc
    return candidates.sort_values(
        ["num_spatial_edges", "mean_tokens", "variant"],
        ascending=[True, True, True],
    ).reset_index(drop=True)


def dominance_safe_select(rows: pd.DataFrame, baseline: str) -> pd.DataFrame:
    if rows.empty:
        return pd.DataFrame()
    rows = rows.copy()
    rows["correct"] = rows["correct"].map(parse_bool)
    summary = summarize_variant_rows(rows)
    base = rows[rows["variant"] == baseline][["record_id", "correct"]].rename(
        columns={"correct": "baseline_correct"}
    )
    records = []
    for variant, group in rows.groupby("variant"):
        merged = group.merge(base, on="record_id", how="left")
        baseline_regressions = int(
            ((merged["baseline_correct"].astype(bool)) & (~merged["correct"].astype(bool))).sum()
        )
        baseline_improvements = int(
            ((~merged["baseline_correct"].astype(bool)) & (merged["correct"].astype(bool))).sum()
        )
        row = summary[summary["variant"] == variant].iloc[0].to_dict()
        row["baseline_regressions"] = baseline_regressions
        row["baseline_improvements"] = baseline_improvements
        row["dominance_safe"] = baseline_regressions == 0
        records.append(row)
    candidates = pd.DataFrame(records)
    candidates = candidates[candidates["dominance_safe"]].copy()
    if candidates.empty:
        return candidates
    return candidates.sort_values(
        ["num_spatial_edges", "baseline_improvements", "mean_tokens", "variant"],
        ascending=[True, False, True, True],
    ).reset_index(drop=True)


def collect_natural_summaries(run_dir: Path) -> list[tuple[str, pd.DataFrame]]:
    out = []
    for name in ["natural_probe", "natural_eval"]:
        df = load_csv(run_dir / f"{name}_variant_summary.csv")
        if not df.empty:
            out.append((name, df))
    for extra_dir in sorted(run_dir.glob("extra_natural_eval_*")):
        df = load_csv(extra_dir / "variant_summary.csv")
        if not df.empty:
            out.append((extra_dir.name, df))
    return out


def markdown_table(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False) if not df.empty else "No rows."


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    metadata = json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))
    selected_tasks = load_csv(run_dir / "selected_generated_tasks.csv")
    generated_candidate_rows = load_csv(run_dir / "generated_candidate_variant_rows.csv")
    if not selected_tasks.empty and not generated_candidate_rows.empty:
        selected_ids = set(int(x) for x in selected_tasks["record_id"].tolist())
        selected_task_rows = generated_candidate_rows[
            generated_candidate_rows["record_id"].astype(int).isin(selected_ids)
        ].copy()
    else:
        selected_task_rows = load_csv(run_dir / "selected_generated_variant_rows.csv")
    selected_task_summary = summarize_variant_rows(selected_task_rows)
    noninferior_candidates = conservative_select(selected_task_summary, args.baseline)
    candidates = dominance_safe_select(selected_task_rows, args.baseline)
    chosen = candidates.iloc[0].to_dict() if not candidates.empty else {}

    generated_all = load_csv(run_dir / "generated_candidate_variant_summary.csv")
    lines = [
        "# Conservative Generated-Task Repair Selection",
        "",
        "## Rule",
        "",
        (
            f"Baseline is `{args.baseline}`. Among variants whose accuracy on selected "
            "generated disagreement tasks is not below the baseline, choose the graph "
            "with the fewest spatial edges; break ties by mean measured tokens."
        ),
        "",
        "## Selected Generated Tasks",
        "",
        markdown_table(
            selected_tasks[
                [
                    "record_id",
                    "family",
                    "target",
                    "num_distinct_predictions",
                    "correctness_disagrees",
                    "selection_score",
                ]
            ]
            if not selected_tasks.empty
            else selected_tasks
        ),
        "",
        "## Generated-Task Summary",
        "",
        markdown_table(selected_task_summary),
        "",
        "## Non-Inferior Sparse Candidates",
        "",
        markdown_table(noninferior_candidates),
        "",
        "## Dominance-Safe Candidates",
        "",
        markdown_table(candidates),
        "",
        "## Chosen Repair",
        "",
        f"- Chosen variant: `{chosen.get('variant')}`",
        f"- Generated accuracy: {chosen.get('accuracy')}",
        f"- Spatial edges: {chosen.get('num_spatial_edges')}",
        f"- Baseline generated accuracy: {float(selected_task_summary.loc[selected_task_summary['variant'] == args.baseline, 'accuracy'].iloc[0]) if not selected_task_summary.empty else 'NA'}",
        "",
        "## Natural Checks Available",
        "",
    ]
    for source, summary in collect_natural_summaries(run_dir):
        lines.extend([f"### {source}", "", markdown_table(summary), ""])

    lines.extend(
        [
            "## Interpretation",
            "",
        (
            "This selector treats generated tasks as diagnostic stress tests, not as final "
            "evaluation. The dominance-safe rule is intentionally conservative: it does "
            "not accept a sparser graph if generated diagnostics reveal any task where "
            "the AL baseline is correct and the repair is wrong. A selected repair is "
            "credible only if it then preserves or improves accuracy on untouched natural tasks."
        ),
        ]
    )
    out_path = run_dir / args.output_name
    out_path.write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "generated_repair_choice.json").write_text(
        json.dumps(
            {
                "chosen": chosen,
                "dominance_safe_candidates": candidates.to_dict("records"),
                "noninferior_candidates": noninferior_candidates.to_dict("records"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {out_path}")
    print(f"Chosen variant: {chosen.get('variant')}")


if __name__ == "__main__":
    main()
