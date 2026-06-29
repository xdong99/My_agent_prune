import argparse
import html
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize graph-discrimination evidence from a generated-task PoC run."
    )
    parser.add_argument("--run_dir", required=True)
    return parser.parse_args()


def load_source(path: Path, source: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["source"] = source
    return df


def per_task_discrimination(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if df.empty:
        return pd.DataFrame(rows)
    for (source, record_id), group in df.groupby(["source", "record_id"]):
        rows.append(
            {
                "source": source,
                "record_id": int(record_id),
                "family": group["family"].iloc[0] if "family" in group else "natural",
                "num_variants": int(group["variant"].nunique()),
                "num_distinct_predictions": int(group["prediction"].astype(str).nunique()),
                "prediction_disagrees": bool(group["prediction"].astype(str).nunique() > 1),
                "correctness_disagrees": bool(group["correct"].astype(bool).nunique() > 1),
                "accuracy_variance": float(np.var(group["correct"].astype(float).to_numpy())),
                "task": group["task"].iloc[0] if "task" in group else "",
                "target": group["target"].iloc[0] if "target" in group else "",
            }
        )
    return pd.DataFrame(rows)


def summarize_rates(task_df: pd.DataFrame) -> pd.DataFrame:
    if task_df.empty:
        return pd.DataFrame()
    return (
        task_df.groupby("source")
        .agg(
            num_tasks=("record_id", "count"),
            prediction_disagreement_rate=("prediction_disagrees", "mean"),
            correctness_disagreement_rate=("correctness_disagrees", "mean"),
            mean_accuracy_variance=("accuracy_variance", "mean"),
        )
        .reset_index()
    )


def plot_rates(summary: pd.DataFrame, out_dir: Path) -> Path:
    width, height = 860, 420
    margin_left, margin_bottom, margin_top = 70, 90, 50
    plot_w = width - margin_left - 30
    plot_h = height - margin_top - margin_bottom
    n = max(1, len(summary))
    group_w = plot_w / n
    bar_w = min(42, group_w / 5)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="26" text-anchor="middle" font-family="Arial" font-size="18">Graph-Variant Discrimination Rate</text>',
        f'<line x1="{margin_left}" y1="{margin_top + plot_h}" x2="{margin_left + plot_w}" y2="{margin_top + plot_h}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_h}" stroke="#333"/>',
        f'<text x="16" y="{margin_top + 8}" font-family="Arial" font-size="12">1.0</text>',
        f'<text x="16" y="{margin_top + plot_h}" font-family="Arial" font-size="12">0.0</text>',
    ]
    for i, row in summary.reset_index(drop=True).iterrows():
        x0 = margin_left + i * group_w + group_w / 2
        pred = float(row["prediction_disagreement_rate"])
        corr = float(row["correctness_disagreement_rate"])
        for offset, value, color, label in [
            (-bar_w * 0.6, pred, "#4C78A8", "prediction"),
            (bar_w * 0.6, corr, "#F58518", "correctness"),
        ]:
            h = value * plot_h
            y = margin_top + plot_h - h
            parts.append(
                f'<rect x="{x0 + offset - bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>'
            )
            parts.append(
                f'<text x="{x0 + offset:.1f}" y="{y - 4:.1f}" text-anchor="middle" font-family="Arial" font-size="11">{value:.2f}</text>'
            )
        source = html.escape(str(row["source"]))
        parts.append(
            f'<text x="{x0:.1f}" y="{height - 48}" text-anchor="middle" font-family="Arial" font-size="11">{source}</text>'
        )
    parts.extend(
        [
            f'<rect x="{width - 220}" y="52" width="12" height="12" fill="#4C78A8"/><text x="{width - 202}" y="63" font-family="Arial" font-size="12">Prediction disagreement</text>',
            f'<rect x="{width - 220}" y="72" width="12" height="12" fill="#F58518"/><text x="{width - 202}" y="83" font-family="Arial" font-size="12">Correctness disagreement</text>',
            "</svg>",
        ]
    )
    path = out_dir / "discrimination_rates.svg"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def plot_generated_correctness(df: pd.DataFrame, out_dir: Path) -> Path | None:
    if df.empty:
        return None
    generated = df[df["source"] == "generated_candidates"].copy()
    if generated.empty:
        return None
    pivot = generated.pivot_table(
        index="record_id",
        columns="variant",
        values="correct",
        aggfunc="first",
    ).astype(float)
    cell_w, cell_h = 145, 32
    left, top = 85, 60
    width = left + cell_w * len(pivot.columns) + 30
    height = top + cell_h * len(pivot.index) + 45
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="26" text-anchor="middle" font-family="Arial" font-size="18">Generated Tasks: Correctness by Graph Variant</text>',
    ]
    for c, col in enumerate(pivot.columns):
        x = left + c * cell_w + cell_w / 2
        parts.append(
            f'<text x="{x:.1f}" y="{top - 14}" text-anchor="middle" font-family="Arial" font-size="11">{html.escape(str(col))}</text>'
        )
    for r, idx in enumerate(pivot.index):
        y = top + r * cell_h
        parts.append(
            f'<text x="{left - 12}" y="{y + 21}" text-anchor="end" font-family="Arial" font-size="12">task {idx}</text>'
        )
        for c, col in enumerate(pivot.columns):
            x = left + c * cell_w
            value = float(pivot.iloc[r, c])
            color = "#54A24B" if value >= 0.5 else "#E45756"
            label = "T" if value >= 0.5 else "F"
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell_w - 4}" height="{cell_h - 4}" fill="{color}" opacity="0.88"/>'
            )
            parts.append(
                f'<text x="{x + cell_w/2 - 2}" y="{y + 19}" text-anchor="middle" font-family="Arial" font-size="12" fill="white">{label}</text>'
            )
    parts.append("</svg>")
    path = out_dir / "generated_correctness_heatmap.svg"
    path.write_text("\n".join(parts), encoding="utf-8")
    return path


def write_report(
    run_dir: Path,
    task_df: pd.DataFrame,
    summary: pd.DataFrame,
    plot_paths: list[Path],
) -> Path:
    report_path = run_dir / "diagnostic_discrimination_report.md"
    generated = summary[summary["source"] == "generated_candidates"]
    lines = [
        "# Diagnostic Discrimination Summary",
        "",
        "This report measures whether a task source separates nearby graph variants.",
        "",
        "## Source-Level Rates",
        "",
        summary.to_markdown(index=False) if not summary.empty else "No rows.",
        "",
        "## Generated Tasks With Correctness Disagreement",
        "",
    ]
    disagreements = task_df[
        (task_df["source"] == "generated_candidates") & task_df["correctness_disagrees"]
    ]
    if disagreements.empty:
        lines.append("No generated task produced correctness disagreement.")
    else:
        for row in disagreements.to_dict("records"):
            lines.append(
                f"- task_id={row['record_id']} family={row['family']} "
                f"distinct_predictions={row['num_distinct_predictions']} target={row['target']}"
            )
    lines.extend(["", "## Plots", ""])
    for path in plot_paths:
        lines.append(f"- `{path}`")
    if not generated.empty:
        rate = float(generated["correctness_disagreement_rate"].iloc[0])
        lines.extend(
            [
                "",
                "## Interpretation",
                "",
                (
                    f"Generated diagnostic candidates produced correctness disagreement "
                    f"on {rate:.1%} of tasks in this run. Treat this as a PoC "
                    "for diagnostic task generation, not as evidence that the current "
                    "generated-task selector already improves over the AL-pruned graph."
                ),
            ]
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    frames = [
        load_source(run_dir / "generated_candidate_variant_rows.csv", "generated_candidates"),
        load_source(run_dir / "natural_probe_variant_rows.csv", "natural_probe"),
        load_source(run_dir / "natural_eval_variant_rows.csv", "natural_eval"),
    ]
    for extra_dir in sorted(
        list(run_dir.glob("extra_natural_eval_*"))
        + list(run_dir.glob("*_variant_eval"))
        + list(run_dir.glob("generated_eval_*"))
    ):
        frames.append(load_source(extra_dir / "variant_rows.csv", extra_dir.name))
    df = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    task_df = per_task_discrimination(df)
    summary = summarize_rates(task_df)
    task_df.to_csv(run_dir / "diagnostic_task_discrimination.csv", index=False)
    summary.to_csv(run_dir / "diagnostic_source_discrimination_summary.csv", index=False)
    plot_paths = [plot_rates(summary, run_dir)]
    heatmap = plot_generated_correctness(df, run_dir)
    if heatmap is not None:
        plot_paths.append(heatmap)
    report = write_report(run_dir, task_df, summary, plot_paths)
    print(f"Wrote {report}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
