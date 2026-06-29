import argparse
import asyncio
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.run_gsm8k_generated_task_poc import (
    Variant,
    build_graph,
    choose_variant,
    evaluate_variants,
    generate_diagnostic_candidates,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate generated diagnostic tasks on saved GSM8K graph variants."
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--num_tasks", type=int, default=12)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument(
        "--families",
        nargs="*",
        default=None,
        help="Optional generated task families to keep; samples until num_tasks from these families are available.",
    )
    parser.add_argument("--output_name", default=None)
    parser.add_argument("--cost_lambda", type=float, default=None)
    parser.add_argument(
        "--include_variants",
        nargs="*",
        default=None,
        help="Optional subset of variant names to evaluate.",
    )
    return parser.parse_args()


def load_metadata(run_dir: Path) -> dict:
    return json.loads((run_dir / "metadata.json").read_text(encoding="utf-8"))


def args_from_metadata(meta: dict) -> SimpleNamespace:
    return SimpleNamespace(
        llm_name=meta["llm_name"],
        seed=int(meta["seed"]),
        mode=meta.get("mode", "FullConnected"),
        decision_method=meta.get("decision_method", "FinalRefer"),
        num_agents=int(meta["num_agents"]),
        num_rounds=int(meta["num_rounds"]),
    )


def variants_from_metadata(meta: dict) -> list[Variant]:
    variants = []
    for row in meta["variants"]:
        variants.append(
            Variant(
                name=row["name"],
                description=row["description"],
                spatial_masks=torch.tensor(row["spatial_masks"], dtype=torch.float32),
                temporal_masks=torch.tensor(row["temporal_masks"], dtype=torch.float32),
                spatial_logits=torch.tensor(row["spatial_logits"], dtype=torch.float32),
                temporal_logits=torch.tensor(row["temporal_logits"], dtype=torch.float32),
            )
        )
    return variants


def task_disagreement(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for record_id, group in rows.groupby("record_id"):
        out.append(
            {
                "record_id": int(record_id),
                "family": group["family"].iloc[0],
                "target": group["target"].iloc[0],
                "prediction_disagrees": bool(group["prediction"].astype(str).nunique() > 1),
                "correctness_disagrees": bool(group["correct"].astype(bool).nunique() > 1),
                "num_distinct_predictions": int(group["prediction"].astype(str).nunique()),
                "task": group["task"].iloc[0],
            }
        )
    return pd.DataFrame(out)


async def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    meta = load_metadata(run_dir)
    graph = build_graph(args_from_metadata(meta))
    variants = variants_from_metadata(meta)
    if args.include_variants:
        keep = set(args.include_variants)
        variants = [variant for variant in variants if variant.name in keep]
        missing = sorted(keep - {variant.name for variant in variants})
        if missing:
            raise ValueError(f"requested variants not found in metadata: {missing}")

    if args.families:
        wanted = set(args.families)
        tasks = []
        multiplier = 2
        while len(tasks) < args.num_tasks:
            pool = generate_diagnostic_candidates(args.num_tasks * multiplier, args.seed)
            tasks = [task for task in pool if task.get("family") in wanted][: args.num_tasks]
            multiplier *= 2
            if multiplier > 64 and len(tasks) < args.num_tasks:
                raise RuntimeError(
                    f"could only generate {len(tasks)} tasks for requested families {sorted(wanted)}"
                )
    else:
        tasks = generate_diagnostic_candidates(args.num_tasks, args.seed)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = run_dir / (args.output_name or f"generated_eval_seed{args.seed}_n{args.num_tasks}_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = await evaluate_variants(
        graph,
        variants,
        tasks,
        num_rounds=int(meta["num_rounds"]),
        max_concurrency=1,
        aggregate_mode=meta.get("aggregate_mode", "all connected"),
    )
    cost_lambda = args.cost_lambda if args.cost_lambda is not None else float(meta.get("cost_lambda", 0.0))
    ranking = choose_variant(summary, cost_lambda)
    disagreement = task_disagreement(rows)
    rows.to_csv(out_dir / "variant_rows.csv", index=False)
    summary.to_csv(out_dir / "variant_summary.csv", index=False)
    ranking.to_csv(out_dir / "variant_ranking.csv", index=False)
    disagreement.to_csv(out_dir / "task_disagreement.csv", index=False)

    report = [
        "# Saved-Variant Generated Diagnostic Evaluation",
        "",
        f"- Source run: `{run_dir}`",
        f"- Generated tasks: {len(tasks)}",
        f"- Seed: {args.seed}",
        f"- Families: {args.families}",
        f"- Aggregate mode: `{meta.get('aggregate_mode', 'all connected')}`",
        "",
        "## Source Metrics",
        "",
        f"- Prediction disagreement rate: {disagreement['prediction_disagrees'].mean():.4f}",
        f"- Correctness disagreement rate: {disagreement['correctness_disagrees'].mean():.4f}",
        "",
        "## Variant Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## Ranking",
        "",
        ranking.to_markdown(index=False),
        "",
        "## Disagreement Tasks",
        "",
        disagreement[disagreement["correctness_disagrees"]].to_markdown(index=False),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote generated evaluation to {out_dir}")
    print(f"Prediction disagreement rate: {disagreement['prediction_disagrees'].mean():.4f}")
    print(f"Correctness disagreement rate: {disagreement['correctness_disagrees'].mean():.4f}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
