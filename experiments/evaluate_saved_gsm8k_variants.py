import argparse
import asyncio
import json
import random
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
    load_gsm8k,
    plot_summaries,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate saved GSM8K graph variants on additional held-out natural tasks."
    )
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--test_json", default="dataset/gsm8k/test.jsonl")
    parser.add_argument("--data_json", default=None)
    parser.add_argument(
        "--positions",
        nargs="*",
        type=int,
        default=None,
        help="Explicit record positions in data_json/test_json to evaluate without shuffling.",
    )
    parser.add_argument("--offset", type=int, default=None)
    parser.add_argument("--limit", type=int, default=8)
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
    path = run_dir / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


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


async def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    meta = load_metadata(run_dir)
    graph_args = args_from_metadata(meta)
    graph = build_graph(graph_args)
    variants = variants_from_metadata(meta)
    if args.include_variants:
        keep = set(args.include_variants)
        variants = [variant for variant in variants if variant.name in keep]
        missing = sorted(keep - {variant.name for variant in variants})
        if missing:
            raise ValueError(f"requested variants not found in metadata: {missing}")

    data_path = args.data_json or args.test_json
    test_set = load_gsm8k(data_path)
    if args.positions:
        offset = None
        records = [test_set[pos] for pos in args.positions]
    else:
        rng = random.Random(int(meta["seed"]))
        rng.shuffle(test_set)
        if args.offset is None:
            offset = int(meta.get("natural_probe_tasks", 0)) + int(meta.get("natural_eval_tasks", 0))
            if offset <= 0:
                offset = 4
        else:
            offset = args.offset
        records = test_set[offset : offset + args.limit]

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = run_dir / (args.output_name or f"extra_natural_eval_{stamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, summary = await evaluate_variants(
        graph,
        variants,
        records,
        num_rounds=int(meta["num_rounds"]),
        max_concurrency=1,
        aggregate_mode=meta.get("aggregate_mode", "all connected"),
    )
    cost_lambda = args.cost_lambda if args.cost_lambda is not None else float(meta.get("cost_lambda", 0.0))
    ranking = choose_variant(summary, cost_lambda)
    rows.to_csv(out_dir / "variant_rows.csv", index=False)
    summary.to_csv(out_dir / "variant_summary.csv", index=False)
    ranking.to_csv(out_dir / "variant_ranking.csv", index=False)
    plot_summaries(summary, summary, out_dir)
    report = [
        "# Extra Natural GSM8K Evaluation",
        "",
        f"- Source run: `{run_dir}`",
        f"- Data: `{data_path}`",
        f"- Offset: {offset}",
        f"- Positions: {args.positions}",
        f"- Limit: {len(records)}",
        f"- Aggregate mode: `{meta.get('aggregate_mode', 'all connected')}`",
        "",
        "## Ranking",
        "",
        ranking.to_markdown(index=False),
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"Wrote extra evaluation to {out_dir}")
    print(ranking.to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
