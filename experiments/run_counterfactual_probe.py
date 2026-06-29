import argparse
import asyncio
import json
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.counterfactual_interventions import (
    EdgeIntervention,
    effect_to_dict,
    paired_counterfactual_rollout,
    summarize_effects,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired counterfactual communication-edge probes."
    )
    parser.add_argument("--domain", choices=["gsm8k"], default="gsm8k")
    parser.add_argument("--llm_name", default="gpt-4.1-nano")
    parser.add_argument("--num_agents", type=int, default=2)
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument("--limit_tasks", type=int, default=2)
    parser.add_argument("--limit_edges", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cost_lambda", type=float, default=0.0)
    parser.add_argument("--logit_magnitude", type=float, default=20.0)
    parser.add_argument(
        "--aggregate_mode",
        choices=["all connected", "last connected"],
        default="all connected",
    )
    parser.add_argument("--output_dir", default="outputs/counterfactual_probe")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Use a deterministic local fake graph; no LLM/API calls.",
    )
    return parser.parse_args()


def load_gsm8k_records(limit: int, seed: int) -> list[dict[str, str]]:
    records = []
    path = Path("dataset/gsm8k/test.jsonl")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            raw_answer = item["answer"]
            answer = raw_answer.split("\n####")[-1].replace(",", "").strip()
            records.append({"task": item["question"], "answer": answer})
    rng = random.Random(seed)
    rng.shuffle(records)
    return records[:limit]


def load_dry_run_records(limit: int) -> list[dict[str, str]]:
    records = [
        {
            "task": "Natalia sold clips to 48 friends in April and half as many in May. How many clips did she sell altogether?",
            "answer": "72",
        },
        {
            "task": "A pond had 2 ducks, then 3 more ducks arrived. How many ducks are there?",
            "answer": "5",
        },
    ]
    return records[:limit]


def build_candidate_edges(num_agents: int, limit_edges: int) -> list[EdgeIntervention]:
    edges = [
        EdgeIntervention("spatial", src=i, dst=j, value=1)
        for i in range(num_agents)
        for j in range(num_agents)
        if i != j
    ]
    return edges[:limit_edges]


def gsm8k_answer_normalizer(value: Any) -> str:
    from dataset.gsm8k_dataset import gsm_get_predict

    return gsm_get_predict(str(value))


def build_gsm8k_graph(llm_name: str, num_agents: int):
    from AgentPrune.graph.graph import Graph

    return Graph(
        domain="gsm8k",
        llm_name=llm_name,
        agent_names=["MathSolver"] * num_agents,
        decision_method="FinalRefer",
        optimized_spatial=False,
        optimized_temporal=False,
    )


class DryRunGraph:
    def __init__(self, num_agents: int):
        import torch

        self.nodes = {f"n{i}": object() for i in range(num_agents)}
        self.spatial_logits = torch.zeros(num_agents * num_agents)
        self.temporal_logits = torch.zeros(num_agents * num_agents)
        self.spatial_masks = torch.ones(num_agents * num_agents)
        self.temporal_masks = torch.ones(num_agents * num_agents)
        self.optimized_spatial = False
        self.optimized_temporal = False

    async def arun(self, input_dict, num_rounds):
        edge_01_on = self.spatial_masks[1].item() > 0.5
        edge_10_on = self.spatial_masks[2].item() > 0.5
        if edge_01_on and "Natalia" in input_dict.get("task", ""):
            return "The answer is 72", []
        if edge_10_on and "ducks" in input_dict.get("task", "").lower():
            return "The answer is 5", []
        return "The answer is 0", []


async def run_probe(args: argparse.Namespace) -> dict[str, Any]:
    records = (
        load_dry_run_records(args.limit_tasks)
        if args.dry_run
        else load_gsm8k_records(args.limit_tasks, args.seed)
    )
    edges = build_candidate_edges(args.num_agents, args.limit_edges)
    graph = DryRunGraph(args.num_agents) if args.dry_run else build_gsm8k_graph(
        args.llm_name, args.num_agents
    )

    rows = []
    effects = []
    for task_id, record in enumerate(records):
        input_dict = {"task": record["task"]}
        for edge in edges:
            effect = await paired_counterfactual_rollout(
                graph=graph,
                input_dict=input_dict,
                target_answer=record["answer"],
                edge=edge,
                num_rounds=args.num_rounds,
                cost_lambda=args.cost_lambda,
                logit_magnitude=args.logit_magnitude,
                answer_normalizer=gsm8k_answer_normalizer,
                arun_kwargs={"aggregate_mode": args.aggregate_mode},
            )
            effects.append(effect)
            row = effect_to_dict(effect)
            row["task_id"] = task_id
            row["target_answer"] = record["answer"]
            row["question_preview"] = record["task"][:160]
            rows.append(row)

    summary = summarize_effects(effects)
    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "domain": args.domain,
        "dry_run": args.dry_run,
        "llm_name": args.llm_name,
        "num_agents": args.num_agents,
        "num_rounds": args.num_rounds,
        "limit_tasks": args.limit_tasks,
        "limit_edges": args.limit_edges,
        "cost_lambda": args.cost_lambda,
        "aggregate_mode": args.aggregate_mode,
        "summary": summary,
        "rows": rows,
    }


def write_reports(report: dict[str, Any], output_dir: str) -> tuple[Path, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"counterfactual_probe_{stamp}"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(render_markdown_report(report), encoding="utf-8")
    return json_path, md_path


def render_markdown_report(report: dict[str, Any]) -> str:
    summary = report["summary"]
    lines = [
        "# Counterfactual Communication Probe Report",
        "",
        f"- Domain: {report['domain']}",
        f"- Dry run: {report['dry_run']}",
        f"- Model: {report['llm_name']}",
        f"- Tasks: {report['limit_tasks']}",
        f"- Edges per task: {report['limit_edges']}",
        f"- Mean effect: {summary['mean_effect']:.4f}",
        f"- Mean loss delta: {summary['mean_loss_delta']:.4f}",
        f"- Mean cost delta: {summary['mean_cost_delta']:.4f}",
        f"- Aggregate mode: {report['aggregate_mode']}",
        f"- Helpful effects: {summary['num_helpful']} / {summary['num_effects']}",
        "",
        "## Top Helpful Edges",
        "",
    ]
    for row in summary["top_helpful"]:
        lines.append(
            f"- `{row['edge']}` effect={row['effect']:.4f}, "
            f"loss_delta={row['loss_delta']:.4f}, cost_delta={row['cost_delta']:.1f}"
        )
    lines.extend(["", "## Top Harmful Edges", ""])
    for row in summary["top_harmful"]:
        lines.append(
            f"- `{row['edge']}` effect={row['effect']:.4f}, "
            f"loss_delta={row['loss_delta']:.4f}, cost_delta={row['cost_delta']:.1f}"
        )
    lines.extend(["", "## Edge-Level Means", ""])
    for row in summary.get("by_edge", []):
        lines.append(
            f"- `{row['edge']}` mean_effect={row['mean_effect']:.4f}, "
            f"mean_loss_delta={row['mean_loss_delta']:.4f}, "
            f"mean_cost_delta={row['mean_cost_delta']:.1f}, "
            f"helpful={row['num_helpful']}/{row['num_effects']}"
        )
    lines.extend(["", "## Per-Probe Rows", ""])
    for row in report["rows"]:
        lines.append(
            f"- task={row['task_id']} edge=`{row['edge']}` "
            f"off={row['off_answer']}({row['off_correct']}) "
            f"on={row['on_answer']}({row['on_correct']}) "
            f"effect={row['effect']:.4f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    report = asyncio.run(run_probe(args))
    json_path, md_path = write_reports(report, args.output_dir)
    print(f"Wrote JSON report: {json_path}")
    print(f"Wrote Markdown report: {md_path}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
