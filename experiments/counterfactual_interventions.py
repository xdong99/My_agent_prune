import copy
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

import torch


@dataclass(frozen=True)
class EdgeIntervention:
    edge_type: str
    src: int
    dst: int
    value: int

    def __post_init__(self) -> None:
        if self.edge_type not in {"spatial", "temporal"}:
            raise ValueError("edge_type must be 'spatial' or 'temporal'.")
        if self.value not in {0, 1}:
            raise ValueError("value must be 0 or 1.")
        if self.src < 0 or self.dst < 0:
            raise ValueError("src and dst must be non-negative.")


@dataclass(frozen=True)
class RolloutOutcome:
    answer: str
    correct: bool
    loss: float
    cost: float
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class CounterfactualEffect:
    edge: EdgeIntervention
    off: RolloutOutcome
    on: RolloutOutcome
    loss_delta: float
    cost_delta: float
    effect: float
    helped: bool


def edge_label(edge: EdgeIntervention) -> str:
    return f"{edge.edge_type}:{edge.src}->{edge.dst}"


def effect_to_dict(effect: CounterfactualEffect) -> dict[str, Any]:
    return {
        "edge": edge_label(effect.edge),
        "edge_type": effect.edge.edge_type,
        "src": effect.edge.src,
        "dst": effect.edge.dst,
        "loss_delta": effect.loss_delta,
        "cost_delta": effect.cost_delta,
        "effect": effect.effect,
        "helped": effect.helped,
        "off_answer": effect.off.answer,
        "off_correct": effect.off.correct,
        "off_loss": effect.off.loss,
        "off_cost": effect.off.cost,
        "on_answer": effect.on.answer,
        "on_correct": effect.on.correct,
        "on_loss": effect.on.loss,
        "on_cost": effect.on.cost,
    }


def summarize_effects(
    effects: list[CounterfactualEffect],
    *,
    top_k: int = 5,
) -> dict[str, Any]:
    rows = [effect_to_dict(effect) for effect in effects]
    if not rows:
        return {
            "num_effects": 0,
            "mean_effect": 0.0,
            "mean_loss_delta": 0.0,
            "mean_cost_delta": 0.0,
            "num_helpful": 0,
            "num_harmful": 0,
            "top_helpful": [],
            "top_harmful": [],
            "by_edge": [],
        }

    mean_effect = sum(row["effect"] for row in rows) / len(rows)
    mean_loss_delta = sum(row["loss_delta"] for row in rows) / len(rows)
    mean_cost_delta = sum(row["cost_delta"] for row in rows) / len(rows)
    helpful = [row for row in rows if row["effect"] > 0.0]
    harmful = [row for row in rows if row["effect"] < 0.0]
    by_edge = _summarize_by_edge(rows)
    return {
        "num_effects": len(rows),
        "mean_effect": mean_effect,
        "mean_loss_delta": mean_loss_delta,
        "mean_cost_delta": mean_cost_delta,
        "num_helpful": len(helpful),
        "num_harmful": len(harmful),
        "top_helpful": sorted(helpful, key=lambda row: row["effect"], reverse=True)[:top_k],
        "top_harmful": sorted(harmful, key=lambda row: row["effect"])[:top_k],
        "by_edge": by_edge,
    }


def _summarize_by_edge(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row["edge"], []).append(row)

    summaries = []
    for edge, edge_rows in grouped.items():
        n = len(edge_rows)
        summaries.append(
            {
                "edge": edge,
                "num_effects": n,
                "mean_effect": sum(row["effect"] for row in edge_rows) / n,
                "mean_loss_delta": sum(row["loss_delta"] for row in edge_rows) / n,
                "mean_cost_delta": sum(row["cost_delta"] for row in edge_rows) / n,
                "num_helpful": sum(1 for row in edge_rows if row["effect"] > 0.0),
                "num_harmful": sum(1 for row in edge_rows if row["effect"] < 0.0),
            }
        )
    return sorted(summaries, key=lambda row: row["mean_effect"], reverse=True)


def apply_edge_intervention(
    graph: Any,
    edge: EdgeIntervention,
    *,
    logit_magnitude: float = 20.0,
) -> Any:
    """Return a copied graph with one communication edge forced on or off."""
    realized_graph = copy.deepcopy(graph)
    logits_attr = f"{edge.edge_type}_logits"
    mask_attrs = (f"{edge.edge_type}_mask", f"{edge.edge_type}_masks")

    if not hasattr(realized_graph, logits_attr):
        raise AttributeError(f"graph has no {logits_attr!r} attribute.")

    logits = getattr(realized_graph, logits_attr)
    if not isinstance(logits, torch.Tensor):
        logits = torch.as_tensor(logits, dtype=torch.float32)
    else:
        logits = logits.detach().clone()

    edge_index = _edge_index(realized_graph, logits, edge)
    logits[edge_index] = logit_magnitude if edge.value == 1 else -logit_magnitude
    setattr(realized_graph, logits_attr, logits)

    for mask_attr in mask_attrs:
        if not hasattr(realized_graph, mask_attr):
            continue
        mask = getattr(realized_graph, mask_attr)
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().clone()
            mask[edge_index] = float(edge.value)
        else:
            mask = copy.deepcopy(mask)
            if isinstance(edge_index, tuple):
                mask[edge_index[0]][edge_index[1]] = edge.value
            else:
                mask[edge_index] = edge.value
        setattr(realized_graph, mask_attr, mask)

    if hasattr(realized_graph, "optimized_spatial"):
        realized_graph.optimized_spatial = False
    if hasattr(realized_graph, "optimized_temporal"):
        realized_graph.optimized_temporal = False

    return realized_graph


def counterfactual_delta(
    *,
    edge: EdgeIntervention,
    off: RolloutOutcome,
    on: RolloutOutcome,
    cost_lambda: float = 0.0,
) -> CounterfactualEffect:
    """Estimate edge value as loss reduction minus communication-cost increase."""
    loss_delta = float(off.loss) - float(on.loss)
    cost_delta = float(on.cost) - float(off.cost)
    effect = loss_delta - float(cost_lambda) * cost_delta
    return CounterfactualEffect(
        edge=edge,
        off=off,
        on=on,
        loss_delta=loss_delta,
        cost_delta=cost_delta,
        effect=effect,
        helped=effect > 0.0,
    )


async def paired_counterfactual_rollout(
    *,
    graph: Any,
    input_dict: Mapping[str, Any],
    target_answer: str,
    edge: EdgeIntervention,
    num_rounds: int = 1,
    cost_lambda: float = 0.0,
    logit_magnitude: float = 20.0,
    answer_normalizer: Callable[[Any], str] | None = None,
    arun_kwargs: Mapping[str, Any] | None = None,
) -> CounterfactualEffect:
    """Run paired do(edge=0/1) rollouts and return the estimated edge effect."""
    off_edge = EdgeIntervention(edge.edge_type, edge.src, edge.dst, 0)
    on_edge = EdgeIntervention(edge.edge_type, edge.src, edge.dst, 1)

    off_graph = apply_edge_intervention(
        graph, off_edge, logit_magnitude=logit_magnitude
    )
    on_graph = apply_edge_intervention(
        graph, on_edge, logit_magnitude=logit_magnitude
    )

    normalizer = answer_normalizer or _normalize_choice
    off = await _run_intervened_graph(
        off_graph, input_dict, target_answer, num_rounds, normalizer, arun_kwargs or {}
    )
    on = await _run_intervened_graph(
        on_graph, input_dict, target_answer, num_rounds, normalizer, arun_kwargs or {}
    )
    return counterfactual_delta(edge=on_edge, off=off, on=on, cost_lambda=cost_lambda)


async def _run_intervened_graph(
    graph: Any,
    input_dict: Mapping[str, Any],
    target_answer: str,
    num_rounds: int,
    answer_normalizer: Callable[[Any], str],
    arun_kwargs: Mapping[str, Any],
) -> RolloutOutcome:
    prompt_before, completion_before = _read_global_token_counts()
    result = await graph.arun(dict(input_dict), num_rounds, **dict(arun_kwargs))
    prompt_after, completion_after = _read_global_token_counts()
    final_out = result[0] if isinstance(result, tuple) else result
    answer = _extract_answer(final_out, answer_normalizer)
    target = answer_normalizer(target_answer)
    correct = answer == target
    loss = 0.0 if correct else 1.0
    cost = _extract_cost(final_out)
    if cost == 0.0:
        cost = (prompt_after - prompt_before) + (completion_after - completion_before)
    return RolloutOutcome(
        answer=answer,
        correct=correct,
        loss=loss,
        cost=cost,
        raw={"final_out": final_out},
    )


def _extract_answer(final_out: Any, answer_normalizer: Callable[[Any], str]) -> str:
    if isinstance(final_out, list):
        final_out = final_out[0] if final_out else ""
    if isinstance(final_out, Mapping):
        for key in ("choice", "answer", "pred", "prediction"):
            if key in final_out:
                return answer_normalizer(final_out[key])
        final_out = final_out.get("text", "")
    return answer_normalizer(final_out)


def _normalize_choice(value: Any) -> str:
    text = str(value).strip().upper()
    match = re.search(r"\b([ABCD])\b", text)
    if match:
        return match.group(1)
    return text[:1] if text else ""


def _extract_cost(final_out: Any) -> float:
    if isinstance(final_out, list):
        final_out = final_out[0] if final_out else {}
    if isinstance(final_out, Mapping):
        for key in ("cost", "total_cost", "total_tokens", "token_count", "tokens"):
            if key in final_out:
                try:
                    return float(final_out[key])
                except (TypeError, ValueError):
                    return 0.0
    return 0.0


def _read_global_token_counts() -> tuple[float, float]:
    try:
        from AgentPrune.utils.globals import CompletionTokens, PromptTokens
    except Exception:
        return 0.0, 0.0
    return float(PromptTokens.instance().value), float(CompletionTokens.instance().value)


def _edge_index(graph: Any, logits: torch.Tensor, edge: EdgeIntervention) -> Any:
    if logits.ndim == 2:
        rows, cols = logits.shape
        if edge.src >= rows or edge.dst >= cols:
            raise IndexError(
                f"edge ({edge.src}, {edge.dst}) is out of bounds for "
                f"{edge.edge_type}_logits shape {(rows, cols)}."
            )
        return edge.src, edge.dst

    if logits.ndim == 1:
        num_nodes = _infer_num_nodes(graph, logits)
        if edge.src >= num_nodes or edge.dst >= num_nodes:
            raise IndexError(
                f"edge ({edge.src}, {edge.dst}) is out of bounds for "
                f"{edge.edge_type}_logits with {num_nodes} nodes."
            )
        return edge.src * num_nodes + edge.dst

    raise ValueError(f"{edge.edge_type}_logits must be a 1D or 2D tensor.")


def _infer_num_nodes(graph: Any, logits: torch.Tensor) -> int:
    if hasattr(graph, "nodes"):
        return len(getattr(graph, "nodes"))
    root = int(logits.numel() ** 0.5)
    if root * root != logits.numel():
        raise IndexError(
            f"cannot infer node count from flattened logits length {logits.numel()}."
        )
    return root
