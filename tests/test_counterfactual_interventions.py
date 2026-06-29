import math
import asyncio
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments.counterfactual_interventions import (
    EdgeIntervention,
    RolloutOutcome,
    apply_edge_intervention,
    counterfactual_delta,
    paired_counterfactual_rollout,
    summarize_effects,
)


class FakeGraph:
    def __init__(self):
        self.spatial_logits = torch.zeros((3, 3), dtype=torch.float32)
        self.temporal_logits = torch.zeros((3, 3), dtype=torch.float32)
        self.spatial_mask = torch.ones((3, 3), dtype=torch.float32)
        self.temporal_mask = torch.ones((3, 3), dtype=torch.float32)
        self.optimized_spatial = True
        self.optimized_temporal = True


class FakeAsyncGraph(FakeGraph):
    async def arun(self, input_dict, num_rounds, **kwargs):
        enabled = self.spatial_logits[0, 1].item() > 0
        answer = "A" if enabled else "B"
        cost = 30.0 if enabled else 10.0
        return {"choice": answer, "cost": cost}, []


class FakeFlatGraph:
    def __init__(self):
        self.spatial_logits = torch.zeros(9, dtype=torch.float32)
        self.temporal_logits = torch.zeros(9, dtype=torch.float32)
        self.spatial_masks = torch.ones(9, dtype=torch.float32)
        self.temporal_masks = torch.ones(9, dtype=torch.float32)
        self.nodes = {"n0": object(), "n1": object(), "n2": object()}
        self.optimized_spatial = True
        self.optimized_temporal = True


def test_apply_edge_intervention_clones_graph_and_forces_spatial_edge():
    graph = FakeGraph()
    edge = EdgeIntervention(edge_type="spatial", src=1, dst=2, value=0)

    intervened = apply_edge_intervention(graph, edge, logit_magnitude=13.0)

    assert intervened is not graph
    assert graph.spatial_logits[1, 2].item() == 0.0
    assert graph.spatial_mask[1, 2].item() == 1.0
    assert intervened.spatial_logits[1, 2].item() == -13.0
    assert intervened.spatial_mask[1, 2].item() == 0.0
    assert intervened.temporal_logits[1, 2].item() == 0.0
    assert intervened.optimized_spatial is False
    assert intervened.optimized_temporal is False


def test_apply_edge_intervention_supports_agentprune_flat_masks():
    graph = FakeFlatGraph()
    edge = EdgeIntervention(edge_type="spatial", src=1, dst=2, value=0)

    intervened = apply_edge_intervention(graph, edge, logit_magnitude=7.0)

    flat_index = 1 * 3 + 2
    assert graph.spatial_logits[flat_index].item() == 0.0
    assert graph.spatial_masks[flat_index].item() == 1.0
    assert intervened.spatial_logits[flat_index].item() == -7.0
    assert intervened.spatial_masks[flat_index].item() == 0.0


def test_counterfactual_delta_rewards_accuracy_gain_and_penalizes_cost_increase():
    edge = EdgeIntervention(edge_type="temporal", src=0, dst=1, value=1)
    off = RolloutOutcome(answer="B", correct=False, loss=1.0, cost=10.0, raw={})
    on = RolloutOutcome(answer="A", correct=True, loss=0.0, cost=30.0, raw={})

    result = counterfactual_delta(edge=edge, off=off, on=on, cost_lambda=0.01)

    assert math.isclose(result.effect, 0.8)
    assert math.isclose(result.loss_delta, 1.0)
    assert math.isclose(result.cost_delta, 20.0)
    assert result.helped is True


def test_paired_counterfactual_rollout_runs_off_and_on_interventions():
    graph = FakeAsyncGraph()
    edge = EdgeIntervention(edge_type="spatial", src=0, dst=1, value=1)

    result = asyncio.run(
        paired_counterfactual_rollout(
            graph=graph,
            input_dict={"question": "synthetic"},
            target_answer="A",
            edge=edge,
            num_rounds=1,
            cost_lambda=0.01,
        )
    )

    assert result.off.answer == "B"
    assert result.on.answer == "A"
    assert math.isclose(result.effect, 0.8)
    assert graph.spatial_logits[0, 1].item() == 0.0


def test_paired_counterfactual_rollout_accepts_custom_answer_normalizer():
    class NumericGraph(FakeGraph):
        async def arun(self, input_dict, num_rounds, **kwargs):
            enabled = self.temporal_logits[0, 2].item() > 0
            text = "The answer is 42" if enabled else "The answer is 41"
            return text, []

    def last_number(value):
        return str(value).split()[-1]

    edge = EdgeIntervention(edge_type="temporal", src=0, dst=2, value=1)
    result = asyncio.run(
        paired_counterfactual_rollout(
            graph=NumericGraph(),
            input_dict={"task": "synthetic"},
            target_answer="42",
            edge=edge,
            num_rounds=2,
            answer_normalizer=last_number,
        )
    )

    assert result.off.answer == "41"
    assert result.on.answer == "42"
    assert result.effect == 1.0


def test_paired_counterfactual_rollout_forwards_arun_kwargs():
    class KwargGraph(FakeGraph):
        async def arun(self, input_dict, num_rounds, **kwargs):
            enabled = self.spatial_logits[0, 1].item() > 0
            aggregate_mode = kwargs.get("aggregate_mode")
            answer = "A" if enabled and aggregate_mode == "last connected" else "B"
            return {"choice": answer}, []

    edge = EdgeIntervention(edge_type="spatial", src=0, dst=1, value=1)
    result = asyncio.run(
        paired_counterfactual_rollout(
            graph=KwargGraph(),
            input_dict={"task": "synthetic"},
            target_answer="A",
            edge=edge,
            num_rounds=1,
            arun_kwargs={"aggregate_mode": "last connected"},
        )
    )

    assert result.off.answer == "B"
    assert result.on.answer == "A"
    assert result.effect == 1.0


def test_summarize_effects_reports_top_edges_and_means():
    effects = [
        counterfactual_delta(
            edge=EdgeIntervention("spatial", 0, 1, 1),
            off=RolloutOutcome("B", False, 1.0, 10.0, {}),
            on=RolloutOutcome("A", True, 0.0, 20.0, {}),
            cost_lambda=0.01,
        ),
        counterfactual_delta(
            edge=EdgeIntervention("spatial", 1, 2, 1),
            off=RolloutOutcome("A", True, 0.0, 10.0, {}),
            on=RolloutOutcome("B", False, 1.0, 10.0, {}),
            cost_lambda=0.01,
        ),
    ]

    summary = summarize_effects(effects)

    assert summary["num_effects"] == 2
    assert math.isclose(summary["mean_effect"], -0.05)
    assert summary["num_helpful"] == 1
    assert summary["top_helpful"][0]["edge"] == "spatial:0->1"
    assert summary["top_harmful"][0]["edge"] == "spatial:1->2"
    assert summary["by_edge"][0]["edge"] == "spatial:0->1"
    assert math.isclose(summary["by_edge"][0]["mean_effect"], 0.9)
