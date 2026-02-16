import torch
from typing import Any, Iterator, List, Optional, Dict, Tuple
import pandas as pd
import numpy as np
import time
import asyncio
from typing import List
import copy

from AgentPrune.graph.graph import Graph
from experiments.accuracy import Accuracy
from AgentPrune.utils.globals import Cost, PromptTokens, CompletionTokens

import itertools, heapq

async def rollout_one_task(
    graph,
    dataset,
    task_idx: int,
    num_rounds: int = 1,
    *,
    return_abcd_probs: bool = True,
    sem: Optional[asyncio.Semaphore] = None,
) -> Dict[str, Any]:
    """
    One rollout on a specified task_idx.

    Returns a dict containing:
      - idx: int
      - choice: str (e.g., "A"/"B"/"C"/"D" or postprocessed)
      - correct: int (0/1)
      - choice_prob: float (P(model chose this choice))  [if available]
      - p_correct: float (P(correct choice))             [if available]
      - log_prob: torch.Tensor scalar WITH grad (for REINFORCE)
    """

    n_data = len(dataset)
    if task_idx < 0 or task_idx >= n_data:
        raise IndexError(f"task_idx out of range: {task_idx} (dataset size={n_data}).")

    record = dataset[int(task_idx)]
    input_dict = dataset.record_to_input(record)
    correct_answer = dataset.record_to_target_answer(record)
    assert isinstance(correct_answer, str), f"correct_answer must be str, got {type(correct_answer)}"

    # Deepcopy graph but keep the SAME parameter tensors so gradients flow back
    realized_graph = copy.deepcopy(graph)
    realized_graph.spatial_logits = graph.spatial_logits
    realized_graph.temporal_logits = graph.temporal_logits

    # Run LLM
    async def _call():
        # We assume arun can optionally return choice probs in its first output when return_choice_logprob=True.
        # And always returns (raw_answer_or_dict, log_prob_tensor).
        return await realized_graph.arun(
            input_dict,
            num_rounds,
            return_choice_logprob=return_abcd_probs,
        )

    if sem is None:
        raw_out, log_prob = await _call()
    else:
        async with sem:
            raw_out, log_prob = await _call()

    # Normalize raw_out to a single object
    raw0 = raw_out[0] if isinstance(raw_out, list) and len(raw_out) > 0 else raw_out

    # Parse model choice string
    if isinstance(raw0, dict):
        pred_str = raw0.get("choice") or raw0.get("text") or ""
        abcd_probs = raw0.get("abcd_probs") or {}
    else:
        pred_str = raw0
        abcd_probs = {}

    choice = dataset.postprocess_answer(pred_str)

    # correctness 0/1
    correct = 1 if choice == dataset.postprocess_answer(correct_answer) else 0

    # probabilities (floats, for logging/selection)
    choice_prob = float(abcd_probs.get(choice, 0.0)) if isinstance(abcd_probs, dict) else 0.0
    p_correct = float(abcd_probs.get(correct_answer, 0.0)) if isinstance(abcd_probs, dict) else 0.0

    # log_prob must be torch scalar tensor with grad for REINFORCE
    if not torch.is_tensor(log_prob):
        raise TypeError(f"log_prob must be a torch.Tensor, got {type(log_prob)}")
    if log_prob.dim() != 0:
        log_prob = log_prob.squeeze()

    return {
        "idx": int(task_idx),
        "choice": choice,
        "correct": int(correct),
        "choice_prob": float(choice_prob),
        "p_correct": float(p_correct),
        "log_prob": log_prob,  # tensor with grad
        "correct_answer": correct_answer,
        "raw_out": raw0,        # optional: for debugging
    }


def reinforce_one_step_from_rollouts(
    graph,
    rollouts: List[Dict[str, Any]],
    *,
    use_baseline: bool = False,
    detach_baseline: bool = True,
) -> Dict[str, Any]:
    """
    One-step REINFORCE gradient estimate (NO optimizer.step, NO loop).

    rollouts: list of dicts from rollout_one_task, each has:
      - log_prob: torch scalar tensor with grad
      - correct: 0/1 reward

    Returns:
      {
        "loss": float,
        "baseline": float,
        "grad_spatial": Tensor (same shape as graph.spatial_logits),
        "grad_temporal": Tensor (same shape as graph.temporal_logits),
        "grad_flat": Tensor (1D)
      }
    """

    if len(rollouts) == 0:
        raise ValueError("rollouts is empty.")

    # Ensure grads are clean
    if graph.spatial_logits.grad is not None:
        graph.spatial_logits.grad.zero_()
    if graph.temporal_logits.grad is not None:
        graph.temporal_logits.grad.zero_()

    # Rewards
    rewards = torch.tensor([r["correct"] for r in rollouts], dtype=torch.float32)

    # Baseline (optional)
    if use_baseline:
        b = rewards.mean()
        baseline_val = float(b.item())
        if detach_baseline:
            adv = rewards - b.detach()
        else:
            adv = rewards - b
    else:
        baseline_val = 0.0
        adv = rewards

    # Log probs (must be tensors with grad)
    log_probs = []
    for r in rollouts:
        lp = r["log_prob"]
        if not torch.is_tensor(lp):
            raise TypeError("Each rollout['log_prob'] must be a torch.Tensor.")
        if lp.dim() != 0:
            lp = lp.squeeze()
        log_probs.append(lp)
    log_probs = torch.stack(log_probs)  # (B,)

    # REINFORCE loss: minimize -E[adv * log_prob]
    loss = -(adv * log_probs).mean()

    # Backprop to fill .grad on graph logits
    loss.backward()

    print("spatial grad is None?", graph.spatial_logits.grad is None)
    print("temporal grad is None?", graph.temporal_logits.grad is None)
    print("loss requires_grad?", loss.requires_grad)
    print("log_probs requires_grad?", all(r["log_prob"].requires_grad for r in rollouts))

    grad_spatial = graph.spatial_logits.grad.detach().clone()
    #grad_temporal = graph.temporal_logits.grad.detach().clone()
    grad_temporal = (
                        graph.temporal_logits.grad.detach().clone()
                        if graph.temporal_logits.grad is not None
                        else torch.zeros_like(graph.temporal_logits)
                    )


    grad_flat = torch.cat([grad_spatial.reshape(-1), grad_temporal.reshape(-1)], dim=0)

    return {
        "loss": float(loss.item()),
        "baseline": baseline_val,
        "grad_spatial": grad_spatial,
        "grad_temporal": grad_temporal,
        "grad_flat": grad_flat,
    }


def _grad_of_single_rollout(graph, r: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    """
    Single-sample REINFORCE gradient:
      loss_i = - reward_i * log_prob_i
    Return detached grads (spatial/temporal/flat).
    """
    params = [graph.spatial_logits, graph.temporal_logits]
    reward = float(r["correct"])  # 0/1
    log_prob = r["log_prob"]      # torch scalar with grad

    loss_i = -log_prob * reward

    # allow_unused=True: temporal may be unused when num_rounds=1
    g_spatial, g_temporal = torch.autograd.grad(
        loss_i,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )

    if g_spatial is None:
        g_spatial = torch.zeros_like(graph.spatial_logits)
    if g_temporal is None:
        g_temporal = torch.zeros_like(graph.temporal_logits)

    g_spatial = g_spatial.detach()
    g_temporal = g_temporal.detach()
    g_flat = torch.cat([g_spatial.reshape(-1), g_temporal.reshape(-1)], dim=0)

    return {"grad_spatial": g_spatial, "grad_temporal": g_temporal, "grad_flat": g_flat}


async def compute_combo_gradients_from_indices(
    graph,
    dataset,
    task_indices: List[int],
    *,
    num_rounds: int = 1,
    max_concurrency: int = 8,
    combo_size: int = 10,
    combo_mode: str = "all",   # "all" => all C(n,k) subsets; "chunk" => consecutive chunks of size k
    max_combos: Optional[int] = None,  # if combo_mode="all", optionally cap the number of combos
    seed: int = 0,
    return_rollouts: bool = False,      # 新增：是否返回 rollouts（默认 True，符合你现在的需求）
) -> Dict[str, Any]:
    """
    1) rollout all tasks in task_indices once
    2) compute per-task reinforce gradients once
    3) aggregate per-task grads to each combo


    Return: list of dicts:
      {
        "combo": [task_idx...],
        "grad_flat": Tensor,
        "grad_spatial": Tensor,
        "grad_temporal": Tensor,
        "grad_norm": float,
        "avg_reward": float,
      }
    """

    if len(task_indices) == 0:
        raise ValueError("task_indices empty")
    if combo_size <= 0 or combo_size > len(task_indices):
        raise ValueError(f"combo_size must be in [1, {len(task_indices)}]")

    sem = asyncio.Semaphore(max_concurrency)

    # ---- 1) rollout once
    rollout_tasks = [
        asyncio.create_task(
            rollout_one_task(
                graph=graph,
                dataset=dataset,
                task_idx=int(idx),
                num_rounds=num_rounds,
                return_abcd_probs=True,
                sem=sem,
            )
        )
        for idx in task_indices
    ]
    rollouts = await asyncio.gather(*rollout_tasks)

    # ---- 2) per-task gradients
    per_task = []
    for r in rollouts:
        g = _grad_of_single_rollout(graph, r)
        per_task.append({
            "idx": r["idx"],
            "correct": r["correct"],
            "choice": r["choice"],
            "choice_prob": r["choice_prob"],
            "p_correct": r["p_correct"],
            **g,
        })

    # helper: map idx -> grad tensors
    idx2g = {d["idx"]: d for d in per_task}

    # ---- 3) build combos & aggregate
    results = []

    if combo_mode == "chunk":
        for i in range(0, len(task_indices), combo_size):
            combo = tuple(task_indices[i:i+combo_size])
            if len(combo) != combo_size:
                continue

            gs = torch.stack([idx2g[j]["grad_spatial"] for j in combo], dim=0).mean(dim=0)
            gt = torch.stack([idx2g[j]["grad_temporal"] for j in combo], dim=0).mean(dim=0)
            gf = torch.cat([gs.reshape(-1), gt.reshape(-1)], dim=0)

            avg_reward = float(sum(idx2g[j]["correct"] for j in combo) / len(combo))
            results.append({
                "combo": list(map(int, combo)),
                "grad_spatial": gs,
                "grad_temporal": gt,
                "grad_flat": gf,
                "grad_norm": float(gf.norm().item()),
                "avg_reward": avg_reward,
            })

    elif combo_mode == "all":
        all_iter = itertools.combinations(task_indices, combo_size)

        # 如果要 cap，就取前 max_combos 个；否则全遍历
        n = 0
        for combo in all_iter:
            gs = torch.stack([idx2g[j]["grad_spatial"] for j in combo], dim=0).mean(dim=0)
            gt = torch.stack([idx2g[j]["grad_temporal"] for j in combo], dim=0).mean(dim=0)
            gf = torch.cat([gs.reshape(-1), gt.reshape(-1)], dim=0)

            avg_reward = float(sum(idx2g[j]["correct"] for j in combo) / len(combo))
            results.append({
                "combo": list(map(int, combo)),
                "grad_spatial": gs,
                "grad_temporal": gt,
                "grad_flat": gf,
                "grad_norm": float(gf.norm().item()),
                "avg_reward": avg_reward,
            })

            n += 1
            if max_combos is not None and n >= max_combos:
                break
    else:
        raise ValueError("combo_mode must be 'all' or 'chunk'")

    return {
        "rollouts": rollouts if return_rollouts else None,
        "per_task": per_task,
        "combos": results,
    }

def _grad_of_single_rollout_minimal(graph, r: Dict[str, Any]) -> Dict[str, Any]:
    """
    单样本 REINFORCE: loss = - log_prob * reward
    reward = r["correct"] (0/1)
    返回 detached 的 grad_spatial/grad_temporal/grad_flat，便于后面组合计算
    """
    reward = float(r["correct"])
    log_prob = r["log_prob"]  # torch scalar (requires grad)

    loss = -log_prob * reward

    params = [graph.spatial_logits, graph.temporal_logits]
    g_sp, g_tp = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=False)

    if g_sp is None:
        g_sp = torch.zeros_like(graph.spatial_logits)
    if g_tp is None:
        g_tp = torch.zeros_like(graph.temporal_logits)

    # 重要：detach，避免把计算图带出函数造成内存问题
    g_sp = g_sp.detach()
    g_tp = g_tp.detach()
    g_flat = torch.cat([g_sp.reshape(-1), g_tp.reshape(-1)], dim=0)

    return {"grad_spatial": g_sp, "grad_temporal": g_tp, "grad_flat": g_flat}


# ============================================================
# 模块1：返回所有任务rollout + 单任务梯度估计（不做组合）
# ============================================================
async def compute_task_rollouts_and_grads(
    graph,
    dataset,
    task_indices: List[int],
    *,
    num_rounds: int = 1,
    max_concurrency: int = 8,
    return_rollouts: bool = True,
    return_raw_out: bool = False,
    grad_fn=_grad_of_single_rollout_minimal,  # 你可以替换成你已有的 _grad_of_single_rollout
) -> Dict[str, Any]:
    """
    返回:
      rollouts: List[dict]   # 每个任务一次 rollout 的原始返回
      per_task: List[dict]   # 每个任务的梯度（detached）+ 正确性等
      idx2grad_flat: dict[int, torch.Tensor]  # idx -> grad_flat (CPU/当前device)
    """
    if len(task_indices) == 0:
        raise ValueError("task_indices empty")

    sem = asyncio.Semaphore(max_concurrency)

    async def _run_one(idx: int):
        r = await rollout_one_task(
            graph=graph,
            dataset=dataset,
            task_idx=int(idx),
            num_rounds=num_rounds,
            return_abcd_probs=True,
            sem=sem,
        )
        if not return_raw_out:
            r = dict(r)
            r.pop("raw_out", None)
        return r

    rollouts = await asyncio.gather(*[asyncio.create_task(_run_one(idx)) for idx in task_indices])

    per_task = []
    idx2grad_flat = {}

    for r in rollouts:
        g = grad_fn(graph, r)  # 必须产出 grad_flat
        d = {
            "idx": int(r["idx"]),
            "correct": int(r["correct"]),
            "choice": r.get("choice", None),
            "choice_prob": float(r.get("choice_prob", 0.0)),
            "p_correct": float(r.get("p_correct", 0.0)),
            "grad_flat": g["grad_flat"],
        }
        # 如果你后续也想要单任务 grad_spatial/grad_temporal，可以加上
        if "grad_spatial" in g:
            d["grad_spatial"] = g["grad_spatial"]
        if "grad_temporal" in g:
            d["grad_temporal"] = g["grad_temporal"]

        per_task.append(d)
        idx2grad_flat[d["idx"]] = d["grad_flat"]

    return {
        "rollouts": rollouts if return_rollouts else None,
        "per_task": per_task,
        "idx2grad_flat": idx2grad_flat,
    }


# ============================================================
# 模块2：排列组合 + 维护 topK（按 L2 最大） + 最终按 sum 选 best
# ============================================================
def select_best_combo_streaming_topk(
    task_indices: List[int],
    idx2grad_flat: Dict[int, torch.Tensor],
    *,
    combo_size: int = 10,
    topk: int = 10,
) -> Dict[str, Any]:
    """
    遍历全部组合 C(n,k)，但只维护一个 topk 池（L2 最大）。
    遍历结束后：
      - 先把 topk 按 L2 降序排序
      - 再从 topk 里选 grad_sum 最大的 combo
    返回:
      {
        "topk_sorted": List[{combo, l2, gsum}],
        "best_combo": List[int],
        "best_l2": float,
        "best_gsum": float,
      }
    """
    n = len(task_indices)
    if combo_size <= 0 or combo_size > n:
        raise ValueError(f"combo_size must be in [1, {n}]")
    if topk <= 0:
        raise ValueError("topk must be positive")

    # 用最小堆维护 topk：堆元素 (l2, gsum, combo_tuple)
    heap: List[Tuple[float, float, Tuple[int, ...]]] = []

    # 组合梯度定义：combo 内 grad_flat 取均值
    # g_combo = mean(g_i)
    for combo in itertools.combinations(task_indices, combo_size):
        gs = torch.stack([idx2grad_flat[i] for i in combo], dim=0).mean(dim=0)
        l2 = float(gs.norm(p=2).item())
        gsum = float(gs.sum().item())

        if len(heap) < topk:
            heapq.heappush(heap, (l2, gsum, combo))
        else:
            # 只按 l2 决定进不进 topk；如果要 tie-break，也可加逻辑
            if l2 > heap[0][0]:
                heapq.heapreplace(heap, (l2, gsum, combo))

    # 遍历结束：heap 是 topk 个（无序），先按 L2 降序排序
    topk_sorted = sorted(heap, key=lambda x: x[0], reverse=True)

    # 再从 topk_sorted 里选 gsum 最大的
    best = max(topk_sorted, key=lambda x: x[1])
    best_l2, best_gsum, best_combo = best[0], best[1], list(best[2])

    topk_infos = [{"combo": list(c), "l2": l2, "gsum": gsum} for (l2, gsum, c) in topk_sorted]

    return {
        "topk_sorted": topk_infos,
        "best_combo": best_combo,
        "best_l2": best_l2,
        "best_gsum": best_gsum,
    }