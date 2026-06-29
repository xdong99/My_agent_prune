import argparse
import asyncio
import copy
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn
from sklearn.cross_decomposition import PLSRegression
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C
from sklearn.gaussian_process.kernels import Matern, WhiteKernel
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from AgentPrune.graph.graph import Graph
from AgentPrune.utils.globals import CompletionTokens, PromptTokens
from dataset.gsm8k_dataset import gsm_data_process, gsm_get_predict


@dataclass
class Variant:
    name: str
    description: str
    spatial_masks: torch.Tensor
    temporal_masks: torch.Tensor
    spatial_logits: torch.Tensor
    temporal_logits: torch.Tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Small proof-of-concept: train an AL-pruned GSM8K graph, build nearby "
            "graph variants, generate diagnostic GSM8K-style tasks, and test "
            "whether generated tasks separate variants better than ordinary tasks."
        )
    )
    parser.add_argument("--train_json", default="dataset/gsm8k/train.jsonl")
    parser.add_argument("--test_json", default="dataset/gsm8k/test.jsonl")
    parser.add_argument("--output_dir", default="outputs/generated_task_poc")
    parser.add_argument("--llm_name", default="gpt-4.1-nano")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num_agents", type=int, default=3)
    parser.add_argument("--num_rounds", type=int, default=1)
    parser.add_argument(
        "--aggregate_mode",
        choices=["all connected", "last connected"],
        default="all connected",
    )
    parser.add_argument("--mode", default="FullConnected")
    parser.add_argument("--decision_method", default="FinalRefer")
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--train_iterations", type=int, default=4)
    parser.add_argument("--imp_per_iterations", type=int, default=2)
    parser.add_argument("--pruning_rate", type=float, default=0.25)
    parser.add_argument("--al_pool_size", type=int, default=24)
    parser.add_argument("--al_rep_budget", type=int, default=8)
    parser.add_argument("--al_init_size", type=int, default=3)
    parser.add_argument("--al_rounds", type=int, default=1)
    parser.add_argument("--al_batch_size", type=int, default=2)
    parser.add_argument("--al_final_topk", type=int, default=4)
    parser.add_argument("--eki_ensemble_size", type=int, default=3)
    parser.add_argument("--eki_prior_std", type=float, default=2.0)
    parser.add_argument("--eki_obs_noise", type=float, default=0.1)
    parser.add_argument("--eki_damping", type=float, default=0.7)
    parser.add_argument("--max_concurrency", type=int, default=6)
    parser.add_argument("--natural_eval_tasks", type=int, default=8)
    parser.add_argument("--natural_probe_tasks", type=int, default=8)
    parser.add_argument("--generated_candidates", type=int, default=12)
    parser.add_argument("--generated_keep", type=int, default=5)
    parser.add_argument("--cost_lambda", type=float, default=0.0005)
    parser.add_argument(
        "--variant_edge_budget",
        type=int,
        default=1,
        help="Number of pruned/kept spatial edges to restore/remove in nearby variants.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_tokens() -> tuple[float, float]:
    return float(PromptTokens.instance().value), float(CompletionTokens.instance().value)


def token_delta(before: tuple[float, float], after: tuple[float, float]) -> float:
    return float((after[0] - before[0]) + (after[1] - before[1]))


def load_gsm8k(path: str) -> list[dict[str, str]]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return gsm_data_process(rows)


def build_graph(args: argparse.Namespace) -> Graph:
    agent_names = ["MathSolver"] * args.num_agents
    kwargs = get_graph_kwargs(args.mode, len(agent_names))
    return Graph(
        domain="gsm8k",
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=args.decision_method,
        optimized_spatial=True,
        optimized_temporal=args.num_rounds > 1,
        **kwargs,
    )


def get_graph_kwargs(mode: str, n_agents: int) -> dict[str, Any]:
    if mode == "FullConnected":
        fixed_spatial_masks = [
            [1 if src != dst else 0 for dst in range(n_agents)]
            for src in range(n_agents)
        ]
        fixed_temporal_masks = [[1 for _ in range(n_agents)] for _ in range(n_agents)]
    elif mode == "Chain":
        fixed_spatial_masks = [
            [1 if dst == src + 1 else 0 for dst in range(n_agents)]
            for src in range(n_agents)
        ]
        fixed_temporal_masks = [[0 for _ in range(n_agents)] for _ in range(n_agents)]
    elif mode == "Star":
        fixed_spatial_masks = [
            [1 if src == 0 and dst != 0 else 0 for dst in range(n_agents)]
            for src in range(n_agents)
        ]
        fixed_temporal_masks = [[1 for _ in range(n_agents)] for _ in range(n_agents)]
    else:
        raise ValueError(f"unsupported mode for PoC script: {mode}")
    return {
        "initial_spatial_probability": 0.5,
        "fixed_spatial_masks": fixed_spatial_masks,
        "initial_temporal_probability": 0.5,
        "fixed_temporal_masks": fixed_temporal_masks,
        "node_kwargs": None,
    }


def vectorize_records(records: list[dict[str, Any]]) -> np.ndarray:
    texts = [str(record["task"]) for record in records]
    vectorizer = HashingVectorizer(
        n_features=256,
        alternate_sign=False,
        norm="l2",
        ngram_range=(1, 2),
    )
    return vectorizer.transform(texts).toarray().astype(np.float64)


def select_farthest_first(X: np.ndarray, k: int) -> list[int]:
    n = X.shape[0]
    k = min(k, n)
    if k <= 0:
        return []
    sim = X @ X.T
    dist = 1.0 - sim
    medoid = int(np.argmin(dist.mean(axis=1)))
    selected = [medoid]
    min_d = dist[medoid].copy()
    min_d[medoid] = -np.inf
    while len(selected) < k:
        nxt = int(np.argmax(min_d))
        selected.append(nxt)
        min_d = np.minimum(min_d, dist[nxt])
        for idx in selected:
            min_d[idx] = -np.inf
    return selected


def flatten_logits(spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
    return torch.cat([spatial.reshape(-1), temporal.reshape(-1)], dim=0)


def unflatten_logits(
    theta: torch.Tensor,
    spatial_shape: torch.Size,
    temporal_shape: torch.Size,
) -> tuple[torch.Tensor, torch.Tensor]:
    s_numel = int(torch.tensor(spatial_shape).prod().item())
    spatial = theta[:s_numel].reshape(spatial_shape)
    temporal = theta[s_numel:].reshape(temporal_shape)
    return spatial, temporal


def gaussian_kl_diag(
    mu0: torch.Tensor,
    var0: torch.Tensor,
    mu1: torch.Tensor,
    var1: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    var0 = var0.clamp_min(eps)
    var1 = var1.clamp_min(eps)
    d = mu0.numel()
    term_log = torch.sum(torch.log(var1) - torch.log(var0))
    term_tr = torch.sum(var0 / var1)
    term_quad = torch.sum((mu1 - mu0) ** 2 / var1)
    return 0.5 * (term_log + term_tr + term_quad - d)


async def eki_score_tasks(
    graph: Graph,
    records: list[dict[str, Any]],
    *,
    num_rounds: int,
    ensemble_size: int,
    prior_std: float,
    obs_noise: float,
    damping: float,
    max_concurrency: int,
    aggregate_mode: str,
    observation_mode: str = "binary_correct",
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if not records:
        return np.asarray([], dtype=np.float64), []

    sem = asyncio.Semaphore(max_concurrency)
    spatial0 = graph.spatial_logits.detach().clone().float()
    temporal0 = graph.temporal_logits.detach().clone().float()
    theta0 = flatten_logits(spatial0, temporal0)
    theta_init = theta0[None, :].repeat(ensemble_size, 1)
    theta_init = theta_init + prior_std * torch.randn(ensemble_size, theta0.numel())

    async def run_member(theta_e: torch.Tensor, record: dict[str, Any]) -> float:
        realized_graph = copy.deepcopy(graph)
        spatial_e, temporal_e = unflatten_logits(theta_e, spatial0.shape, temporal0.shape)
        realized_graph.spatial_logits = torch.nn.Parameter(
            spatial_e.detach().clone(), requires_grad=False
        )
        realized_graph.temporal_logits = torch.nn.Parameter(
            temporal_e.detach().clone(), requires_grad=False
        )
        realized_graph.optimized_spatial = True
        realized_graph.optimized_temporal = num_rounds > 1
        async with sem:
            final_answers, _log_probs = await realized_graph.arun(
                {"task": record["task"]},
                num_rounds,
                aggregate_mode=aggregate_mode,
            )
        answer_text = final_answers[0] if isinstance(final_answers, list) else final_answers
        return gsm_observation_value(str(answer_text), str(record["answer"]), mode=observation_mode)

    kl_scores: list[float] = []
    details: list[dict[str, Any]] = []
    for task_id, record in enumerate(records):
        theta = theta_init.detach().clone()
        jobs = [asyncio.create_task(run_member(theta[i], record)) for i in range(ensemble_size)]
        f_vals = torch.tensor(await asyncio.gather(*jobs), dtype=torch.float32)

        mu_old = theta.mean(dim=0)
        var_old = theta.var(dim=0, unbiased=True)
        pred_var = torch.var(f_vals, unbiased=True)

        with torch.no_grad():
            theta_mean = theta.mean(dim=0)
            f_mean = f_vals.mean()
            theta_a = theta - theta_mean[None, :]
            f_a = (f_vals - f_mean).reshape(ensemble_size, 1)
            c_theta_f = (theta_a.T @ f_a) / max(1, ensemble_size - 1)
            c_ff = (f_a.T @ f_a) / max(1, ensemble_size - 1)
            r = torch.tensor([[obs_noise**2]], dtype=torch.float32)
            k_gain = c_theta_f @ torch.linalg.inv(c_ff + r)
            eps = obs_noise * torch.randn(ensemble_size, 1)
            innovation = (torch.tensor(1.0) + eps) - f_vals.reshape(ensemble_size, 1)
            theta = theta + damping * (innovation @ k_gain.T)

        mu_new = theta.mean(dim=0)
        var_new = theta.var(dim=0, unbiased=True)
        kl = gaussian_kl_diag(mu_old, var_old, mu_new, var_new)
        score = float(kl.item())
        kl_scores.append(score)
        details.append(
            {
                "local_task_id": task_id,
                "kl": score,
                "pred_var": float(pred_var.item()),
                "ensemble_mean_correct": float(f_vals.mean().item()),
                "observation_mode": observation_mode,
                "ensemble_values": [float(x) for x in f_vals.tolist()],
            }
        )
    return np.asarray(kl_scores, dtype=np.float64), details


def gsm_observation_value(answer_text: str, target_answer: str, *, mode: str = "binary_correct") -> float:
    pred = gsm_get_predict(str(answer_text))
    if mode == "binary_correct":
        try:
            return float(float(pred) == float(target_answer))
        except Exception:
            return float(str(pred).strip() == str(target_answer).strip())
    if mode == "soft_numeric":
        try:
            pred_value = float(str(pred).replace(",", "").strip())
            target_value = float(str(target_answer).replace(",", "").strip())
        except Exception:
            return float(str(pred).strip() == str(target_answer).strip())
        scale = max(1.0, abs(target_value) * 0.1)
        return float(math.exp(-abs(pred_value - target_value) / scale))
    raise ValueError(f"unknown GSM observation mode: {mode}")


def fit_pls_gp(
    X_all: np.ndarray,
    y_obs: list[float],
    obs_pos: list[int],
    seed: int,
) -> tuple[np.ndarray, GaussianProcessRegressor, np.ndarray, np.ndarray]:
    obs = np.asarray(obs_pos, dtype=int)
    X_obs = X_all[obs]
    y = np.asarray(y_obs, dtype=np.float64)
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()
    X_obs_s = x_scaler.fit_transform(X_obs)
    y_t = np.arcsinh(y)
    y_s = y_scaler.fit_transform(y_t.reshape(-1, 1)).ravel()
    n_components = max(1, min(3, X_obs_s.shape[0], X_obs_s.shape[1]))
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_obs_s, y_s)
    Z_all = pls.transform(x_scaler.transform(X_all))
    z_scaler = StandardScaler()
    Z_all_s = z_scaler.fit_transform(Z_all)
    d = Z_all_s.shape[1]
    kernel = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(d),
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5,
    ) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e-1))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-4,
        normalize_y=True,
        n_restarts_optimizer=1,
        random_state=seed,
    )
    gp.fit(Z_all_s[obs], y_s)
    mu, std = gp.predict(Z_all_s, return_std=True)
    return Z_all_s, gp, mu, std


async def select_al_training_subset(
    graph: Graph,
    train_set: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    pool_size = min(args.al_pool_size, len(train_set))
    pool_positions = rng.choice(len(train_set), size=pool_size, replace=False)
    pool_records = [train_set[int(pos)] for pos in pool_positions]
    X_pool = vectorize_records(pool_records)
    rep_positions = np.asarray(
        select_farthest_first(X_pool, min(args.al_rep_budget, pool_size)), dtype=int
    )
    representative_records = [pool_records[int(pos)] for pos in rep_positions]
    X_all = X_pool[rep_positions]

    init_size = min(args.al_init_size, len(representative_records))
    obs_pos = rng.choice(len(representative_records), size=init_size, replace=False).tolist()
    init_records = [representative_records[int(pos)] for pos in obs_pos]
    y0, init_details = await eki_score_tasks(
        graph,
        init_records,
        num_rounds=args.num_rounds,
        ensemble_size=args.eki_ensemble_size,
        prior_std=args.eki_prior_std,
        obs_noise=args.eki_obs_noise,
        damping=args.eki_damping,
        max_concurrency=args.max_concurrency,
        aggregate_mode=args.aggregate_mode,
    )
    y_obs = [float(v) for v in y0.tolist()]
    observed = np.zeros(len(representative_records), dtype=bool)
    observed[np.asarray(obs_pos, dtype=int)] = True
    round_logs = []

    for round_id in range(args.al_rounds):
        candidates = np.where(~observed)[0]
        if len(candidates) == 0:
            break
        Z_all, gp, mu, std = fit_pls_gp(X_all, y_obs, obs_pos, seed=args.seed + round_id)
        samples = gp.sample_y(
            Z_all,
            n_samples=1,
            random_state=args.seed + 100 + round_id,
        ).ravel()
        order = candidates[np.argsort(samples[candidates])[::-1]]
        picked = order[: min(args.al_batch_size, len(order))].tolist()
        picked_records = [representative_records[int(pos)] for pos in picked]
        y_new, details = await eki_score_tasks(
            graph,
            picked_records,
            num_rounds=args.num_rounds,
            ensemble_size=args.eki_ensemble_size,
            prior_std=args.eki_prior_std,
            obs_noise=args.eki_obs_noise,
            damping=args.eki_damping,
            max_concurrency=args.max_concurrency,
            aggregate_mode=args.aggregate_mode,
        )
        for pos, score in zip(picked, y_new.tolist()):
            obs_pos.append(int(pos))
            y_obs.append(float(score))
            observed[int(pos)] = True
        round_logs.append(
            {
                "round": round_id,
                "picked_rep_positions": picked,
                "picked_train_positions": [
                    int(pool_positions[int(rep_positions[int(pos)])]) for pos in picked
                ],
                "picked_scores": [float(v) for v in y_new.tolist()],
                "details": details,
            }
        )

    Z_all, gp, mu, std = fit_pls_gp(X_all, y_obs, obs_pos, seed=args.seed + 999)
    final_topk = min(args.al_final_topk, len(representative_records))
    final_pos = np.argsort(mu)[::-1][:final_topk].tolist()
    selected_records = [representative_records[int(pos)] for pos in final_pos]
    return {
        "embedding_backend": "sklearn.HashingVectorizer(256)+PLS+GP",
        "pool_train_positions": [int(x) for x in pool_positions.tolist()],
        "representative_pool_positions": [int(x) for x in rep_positions.tolist()],
        "representative_train_positions": [
            int(pool_positions[int(pos)]) for pos in rep_positions.tolist()
        ],
        "observed_rep_positions": [int(x) for x in obs_pos],
        "observed_scores": [float(x) for x in y_obs],
        "init_details": init_details,
        "round_logs": round_logs,
        "final_rep_positions": [int(x) for x in final_pos],
        "final_train_positions": [
            int(pool_positions[int(rep_positions[int(pos)])]) for pos in final_pos
        ],
        "final_mu": [float(mu[int(pos)]) for pos in final_pos],
        "selected_records": selected_records,
    }


async def train_graph(
    graph: Graph,
    records: list[dict[str, Any]],
    args: argparse.Namespace,
) -> dict[str, Any]:
    optimizer = torch.optim.Adam([graph.spatial_logits, graph.temporal_logits], lr=args.lr)
    rows = []
    for i_iter in range(args.train_iterations):
        batch = [records[(i_iter * args.batch_size + j) % len(records)] for j in range(args.batch_size)]
        jobs = []
        answers = []
        for record in batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits
            jobs.append(
                asyncio.create_task(
                    realized_graph.arun(
                        {"task": record["task"]},
                        args.num_rounds,
                        aggregate_mode=args.aggregate_mode,
                    )
                )
            )
            answers.append(str(record["answer"]).strip())
        raw_results = await asyncio.gather(*jobs)
        loss_list = []
        solved = 0
        for record, target, result in zip(batch, answers, raw_results):
            final_answers, log_prob = result
            answer_text = final_answers[0] if isinstance(final_answers, list) else final_answers
            pred = gsm_get_predict(str(answer_text))
            try:
                correct = float(pred) == float(target)
            except Exception:
                correct = str(pred).strip() == target
            utility = float(correct)
            solved += int(correct)
            loss_list.append(-log_prob * utility)
            rows.append(
                {
                    "iteration": i_iter,
                    "task": record["task"],
                    "target": target,
                    "prediction": pred,
                    "correct": bool(correct),
                    "utility": utility,
                }
            )
        loss = torch.mean(torch.stack(loss_list))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (i_iter + 1) % args.imp_per_iterations == 0:
            graph.update_masks(args.pruning_rate)
        print(
            f"[train] iter={i_iter} acc={solved / len(batch):.3f} "
            f"spatial_edges={int((graph.spatial_masks > 0).sum().item())}"
        )
    return {
        "num_rows": len(rows),
        "accuracy": sum(int(row["correct"]) for row in rows) / max(1, len(rows)),
        "rows": rows,
    }


def make_variants(graph: Graph, args: argparse.Namespace) -> list[Variant]:
    spatial_masks = graph.spatial_masks.detach().clone().float()
    temporal_masks = graph.temporal_masks.detach().clone().float()
    spatial_logits = graph.spatial_logits.detach().clone().float()
    temporal_logits = graph.temporal_logits.detach().clone().float()

    n = args.num_agents
    valid_spatial = [idx for idx in range(n * n) if idx // n != idx % n]
    kept = [idx for idx in valid_spatial if spatial_masks[idx].item() > 0.5]
    pruned = [idx for idx in valid_spatial if spatial_masks[idx].item() <= 0.5]
    kept_low = sorted(kept, key=lambda idx: float(spatial_logits[idx].item()))
    pruned_high = sorted(pruned, key=lambda idx: float(spatial_logits[idx].item()), reverse=True)

    variants = [
        Variant(
            name="al_pruned",
            description="Original AL-trained graph after logit-based pruning.",
            spatial_masks=spatial_masks,
            temporal_masks=temporal_masks,
            spatial_logits=spatial_logits,
            temporal_logits=temporal_logits,
        )
    ]

    if kept_low:
        mask = spatial_masks.clone()
        for idx in kept_low[: args.variant_edge_budget]:
            mask[idx] = 0.0
        variants.append(
            Variant(
                name="one_edge_removed",
                description=f"AL graph with low-logit kept edge(s) removed: {kept_low[:args.variant_edge_budget]}.",
                spatial_masks=mask,
                temporal_masks=temporal_masks,
                spatial_logits=spatial_logits,
                temporal_logits=temporal_logits,
            )
        )

    if pruned_high:
        mask = spatial_masks.clone()
        for idx in pruned_high[: args.variant_edge_budget]:
            mask[idx] = 1.0
        variants.append(
            Variant(
                name="one_edge_restored",
                description=f"AL graph with high-logit pruned edge(s) restored: {pruned_high[:args.variant_edge_budget]}.",
                spatial_masks=mask,
                temporal_masks=temporal_masks,
                spatial_logits=spatial_logits,
                temporal_logits=temporal_logits,
            )
        )

    if pruned_high:
        mask = spatial_masks.clone()
        restore_n = min(max(2, args.variant_edge_budget + 1), len(pruned_high))
        for idx in pruned_high[:restore_n]:
            mask[idx] = 1.0
        variants.append(
            Variant(
                name="slightly_denser",
                description=f"AL graph with top {restore_n} pruned spatial edge(s) restored.",
                spatial_masks=mask,
                temporal_masks=temporal_masks,
                spatial_logits=spatial_logits,
                temporal_logits=temporal_logits,
            )
        )

    return variants


def instantiate_variant_graph(base_graph: Graph, variant: Variant) -> Graph:
    graph = copy.deepcopy(base_graph)
    graph.spatial_masks = torch.nn.Parameter(variant.spatial_masks.clone(), requires_grad=False)
    graph.temporal_masks = torch.nn.Parameter(variant.temporal_masks.clone(), requires_grad=False)
    graph.spatial_logits = torch.nn.Parameter(variant.spatial_logits.clone(), requires_grad=False)
    graph.temporal_logits = torch.nn.Parameter(variant.temporal_logits.clone(), requires_grad=False)
    graph.optimized_spatial = False
    graph.optimized_temporal = False
    return graph


async def evaluate_variants(
    base_graph: Graph,
    variants: list[Variant],
    records: list[dict[str, Any]],
    *,
    num_rounds: int,
    max_concurrency: int,
    aggregate_mode: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del max_concurrency
    rows = []

    async def run_one(variant: Variant, record_id: int, record: dict[str, Any]) -> None:
        graph = instantiate_variant_graph(base_graph, variant)
        before = read_tokens()
        final_answers, _log_probs = await graph.arun(
            {"task": record["task"]},
            num_rounds,
            aggregate_mode=aggregate_mode,
        )
        after = read_tokens()
        answer_text = final_answers[0] if isinstance(final_answers, list) else final_answers
        pred = gsm_get_predict(str(answer_text))
        target = str(record["answer"]).strip()
        try:
            correct = float(pred) == float(target)
        except Exception:
            correct = str(pred).strip() == target
        rows.append(
            {
                "variant": variant.name,
                "record_id": record_id,
                "family": record.get("family", "natural"),
                "task": record["task"],
                "target": target,
                "prediction": pred,
                "correct": bool(correct),
                "tokens": token_delta(before, after),
                "num_spatial_edges": int((variant.spatial_masks > 0).sum().item()),
                "description": variant.description,
            }
        )

    for record_id, record in enumerate(records):
        for variant in variants:
            await run_one(variant, record_id, record)
    df = pd.DataFrame(rows).sort_values(["record_id", "variant"]).reset_index(drop=True)
    summary = (
        df.groupby("variant")
        .agg(
            accuracy=("correct", "mean"),
            mean_tokens=("tokens", "mean"),
            num_tasks=("record_id", "nunique"),
            num_spatial_edges=("num_spatial_edges", "first"),
        )
        .reset_index()
    )
    return df, summary


def generate_diagnostic_candidates(n: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    candidates = []
    families = [
        make_percent_base_task,
        make_average_added_score_task,
        make_setup_capacity_task,
        make_reverse_fraction_task,
        make_defective_rate_task,
        make_overlap_count_task,
        make_two_step_fraction_inverse_task,
        make_coupon_tax_order_task,
        make_lcm_meeting_task,
        make_growth_difference_task,
    ]
    while len(candidates) < n:
        family_fn = families[len(candidates) % len(families)]
        task = family_fn(rng, len(candidates))
        candidates.append(task)
    return candidates


def make_percent_base_task(rng: random.Random, idx: int) -> dict[str, Any]:
    purchase = rng.choice([120, 160, 200, 240, 300])
    pct = rng.choice([50, 75, 100, 125, 150])
    repair = rng.choice([20, 30, 40, 50, 60])
    market_value = purchase + purchase * pct // 100
    answer = market_value - purchase - repair
    if answer <= 0:
        answer = abs(answer) + 10
        repair = purchase * pct // 100 - answer
    task = (
        f"A store bought a used machine for ${purchase} and spent ${repair} repairing it. "
        f"After the repairs, the machine's market value was {pct}% higher than the original "
        f"purchase price, not {pct}% higher than the total money spent. The store sold it for "
        f"that market value. How many dollars of profit did the store make?"
    )
    return {
        "task": task,
        "answer": str(int(answer)),
        "family": "percent_base_trap",
        "generator_id": idx,
    }


def make_average_added_score_task(rng: random.Random, idx: int) -> dict[str, Any]:
    n = rng.choice([4, 5, 6, 8])
    old_avg = rng.choice([62, 65, 70, 75, 80])
    new_avg = old_avg + rng.choice([3, 4, 5, 6])
    answer = (n + 1) * new_avg - n * old_avg
    task = (
        f"The average score of {n} quizzes is {old_avg}. After one more quiz is added, "
        f"the average becomes {new_avg}. What was the score on the added quiz?"
    )
    return {
        "task": task,
        "answer": str(int(answer)),
        "family": "average_inverse",
        "generator_id": idx,
    }


def make_setup_capacity_task(rng: random.Random, idx: int) -> dict[str, Any]:
    total = rng.choice([180, 210, 240, 270, 300])
    rate = rng.choice([3, 4, 5])
    day_limit = rng.choice([70, 80, 90])
    setup = rng.choice([10, 15, 20])
    capacity = (day_limit - setup) * rate
    answer = math.ceil(total / capacity)
    task = (
        f"A team must inspect {total} boxes. On each workday they spend {setup} minutes setting up, "
        f"then inspect boxes at {rate} boxes per minute. They can spend at most {day_limit} minutes "
        f"per day including setup. What is the minimum number of workdays needed?"
    )
    return {
        "task": task,
        "answer": str(int(answer)),
        "family": "capacity_ceiling",
        "generator_id": idx,
    }


def make_reverse_fraction_task(rng: random.Random, idx: int) -> dict[str, Any]:
    denom = rng.choice([3, 4, 5])
    kept_num = denom - 1
    added = rng.choice([6, 8, 10, 12])
    final = rng.choice([30, 36, 42, 48])
    base_after_spend = final - added
    original = base_after_spend * denom // kept_num
    if base_after_spend * denom % kept_num != 0:
        final = kept_num * rng.choice([8, 10, 12]) + added
        base_after_spend = final - added
        original = base_after_spend * denom // kept_num
    task = (
        f"Jia had some stickers. She gave away 1/{denom} of them, then bought {added} more. "
        f"She then had {final} stickers. How many stickers did Jia have at the start?"
    )
    return {
        "task": task,
        "answer": str(int(original)),
        "family": "reverse_fraction",
        "generator_id": idx,
    }


def make_defective_rate_task(rng: random.Random, idx: int) -> dict[str, Any]:
    a_rate = rng.choice([12, 15, 18])
    a_hours = rng.choice([3, 4, 5])
    b_rate = rng.choice([10, 14, 16])
    b_hours = rng.choice([2, 3, 4])
    defective_pct = rng.choice([10, 20, 25])
    total = a_rate * a_hours + b_rate * b_hours
    good = total * (100 - defective_pct) // 100
    task = (
        f"Machine A makes {a_rate} parts per hour for {a_hours} hours. Machine B makes "
        f"{b_rate} parts per hour for {b_hours} hours. After production, {defective_pct}% "
        f"of all parts are discarded as defective. How many good parts remain?"
    )
    return {
        "task": task,
        "answer": str(int(good)),
        "family": "defective_rate",
        "generator_id": idx,
    }


def make_overlap_count_task(rng: random.Random, idx: int) -> dict[str, Any]:
    both = rng.choice([6, 8, 10, 12])
    soccer_only = rng.choice([11, 13, 15, 17])
    chess_only = rng.choice([9, 14, 16, 18])
    neither = rng.choice([3, 5, 7])
    soccer = soccer_only + both
    chess = chess_only + both
    total = soccer + chess - both + neither
    task = (
        f"In a club, {soccer} students play soccer, {chess} students play chess, "
        f"and {both} students do both. There are also {neither} students who do neither activity. "
        f"How many students are in the club altogether?"
    )
    return {
        "task": task,
        "answer": str(int(total)),
        "family": "overlap_inclusion_exclusion",
        "generator_id": idx,
    }


def make_two_step_fraction_inverse_task(rng: random.Random, idx: int) -> dict[str, Any]:
    denom1 = rng.choice([3, 4, 5])
    denom2 = rng.choice([3, 4])
    original = rng.choice([120, 180, 240, 300])
    after_first = original * (denom1 - 1) // denom1
    if original * (denom1 - 1) % denom1 != 0:
        original = denom1 * rng.choice([40, 50, 60])
        after_first = original * (denom1 - 1) // denom1
    after_second = after_first * (denom2 - 1) // denom2
    if after_first * (denom2 - 1) % denom2 != 0:
        after_first = denom2 * rng.choice([30, 40, 50])
        original = after_first * denom1 // (denom1 - 1)
        after_second = after_first * (denom2 - 1) // denom2
    task = (
        f"After a shipment arrived, a store sold 1/{denom1} of its notebooks in the morning. "
        f"In the afternoon it sold 1/{denom2} of the notebooks that were left after the morning sale. "
        f"At closing time, {after_second} notebooks remained. How many notebooks were in the shipment?"
    )
    return {
        "task": task,
        "answer": str(int(original)),
        "family": "two_step_fraction_inverse",
        "generator_id": idx,
    }


def make_coupon_tax_order_task(rng: random.Random, idx: int) -> dict[str, Any]:
    price = rng.choice([160, 200, 240, 320])
    coupon_pct = rng.choice([10, 20, 25])
    tax_pct = rng.choice([5, 10, 20])
    gift_card = rng.choice([10, 20, 30])
    discounted = price * (100 - coupon_pct) // 100
    taxed = discounted * (100 + tax_pct) // 100
    answer = taxed - gift_card
    task = (
        f"A jacket costs ${price}. A {coupon_pct}% coupon is applied first. Then a {tax_pct}% sales tax "
        f"is charged on the discounted price. Finally, Mina uses a ${gift_card} gift card. "
        f"How many dollars does Mina still have to pay?"
    )
    return {
        "task": task,
        "answer": str(int(answer)),
        "family": "coupon_tax_order",
        "generator_id": idx,
    }


def make_lcm_meeting_task(rng: random.Random, idx: int) -> dict[str, Any]:
    a = rng.choice([4, 6, 8])
    b = rng.choice([6, 9, 10])
    horizon = rng.choice([48, 60, 72, 90])
    lcm = abs(a * b) // math.gcd(a, b)
    answer = horizon // lcm + 1
    task = (
        f"Two reminder lights flash together at noon today. One flashes every {a} days and the other "
        f"flashes every {b} days. Counting today, how many times will they flash together during the "
        f"next {horizon} days?"
    )
    return {
        "task": task,
        "answer": str(int(answer)),
        "family": "lcm_inclusive_count",
        "generator_id": idx,
    }


def make_growth_difference_task(rng: random.Random, idx: int) -> dict[str, Any]:
    a0 = rng.choice([80, 100, 120])
    b0 = rng.choice([60, 90, 110])
    a_pct = rng.choice([25, 50, 75])
    b_pct = rng.choice([20, 40, 60])
    a_final = a0 + a0 * a_pct // 100
    b_final = b0 + b0 * b_pct // 100
    answer = abs(a_final - b_final)
    task = (
        f"Store A started the month with {a0} orders and increased its orders by {a_pct}%. "
        f"Store B started with {b0} orders and increased its orders by {b_pct}%. "
        f"At the end of the month, how many more orders did the busier store have than the other store?"
    )
    return {
        "task": task,
        "answer": str(int(answer)),
        "family": "growth_difference",
        "generator_id": idx,
    }


def select_disagreement_tasks(df: pd.DataFrame, keep: int) -> pd.DataFrame:
    rows = []
    for record_id, group in df.groupby("record_id"):
        answers = set(str(x) for x in group["prediction"].tolist())
        correctness = set(bool(x) for x in group["correct"].tolist())
        acc_var = float(np.var(group["correct"].astype(float).to_numpy()))
        token_spread = float(group["tokens"].max() - group["tokens"].min())
        rows.append(
            {
                "record_id": int(record_id),
                "family": group["family"].iloc[0],
                "task": group["task"].iloc[0],
                "target": group["target"].iloc[0],
                "num_distinct_predictions": len(answers),
                "correctness_disagrees": len(correctness) > 1,
                "accuracy_variance": acc_var,
                "token_spread": token_spread,
                "selection_score": len(answers) + 3.0 * float(len(correctness) > 1) + acc_var,
            }
        )
    ranked = pd.DataFrame(rows).sort_values(
        ["correctness_disagrees", "selection_score", "token_spread"],
        ascending=[False, False, False],
    )
    return ranked.head(keep).reset_index(drop=True)


def choose_variant(summary: pd.DataFrame, cost_lambda: float) -> pd.DataFrame:
    chosen = summary.copy()
    chosen["utility"] = chosen["accuracy"] - cost_lambda * chosen["mean_tokens"]
    return chosen.sort_values(["utility", "accuracy"], ascending=[False, False]).reset_index(drop=True)


def plot_summaries(
    generated_summary: pd.DataFrame,
    natural_summary: pd.DataFrame,
    out_dir: Path,
) -> list[str]:
    paths = []
    for name, summary in [("generated", generated_summary), ("natural", natural_summary)]:
        fig, ax1 = plt.subplots(figsize=(8, 4))
        x = np.arange(len(summary))
        ax1.bar(x - 0.2, summary["accuracy"], width=0.4, label="Accuracy")
        ax1.set_ylim(0, 1.05)
        ax1.set_ylabel("Accuracy")
        ax1.set_xticks(x)
        ax1.set_xticklabels(summary["variant"], rotation=25, ha="right")
        ax2 = ax1.twinx()
        ax2.bar(x + 0.2, summary["mean_tokens"], width=0.4, color="tab:orange", label="Mean tokens")
        ax2.set_ylabel("Mean tokens")
        ax1.set_title(f"{name.title()} task evaluation by graph variant")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")
        fig.tight_layout()
        path = out_dir / f"{name}_variant_accuracy_tokens.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        paths.append(str(path))
    return paths


def write_markdown_report(
    out_dir: Path,
    metadata: dict[str, Any],
    generated_choice: pd.DataFrame,
    natural_choice: pd.DataFrame,
    selected_tasks: pd.DataFrame,
    plot_paths: list[str],
) -> Path:
    path = out_dir / "report.md"
    best_generated = generated_choice.iloc[0].to_dict() if len(generated_choice) else {}
    al_generated = generated_choice[generated_choice["variant"] == "al_pruned"]
    al_generated_row = al_generated.iloc[0].to_dict() if len(al_generated) else {}
    best_natural = natural_choice.iloc[0].to_dict() if len(natural_choice) else {}
    lines = [
        "# Generated Diagnostic Task PoC",
        "",
        "## Claim Tested",
        "",
        (
            "Can purposefully generated GSM8K-style diagnostic tasks separate nearby "
            "LLM-MAS graph variants after the original active-learning/pruning pipeline?"
        ),
        "",
        "## Configuration",
        "",
        f"- Model: `{metadata['llm_name']}`",
        f"- Agents: {metadata['num_agents']}",
        f"- Aggregate mode: `{metadata['aggregate_mode']}`",
        f"- AL embedding backend: `{metadata['al_summary']['embedding_backend']}`",
        f"- Selected AL train positions: {metadata['al_summary']['final_train_positions']}",
        f"- Train iterations: {metadata['train_iterations']}",
        f"- Generated candidates evaluated: {metadata['generated_candidates']}",
        "",
        "## Generated-Task Selection",
        "",
    ]
    if selected_tasks.empty:
        lines.append("No generated tasks produced graph disagreement under this run.")
    else:
        for row in selected_tasks.to_dict("records"):
            lines.append(
                f"- task_id={row['record_id']} family={row['family']} "
                f"distinct_predictions={row['num_distinct_predictions']} "
                f"correctness_disagrees={row['correctness_disagrees']} "
                f"target={row['target']}"
            )
    lines.extend(
        [
            "",
            "## Generated-Task Winner",
            "",
            f"- Best variant by generated utility: `{best_generated.get('variant')}`",
            f"- Generated accuracy: {best_generated.get('accuracy')}",
            f"- Generated mean tokens: {best_generated.get('mean_tokens')}",
            f"- AL-pruned generated accuracy: {al_generated_row.get('accuracy')}",
            f"- AL-pruned generated mean tokens: {al_generated_row.get('mean_tokens')}",
            "",
            "## Natural Held-Out Check",
            "",
            f"- Best natural variant by same utility: `{best_natural.get('variant')}`",
            f"- Natural accuracy: {best_natural.get('accuracy')}",
            f"- Natural mean tokens: {best_natural.get('mean_tokens')}",
            "",
            "## Plots",
            "",
        ]
    )
    for plot_path in plot_paths:
        lines.append(f"- `{plot_path}`")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


async def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) / f"run_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_set = load_gsm8k(args.train_json)
    test_set = load_gsm8k(args.test_json)
    rng = random.Random(args.seed)
    rng.shuffle(test_set)
    natural_probe = test_set[: args.natural_probe_tasks]
    natural_eval = test_set[args.natural_probe_tasks : args.natural_probe_tasks + args.natural_eval_tasks]

    graph = build_graph(args)
    p_start = read_tokens()
    al_summary = await select_al_training_subset(graph, train_set, args)
    p_after_al = read_tokens()
    selected_train_records = al_summary.pop("selected_records")
    train_summary = await train_graph(graph, selected_train_records, args)
    p_after_train = read_tokens()

    variants = make_variants(graph, args)
    variants_json = [
        {
            "name": variant.name,
            "description": variant.description,
            "spatial_masks": [float(x) for x in variant.spatial_masks.tolist()],
            "temporal_masks": [float(x) for x in variant.temporal_masks.tolist()],
            "spatial_logits": [float(x) for x in variant.spatial_logits.tolist()],
            "temporal_logits": [float(x) for x in variant.temporal_logits.tolist()],
            "num_spatial_edges": int((variant.spatial_masks > 0).sum().item()),
        }
        for variant in variants
    ]

    natural_probe_df, natural_probe_summary = await evaluate_variants(
        graph,
        variants,
        natural_probe,
        num_rounds=args.num_rounds,
        max_concurrency=args.max_concurrency,
        aggregate_mode=args.aggregate_mode,
    )
    generated_candidates = generate_diagnostic_candidates(args.generated_candidates, args.seed)
    generated_df, generated_summary_all = await evaluate_variants(
        graph,
        variants,
        generated_candidates,
        num_rounds=args.num_rounds,
        max_concurrency=args.max_concurrency,
        aggregate_mode=args.aggregate_mode,
    )
    selected_generated = select_disagreement_tasks(generated_df, args.generated_keep)
    selected_records = [
        generated_candidates[int(record_id)] for record_id in selected_generated["record_id"].tolist()
    ]
    if selected_records:
        selected_generated_df, selected_generated_summary = await evaluate_variants(
            graph,
            variants,
            selected_records,
            num_rounds=args.num_rounds,
            max_concurrency=args.max_concurrency,
            aggregate_mode=args.aggregate_mode,
        )
    else:
        selected_generated_df = generated_df.iloc[0:0].copy()
        selected_generated_summary = generated_summary_all.iloc[0:0].copy()

    generated_choice = choose_variant(selected_generated_summary, args.cost_lambda)
    natural_df, natural_summary = await evaluate_variants(
        graph,
        variants,
        natural_eval,
        num_rounds=args.num_rounds,
        max_concurrency=args.max_concurrency,
        aggregate_mode=args.aggregate_mode,
    )
    natural_choice = choose_variant(natural_summary, args.cost_lambda)
    p_end = read_tokens()

    plot_paths = plot_summaries(
        selected_generated_summary if len(selected_generated_summary) else generated_summary_all,
        natural_summary,
        out_dir,
    )

    metadata = {
        "created_at": stamp,
        "llm_name": args.llm_name,
        "seed": args.seed,
        "mode": args.mode,
        "decision_method": args.decision_method,
        "num_agents": args.num_agents,
        "num_rounds": args.num_rounds,
        "aggregate_mode": args.aggregate_mode,
        "train_iterations": args.train_iterations,
        "natural_probe_tasks": args.natural_probe_tasks,
        "natural_eval_tasks": args.natural_eval_tasks,
        "generated_candidates": args.generated_candidates,
        "cost_lambda": args.cost_lambda,
        "token_counts": {
            "al_tokens": token_delta(p_start, p_after_al),
            "train_tokens": token_delta(p_after_al, p_after_train),
            "total_tokens": token_delta(p_start, p_end),
        },
        "al_summary": al_summary,
        "train_summary": {
            "accuracy": train_summary["accuracy"],
            "num_rows": train_summary["num_rows"],
        },
        "variants": variants_json,
        "natural_probe_summary": natural_probe_summary.to_dict("records"),
        "generated_summary_all": generated_summary_all.to_dict("records"),
        "selected_generated_summary": selected_generated_summary.to_dict("records"),
        "generated_choice": generated_choice.to_dict("records"),
        "natural_summary": natural_summary.to_dict("records"),
        "natural_choice": natural_choice.to_dict("records"),
    }

    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    pd.DataFrame(al_summary["init_details"]).to_csv(out_dir / "al_init_details.csv", index=False)
    pd.DataFrame(train_summary["rows"]).to_csv(out_dir / "train_rows.csv", index=False)
    natural_probe_df.to_csv(out_dir / "natural_probe_variant_rows.csv", index=False)
    natural_probe_summary.to_csv(out_dir / "natural_probe_variant_summary.csv", index=False)
    generated_df.to_csv(out_dir / "generated_candidate_variant_rows.csv", index=False)
    generated_summary_all.to_csv(out_dir / "generated_candidate_variant_summary.csv", index=False)
    selected_generated.to_csv(out_dir / "selected_generated_tasks.csv", index=False)
    selected_generated_df.to_csv(out_dir / "selected_generated_variant_rows.csv", index=False)
    selected_generated_summary.to_csv(out_dir / "selected_generated_variant_summary.csv", index=False)
    natural_df.to_csv(out_dir / "natural_eval_variant_rows.csv", index=False)
    natural_summary.to_csv(out_dir / "natural_eval_variant_summary.csv", index=False)
    generated_choice.to_csv(out_dir / "generated_choice.csv", index=False)
    natural_choice.to_csv(out_dir / "natural_choice.csv", index=False)
    report_path = write_markdown_report(
        out_dir,
        metadata,
        generated_choice,
        natural_choice,
        selected_generated,
        plot_paths,
    )

    print(f"Wrote artifacts to {out_dir}")
    print(f"Wrote report to {report_path}")
    print("Generated-task ranking:")
    print(generated_choice.to_string(index=False))
    print("Natural held-out ranking:")
    print(natural_choice.to_string(index=False))


if __name__ == "__main__":
    asyncio.run(main())
