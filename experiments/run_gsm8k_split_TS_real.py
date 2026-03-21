import sys
import os
import argparse
import yaml
import json
import time
import asyncio
from pathlib import Path
import torch
import copy
from typing import List, Union, Literal, Optional
import random
import math

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C, WhiteKernel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from AgentPrune.utils.const import AgentPrune_ROOT
from AgentPrune.graph.graph import Graph
from AgentPrune.tools.reader.readers import JSONLReader
from AgentPrune.utils.globals import Time
from dataset.gsm8k_dataset import gsm_data_process, gsm_get_predict
from AgentPrune.utils.globals import PromptTokens, CompletionTokens
from experiments.train_gsm8k import eki_metrics_per_task_one_step_give_task
from experiments.embedding_mmlu import encode_texts, select_farthest_first


def save_json(path: Path, data):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


#def dataloader(data_list, batch_size, i_batch):
#    return data_list[i_batch * batch_size: i_batch * batch_size + batch_size]
def dataloader(data_list, batch_size, i_batch, cycle=False):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    n = len(data_list)
    if n == 0:
        return []

    if not cycle:
        start = i_batch * batch_size
        end = start + batch_size
        return data_list[start:end]

    start = i_batch * batch_size
    return [data_list[(start + j) % n] for j in range(batch_size)]

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(
        description="AgentPrune Experiments on GSM8K with explicit train/test split and TS-real active learning"
    )

    parser.add_argument("--train_json", type=str, default="dataset/gsm8k/train.jsonl",
                        help="Path to training set json file.")
    parser.add_argument("--test_json", type=str, default="dataset/gsm8k/test.jsonl",
                        help="Path to test set json file.")
    parser.add_argument("--result_file", type=str, default=None)

    parser.add_argument("--llm_name", type=str, default="gpt-4.1-nano")#"gpt-4o-mini")#
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered', 'Star'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size")
    parser.add_argument('--imp_per_iterations', type=int, default=5, help="Prune every few iterations. Default 5.")
    parser.add_argument('--num_rounds', type=int, default=1, help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25, help="The Rate of Pruning. Default 0.25.")
    parser.add_argument('--train_num_iterations', type=int, default=10, help="The number of training batches to run.")
    parser.add_argument('--domain', type=str, default="gsm8k", help="Domain (the same as dataset name), default 'gsm8k'")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                        help='The decision method of AgentPrune')
    parser.add_argument('--optimized_spatial', action='store_true')
    parser.add_argument('--optimized_temporal', action='store_true')

    parser.add_argument('--shuffle_train', action='store_true',
                        help='Shuffle training set before training.')
    parser.add_argument('--eval_batch_size', type=int, default=None,
                        help='Batch size used during test evaluation. Defaults to batch_size.')
    parser.add_argument('--train_seed', type=int, default=None,
                        help='Local random seed used only to shuffle the training set.')
    parser.add_argument('--test_num_iterations', type=int, default=None,
                        help='Number of test batches to run. Default None means evaluate the full test set.')

    parser.add_argument('--test_ratio', type=float, default=0.35,
                        help='Fraction of test_json kept as the actual evaluation pool.')
    parser.add_argument('--test_split_seed', type=int, default=0,
                        help='Random seed used for splitting test_json if test_shuffle is enabled.')
    parser.add_argument('--test_shuffle', action='store_true',
                        help='Shuffle test_json before taking the tail portion as evaluation pool.')

    # TS-real active learning args
    parser.add_argument('--active_learning', action='store_true',
                        help='Enable TS-real style active learning to choose training tasks from the train pool.')
    parser.add_argument('--al_seed', type=int, default=1098,
                        help='Seed used by active learning.')
    parser.add_argument('--al_pool_size', type=int, default=1000,
                        help='Candidate pool size for active learning before representative selection. If larger than train size, all tasks are used.')
    parser.add_argument('--al_rep_budget', type=int, default=50,
                        help='Number of representative tasks selected from the candidate pool before TS.')
    parser.add_argument('--al_init_size', type=int, default=10,
                        help='Initial number of randomly observed tasks.')
    parser.add_argument('--al_num_rounds', type=int, default=3,
                        help='Number of TS rounds.')
    parser.add_argument('--al_batch_size', type=int, default=5,
                        help='Number of tasks queried in each TS round.')
    parser.add_argument('--al_final_topk', type=int, default=10,
                        help='Final number of selected tasks for training.')
    parser.add_argument('--embedding_model', type=str, default='sentence-transformers/all-MiniLM-L6-v2',
                        help='Sentence-transformer model used by the same embedding backend as MMLU.')
    parser.add_argument('--embedding_batch_size', type=int, default=64,
                        help='Batch size for text embedding.')
    parser.add_argument('--embedding_normalize', action='store_true', default=True,
                        help='Whether to normalize embeddings. Default True.')
    parser.add_argument('--eki_ensemble_size', type=int, default=6,
                        help='Ensemble size for GSM8K EKI task scoring.')
    parser.add_argument('--eki_prior_std', type=float, default=2.0)
    parser.add_argument('--eki_obs_noise', type=float, default=0.1)
    parser.add_argument('--eki_damping', type=float, default=0.7)
    parser.add_argument('--eki_max_concurrency', type=int, default=8)

    args = parser.parse_args()
    result_path = AgentPrune_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args


def perturb_train_set(train_set, shuffle_train: bool, train_seed: Optional[int]):
    train_set = list(train_set)
    if shuffle_train:
        rng = random.Random(train_seed)
        rng.shuffle(train_set)
    return train_set


def read_tokens():
    p = int(PromptTokens.instance().value)
    c = int(CompletionTokens.instance().value)
    return p, c


def build_eval_test_set(full_test_set, test_ratio: float, test_split_seed: int, test_shuffle: bool):
    full_test_set = list(full_test_set)
    if test_shuffle:
        rng = random.Random(test_split_seed)
        rng.shuffle(full_test_set)

    #split_idx = int(len(full_test_set) * (1 - test_ratio))
    #eval_test_set = full_test_set[split_idx:]
    keep_n = int(len(full_test_set) * test_ratio)
    eval_test_set = full_test_set[:keep_n]
    return eval_test_set


async def run_batches(
    graph: Graph,
    dataset,
    batch_size: int,
    num_rounds: int,
    result_file: Path,
    phase: str,
    optimizer=None,
    prune_every: int = None,
    pruning_rate: float = None,
    max_batches: int = None,
    allow_optimization: bool = False,
    cycle_dataset: bool = False,
):
    if len(dataset) == 0:
        num_batches = 0
    elif cycle_dataset:
        if max_batches is None:
            num_batches = math.ceil(len(dataset) / batch_size)
        else:
            num_batches = max_batches
    else:
        num_batches = math.ceil(len(dataset) / batch_size)
        if max_batches is not None:
            num_batches = min(num_batches, max_batches)

    total_solved = 0
    total_executed = 0
    all_records = []
    batch_metrics = []

    for i_batch in range(num_batches):
        print(f"[{phase}] Batch {i_batch}", 80 * '-')
        start_ts = time.time()
        answer_log_probs = []
        answers = []
        steps = []
        #current_batch = dataloader(dataset, batch_size, i_batch)
        current_batch = dataloader(dataset, batch_size, i_batch, cycle=cycle_dataset)
        if not current_batch:
            break

        for record in current_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits

            task = record["task"]
            step = record.get("step")
            answer = record["answer"]
            answers.append(answer)
            steps.append(step)
            input_dict = {"task": task}
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds)))

        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)

        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        batch_solved = 0

        for record, step, answer, log_prob, true_answer in zip(current_batch, steps, raw_answers, log_probs, answers):
            predict_answer = gsm_get_predict(answer[0])
            try:
                is_solved = float(predict_answer) == float(true_answer)
            except Exception:
                is_solved = False

            total_solved += int(is_solved)
            total_executed += 1
            batch_solved += int(is_solved)
            cumulative_accuracy = total_solved / total_executed
            utility = float(is_solved)
            utilities.append(utility)

            updated_item = {
                "Phase": phase,
                "Question": record,
                "Answer": true_answer,
                "Step": step,
                "Response": answer,
                "Attempt answer": predict_answer,
                "Solved": bool(is_solved),
                "Total solved": total_solved,
                "Total executed": total_executed,
                "Cumulative accuracy": cumulative_accuracy,
            }
            all_records.append(updated_item)

            if allow_optimization:
                single_loss = -log_prob * utility
                loss_list.append(single_loss)

        batch_accuracy = batch_solved / len(current_batch)
        batch_info = {
            "phase": phase,
            "batch_index": i_batch,
            "batch_size": len(current_batch),
            "batch_accuracy": batch_accuracy,
            "cumulative_accuracy": total_solved / total_executed,
            "solved_in_batch": batch_solved,
            "total_solved": total_solved,
            "total_executed": total_executed,
        }
        batch_metrics.append(batch_info)

        if allow_optimization and loss_list:
            total_loss = torch.mean(torch.stack(loss_list))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if prune_every is not None and prune_every > 0 and (i_batch + 1) % prune_every == 0:
                graph.update_masks(pruning_rate)
        else:
            total_loss = None

        print(f"[{phase}] Batch time {time.time() - start_ts:.3f}")
        print(f"[{phase}] Batch accuracy: {batch_accuracy:.4f}")
        print(f"[{phase}] Cumulative accuracy: {total_solved / total_executed:.4f}")
        print(f"[{phase}] utilities: {utilities}")
        if total_loss is not None:
            print(f"[{phase}] Loss: {float(total_loss.detach().cpu()):.6f}")

    summary = {
        "phase": phase,
        "num_examples_used": total_executed,
        "num_batches_used": len(batch_metrics),
        "solved": total_solved,
        "accuracy": (total_solved / total_executed) if total_executed > 0 else None,
        "batch_metrics": batch_metrics,
        "records": all_records,
    }
    save_json(result_file, summary)
    return summary


def encode_gsm8k_items(records, model: str, batch_size: int, normalize: bool) -> np.ndarray:
    texts = [str(r.get("task", "")) for r in records]
    X = encode_texts(
        texts,
        model=model,
        batch_size=batch_size,
        normalize=normalize,
    )
    return np.asarray(X, dtype=np.float64)


def select_gsm8k_representatives_from_pool(X_pool: np.ndarray, budget: int):
    """
    Select representative positions from a GSM8K candidate pool using the same
    farthest-first core routine as MMLU, but without subject-wise quota.
    Returns positions inside the candidate pool and the corresponding embeddings.
    """
    n_pool = X_pool.shape[0]
    budget = min(int(budget), n_pool)
    if budget <= 0:
        return np.array([], dtype=int), np.empty((0, X_pool.shape[1]), dtype=X_pool.dtype)

    rep_pos = np.asarray(select_farthest_first(np.asarray(X_pool, dtype=np.float32), budget), dtype=int)
    rep_emb = X_pool[rep_pos]
    return rep_pos, rep_emb


def fit_pls_gp(X_all, y_obs, obs_pos, n_components=3, seed=0):
    X_obs = X_all[obs_pos]
    y_obs = np.asarray(y_obs, dtype=np.float64)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_obs_s = x_scaler.fit_transform(X_obs)
    y_t = np.arcsinh(y_obs)
    y_s = y_scaler.fit_transform(y_t.reshape(-1, 1)).ravel()

    n_components = max(1, min(n_components, X_obs_s.shape[0], X_obs_s.shape[1]))
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_obs_s, y_s)

    X_all_s = x_scaler.transform(X_all)
    Z_all = pls.transform(X_all_s)
    z_scaler = StandardScaler()
    Z_all_s = z_scaler.fit_transform(Z_all)

    d = Z_all_s.shape[1]
    kernel = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(d),
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5,
    ) + WhiteKernel(
        noise_level=1e-3,
        noise_level_bounds=(1e-6, 1e-1),
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-4,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=seed,
    )
    gp.fit(Z_all_s[obs_pos], y_s)
    mu, std = gp.predict(Z_all_s, return_std=True)
    return Z_all_s, gp, mu, std



def ts_select_batch(gp, Z_all, candidates_pos, batch_size=5, seed=0):
    samples = gp.sample_y(Z_all, n_samples=1, random_state=seed).ravel()
    cand_scores = samples[candidates_pos]
    order = np.argsort(cand_scores)[::-1]
    pick = candidates_pos[order[:batch_size]]
    return pick, samples



def batch_ts_topk(gp, Z_all, k=10, n_draws=200, seed=0):
    rng = np.random.default_rng(seed)
    counts = np.zeros(Z_all.shape[0], dtype=int)
    for _ in range(n_draws):
        s = gp.sample_y(Z_all, n_samples=1, random_state=int(rng.integers(1_000_000_000))).ravel()
        topk = np.argsort(s)[::-1][:k]
        counts[topk] += 1
    order = np.argsort(counts)[::-1]
    return order[:k], counts / n_draws



def interleave_ends(lst):
    res = []
    i, j = 0, len(lst) - 1
    while i <= j:
        if i == j:
            res.append(lst[i])
        else:
            res.append(lst[i])
            res.append(lst[j])
        i += 1
        j -= 1
    return res


async def score_gsm8k_tasks(graph, tasks, args):
    metrics = await eki_metrics_per_task_one_step_give_task(
        graph=graph,
        dataset=None,
        tasks=tasks,
        num_rounds=args.num_rounds,
        ensemble_size=args.eki_ensemble_size,
        prior_std=args.eki_prior_std,
        obs_noise=args.eki_obs_noise,
        damping=args.eki_damping,
        max_concurrency=args.eki_max_concurrency,
    )
    return np.asarray(metrics["kl"], dtype=np.float64), metrics


async def select_training_subset_ts_real(graph, train_set, args):
    pool_size = min(args.al_pool_size, len(train_set))
    if pool_size <= 0:
        raise ValueError("Active learning pool is empty.")

    rng = np.random.default_rng(args.al_seed)
    if pool_size < len(train_set):
        pool_positions = rng.choice(len(train_set), size=pool_size, replace=False)
        pool_positions = np.asarray(pool_positions, dtype=int)
    else:
        pool_positions = np.arange(len(train_set), dtype=int)

    candidate_pool = [train_set[int(i)] for i in pool_positions]
    X_pool = encode_gsm8k_items(
        candidate_pool,
        model=args.embedding_model,
        batch_size=args.embedding_batch_size,
        normalize=args.embedding_normalize,
    )

    rep_budget = min(args.al_rep_budget, len(candidate_pool))
    rep_pool_positions, X_all = select_gsm8k_representatives_from_pool(X_pool, rep_budget)
    idx_all = np.arange(len(rep_pool_positions), dtype=int)
    representative_records = [candidate_pool[int(p)] for p in rep_pool_positions]

    if len(representative_records) == 0:
        raise ValueError("Representative pool is empty after selection.")

    init_size = min(args.al_init_size, len(representative_records))
    if init_size == 0:
        raise ValueError("al_init_size resolves to zero.")

    init_pos = rng.choice(len(representative_records), size=init_size, replace=False)
    obs_pos = list(map(int, init_pos.tolist()))
    init_tasks = [representative_records[p] for p in obs_pos]
    y0, _m0 = await score_gsm8k_tasks(graph, init_tasks, args)
    y_obs = list(y0.tolist())

    observed_mask = np.zeros(len(representative_records), dtype=bool)
    observed_mask[obs_pos] = True
    round_logs = []

    for t in range(args.al_num_rounds):
        candidates_pos = np.where(~observed_mask)[0]
        if len(candidates_pos) == 0:
            break

        Z_all, gp, mu, std = fit_pls_gp(X_all, y_obs, np.array(obs_pos), n_components=3, seed=args.al_seed + 10 * t + 1)
        pick_pos, _samples = ts_select_batch(
            gp,
            Z_all,
            candidates_pos,
            batch_size=min(args.al_batch_size, len(candidates_pos)),
            seed=args.al_seed + 20 * t + 3,
        )
        pick_pos = np.asarray(pick_pos, dtype=int)
        pick_tasks = [representative_records[p] for p in pick_pos.tolist()]
        y_new, _m_new = await score_gsm8k_tasks(graph, pick_tasks, args)

        for p, yv in zip(pick_pos.tolist(), y_new.tolist()):
            obs_pos.append(int(p))
            y_obs.append(float(yv))
            observed_mask[p] = True

        remaining_pos = np.where(~observed_mask)[0]
        ranked_remaining = []
        if len(remaining_pos) > 0:
            ranked_remaining = remaining_pos[np.argsort(mu[remaining_pos])[::-1][:10]].tolist()

        round_logs.append({
            "round": t,
            "picked_representative_positions": pick_pos.tolist(),
            "picked_pool_positions": rep_pool_positions[pick_pos].tolist(),
            "picked_train_positions": pool_positions[rep_pool_positions[pick_pos]].tolist(),
            "picked_scores": y_new.tolist(),
            "top_remaining_representative_positions": ranked_remaining,
            "top_remaining_pool_positions": rep_pool_positions[np.asarray(ranked_remaining, dtype=int)].tolist() if ranked_remaining else [],
            "top_remaining_train_positions": pool_positions[rep_pool_positions[np.asarray(ranked_remaining, dtype=int)]].tolist() if ranked_remaining else [],
        })
        print(f"[AL round {t}] picked representative positions: {pick_pos.tolist()}")
        print(f"[AL round {t}] picked train positions: {pool_positions[rep_pool_positions[pick_pos]].tolist()}")
        print(f"[AL round {t}] picked scores: {y_new.tolist()}")

    Z_all, gp, mu, std = fit_pls_gp(X_all, y_obs, np.array(obs_pos), n_components=3, seed=args.al_seed + 999)
    final_topk = min(args.al_final_topk, len(representative_records))
    topk_pos, prob = batch_ts_topk(gp, Z_all, k=final_topk, n_draws=200, seed=args.al_seed + 12345)
    topk_pos = np.asarray(topk_pos, dtype=int)
    training_order_rep = interleave_ends(topk_pos.tolist())
    final_topk_pool_positions = rep_pool_positions[topk_pos].tolist()
    final_topk_train_positions = pool_positions[rep_pool_positions[topk_pos]].tolist()
    ordered_pool_positions = rep_pool_positions[np.asarray(training_order_rep, dtype=int)].tolist()
    ordered_train_positions = pool_positions[rep_pool_positions[np.asarray(training_order_rep, dtype=int)]].tolist()
    selected_records = [representative_records[p] for p in training_order_rep]

    print("=== Representative-selected train positions ===", ordered_train_positions)

    return {
        "candidate_pool_positions": pool_positions.tolist(),
        "candidate_pool_size": len(candidate_pool),
        "representative_pool_positions": rep_pool_positions.tolist(),
        "representative_train_positions": pool_positions[rep_pool_positions].tolist(),
        "representative_pool_size": len(representative_records),
        "observed_representative_positions": obs_pos,
        "observed_pool_positions": rep_pool_positions[np.asarray(obs_pos, dtype=int)].tolist(),
        "observed_train_positions": pool_positions[rep_pool_positions[np.asarray(obs_pos, dtype=int)]].tolist(),
        "observed_scores": y_obs,
        "round_logs": round_logs,
        "final_topk_representative_positions": topk_pos.tolist(),
        "final_topk_pool_positions": final_topk_pool_positions,
        "final_topk_train_positions": final_topk_train_positions,
        "final_training_order_representative_positions": training_order_rep,
        "final_training_order_pool_positions": ordered_pool_positions,
        "final_training_order_train_positions": ordered_train_positions,
        "final_topk_prob": prob[topk_pos].tolist(),
        "selected_records": selected_records,
    }


async def main():
    args = parse_args()
    train_set = JSONLReader.parse_file(args.train_json)
    train_set = gsm_data_process(train_set)

    full_test_set = JSONLReader.parse_file(args.test_json)
    full_test_set = gsm_data_process(full_test_set)

    train_set = perturb_train_set(train_set, args.shuffle_train, args.train_seed)
    test_set = build_eval_test_set(full_test_set, args.test_ratio, args.test_split_seed, args.test_shuffle)

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{AgentPrune_ROOT}/result/gsm8k")
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.result_file is not None:
        base_path = Path(args.result_file)
        if base_path.suffix:
            stem = base_path.stem
            parent = base_path.parent
            suffix = base_path.suffix
        else:
            stem = base_path.name
            parent = base_path.parent
            suffix = ".json"
        parent.mkdir(parents=True, exist_ok=True)
        train_result_file = parent / f"{stem}_train{suffix}"
        test_result_file = parent / f"{stem}_test{suffix}"
        summary_file = parent / f"{stem}_summary{suffix}"
    else:
        prefix = f"{args.domain}_{args.llm_name}_{current_time}"
        train_result_file = result_dir / f"{prefix}_train.json"
        test_result_file = result_dir / f"{prefix}_test.json"
        summary_file = result_dir / f"{prefix}_summary.json"

    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(args.mode, len(agent_names))
    graph = Graph(
        domain=args.domain,
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=args.decision_method,
        optimized_spatial=args.optimized_spatial,
        optimized_temporal=args.optimized_temporal,
        **kwargs,
    )

    do_train = args.optimized_spatial or args.optimized_temporal
    optimizer = None
    train_summary = None
    active_learning_summary = None

    print(f"Train set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Training batches requested: {args.train_num_iterations}")
    print(f"Testing batches requested: {args.test_num_iterations}")

    pn1, cn1 = read_tokens()
    train_dataset_for_optimization = train_set

    if do_train and args.active_learning:
        active_learning_summary = await select_training_subset_ts_real(graph, train_set, args)
        train_dataset_for_optimization = active_learning_summary["selected_records"]
        print(f"Active-learning train subset size: {len(train_dataset_for_optimization)}")
        #fixed_positions = [4660, 4179, 789, 6957, 3348, 4015, 3193, 4498, 4794, 2149]
        #train_dataset_for_optimization = [train_set[p] for p in fixed_positions]
        #print("=== Using fixed train positions ===", fixed_positions)
        #print(f"Fixed train subset size: {len(train_dataset_for_optimization)}")

    p0, c0 = read_tokens()
    task_prompt = p0 - pn1
    task_completion = c0 - cn1
    if do_train:
        optimizer = torch.optim.Adam([graph.spatial_logits, graph.temporal_logits], lr=args.lr)
        train_summary = await run_batches(
            graph=graph,
            dataset=train_dataset_for_optimization,
            batch_size=args.batch_size,
            num_rounds=args.num_rounds,
            result_file=train_result_file,
            phase="train",
            optimizer=optimizer,
            prune_every=args.imp_per_iterations,
            pruning_rate=args.pruning_rate,
            max_batches=args.train_num_iterations,
            allow_optimization=do_train,
            cycle_dataset=True,
        )

    p1, c1 = read_tokens()
    train_prompt = p1 - p0
    train_completion = c1 - c0

    eval_batch_size = args.eval_batch_size or args.batch_size
    test_summary = await run_batches(
        graph=graph,
        dataset=test_set,
        batch_size=eval_batch_size,
        num_rounds=args.num_rounds,
        result_file=test_result_file,
        phase="test",
        optimizer=None,
        prune_every=None,
        pruning_rate=None,
        max_batches=args.test_num_iterations,
        allow_optimization=False,
    )
    p2, c2 = read_tokens()

    final_summary = {
        "train_json": args.train_json,
        "test_json": args.test_json,
        "mode": args.mode,
        "llm_name": args.llm_name,
        "domain": args.domain,
        "agent_names": args.agent_names,
        "agent_nums": args.agent_nums,
        "expanded_agent_names": agent_names,
        "optimized_spatial": args.optimized_spatial,
        "optimized_temporal": args.optimized_temporal,
        "active_learning": args.active_learning,
        "embedding_model": args.embedding_model,
        "embedding_batch_size": args.embedding_batch_size,
        "embedding_normalize": args.embedding_normalize,
        "active_learning_summary": {
            k: v for k, v in (active_learning_summary or {}).items() if k != "selected_records"
        },
        "train_seed": args.train_seed,
        "shuffle_train": args.shuffle_train,
        "batch_size": args.batch_size,
        "eval_batch_size": eval_batch_size,
        "num_rounds": args.num_rounds,
        "train_num_iterations": args.train_num_iterations,
        "test_num_iterations": args.test_num_iterations,
        "test_ratio": args.test_ratio,
        "test_split_seed": args.test_split_seed,
        "test_shuffle": args.test_shuffle,
        "full_test_size": len(full_test_set),
        "eval_test_size": len(test_set),
        "train_size": len(train_set),
        "train_size_used": len(train_dataset_for_optimization) if do_train else 0,
        "test_size": len(test_set),
        "train_summary": {
            "num_examples_used": train_summary["num_examples_used"] if do_train else 0,
            "num_batches_used": train_summary["num_batches_used"] if do_train else 0,
            "accuracy": train_summary["accuracy"] if do_train else None,
        },
        "test_summary": {
            "num_examples_used": test_summary["num_examples_used"],
            "num_batches_used": test_summary["num_batches_used"],
            "accuracy": test_summary["accuracy"],
        },
        "train_result_file": str(train_result_file),
        "test_result_file": str(test_result_file),
    }
    save_json(summary_file, final_summary)

    print("=" * 80)
    if do_train:
        print(f"Train accuracy: {train_summary['accuracy']}")
    print(f"Train set size: {len(train_set)}")
    print(f"Full test set size: {len(full_test_set)}")
    print(f"Evaluation test pool size: {len(test_set)}")
    print(f"Training batches requested: {args.train_num_iterations}")
    print(f"Testing batches requested: {args.test_num_iterations}")
    print(f"Saved test results to: {test_result_file}")
    print(f"Saved summary to: {summary_file}")

    eval_prompt = p2 - p1
    eval_completion = c2 - c1
    
    print(f"Test accuracy: {test_summary['accuracy']}")
    print("=== AL tokens ===", task_prompt, task_completion, task_prompt + task_completion)
    print("=== Train tokens ===", train_prompt, train_completion, train_prompt + train_completion)
    print("=== Eval tokens  ===", eval_prompt, eval_completion, eval_prompt + eval_completion)



def get_kwargs(mode: Union[Literal['DirectAnswer'], Literal['FullConnected'], Literal['Random'], Literal['Chain'], Literal['Debate'], Literal['Layered'], Literal['Star']],
               N: int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks: List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks: List[List[int]] = None
    node_kwargs = None

    def generate_layered_graph(N, layer_num=2):
        adj_matrix = [[0 for _ in range(N)] for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            current_layer = layers[i]
            for j in range(N):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix

    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        for j in range(1, n):
            matrix[0][j] = 1
        return matrix

    if mode == 'DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role': 'Programming Expert'}]
    elif mode == 'FullConnected':
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == 'Random':
        fixed_spatial_masks = [[random.randint(0, 1) if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode == 'Chain':
        fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i == 0 and j == N - 1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for _ in range(N)] for _ in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]

    return {
        "initial_spatial_probability": initial_spatial_probability,
        "fixed_spatial_masks": fixed_spatial_masks,
        "initial_temporal_probability": initial_temporal_probability,
        "fixed_temporal_masks": fixed_temporal_masks,
        "node_kwargs": node_kwargs,
    }


if __name__ == '__main__':
    asyncio.run(main())
