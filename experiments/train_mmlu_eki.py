import torch
import numpy as np
import asyncio
import copy
import time
from typing import List, Any, Dict, Tuple


from experiments.accuracy import Accuracy
from experiments.SoftAccuracy import SoftAccuracy

import torch.nn.functional as Fnn


def _flatten_logits(spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
    return torch.cat([spatial.reshape(-1), temporal.reshape(-1)], dim=0)


def _unflatten_logits(theta: torch.Tensor, spatial_shape: torch.Size, temporal_shape: torch.Size) -> Tuple[torch.Tensor, torch.Tensor]:
    s_numel = int(torch.tensor(spatial_shape).prod().item())
    spatial = theta[:s_numel].reshape(spatial_shape)
    temporal = theta[s_numel:].reshape(temporal_shape)
    return spatial, temporal


async def _eval_member_scalar_utility(
    graph_template,
    theta_e: torch.Tensor,
    batch_records: List[Any],
    dataset,
    num_rounds: int,
    spatial_shape: torch.Size,
    temporal_shape: torch.Size,
    sem: asyncio.Semaphore,
) -> float:
    """返回该 member 在 batch 上的平均 utility（标量）。"""

    spatial_e, temporal_e = _unflatten_logits(theta_e, spatial_shape, temporal_shape)
    utilities: List[float] = []

    tem_iter = 0
    for record in batch_records:
        realized_graph = copy.deepcopy(graph_template)

        # EKI: 不需要 autograd
        realized_graph.optimized_spatial = False
        realized_graph.optimized_temporal = False

        realized_graph.spatial_logits = spatial_e.detach()
        realized_graph.temporal_logits = temporal_e.detach()

        input_dict = dataset.record_to_input(record)
        correct_answer = dataset.record_to_target_answer(record)

        # 限制并发（重要：避免 rate limit）
        async with sem:
            final_answers, _log_probs = await realized_graph.arun(input_dict, num_rounds)

        # 你的 arun 返回 final_answers 是 list，这里沿用你原训练里对 raw_answer 的处理方式
        raw_answer = final_answers
        answer = dataset.postprocess_answer(raw_answer)

        acc = Accuracy()
        acc.update(answer, correct_answer)
        utilities.append(float(acc.get()))
        tem_iter += 1
        print(f"Iter {tem_iter}")

    return float(np.mean(utilities))


async def train_eki(
    graph,
    dataset,
    num_iters: int = 4,         # 迭代步数
    num_rounds: int = 1,
    batch_size: int = 2,
    ensemble_size: int = 6,
    prior_std: float = 0.2,
    obs_noise: float = 0.1,     # 越大更新越保守，越稳
    damping: float = 0.7,       # 0~1
    max_concurrency: int = 8,   # 控并发
    imp_per_iters: int = 1,
    pruning_rate: float = 0.05,
) -> None:

    def infinite_data_loader():
        perm = np.random.permutation(len(dataset))
        while True:
            for idx in perm:
                yield dataset[idx.item()]

    loader = infinite_data_loader()
    sem = asyncio.Semaphore(max_concurrency)

    # parameter vector theta0
    spatial0 = graph.spatial_logits.detach().clone().float()
    temporal0 = graph.temporal_logits.detach().clone().float()
    spatial_shape, temporal_shape = spatial0.shape, temporal0.shape

    theta0 = _flatten_logits(spatial0, temporal0)
    D = theta0.numel()
    E = ensemble_size

    # init ensemble
    Theta = theta0[None, :].repeat(E, 1) + prior_std * torch.randn(E, D)

    for i_iter in range(num_iters):
        print(f"Iter {i_iter}", 80 * "-")
        t0 = time.time()

        # sample a batch
        batch_records = [next(loader) for _ in range(batch_size)]


        ## forward: evaluate each member
        #F_vals: List[float] = []
        #for e in range(E):
        #    f_e = await _eval_member_scalar_utility(
        #        graph_template=graph,
        #        theta_e=Theta[e],
        #        batch_records=batch_records,
        #        dataset=dataset,
        #        num_rounds=num_rounds,
        #        spatial_shape=spatial_shape,
        #        temporal_shape=temporal_shape,
        #        sem=sem,
        #    )
        #    F_vals.append(f_e)
        #
        #F = torch.tensor(F_vals, dtype=torch.float32)  # (E,)

        # ============================================================
        # [MODIFIED] forward: evaluate ALL (member, record) in parallel
        # ============================================================
        async def _run_one(theta_e: torch.Tensor, record):
            # IMPORTANT: do NOT reuse graph across records (memory/outputs mutate)
            realized_graph = copy.deepcopy(graph)

            # EKI does not need autograd; also avoids building useless graphs in construct_*()
            if hasattr(realized_graph, "optimized_spatial"):
                realized_graph.optimized_spatial = False
            if hasattr(realized_graph, "optimized_temporal"):
                realized_graph.optimized_temporal = False

            spatial_e, temporal_e = _unflatten_logits(theta_e, spatial_shape, temporal_shape)
            realized_graph.spatial_logits = spatial_e.detach()
            realized_graph.temporal_logits = temporal_e.detach()

            input_dict = dataset.record_to_input(record)
            correct_answer = dataset.record_to_target_answer(record)

            # concurrency gate: at most max_concurrency arun() running at once
            async with sem:
                final_answers, _log_probs = await realized_graph.arun(input_dict, num_rounds)

            answer = dataset.postprocess_answer(final_answers)

            acc = Accuracy()
            acc.update(answer, correct_answer)
            return float(acc.get())  # 0/1

        tasks = []
        for e in range(E):
            for record in batch_records:
                tasks.append(asyncio.create_task(_run_one(Theta[e], record)))

        # results length = E*B, ordered by (e, record)
        results = await asyncio.gather(*tasks)

        # reshape to (E, B)
        F_matrix = torch.tensor(results, dtype=torch.float32).reshape(E, batch_size)

        # scalar output per member: mean utility over batch => (E,)
        F = F_matrix.mean(dim=1)

        y = torch.tensor(1.0, dtype=torch.float32)     # target mean accuracy = 1

        # ensemble stats
        Theta_mean = Theta.mean(dim=0)      # (D,)
        F_mean = F.mean()                  # scalar

        Theta_A = Theta - Theta_mean[None, :]  # (E, D)
        F_A = (F - F_mean).reshape(E, 1)       # (E, 1)

        # covariances (D×1 and 1×1)
        C_thetaF = (Theta_A.T @ F_A) / (E - 1)          # (D, 1)
        C_FF = (F_A.T @ F_A) / (E - 1)                  # (1, 1)

        R = torch.tensor([[obs_noise ** 2]], dtype=torch.float32)  # (1,1)
        inv = torch.linalg.inv(C_FF + R)                           # (1,1)
        K = C_thetaF @ inv                                         # (D,1)

        # perturbed obs
        eps = obs_noise * torch.randn(E, 1)
        innovation = (y + eps.reshape(E, 1)) - F.reshape(E, 1)      # (E,1)

        Theta = Theta + damping * (innovation @ K.T)                # (E,D)

        # write back: ensemble mean
        theta_new = Theta.mean(dim=0)
        spatial_new, temporal_new = _unflatten_logits(theta_new, spatial_shape, temporal_shape)

        graph.spatial_logits = spatial_new
        graph.temporal_logits = temporal_new

        if (i_iter + 1) % imp_per_iters == 0:
            graph.update_masks(pruning_rate)

        #print("F (member mean utilities):", F_vals)
        print("F:", F)
        print(f"Iter time: {time.time() - t0:.3f}s")


def _gaussian_kl_diag(
    mu0: torch.Tensor, var0: torch.Tensor,
    mu1: torch.Tensor, var1: torch.Tensor,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    KL( N(mu0, diag(var0)) || N(mu1, diag(var1)) )
    mu*: (D,), var*: (D,)
    """
    var0 = var0.clamp_min(eps)
    var1 = var1.clamp_min(eps)
    D = mu0.numel()

    term_log = torch.sum(torch.log(var1) - torch.log(var0))
    term_tr = torch.sum(var0 / var1)
    term_quad = torch.sum((mu1 - mu0) ** 2 / var1)
    return 0.5 * (term_log + term_tr + term_quad - D)


async def train_eki_final(
    graph,
    dataset,
    task_indices,
    num_iters: int = 4,
    num_rounds: int = 1,
    batch_size: int = 2,
    ensemble_size: int = 6,
    prior_std: float = 0.2,
    obs_noise: float = 0.1,
    damping: float = 0.7,
    max_concurrency: int = 8,
    imp_per_iters: int = 1,      # 可选：每隔多少 iter 做一次 pruning
    pruning_rate: float = 0.05,  # 可选：pruning rate
) -> None:
    """
    Final training: sequential EKI updates across iters; write back ensemble mean to `graph`.

    - Deterministic task selection by provided indices (with wrap-around).
    - Updates Theta across iterations (carries over).
    - Writes back ensemble mean logits to `graph` each iteration.
    - Optionally prunes masks periodically.
    """

    if task_indices is None or len(task_indices) == 0:
        raise ValueError("task_indices is empty.")

    # fail-fast range check
    n_data = len(dataset)
    for idx in task_indices:
        if idx < 0 or idx >= n_data:
            raise IndexError(f"task_indices contains out-of-range index {idx} (dataset size={n_data}).")

    sem = asyncio.Semaphore(max_concurrency)

    # parameter vector theta0 (read only)
    spatial0 = graph.spatial_logits.detach().clone().float()
    temporal0 = graph.temporal_logits.detach().clone().float()
    spatial_shape, temporal_shape = spatial0.shape, temporal0.shape

    theta0 = _flatten_logits(spatial0, temporal0)
    D = theta0.numel()
    E = ensemble_size

    # init ensemble
    Theta = theta0[None, :].repeat(E, 1) + prior_std * torch.randn(E, D)

    # pointer for cycling through task_indices
    idx_ptr = 0
    n_idx = len(task_indices)

    def get_next_batch_records():
        """
        Select next batch_size records from dataset using task_indices in order.
        Works for any n_idx relative to batch_size (n_idx > batch_size or n_idx < batch_size).
        Always returns exactly batch_size records by wrap-around.
        """
        nonlocal idx_ptr
        chosen = []
        for _ in range(batch_size):
            data_idx = task_indices[idx_ptr % n_idx]
            chosen.append(dataset[data_idx])
            idx_ptr += 1
        return chosen

    for i_iter in range(num_iters):
        print(f"Iter {i_iter}", 80 * "-")
        t0 = time.time()

        batch_records = get_next_batch_records()

        async def _run_one(theta_e: torch.Tensor, record):
            # do NOT reuse graph across records (memory/outputs mutate)
            realized_graph = copy.deepcopy(graph)

            # EKI does not need autograd; also avoids building useless graphs in construct_*()
            if hasattr(realized_graph, "optimized_spatial"):
                realized_graph.optimized_spatial = False
            if hasattr(realized_graph, "optimized_temporal"):
                realized_graph.optimized_temporal = False

            spatial_e, temporal_e = _unflatten_logits(theta_e, spatial_shape, temporal_shape)
            realized_graph.spatial_logits = spatial_e.detach()
            realized_graph.temporal_logits = temporal_e.detach()

            input_dict = dataset.record_to_input(record)
            correct_answer = dataset.record_to_target_answer(record)

            async with sem:
                final_answers, _log_probs = await realized_graph.arun(input_dict, num_rounds)

            answer = dataset.postprocess_answer(final_answers)

            acc = Accuracy()
            acc.update(answer, correct_answer)
            return float(acc.get())  # 0/1

        # evaluate (E,B)
        tasks = []
        for e in range(E):
            for record in batch_records:
                tasks.append(asyncio.create_task(_run_one(Theta[e], record)))

        results = await asyncio.gather(*tasks)

        # reshape to (E, B)
        F_matrix = torch.tensor(results, dtype=torch.float32).reshape(E, batch_size)

        # scalar output per member: mean utility over batch => (E,)
        F_vals = F_matrix.mean(dim=1)

        # EKI update (Theta only; do NOT update graph)
        y = torch.tensor(1.0, dtype=torch.float32)  # target mean accuracy = 1

        with torch.no_grad():
            Theta_mean = Theta.mean(dim=0)                   # (D,)
            F_mean = F_vals.mean()                          # scalar

            Theta_A = Theta - Theta_mean[None, :]           # (E, D)
            F_A = (F_vals - F_mean).reshape(E, 1)           # (E, 1)

            C_thetaF = (Theta_A.T @ F_A) / (E - 1)          # (D, 1)
            C_FF = (F_A.T @ F_A) / (E - 1)                  # (1, 1)

            R = torch.tensor([[obs_noise ** 2]], dtype=torch.float32)  # (1,1)
            inv = torch.linalg.inv(C_FF + R)                           # (1,1)
            K = C_thetaF @ inv                                         # (D,1)

            eps = obs_noise * torch.randn(E, 1)
            innovation = (y + eps.reshape(E, 1)) - F_vals.reshape(E, 1)  # (E,1)
            Theta = Theta + damping * (innovation @ K.T)                 # (E,D)

        # write back: ensemble mean
        theta_new = Theta.mean(dim=0)
        spatial_new, temporal_new = _unflatten_logits(theta_new, spatial_shape, temporal_shape)

        graph.spatial_logits = spatial_new
        graph.temporal_logits = temporal_new

        if (i_iter + 1) % imp_per_iters == 0:
            graph.update_masks(pruning_rate)

        #print("F (member mean utilities):", F_vals)
        print("F_vals:", F_vals)
        print(f"Iter time: {time.time() - t0:.3f}s")

async def train_eki_final2(
    graph,
    dataset,
    task_indices,
    num_iters: int = 4,
    num_rounds: int = 1,
    batch_size: int = 2,
    ensemble_size: int = 6,
    prior_std: float = 0.2,
    obs_noise: float = 0.1,
    damping: float = 0.7,
    max_concurrency: int = 8,
    imp_per_iters: int = 1,
    pruning_rate: float = 0.25,
    clamp_min: float = 1e-20,   # 和 metrics 里一致
) -> None:
    """
    Final training (logprob version):
      - Deterministic task selection by provided indices (with wrap-around).
      - F is log p(correct) per record (from return_choice_logprob=True), averaged over batch.
      - Target y = 0.0.
      - Writes back ensemble mean logits to `graph` each iteration.
      - Optionally prunes masks periodically.
    """
    if task_indices is None or len(task_indices) == 0:
        raise ValueError("task_indices is empty.")

    n_data = len(dataset)
    for idx in task_indices:
        if idx < 0 or idx >= n_data:
            raise IndexError(f"task_indices contains out-of-range index {idx} (dataset size={n_data}).")

    sem = asyncio.Semaphore(max_concurrency)

    # theta0 + shapes
    spatial0 = graph.spatial_logits.detach().clone().float()
    temporal0 = graph.temporal_logits.detach().clone().float()
    spatial_shape, temporal_shape = spatial0.shape, temporal0.shape

    theta0 = _flatten_logits(spatial0, temporal0)
    D = theta0.numel()
    E = ensemble_size

    # init ensemble
    Theta = theta0[None, :].repeat(E, 1) + prior_std * torch.randn(E, D)

    # cycle through task_indices
    idx_ptr = 0
    n_idx = len(task_indices)

    def get_next_batch_records():
        nonlocal idx_ptr
        chosen = []
        for _ in range(batch_size):
            data_idx = task_indices[idx_ptr % n_idx]
            chosen.append(dataset[data_idx])
            idx_ptr += 1
        return chosen

    for i_iter in range(num_iters):
        print(f"Iter {i_iter}", 80 * "-")
        t0 = time.time()

        batch_records = get_next_batch_records()

        async def _run_one(theta_e: torch.Tensor, record):
            realized_graph = copy.deepcopy(graph)

            # EKI does not need autograd; avoid building useless graphs in construct_*()
            #if hasattr(realized_graph, "optimized_spatial"):
            #    realized_graph.optimized_spatial = False
            #if hasattr(realized_graph, "optimized_temporal"):
            #    realized_graph.optimized_temporal = False

            spatial_e, temporal_e = _unflatten_logits(theta_e, spatial_shape, temporal_shape)
            realized_graph.spatial_logits = spatial_e.detach()
            realized_graph.temporal_logits = temporal_e.detach()

            input_dict = dataset.record_to_input(record)
            correct_answer = dataset.record_to_target_answer(record)
            key = str(correct_answer).strip()

            async with sem:
                final_out, _log_probs = await realized_graph.arun(
                    input_dict,
                    num_rounds,
                    return_choice_logprob=True,
                )

            # keep same robustness as metrics version
            if isinstance(final_out, list):
                final_out0 = final_out[0] if len(final_out) > 0 else ""
            else:
                final_out0 = final_out

            abcd_probs = (final_out0.get("abcd_probs") or {}) if isinstance(final_out0, dict) else {}
            p_correct = float(
                abcd_probs.get(key,
                               abcd_probs.get(key.upper(),
                                              abcd_probs.get(key.lower(), 0.0)))
            )

            p_correct_t = torch.tensor(p_correct, dtype=torch.float32).clamp_min(clamp_min)
            return float(torch.log(p_correct_t).item())

        # evaluate (E,B) in parallel
        tasks = []
        for e in range(E):
            for record in batch_records:
                tasks.append(asyncio.create_task(_run_one(Theta[e], record)))

        results = await asyncio.gather(*tasks)

        # (E,B) -> per-member scalar (E,)
        F_matrix = torch.tensor(results, dtype=torch.float32).reshape(E, batch_size)
        F_vals = F_matrix.mean(dim=1)

        # EKI update
        y = torch.tensor(0.0, dtype=torch.float32)

        with torch.no_grad():
            Theta_mean = Theta.mean(dim=0)
            F_mean = F_vals.mean()

            Theta_A = Theta - Theta_mean[None, :]
            F_A = (F_vals - F_mean).reshape(E, 1)

            C_thetaF = (Theta_A.T @ F_A) / (E - 1)
            C_FF = (F_A.T @ F_A) / (E - 1)

            R = torch.tensor([[obs_noise ** 2]], dtype=torch.float32)
            inv = torch.linalg.inv(C_FF + R)
            K = C_thetaF @ inv

            eps = obs_noise * torch.randn(E, 1)
            innovation = (y + eps.reshape(E, 1)) - F_vals.reshape(E, 1)
            Theta = Theta + damping * (innovation @ K.T)

        # write back ensemble mean
        theta_new = Theta.mean(dim=0)
        spatial_new, temporal_new = _unflatten_logits(theta_new, spatial_shape, temporal_shape)

        graph.spatial_logits = spatial_new
        graph.temporal_logits = temporal_new

        if (i_iter + 1) % imp_per_iters == 0:
            graph.update_masks(pruning_rate)

        print("F_vals:", F_vals)
        print(f"Iter time: {time.time() - t0:.3f}s")


async def eki_metrics_per_task_one_step(
    graph,
    dataset,
    task_indices,
    num_rounds: int = 1,
    ensemble_size: int = 6,
    prior_std: float = 2.0,#0.2,
    obs_noise: float = 0.1,
    damping: float = 0.7,
    max_concurrency: int = 8,
) -> dict:
    """
    For each task index:
      - start from the SAME initial graph theta0 and SAME initial ensemble Theta_init
      - run exactly ONE EKI update step using ONLY that single task (batch_size=1)
      - compute metrics: KL(old||new), predictive variance of F across ensemble, cosine(mean_old, mean_new)

    Returns:
      {"kl":[...], "pred_var":[...], "cosine":[...]}
    """

    if task_indices is None or len(task_indices) == 0:
        raise ValueError("task_indices is empty.")

    n_data = len(dataset)
    for idx in task_indices:
        if idx < 0 or idx >= n_data:
            raise IndexError(f"task_indices contains out-of-range index {idx} (dataset size={n_data}).")

    sem = asyncio.Semaphore(max_concurrency)

    # fixed initial theta0 and shapes
    spatial0 = graph.spatial_logits.detach().clone().float()
    temporal0 = graph.temporal_logits.detach().clone().float()
    spatial_shape, temporal_shape = spatial0.shape, temporal0.shape

    theta0 = _flatten_logits(spatial0, temporal0)
    D = theta0.numel()
    E = ensemble_size

    # IMPORTANT: use the same initial ensemble for every task
    Theta_init = theta0[None, :].repeat(E, 1) + prior_std * torch.randn(E, D)

    kl_list = []
    pred_var_list = []
    cosine_list = []

    async def _run_one(theta_e: torch.Tensor, record):
        realized_graph = copy.deepcopy(graph)

        #if hasattr(realized_graph, "optimized_spatial"):
        #    realized_graph.optimized_spatial = False
        #if hasattr(realized_graph, "optimized_temporal"):
        #    realized_graph.optimized_temporal = False

        spatial_e, temporal_e = _unflatten_logits(theta_e, spatial_shape, temporal_shape)
        realized_graph.spatial_logits = spatial_e.detach()
        realized_graph.temporal_logits = temporal_e.detach()

        ## test ============
        #print("logits norm:", realized_graph.spatial_logits.norm().item(),
        #           realized_graph.temporal_logits.norm().item())


        input_dict = dataset.record_to_input(record)
        correct_answer = dataset.record_to_target_answer(record)

        async with sem:
            final_out, _log_probs = await realized_graph.arun(input_dict, num_rounds, return_choice_logprob=True,)


        if isinstance(final_out, list):
            final_out0 = final_out[0] if len(final_out) > 0 else ""
        else:
            final_out0 = final_out

        # final_out0 可能是 dict（return_choice_logprob=True 时）
        if isinstance(final_out0, dict):
            # 优先用明确的单字符 choice；否则用 text
            pred_str = final_out0.get("choice") or final_out0.get("text") or ""
        else:
            pred_str = final_out0

        predicted = dataset.postprocess_answer(pred_str)

        print("correct_answer raw:", repr(correct_answer), type(correct_answer))
        #print("final_out type:", type(final_out), "len:", len(final_out) if isinstance(final_out, list) else None)
        print("model answer", pred_str)


        #print('final_out', final_out)

        #p_correct = (final_out.get("abcd_probs") or {}).get(correct_answer, 0.0) if isinstance(final_out, dict) else 0.0
        abcd_probs = (final_out0.get("abcd_probs") or {}) if isinstance(final_out0, dict) else {}
        print('abcd_probs', abcd_probs)
        p_correct = float(abcd_probs.get(correct_answer, 0.0))
        print('p_correct', p_correct)

        # 3) log p_correct 作为 F（避免 log(0)）
        p_correct_t = torch.tensor(p_correct, dtype=torch.float32).clamp_min(1e-20)
        F_val = float(torch.log(p_correct_t).item())
        print('F_val', F_val)

        return F_val

        #answer = dataset.postprocess_answer(final_answers)
        #
        #acc = Accuracy()
        #acc.update(answer, correct_answer)
        #
        #
        ## test ============
        #print("correct:", correct_answer)
        #print("answer :", answer)
        #print("raw final_answers:", final_answers)
        #print("_log_probs:", _log_probs)
        #
        #
        #return float(acc.get())  # 0/1

    for idx in task_indices:
        record = dataset[idx]


        # test ============
        print("idx:", idx, "record_id:", id(record), "record_type:", type(record))
        if isinstance(record, dict):
            print("keys:", record.keys())
            for k in ["question", "prompt", "input", "subject", "id"]:
                if k in record:
                    print(k, str(record[k])[:120])


        # reset ensemble for this task
        Theta = Theta_init.detach().clone()

        ## test ============
        #print("Theta std(mean over dims):", Theta.std(dim=0).mean().item())
        #print("Theta[0][:5] vs Theta[1][:5]:", Theta[0, :5], Theta[1, :5])

        # evaluate F for this single task: total E runs
        tasks = [asyncio.create_task(_run_one(Theta[e], record)) for e in range(E)]

        results = await asyncio.gather(*tasks)

        F_vals = torch.tensor(results, dtype=torch.float32)  # (E,)

        # test ============
        print("results:", results)
        print("F_vals unique:", torch.unique(torch.tensor(results)))


        # predictive variance across ensemble
        pred_var = torch.var(F_vals, unbiased=True)
        pred_var_list.append(float(pred_var.item()))

        # old ensemble stats
        mu_old = Theta.mean(dim=0)
        var_old = Theta.var(dim=0, unbiased=True)

        # one-step EKI update
        y = torch.tensor(0.0, dtype=torch.float32)

        with torch.no_grad():
            Theta_mean = Theta.mean(dim=0)
            F_mean = F_vals.mean()

            Theta_A = Theta - Theta_mean[None, :]
            F_A = (F_vals - F_mean).reshape(E, 1)

            C_thetaF = (Theta_A.T @ F_A) / (E - 1)
            C_FF = (F_A.T @ F_A) / (E - 1)

            R = torch.tensor([[obs_noise ** 2]], dtype=torch.float32)
            inv = torch.linalg.inv(C_FF + R)
            K = C_thetaF @ inv

            eps = obs_noise * torch.randn(E, 1)
            innovation = (y + eps) - F_vals.reshape(E, 1)
            Theta = Theta + damping * (innovation @ K.T)

        # test ============
        print("C_FF:", C_FF.item())
        print("C_thetaF norm:", C_thetaF.norm().item())
        print("K norm:", K.norm().item())
        print("innovation std:", innovation.std().item())

        # new ensemble stats
        mu_new = Theta.mean(dim=0)
        var_new = Theta.var(dim=0, unbiased=True)

        # KL and cosine
        kl = _gaussian_kl_diag(mu_old, var_old, mu_new, var_new)
        kl_list.append(float(kl.item()))

        cosine = Fnn.cosine_similarity(mu_old.unsqueeze(0), mu_new.unsqueeze(0), dim=1)
        cosine_list.append(float(cosine.item()))

    return {"kl": kl_list, "pred_var": pred_var_list, "cosine": cosine_list}