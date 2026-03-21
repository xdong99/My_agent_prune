import asyncio
import copy
from typing import Any, List, Tuple

import torch
import torch.nn.functional as Fnn

from dataset.gsm8k_dataset import gsm_get_predict


def _flatten_logits(spatial: torch.Tensor, temporal: torch.Tensor) -> torch.Tensor:
    return torch.cat([spatial.reshape(-1), temporal.reshape(-1)], dim=0)


def _unflatten_logits(
    theta: torch.Tensor,
    spatial_shape: torch.Size,
    temporal_shape: torch.Size,
) -> Tuple[torch.Tensor, torch.Tensor]:
    s_numel = int(torch.tensor(spatial_shape).prod().item())
    spatial = theta[:s_numel].reshape(spatial_shape)
    temporal = theta[s_numel:].reshape(temporal_shape)
    return spatial, temporal


def _gaussian_kl_diag(
    mu0: torch.Tensor,
    var0: torch.Tensor,
    mu1: torch.Tensor,
    var1: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    KL( N(mu0, diag(var0)) || N(mu1, diag(var1)) )
    mu*: (D,), var*: (D,)
    """
    var0 = var0.clamp_min(eps)
    var1 = var1.clamp_min(eps)
    d = mu0.numel()

    term_log = torch.sum(torch.log(var1) - torch.log(var0))
    term_tr = torch.sum(var0 / var1)
    term_quad = torch.sum((mu1 - mu0) ** 2 / var1)
    return 0.5 * (term_log + term_tr + term_quad - d)


async def eki_metrics_per_task_one_step_give_task(
    graph,
    dataset,
    tasks: List[Any],
    num_rounds: int = 1,
    ensemble_size: int = 6,
    prior_std: float = 2.0,
    obs_noise: float = 0.1,
    damping: float = 0.7,
    max_concurrency: int = 8,
) -> dict:
    """
    GSM8K version of `eki_metrics_per_task_one_step_give_task`.

    Differences from the MMLU version:
      - does NOT require `dataset.record_to_input(...)`
      - does NOT require `dataset.record_to_target_answer(...)`
      - reads GSM8K records directly via:
            record["task"]
            record["answer"]
      - uses 0/1 correctness as F_val

    Expected record format:
      {
        "task": <question string>,
        "answer": <gold final numeric answer as string>,
        ...
      }

    Returns:
      {
        "kl": [...],
        "pred_var": [...],
        "cosine": [...],
        "var_list": [...]
      }

    Note:
      `dataset` is kept in the signature for compatibility with existing call sites,
      but is not used inside this GSM8K-specific implementation.
    """
    del dataset

    if tasks is None or len(tasks) == 0:
        raise ValueError("tasks is empty.")

    sem = asyncio.Semaphore(max_concurrency)

    spatial0 = graph.spatial_logits.detach().clone().float()
    temporal0 = graph.temporal_logits.detach().clone().float()
    spatial_shape, temporal_shape = spatial0.shape, temporal0.shape

    theta0 = _flatten_logits(spatial0, temporal0)
    d = theta0.numel()
    e_size = ensemble_size

    # Use the same initial ensemble for every task to make the per-task metrics comparable.
    theta_init = theta0[None, :].repeat(e_size, 1) + prior_std * torch.randn(e_size, d)

    async def _run_one(theta_e: torch.Tensor, record: Any) -> float:
        realized_graph = copy.deepcopy(graph)

        if hasattr(realized_graph, "optimized_spatial"):
            realized_graph.optimized_spatial = False
        if hasattr(realized_graph, "optimized_temporal"):
            realized_graph.optimized_temporal = False

        spatial_e, temporal_e = _unflatten_logits(theta_e, spatial_shape, temporal_shape)
        realized_graph.spatial_logits = spatial_e.detach()
        realized_graph.temporal_logits = temporal_e.detach()

        task = record["task"]
        correct_answer = str(record["answer"]).strip()
        input_dict = {"task": task}

        async with sem:
            final_out, _log_probs = await realized_graph.arun(input_dict, num_rounds)

        if isinstance(final_out, list):
            final_answer_text = final_out[0] if len(final_out) > 0 else ""
        else:
            final_answer_text = final_out

        if isinstance(final_answer_text, dict):
            # Fallback in case some decision methods return a dict-like payload.
            final_answer_text = (
                final_answer_text.get("answer")
                or final_answer_text.get("final_answer")
                or str(final_answer_text)
            )

        predicted_answer = gsm_get_predict(str(final_answer_text))

        try:
            is_solved = float(predicted_answer) == float(correct_answer)
        except Exception:
            is_solved = predicted_answer == correct_answer

        return float(is_solved)

    kl_list = []
    pred_var_list = []
    cosine_list = []
    var_list = []

    for record in tasks:
        theta = theta_init.detach().clone()

        jobs = [asyncio.create_task(_run_one(theta[i], record)) for i in range(e_size)]
        results = await asyncio.gather(*jobs)

        f_vals = torch.tensor(results, dtype=torch.float32)

        pred_var = torch.var(f_vals, unbiased=True)
        pred_var_list.append(float(pred_var.item()))

        mu_old = theta.mean(dim=0)
        var_old = theta.var(dim=0, unbiased=True)

        # For 0/1 utility we target perfect correctness.
        y = torch.tensor(1.0, dtype=torch.float32)

        with torch.no_grad():
            theta_mean = theta.mean(dim=0)
            f_mean = f_vals.mean()

            theta_a = theta - theta_mean[None, :]
            f_a = (f_vals - f_mean).reshape(e_size, 1)

            c_theta_f = (theta_a.T @ f_a) / (e_size - 1)
            c_ff = (f_a.T @ f_a) / (e_size - 1)

            r = torch.tensor([[obs_noise ** 2]], dtype=torch.float32)
            inv = torch.linalg.inv(c_ff + r)
            k_gain = c_theta_f @ inv

            eps = obs_noise * torch.randn(e_size, 1)
            innovation = (y + eps) - f_vals.reshape(e_size, 1)
            theta = theta + damping * (innovation @ k_gain.T)

        mu_new = theta.mean(dim=0)
        var_new = theta.var(dim=0, unbiased=True)
        var_list.append(float(var_new.sum().item()))

        kl = _gaussian_kl_diag(mu_old, var_old, mu_new, var_new)
        kl_list.append(float(kl.item()))

        cosine = Fnn.cosine_similarity(mu_old.unsqueeze(0), mu_new.unsqueeze(0), dim=1)
        cosine_list.append(float(cosine.item()))

    return {
        "kl": kl_list,
        "pred_var": pred_var_list,
        "cosine": cosine_list,
        "var_list": var_list,
    }
