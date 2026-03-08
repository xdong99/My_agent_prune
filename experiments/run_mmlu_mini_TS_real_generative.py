import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
from typing import Union, Literal, List, Any, Optional, Dict, Tuple
import argparse
import random

from AgentPrune.graph.graph import Graph
from dataset.mmlu_dataset import MMLUDataset
from dataset.MMLU.download import download
from experiments.evaluate_mmlu import evaluate
from AgentPrune.utils.const import AgentPrune_ROOT
from AgentPrune.utils.globals import PromptTokens, CompletionTokens

import torch
import pandas as pd
import numpy as np

from experiments.train_mmlu import train, train2, train3, train4
from experiments.train_mmlu_eki import eki_metrics_per_task_one_step, eki_metrics_per_task_one_step_give_task
from experiments.embedding_mmlu import select_mmlu_representatives, encode_mmlu_items
from experiments.metric_cache import MetricCache, eki_metrics_per_task_one_step_cached

from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel, Matern
import json
from pathlib import Path

from AgentPrune.llm.gpt_chat import GPTChat
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered','Star', 'Mesh',
                                 'FakeFullConnected','FakeRandom','FakeChain','FakeStar','FakeMesh','FakeAGRandom','FakeAGFull'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4,
                        help="batch size")
    parser.add_argument('--batch_size_train', type=int, default=4,
                        help="batch size")
    parser.add_argument('--batch_size_test', type=int, default=4,
                        help="batch size")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['AnalyzeAgent'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[5],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--num_iterations', type=int, default=10,
                        help="Number of optimization iterations. Default 10.")
    parser.add_argument('--imp_per_iterations', type=int, default=5,
                        help="Prune every few iterations. Default 5.")
    parser.add_argument('--num_rounds',type=int,default=1,
                        help="Number of optimization/inference rounds for one query")
    parser.add_argument('--return_choice_logprob',type=bool,default=True,
                        help="Return logprob for answer or not")

    parser.add_argument('--eki_iterations',type=int,default=4,
                        help="Number of eki iterations for one query")
    parser.add_argument('--ensemble_size',type=int,default=6,
                        help="Number of ensemble members")
    parser.add_argument('--batch_size_eki',type=int,default=2,
                        help="Number of task for eki")

    parser.add_argument('--pruning_rate', type=float, default=0.25,
                        help="The Rate of Pruning. Default 0.05.")
    #parser.add_argument('--llm_name', type=str, default="gpt-4-1106-preview",
    #                    help="Model name, None runs the default ChatGPT4")
    #parser.add_argument('--llm_name', type=str, default="gpt-5-nano",
    #                    help="Model name, None runs the gpt-5-nano")
    #parser.add_argument('--llm_name', type=str, default="gpt-4o-mini",
    #                    help="Model name, None runs ChatGPT4o-mini")
    parser.add_argument('--llm_task_generator_name', type=str, default="gpt-5",
                        help="Model name, None runs ChatGPT4.1-nano")
    parser.add_argument('--llm_name', type=str, default="gpt-4.1-nano",
                        help="Model name, None runs ChatGPT4.1-nano")
    #parser.add_argument('--llm_name', type=str, default="gemini-2.5-flash-preview-09-2025",
    #                    help="Model name, None runs the default gemini-2.5-flash-preview-09-2025")
    parser.add_argument('--domain', type=str, default="mmlu",
                        help="Domain (the same as dataset name), default 'MMLU'")
    parser.add_argument('--decision_method', type=str, default="FinalRefer",
                        help="the decision method of the final node")
    parser.add_argument('--optimized_spatial',action='store_true')
    parser.add_argument('--optimized_temporal',action='store_true')
    args = parser.parse_args()
    result_path = AgentPrune_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
        
    return args


def read_tokens():
    p = int(PromptTokens.instance().value)
    c = int(CompletionTokens.instance().value)
    return p, c


async def eval_and_cache_metrics(
    graph,
    dataset_train,
    task_indices,                 # List[int]
    ensemble_size=6,
    out_dir="outputs/metric_cache",
    tag="mmlu50_seed0",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 评估（一次性把全部 idx 丢进去）
    metrics = await eki_metrics_per_task_one_step(
        graph, dataset_train, task_indices, ensemble_size=ensemble_size
    )

    # 2) 组装成 per_task 表（强烈建议保存成表格，后面用起来最方便）
    per_task = []
    for i, idx in enumerate(task_indices):
        row = {"idx": int(idx)}
        # 把 metrics 里所有键都存下来（以后你扩指标不需要改逻辑）
        for k, v in metrics.items():
            # v 可能是 list / np.ndarray；确保可以用 v[i]
            row[k] = float(v[i]) if np.isscalar(v[i]) or np.ndim(v) == 1 else v[i]
        per_task.append(row)

    df = pd.DataFrame(per_task)

    # 3) 保存：推荐 “csv + npz + json(meta)” 三件套
    csv_path = out_dir / f"{tag}.csv"
    npz_path = out_dir / f"{tag}.npz"
    meta_path = out_dir / f"{tag}.meta.json"

    df.to_csv(csv_path, index=False)

    # npz 适合快速载入 numpy 数组
    save_dict = {"idx": df["idx"].to_numpy(dtype=np.int64)}
    for k in metrics.keys():
        # 统一保存为 float64 数组（你也可以改成 float32）
        save_dict[k] = df[k].to_numpy(dtype=np.float64)
    np.savez_compressed(npz_path, **save_dict)

    # meta 记录关键信息，方便以后确认缓存是否匹配
    meta = {
        "tag": tag,
        "n_tasks": len(task_indices),
        "ensemble_size": ensemble_size,
        "metric_keys": list(metrics.keys()),
        "idx_head": task_indices[:10],
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Saved:", csv_path, npz_path, meta_path)
    return df, metrics


def fit_pls_gp(X_all, y_obs, obs_pos, n_components=3, seed=0):
    """
    用当前已观测点 (obs_pos) 的 (X,y) 拟合：StandardScaler -> PLS -> GP
    返回：Z_all, gp, (mu,std)
    """
    X_obs = X_all[obs_pos]
    y_obs = np.asarray(y_obs, dtype=np.float64)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_obs_s = x_scaler.fit_transform(X_obs)
    y_t = np.arcsinh(y_obs)
    y_s = y_scaler.fit_transform(y_t.reshape(-1, 1)).ravel()

    pls = PLSRegression(n_components=n_components)
    pls.fit(X_obs_s, y_s)

    X_all_s = x_scaler.transform(X_all)
    Z_all = pls.transform(X_all_s)
    z_scaler = StandardScaler()
    Z_all_s = z_scaler.fit_transform(Z_all)
    Z_obs_s = Z_all_s[obs_pos]

    d = Z_obs_s.shape[1]

    #kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e3)) \
    #         + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0))
    #kernel = C(1.0, (1e-3, 1e3)) * Matern(
    #    length_scale=np.ones(d),
    #    length_scale_bounds=(1e-2, 1e3),
    #    nu=2.5
    #) + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0))
    #
    #gp = GaussianProcessRegressor(
    #    kernel=kernel,
    #    alpha=0.0,
    #    normalize_y=True,
    #    n_restarts_optimizer=5,
    #    random_state=seed,
    #)

    kernel = C(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(d),
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5
    ) + WhiteKernel(
        noise_level=1e-3,
        noise_level_bounds=(1e-6, 1e-1)
    )

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=1e-4,
        normalize_y=True,
        n_restarts_optimizer=3,
        random_state=seed,
    )

    gp.fit(Z_obs_s, y_s)

    mu, std = gp.predict(Z_all_s, return_std=True)
    return Z_all_s, gp, mu, std


def ts_select_batch(gp, Z_all, candidates_pos, batch_size=5, seed=0):
    """
    Thompson sampling: 从后验抽一条函数样本，在 candidates 里取样本值最大的前 batch_size 个
    """
    samples = gp.sample_y(Z_all, n_samples=1, random_state=seed).ravel()
    cand_scores = samples[candidates_pos]
    order = np.argsort(cand_scores)[::-1]                 # 降序
    pick = candidates_pos[order[:batch_size]]
    return pick, samples


def rank_remaining(mu, candidates_pos):
    """
    给剩余任务一个排序（这里用后验均值 mu；你也可以换成 UCB: mu + beta*std）
    """
    order = np.argsort(mu[candidates_pos])[::-1]
    return candidates_pos[order]


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

def batch_ts_topk(gp, Z_all, k=10, n_draws=200, seed=0):
    rng = np.random.default_rng(seed)
    counts = np.zeros(Z_all.shape[0], dtype=int)
    for d in range(n_draws):
        s = gp.sample_y(Z_all, n_samples=1, random_state=int(rng.integers(1e9))).ravel()
        topk = np.argsort(s)[::-1][:k]
        counts[topk] += 1
    # 频率最高的 k 个作为最终 top-k（也可输出概率）
    order = np.argsort(counts)[::-1]
    return order[:k], counts / n_draws

def count_common(a, b):
    return len(set(a) & set(b))

# ======================== extract information ===========

def export_tasks_by_index_with_subject(
    dataset,
    task_indices: Union[int, List[int]],
    *,
    include_input: bool = True,
    include_target: bool = True,
    include_raw_record: bool = True,) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Version that merges Version-1 advantages:
      - strict index validation (type + range)
      - robust record_to_jsonable conversion
      - safe try/except around record_to_input / record_to_target_answer
      - lifts common keys from record_raw to top-level so pretty works reliably
      - always provides item["pretty"]

    Plus:
      - adds subject, preferring dataset._total_df_with_subject.iloc[idx]["subject"] when available
        and falling back to record_raw["subject"] if present.
    """

    # ---- normalize indices ----
    if isinstance(task_indices, int):
        indices = [task_indices]
        single = True
    else:
        indices = list(task_indices)
        single = False

    if len(indices) == 0:
        raise ValueError("task_indices is empty.")

    n_data = len(dataset)
    for idx in indices:
        if not isinstance(idx, int):
            raise TypeError(f"Index must be int, got {type(idx)}: {idx}")
        if idx < 0 or idx >= n_data:
            raise IndexError(f"Index {idx} out of range (dataset size={n_data}).")

    # ---- helpers ----
    def record_to_jsonable(record: Any) -> Any:
        """Convert common pandas objects to plain python for printing/serialization."""
        if isinstance(record, pd.DataFrame):
            if len(record) == 1:
                return record.iloc[0].to_dict()
            return record.to_dict(orient="records")
        if isinstance(record, pd.Series):
            return record.to_dict()
        if isinstance(record, dict):
            return record
        if hasattr(record, "to_dict"):
            try:
                return record.to_dict()
            except Exception:
                pass
        return record

    has_record_to_input = hasattr(dataset, "record_to_input") and callable(getattr(dataset, "record_to_input"))
    has_record_to_target = hasattr(dataset, "record_to_target_answer") and callable(getattr(dataset, "record_to_target_answer"))
    has_postprocess = hasattr(dataset, "postprocess_answer") and callable(getattr(dataset, "postprocess_answer"))

    # ---- subject source (preferred) ----
    df_with_subject = getattr(dataset, "_total_df_with_subject", None)
    can_df_subject = (
        df_with_subject is not None
        and hasattr(df_with_subject, "iloc")
        and hasattr(df_with_subject, "columns")
        and ("subject" in df_with_subject.columns)
    )

    outputs: List[Dict[str, Any]] = []

    for idx in indices:
        record = dataset[int(idx)]
        rr = record_to_jsonable(record)

        item: Dict[str, Any] = {"index": int(idx)}

        # ---- subject (prefer df, fallback to record) ----
        subject = None
        if can_df_subject:
            try:
                subject = df_with_subject.iloc[int(idx)]["subject"]
            except Exception:
                subject = None
        if subject is None and isinstance(rr, dict) and "subject" in rr:
            subject = rr["subject"]
        item["subject"] = subject

        # ---- raw record ----
        if include_raw_record:
            item["record_raw"] = rr

        # ---- input ----
        if include_input:
            if not has_record_to_input:
                item["input_dict"] = None
                item["input_error"] = "dataset.record_to_input(record) not found"
            else:
                try:
                    item["input_dict"] = dataset.record_to_input(record)
                except Exception as e:
                    item["input_dict"] = None
                    item["input_error"] = f"{type(e).__name__}: {e}"

        # ---- target ----
        if include_target:
            if not has_record_to_target:
                item["target_answer"] = None
                item["target_error"] = "dataset.record_to_target_answer(record) not found"
            else:
                try:
                    target = dataset.record_to_target_answer(record)
                    item["target_answer"] = target
                    if not isinstance(target, str):
                        item["target_warning"] = f"Expected str target, got {type(target)}"
                except Exception as e:
                    item["target_answer"] = None
                    item["target_error"] = f"{type(e).__name__}: {e}"

        # ---- lift common fields to top-level for convenience + pretty ----
        # Important: do NOT override subject that we already set from df_with_subject.
        if isinstance(rr, dict):
            for k in ["category", "question", "prompt", "options", "choices", "answer", "id", "A", "B", "C", "D"]:
                if k in rr and k not in item:
                    item[k] = rr[k]
            if item.get("subject") is None and "subject" in rr:
                item["subject"] = rr["subject"]

        # ---- pretty view ----
        item["pretty"] = _pretty_task_view(item, has_postprocess=has_postprocess)

        outputs.append(item)

    return outputs[0] if single else outputs


def _pretty_task_view(item: Dict[str, Any], *, has_postprocess: bool) -> str:
    """
    Best-effort human readable dump.
    No dependency on your dataset schema, but will look better if record has question/options.
    """
    lines: List[str] = []
    lines.append(f"Index: {item.get('index')}")

    # Try to use extracted fields
    subject = item.get("subject")
    if subject is not None:
        lines.append(f"Subject: {subject}")

    question = item.get("question") or item.get("prompt")
    if question is not None:
        lines.append("Question:")
        lines.append(str(question))

    options = item.get("options") or item.get("choices")
    if options is not None:
        lines.append("Options:")
        if isinstance(options, dict):
            for k, v in options.items():
                lines.append(f"  {k}. {v}")
        elif isinstance(options, (list, tuple)):
            for i, v in enumerate(options):
                lines.append(f"  {i}. {v}")
        else:
            lines.append(str(options))

    target = item.get("target_answer")
    if target is not None:
        lines.append(f"Target: {target}")

    # Always include whether input_dict exists
    if item.get("input_dict") is not None:
        lines.append("Input_dict: (available)")
    else:
        err = item.get("input_error")
        if err:
            lines.append(f"Input_dict: (unavailable: {err})")
        else:
            lines.append("Input_dict: (unavailable)")

    return "\n".join(lines)

def _as_pretty_item_from_dataset_record(record, *, index=None, subject=None):
    # record: pd.Series or dict with keys question/A/B/C/D/correct_answer
    return {
        "index": index,
        "subject": subject,
        "question": record.get("question"),
        "options": {
            "A": record.get("A"),
            "B": record.get("B"),
            "C": record.get("C"),
            "D": record.get("D"),
        },
        "target_answer": record.get("correct_answer"),
        # 可选：也可以塞 input_dict，方便你知道能不能喂给 graph
        # "input_dict": dataset_train.record_to_input(record),
    }

def _as_pretty_item_from_exported_task(task_dict):
    # task_dict: export_tasks_by_index_with_subject(...) 的输出
    # 兼容你模板里的字段名：QUESTION_STEM / CHOICE_A...
    return {
        "index": task_dict.get("index"),
        "subject": task_dict.get("subject"),
        "question": task_dict.get("question") or task_dict.get("question_stem"),
        "options": {
            "A": task_dict.get("A") or task_dict.get("choice_a") or task_dict.get("CHOICE_A"),
            "B": task_dict.get("B") or task_dict.get("choice_b") or task_dict.get("CHOICE_B"),
            "C": task_dict.get("C") or task_dict.get("choice_c") or task_dict.get("CHOICE_C"),
            "D": task_dict.get("D") or task_dict.get("choice_d") or task_dict.get("CHOICE_D"),
        },
        "target_answer": task_dict.get("correct_answer") or task_dict.get("CORRECT_LETTER"),
    }
# ======================== generate text =================

PROMPT_TEMPLATE = """You are an expert exam item writer for MMLU-style multiple-choice questions.

Goal:
Given ONE MMLU-format seed question (known to be relatively high-information), you will:
(1) summarize in natural language why this seed question is relatively high-information,
(2) generate 10 NEW MMLU-format questions in the SAME SUBJECT that aim to be EVEN MORE high-information.

You are allowed to browse the web ONLY for factual verification.
- Do NOT copy any existing question stems or answer choices from any source you find.
- Do NOT paraphrase or lightly rewrite existing published questions.
- Use browsing only to confirm factual claims (definitions, dates, laws, scientific facts, terminology).
- If a detail cannot be verified confidently, rewrite the question to avoid that fragile fact.

Input (seed question):
Subject: {SUBJECT}
Question: {QUESTION_STEM}
Choices:
A) {CHOICE_A}
B) {CHOICE_B}
C) {CHOICE_C}
D) {CHOICE_D}
Correct answer: {CORRECT_LETTER}

Part 1: Why the seed is relatively high-information
Write 5–8 bullet points. Each bullet must be an actionable design factor, such as:
- "Distractors are near-miss alternatives that match surface cues but fail under a key condition."
- "Requires linking concept X and concept Y rather than recalling a single fact."
- "Targets misconception Z by including a tempting but wrong option."
Be specific to the seed, but do NOT fully solve the seed.

Part 2: Generate 10 higher-information questions (same subject)
For each new question:
- Keep the same subject: {SUBJECT}.
- Keep the MMLU style: one question + four choices (A–D) + exactly one correct answer.
- Make it novel: do not reuse the same scenario, named entities, or numbers from the seed.
- Increase information content by intentionally:
  (i) making at least one distractor a strong near-miss,
  (ii) requiring at least two linked concepts or reasoning steps,
  (iii) targeting a common misconception/failure mode,
  (iv) adding a subtle constraint that flips the conclusion compared with a naive reading.
- Keep length reasonable (avoid long passages).
- Provide a short explanation (2–4 sentences, <= 80 words) that:
  * justifies the correct choice, and
  * explains why the strongest distractor is tempting but wrong.

Mandatory self-check (do this silently before output):
For each generated item, verify:
1) There is exactly ONE best answer; the other three are clearly wrong under the intended interpretation.
2) No ambiguity or missing information.
3) Distractors are plausible and not obviously irrelevant.
4) The correct answer letter matches the content.
5) No copied wording from web sources; originality preserved.
6) Any factual claims are either verified or phrased in a robust, widely accepted way.

Output format (STRICT):
First print:
SEED HIGH-INFORMATION FACTORS:
- ...
- ...

Then print exactly 10 items, numbered 1–10, each using this exact structure:

[1]
Subject: {SUBJECT}
Question: ...
A) ...
B) ...
C) ...
D) ...
Correct answer: X
Explanation: ...

(Repeat for [2] ... [10])
"""

def _normalize_text(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    return " ".join(s.split())


def _extract_seed_fields(task: Dict[str, Any]) -> Tuple[str, str, str, str, str, str]:
    """
    Returns: (subject, question, A, B, C, D, correct_letter)
    Tries multiple schemas:
      - task top-level: subject/question/A/B/C/D
      - task["record_raw"] dict: subject/question/A/B/C/D/answer
      - task["target_answer"] for correct answer (letter or text)
    """
    rr = task.get("record_raw")
    rr0 = rr[0] if isinstance(rr, list) and rr and isinstance(rr[0], dict) else rr
    rr0 = rr0 if isinstance(rr0, dict) else {}

    subject = task.get("subject", None)
    if subject is None:
        subject = rr0.get("subject", None)
    subject = _normalize_text(subject)

    question = task.get("question", None) or task.get("prompt", None)
    if not question:
        question = rr0.get("question", None) or rr0.get("prompt", None)
    question = _normalize_text(question)

    # Choices: prefer MMLU A/B/C/D columns
    A = task.get("A", None) or rr0.get("A", None)
    B = task.get("B", None) or rr0.get("B", None)
    C = task.get("C", None) or rr0.get("C", None)
    D = task.get("D", None) or rr0.get("D", None)
    A, B, C, D = map(_normalize_text, (A, B, C, D))

    if not (A and B and C and D):
        # Fallback: options/choices could be dict or list
        opts = task.get("options") or task.get("choices") or rr0.get("options") or rr0.get("choices")
        if isinstance(opts, dict):
            A = A or _normalize_text(opts.get("A") or opts.get("a") or opts.get(0))
            B = B or _normalize_text(opts.get("B") or opts.get("b") or opts.get(1))
            C = C or _normalize_text(opts.get("C") or opts.get("c") or opts.get(2))
            D = D or _normalize_text(opts.get("D") or opts.get("d") or opts.get(3))
        elif isinstance(opts, (list, tuple)) and len(opts) >= 4:
            A = A or _normalize_text(opts[0])
            B = B or _normalize_text(opts[1])
            C = C or _normalize_text(opts[2])
            D = D or _normalize_text(opts[3])

    # Correct answer: try target_answer first, then rr answer
    target = task.get("target_answer", None)
    if target is None:
        target = rr0.get("answer", None)
    target = _normalize_text(target)

    # Derive correct letter
    correct_letter: Optional[str] = None
    if target in {"A", "B", "C", "D"}:
        correct_letter = target
    else:
        # If target is like "A)" or "Answer: B"
        t0 = target.replace("Answer:", "").replace("answer:", "").strip()
        if len(t0) >= 1 and t0[0].upper() in {"A", "B", "C", "D"} and (len(t0) == 1 or t0[1] in {")", ".", " "}):
            correct_letter = t0[0].upper()

    # If target is the full correct option text, match to A/B/C/D
    if correct_letter is None and target:
        choices = {"A": A, "B": B, "C": C, "D": D}
        # exact (normalized) match
        for k, v in choices.items():
            if v and _normalize_text(v) == _normalize_text(target):
                correct_letter = k
                break
        # substring match (last resort)
        if correct_letter is None:
            for k, v in choices.items():
                if v and _normalize_text(target) in _normalize_text(v):
                    correct_letter = k
                    break

    if correct_letter is None:
        raise ValueError(
            "Could not determine correct answer letter. "
            "Ensure task['target_answer'] is 'A'/'B'/'C'/'D' OR equals one of the choice texts, "
            "or record_raw contains an 'answer' field."
        )

    # Basic sanity
    if not subject:
        raise ValueError("Missing subject (task['subject'] or record_raw['subject']).")
    if not question:
        raise ValueError("Missing question stem (task['question']/'prompt' or record_raw['question'/'prompt']).")
    if not (A and B and C and D):
        raise ValueError("Missing choices A/B/C/D (task or record_raw).")

    return subject, question, A, B, C, D, correct_letter


def build_generation_prompt_from_task(
    task: Dict[str, Any],
    template: str = PROMPT_TEMPLATE,
) -> str:
    subject, q, A, B, C, D, correct = _extract_seed_fields(task)
    return template.format(
        SUBJECT=subject,
        QUESTION_STEM=q,
        CHOICE_A=A,
        CHOICE_B=B,
        CHOICE_C=C,
        CHOICE_D=D,
        CORRECT_LETTER=correct,
    )

_ITEM_RE = re.compile(r"^\[(\d+)\]\s*$", re.MULTILINE)

def parse_generated_mmlu_items(gen_text: str) -> List[Dict[str, Any]]:
    """
    Parse the LLM output into 10 MMLU-style items.
    Returns list of dicts: {subject, question, A,B,C,D, correct_answer, explanation}
    """
    # split by [k]
    hits = list(_ITEM_RE.finditer(gen_text))
    if len(hits) < 10:
        raise ValueError(f"Found only {len(hits)} item markers like [1]. Output format may be broken.")

    items = []
    for i in range(10):
        start = hits[i].end()
        end = hits[i + 1].start() if i + 1 < len(hits) else len(gen_text)
        block = gen_text[start:end].strip()

        def grab(prefix: str) -> str:
            m = re.search(rf"^{re.escape(prefix)}\s*(.*)$", block, flags=re.MULTILINE)
            return m.group(1).strip() if m else ""

        subject = grab("Subject:")
        question = grab("Question:")
        A = grab("A)")
        B = grab("B)")
        C = grab("C)")
        D = grab("D)")
        correct = grab("Correct answer:")
        explanation = grab("Explanation:")

        if not (subject and question and A and B and C and D and correct):
            raise ValueError(f"Item {i+1} missing fields. Got:\n{block}")

        correct = correct.strip()
        if correct not in {"A", "B", "C", "D"}:
            # 容错：比如输出成 "A." / "A)" / "Answer: A"
            c0 = correct.replace("Answer:", "").strip()
            if c0 and c0[0].upper() in {"A", "B", "C", "D"}:
                correct = c0[0].upper()
            else:
                raise ValueError(f"Item {i+1} bad correct answer field: {correct!r}")

        items.append({
            "subject": subject,
            "question": question,
            "A": A, "B": B, "C": C, "D": D,
            "correct_answer": correct,
            "explanation": explanation,
        })
    return items

def load_gen_text(path="generated_task.json") -> str:
    obj = json.load(open(path, "r", encoding="utf-8"))
    if "text" not in obj:
        raise KeyError("generated_task.json must contain key 'text'.")
    return obj["text"]

def gen_item_to_record(item: dict):
    # item: {"subject","question","A","B","C","D","correct_answer"...}
    return {
        "question": item["question"],
        "A": item["A"],
        "B": item["B"],
        "C": item["C"],
        "D": item["D"],
        "correct_answer": item["correct_answer"],
    }
# ======================== load pls and gp ===============
def fit_pls_gp_bundle(X_all, y_obs, obs_pos, n_components=3, seed=0):
    """
    在已观测点上拟合 x_scaler -> PLS -> z_scaler -> GP
    返回一个 bundle，里面包含所有可复用组件 + 对 X_all 的 (mu,std)
    """
    X_obs = X_all[obs_pos]
    y_obs = np.asarray(y_obs, dtype=np.float64)

    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_obs_s = x_scaler.fit_transform(X_obs)
    y_t = np.arcsinh(y_obs)
    y_s = y_scaler.fit_transform(y_t.reshape(-1, 1)).ravel()

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
        nu=2.5
    ) + WhiteKernel(
        noise_level=1e-3,
        noise_level_bounds=(1e-6, 1e-1)
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

    return {
        "x_scaler": x_scaler,
        "y_scaler": y_scaler,
        "pls": pls,
        "z_scaler": z_scaler,
        "gp": gp,
        "Z_all_s": Z_all_s,
        "mu_all": mu,
        "std_all": std,
    }

def predict_with_bundle(bundle, X_new):
    """
    用同一套 x_scaler/pls/z_scaler 把新点映射到 3D，然后用 gp 预测
    """
    x_scaler = bundle["x_scaler"]
    pls = bundle["pls"]
    z_scaler = bundle["z_scaler"]
    gp = bundle["gp"]

    X_new_s = x_scaler.transform(X_new)
    Z_new = pls.transform(X_new_s)
    Z_new_s = z_scaler.transform(Z_new)
    mu_new, std_new = gp.predict(Z_new_s, return_std=True)
    return Z_new_s, mu_new, std_new

# ======================== main ==========================

async def main():
    args = parse_args()
    
    mode = args.mode
    decision_method = args.decision_method
    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(mode,len(agent_names))
    limit_questions = 153
    
    graph = Graph(domain=args.domain,
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=decision_method,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal,
                  **kwargs)
    #print("roles:", [node.role for node in graph.nodes.values()])
    #print("roles_by_id:", {node_id: node.role for node_id, node in graph.nodes.items()})
    print("agent+role:", [(node.agent_name, node.role, node.id) for node in graph.nodes.values()])




    download()
    dataset_train = MMLUDataset('test')#dev')
    dataset_val = MMLUDataset('val')

    ## ======================== 生成新的任务 ===================================
    #task = export_tasks_by_index_with_subject(dataset_train, 1982)
    #print(task["pretty"])
    #print(task["record_raw"])
    #print(task["input_dict"])
    #print(task["target_answer"])
    #
    ### 多个任务（保持你给的顺序）
    ##tasks = export_tasks_by_index_with_subject(dataset_train, [3, 7, 12])
    ##for t in tasks:
    ##    print("=" * 40)
    ##    print(t["pretty"])
    #
    #prompt = build_generation_prompt_from_task(task)
    #print(prompt)
    #
    #
    #task_llm = GPTChat(model_name=args.llm_task_generator_name)
    #
    #gen_text = await task_llm.agen(
    #    messages=[{"role": "user", "content": prompt}],
    #    # 可选参数：temperature / max_tokens
    #)
    #
    #print("=== task generator output ===")
    #print(gen_text)
    #
    #breakpoint()

    ## ======================== 测试估计分数 ===================================
    #
    ## 1) load + parse 10 generated MMLU items
    #gen_text = load_gen_text("generated_task.json")
    #gen_items = parse_generated_mmlu_items(gen_text)  # list of 10 dicts with question/A/B/C/D...
    #
    ## 2) embed with the SAME encoder config as pool
    #X_new10 = encode_mmlu_items(
    #    gen_items,
    #    model="sentence-transformers/all-MiniLM-L6-v2",
    #    batch_size=64,
    #    normalize=True,
    #)  # (10,384)
    #
    ## 3) dry-run: random init y to fit PLS+GP (so the pipeline runs)
    #rng_tmp = np.random.default_rng(seed_number if "seed_number" in locals() else 0)
    #y_dummy = rng_tmp.standard_normal(size=(X_new10.shape[0],)).astype(np.float64)
    #obs_pos_dummy = np.arange(X_new10.shape[0], dtype=int)
    #
    #Z_new10, gp_new10, mu_new10, std_new10 = fit_pls_gp(
    #    X_new10, y_dummy, obs_pos_dummy, n_components=3, seed=0
    #)
    #
    #print("\n=== [Dry-run] Predicted scores for 10 generated tasks (meaningless values, pipeline sanity check) ===")
    #for i, it in enumerate(gen_items):
    #    print(f"[gen {i+1:02d}] mu={mu_new10[i]: .4f} std={std_new10[i]: .4f} | subject={it['subject']}")
    #
    #breakpoint()

    ## ========================= train4能不能用
    #
    ## 读 gen_text 并 parse 成 10 个 item dict（你已经有这两个函数）
    #gen_text = load_gen_text("generated_task.json")
    #gen_items = parse_generated_mmlu_items(gen_text)
    #
    ## 随便取 3 个做最小测试
    #picked = gen_items[:3]
    #
    ## 转成 record（dict 或 Series 均可）
    #external_records = [pd.Series(gen_item_to_record(it)) for it in picked]
    #
    ## 最小跑通：不需要 task_indices，只训练库外任务
    #await train4(
    #    graph,
    #    dataset_train,
    #    external_records=external_records,
    #    task_indices=None,
    #    num_iters=2,          # 先跑 1-2 次看看通不通
    #    num_rounds=args.num_rounds,
    #    lr=0.1,
    #    batch_size=2,
    #    imp_per_iters=1,
    #    pruning_rate=args.pruning_rate,
    #)

    # ===== Active learning, choosing tasks
    pn1, cn1 = read_tokens()
    #task_indices = [7816, 13422, 9658, 13025, 4173, 8510, 11980, 1521, 2052, 11223, 4, 2101, 11810, 2459, 1729, 9450, 9937, 321, 10269, 13551]
    # 选20个

    # 选40个
    #task_indices = [ 7816, 11690,  8731,  4683, 10101,    74, 12221, 11980,  4646,
    #    8226,  9076, 11223,  8358,  4061, 10614, 12353,  2600,  2801,
    #    6685, 12103,   889,  7994, 10324, 11810, 10268, 10103,  8608,
    #    8510, 10157,  2391, 13025, 12073,  8180,  7355,  2171, 11895,
    #   11260, 13551, 10824,  7873]

    # 选出前N个池

    seed_number = 1098 #2026 #42 #0 #123 #
    selected_idx, selected_emb = select_mmlu_representatives(split="test", sample_n=1000, seed=seed_number, budget=50, return_selected_embeddings=True)
    idx_all = np.array(selected_idx)              # (50,)
    X_all = np.asarray(selected_emb)#, dtype=np.float64)  # (50,384)
    
    # 测试：计算全部的数据
    #df50, metrics50 = await eval_and_cache_metrics(
    #    graph, dataset_train, selected_idx, ensemble_size=6,
    #    out_dir="outputs/metric_cache2",
    #    tag="selected50_es6"
    #)
    #breakpoint()

    # TS部分
    # 

    #m = cache.query(selected_idx, keys=["kl"])          # 取出这50个的KL（顺序与 selected_idx 对齐）
    #kl = np.asarray(m["kl"], dtype=np.float64)
    #
    #order = np.argsort(kl)[::-1]                        # KL 从大到小
    #top10_idx = np.asarray(selected_idx)[order[:10]].tolist()
    #print(top10_idx)
    #
    #breakpoint()

    # 随机选10个
    rng_1 = np.random.default_rng(seed_number)
    init_pos = rng_1.choice(len(idx_all), size=10, replace=False)
    #init_task_indices = idx_all[init_pos].tolist()
    obs_pos = list(init_pos)

    # 从缓存读初始 y
    init_task_indices = idx_all[init_pos].tolist()
    
    m0 = await eki_metrics_per_task_one_step(graph, dataset_train, init_task_indices, ensemble_size=6)
    y_obs = list(np.asarray(m0["kl"], dtype=np.float64))  # 与 obs_pos 对齐

    observed_mask = np.zeros(len(idx_all), dtype=bool)
    observed_mask[obs_pos] = True

    # 你想做几轮，每轮选 5 个
    num_rounds = 3
    batch_size = 5

    for t in range(num_rounds):
        # 1) 用当前观测拟合 PLS+GP，并得到全体的 mu/std
        Z_all, gp, mu, std = fit_pls_gp(X_all, y_obs, np.array(obs_pos), n_components=3, seed=t*10+1)

        # 2) candidates = 未观测
        candidates_pos = np.where(~observed_mask)[0]
        if len(candidates_pos) == 0:
            break

        # 3) TS 选本轮要评估的 5 个（在 candidates 内按 samples 取前5）
        pick_pos, samples = ts_select_batch(gp, Z_all, candidates_pos, batch_size=min(batch_size, len(candidates_pos)), seed=t*20+3)
        pick_idx = idx_all[pick_pos].tolist()

        # 4) 从缓存查询这 5 个的真实 KL，当作新观测加入
        #m_new = cache.query(pick_idx, keys=["kl"])
        #y_new = np.asarray(m_new["kl"], dtype=np.float64)
        
        m_new = await eki_metrics_per_task_one_step(graph, dataset_train, pick_idx, ensemble_size=6)
        y_new = np.asarray(m_new["kl"], dtype=np.float64)

        # 更新观测集合
        for p, yv in zip(pick_pos.tolist(), y_new.tolist()):
            obs_pos.append(int(p))
            y_obs.append(float(yv))
            observed_mask[p] = True

        # 5) 可选：输出本轮“剩余任务”的排序（用 mu）
        remaining_pos = np.where(~observed_mask)[0]
        if len(remaining_pos) > 0:
            ranked_pos = rank_remaining(mu, remaining_pos)
            ranked_idx = idx_all[ranked_pos].tolist()
            print(f"[Round {t}] picked idx:", pick_idx)
            print(f"[Round {t}] top-10 remaining by mean:", ranked_idx[:10])
        else:
            print(f"[Round {t}] picked idx:", pick_idx)
            print(f"[Round {t}] no remaining tasks.")

    Z_all, gp, mu, std = fit_pls_gp(X_all, y_obs, np.array(obs_pos), n_components=3, seed=t*30+5)
    #mu, std = gp.predict(Z_all, return_std=True)
    #order = np.argsort(mu)[::-1]
    #top10_pos = order[:10]
    #top10_idx = idx_all[top10_pos].tolist()
    top10_pos, prob = batch_ts_topk(gp, Z_all, k=10, n_draws=200, seed=t*50+15)
    top10_idx = idx_all[top10_pos].tolist()

    # ===== train
    print("Before train: graph")
    print(graph.spatial_logits.detach())
    print(graph.temporal_logits.detach())

    top10_idx_resort = interleave_ends(top10_idx)#_true)
    print("=== Chosen task resort ===", top10_idx_resort)
    print("=== Chosen task unresorted ===", top10_idx)

    
    pn2, cn2 = read_tokens()

    # ===================== NEW: replace worst-3 in top10 with generated tasks =====================

    # 0) fit a reusable (x_scaler + pls + z_scaler + gp) bundle from the observed pool tasks
    #    use EXACTLY the same X_all / y_obs / obs_pos you already used in TS
    bundle = fit_pls_gp_bundle(X_all, y_obs, np.array(obs_pos), n_components=3, seed=999)

    # 1) decide worst-3 within top10 (use mu on those positions)
    mu_top10 = mu[top10_pos]  # mu is from fit_pls_gp() right above
    worst3_local = np.argsort(mu_top10)[:3]          # indices in [0..9]
    worst3_idx = [top10_idx[i] for i in worst3_local]  # dataset indices
    print("=== top10 (idx, mu) ===", list(zip(top10_idx, mu_top10.tolist())))
    print("=== worst3_idx to replace ===", worst3_idx)

    external_records = []  # 3 out-of-pool tasks (pd.Series)
    kept_inpool_idx = [x for x in top10_idx if x not in set(worst3_idx)]  # 7 remaining

    task_llm = GPTChat(model_name=args.llm_task_generator_name)

    for old_idx in worst3_idx:
        # 2.1) LLM generates 10 tasks following the seed task old_idx
        seed_task = export_tasks_by_index_with_subject(dataset_train, int(old_idx))
        prompt = build_generation_prompt_from_task(seed_task)

        gen_text = await task_llm.agen(messages=[{"role": "user", "content": prompt}])
        gen_items = parse_generated_mmlu_items(gen_text)  # length=10

        # 2.2) embedding -> PLS+GP predict -> pick top-3 by predicted mean
        X_gen10 = encode_mmlu_items(
            gen_items,
            model="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=64,
            normalize=True,
        )  # (10,384)

        _, mu_gen10, std_gen10 = predict_with_bundle(bundle, X_gen10)
        top3_pos = np.argsort(mu_gen10)[::-1][:3]

        cand_records = [pd.Series(gen_item_to_record(gen_items[i])) for i in top3_pos]

        print(f"[gen] old_idx={old_idx} pred_top3(mu,std) =",
              [(float(mu_gen10[i]), float(std_gen10[i])) for i in top3_pos])

        # 2.3) EKI true scoring on these 3 candidates, pick the best
        m = await eki_metrics_per_task_one_step_give_task(
            graph=graph,
            dataset=dataset_train,
            tasks=cand_records,
            num_rounds=args.num_rounds,
            ensemble_size=args.ensemble_size,
        )
        true_scores = np.asarray(m["kl"], dtype=np.float64)  # you used KL as score elsewhere
        best_j = int(np.argmax(true_scores))
        best_record = cand_records[best_j]

        old_task_export = export_tasks_by_index_with_subject(dataset_train, int(old_idx))
        old_record = dataset_train[int(old_idx)]  # 原始 dataset record

        print("\n" + "="*90)
        print(f"[replace] old_idx={old_idx} best_j={best_j}")
        print(f"[replace] true_scores_top3={true_scores.tolist()}")

        print("\n--- OLD TASK (export view) ---")
        print(_pretty_task_view(_as_pretty_item_from_exported_task(old_task_export), has_postprocess=False))

        print("\n--- OLD TASK (dataset record view) ---")
        print(_pretty_task_view(_as_pretty_item_from_dataset_record(old_record, index=int(old_idx), subject=old_task_export.get("subject")), has_postprocess=False))

        print("\n--- NEW TASK (chosen) ---")
        print(_pretty_task_view(_as_pretty_item_from_dataset_record(best_record, index=None, subject=old_task_export.get("subject")), has_postprocess=False))

        print(f"[eki] old_idx={old_idx} true_scores_top3={true_scores.tolist()} best_j={best_j}")

        # 2.4) replace: store best out-of-pool record
        external_records.append(best_record)

    # After replacement, we still train on 10 tasks: 7 in-pool + 3 out-of-pool
    task_indices_for_train = interleave_ends(kept_inpool_idx)
    print("=== kept_inpool_idx (7) ===", task_indices_for_train)
    print("=== external_records (3) ready ===", len(external_records))

    #top10_idx_resort = [13035 if x == 13466 else x for x in top10_idx_resort]
    #top10_idx_resort = [13348 if x == 9937 else x for x in top10_idx_resort]
    #breakpoint()

    #top10_idx_resort = top10_idx_resort[:5] * 2
    #top10_idx_resort = top10_idx_resort[-5:] * 2

    p0, c0 = read_tokens()
    if args.optimized_spatial or args.optimized_temporal:
        #await train(graph=graph,dataset=dataset_train,num_iters=args.num_iterations,num_rounds=args.num_rounds,
        #            lr=args.lr,batch_size=args.batch_size_train,imp_per_iters=args.imp_per_iterations,pruning_rate=args.pruning_rate)

        #task_indices = [13551, 9450, 7816, 4, 11223, 10269, 1729, 11980, 2459, 4173]
        
        #random_list = random.sample(list(idx_all), 10)
        #print("random_list", random_list)
        #
        #await train2(graph=graph,dataset=dataset_train,
        #            #task_indices= random_list,
        #            task_indices= top10_idx_resort,
        #            num_iters=args.num_iterations,num_rounds=args.num_rounds,
        #            lr=args.lr,batch_size=args.batch_size_train,imp_per_iters=args.imp_per_iterations,pruning_rate=args.pruning_rate)

        await train4(
            graph=graph,
            dataset=dataset_train,
            task_indices=task_indices_for_train,     # 7 in-pool
            external_records=external_records,       # 3 out-of-pool
            num_iters=args.num_iterations,
            num_rounds=args.num_rounds,
            lr=args.lr,
            batch_size=args.batch_size_train,
            imp_per_iters=args.imp_per_iterations,
            pruning_rate=args.pruning_rate,
        )
        #await train3(graph=graph,dataset=dataset_train,num_iters=args.num_iterations,num_rounds=args.num_rounds,
        #            lr=args.lr,batch_size=args.batch_size_train,imp_per_iters=args.imp_per_iterations,pruning_rate=args.pruning_rate)
        #breakpoint()
    #
    print("After train: graph")
    print(graph.spatial_logits.detach())
    print(graph.temporal_logits.detach())

    p1, c1 = read_tokens()
    train_prompt = p1 - p0
    train_completion = c1 - c0

    p2, c2 = read_tokens()

    score = await evaluate(graph=graph,dataset=dataset_val,num_rounds=args.num_rounds,limit_questions=limit_questions,eval_batch_size=args.batch_size_test)
    
    p3, c3 = read_tokens()
    eval_prompt = p3 - p2
    eval_completion = c3 - c2
    #prompt = PromptTokens.instance().value
    #completion = CompletionTokens.instance().value
    #total = prompt + completion
    #print("Prompt tokens:", int(prompt))
    #print("Completion tokens:", int(completion))
    #print("Total tokens:", int(total))
    
    #print("=== Chosen task resorted ===", top10_idx_resort)
    print("=== final training tasks ===", task_indices_for_train)
    task_prompt = pn2 - pn1
    task_completion = cn2 - cn1
    print("=== Task evaluation tokens ===", task_prompt, task_completion, task_prompt + task_completion)
    task_prompt = p0 - pn2
    task_completion = c0 - cn2
    print("=== Task generation tokens ===", task_prompt, task_completion, task_prompt + task_completion)
    print("=== Train tokens ===", train_prompt, train_completion, train_prompt + train_completion)
    print("=== Eval tokens  ===", eval_prompt, eval_completion, eval_prompt + eval_completion)
    print("=== Total tokens ===", p3, c3, p3 + c3)
    print(f"Score: {score}")



def get_kwargs(mode:Union[Literal['DirectAnswer'],Literal['FullConnected'],Literal['Random'],Literal['Chain'],Literal['Debate'],Literal['Layered'],Literal['Star'],Literal['Mesh'],
                          Literal['FakeFullConnected'],Literal['FakeRandom'],Literal['FakeChain'],Literal['FakeStar'],Literal['FakeMesh'],Literal['FakeAGRandom'],Literal['FakeAGFull']],
               N:int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks:List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks:List[List[int]] = None
    node_kwargs = None
    
    def generate_layered_graph(N,layer_num=2):
        adj_matrix = [[0]*N for _ in range(N)]
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
    
    def generate_mesh_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(0, N):
            for j in range(i+1,N):
                adj_matrix[i][j] = 1
        return adj_matrix
    
    def generate_star_graph(N):
        adj_matrix = [[0] * N for _ in range(N)]
        for i in range(1,N):
            adj_matrix[0][i] = 1
        return adj_matrix
    
    if mode=='DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role':'Normal'}]
    elif mode=='FullConnected' or mode == 'FakeFullConnected' or mode=='FakeAGFull':
        fixed_spatial_masks = [[1 if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode=='Random' or mode == 'FakeRandom' or mode == 'FakeAGRandom':
        fixed_spatial_masks = [[random.randint(0, 1)  if i!=j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode=='Chain' or mode == 'FakeChain':
        fixed_spatial_masks = [[1 if i==j+1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i==0 and j==N-1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Mesh' or mode=='FakeMesh':
        fixed_spatial_masks = generate_mesh_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    elif mode == 'Star' or mode=='FakeStar':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for i in range(N)] for j in range(N)]
    
    if 'Fake' in mode and 'AG' not in mode:
        node_kwargs = [{'role':'Fake'} if i % 2 == N % 2 else {'role':'Normal'} for i in range(N)]
    elif 'Fake' in mode and 'AG' in mode:
        node_kwargs = [{'role':'Fake'} if i % 2 == N % 2 else {'role':None} for i in range(N)]
        
    return {"initial_spatial_probability": initial_spatial_probability,
            "fixed_spatial_masks": fixed_spatial_masks,
            "initial_temporal_probability": initial_temporal_probability,
            "fixed_temporal_masks": fixed_temporal_masks,
            "node_kwargs":node_kwargs}    

def summarize_graph(graph, tag: str = ""):
    with torch.no_grad():
        print(f"\n===== {tag} =====")
        if hasattr(graph, "spatial_logits") and graph.spatial_logits is not None:
            s = graph.spatial_logits.detach().float().cpu()
            print(f"spatial_logits: shape={tuple(s.shape)} "
                  f"min={s.min().item():.6e} max={s.max().item():.6e} mean={s.mean().item():.6e} "
                  f"sigmoid_mean={torch.sigmoid(s).mean().item():.6e} "
                  f"requires_grad={graph.spatial_logits.requires_grad}")
        else:
            print("spatial_logits: <missing>")

        if hasattr(graph, "temporal_logits") and graph.temporal_logits is not None:
            t = graph.temporal_logits.detach().float().cpu()
            print(f"temporal_logits: shape={tuple(t.shape)} "
                  f"min={t.min().item():.6e} max={t.max().item():.6e} mean={t.mean().item():.6e} "
                  f"sigmoid_mean={torch.sigmoid(t).mean().item():.6e} "
                  f"requires_grad={graph.temporal_logits.requires_grad}")
        else:
            print("temporal_logits: <missing>")

        # 可选：如果有 mask，就顺便打印稀疏度
        if hasattr(graph, "spatial_mask"):
            sm = getattr(graph, "spatial_mask")
            if sm is not None:
                sm = sm.detach().cpu()
                print(f"spatial_mask: active={int(sm.sum().item())}/{sm.numel()}")

        if hasattr(graph, "temporal_mask"):
            tm = getattr(graph, "temporal_mask")
            if tm is not None:
                tm = tm.detach().cpu()
                print(f"temporal_mask: active={int(tm.sum().item())}/{tm.numel()}")


if __name__ == "__main__":
    asyncio.run(main())
