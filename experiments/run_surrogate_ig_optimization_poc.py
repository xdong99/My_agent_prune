import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import asdict, dataclass
from fractions import Fraction
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import aiohttp
import numpy as np
import pandas as pd
import torch
from openai import AsyncOpenAI
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from AgentPrune.graph.graph import Graph
from AgentPrune.llm.gpt_chat import _extract_usage_tokens
from AgentPrune.utils.globals import CompletionTokens, PromptTokens
from experiments.run_gsm8k_generated_task_poc import build_graph, eki_score_tasks


SPEC_SPACE: dict[str, list[str]] = {
    "task_family": [
        "percent_base",
        "average_inverse",
        "capacity_ceiling",
        "reverse_fraction",
        "defective_rate",
        "overlap_count",
        "coupon_tax",
        "lcm_meeting",
        "growth_difference",
    ],
    "reasoning_depth": ["1", "2", "3"],
    "distractor_type": [
        "none",
        "wrong_base",
        "wrong_denominator",
        "off_by_one",
        "irrelevant_extra",
    ],
    "numeric_scale": ["small", "medium", "large"],
    "difficulty_band": ["easy", "medium", "hard"],
    "answer_magnitude": ["one_digit", "two_digit", "three_digit"],
}
FIELD_ORDER = list(SPEC_SPACE.keys())


@dataclass(frozen=True)
class TaskSpec:
    task_family: str
    reasoning_depth: str
    distractor_type: str
    numeric_scale: str
    difficulty_band: str
    answer_magnitude: str


class IGSurrogate(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Surrogate-guided task-spec optimization for high EKI/KL graph IG."
    )
    parser.add_argument("--metadata", default="outputs/generated_task_poc/run_20260601_180545/metadata.json")
    parser.add_argument("--output_dir", default="outputs/surrogate_ig_poc")
    parser.add_argument("--llm_name", default=None)
    parser.add_argument("--renderer", default="template", choices=["template", "llm"])
    parser.add_argument("--generator_model", default="gpt-4.1-nano")
    parser.add_argument("--generator_conditioning", default="freeform", choices=["freeform", "canonical"])
    parser.add_argument("--generator_validation_model", default=None)
    parser.add_argument("--generator_temperature", type=float, default=0.7)
    parser.add_argument("--generator_max_tokens", type=int, default=700)
    parser.add_argument("--generator_concurrency", type=int, default=4)
    parser.add_argument("--generator_retries", type=int, default=3)
    parser.add_argument("--generator_timeout", type=float, default=float(os.getenv("OPENAI_REQUEST_TIMEOUT", "60")))
    parser.add_argument("--disable_generator_validation", action="store_true")
    parser.add_argument("--allow_generator_answer_correction", action="store_true")
    parser.add_argument("--fresh_graph", action="store_true")
    parser.add_argument("--num_agents", type=int, default=4)
    parser.add_argument("--num_rounds", type=int, default=2)
    parser.add_argument("--mode", default="FullConnected", choices=["FullConnected", "Chain", "Star"])
    parser.add_argument("--decision_method", default="FinalRefer")
    parser.add_argument("--variant_name", default="al_pruned")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--initial_specs", type=int, default=30)
    parser.add_argument(
        "--initial_data_dir",
        default=None,
        help="Optional run directory or specs_initial.jsonl file with precomputed real_ig labels.",
    )
    parser.add_argument("--initial_only", action="store_true")
    parser.add_argument("--initial_batch_size", type=int, default=0)
    parser.add_argument("--spec_sampling", default="uniform", choices=["uniform", "informative"])
    parser.add_argument("--random_spec_sampling", default=None, choices=["uniform", "informative"])
    parser.add_argument("--pair_count", type=int, default=6)
    parser.add_argument("--random_baseline_count", type=int, default=None)
    parser.add_argument("--optimize_starts", type=int, default=32)
    parser.add_argument("--discrete_topk", type=int, default=16)
    parser.add_argument("--ensemble_size", type=int, default=2)
    parser.add_argument("--ig_repeats", type=int, default=1)
    parser.add_argument("--verification_ensemble_size", type=int, default=None)
    parser.add_argument("--verification_repeats", type=int, default=1)
    parser.add_argument("--prior_std", type=float, default=2.0)
    parser.add_argument("--obs_noise", type=float, default=0.1)
    parser.add_argument("--damping", type=float, default=0.7)
    parser.add_argument("--max_concurrency", type=int, default=4)
    parser.add_argument("--ig_observation", default="binary_correct", choices=["binary_correct", "soft_numeric"])
    parser.add_argument("--surrogate_epochs", type=int, default=500)
    parser.add_argument("--surrogate_lr", type=float, default=0.01)
    parser.add_argument("--surrogate_device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--opt_steps", type=int, default=160)
    parser.add_argument("--opt_lr", type=float, default=0.12)
    parser.add_argument("--drift_penalty", type=float, default=0.01)
    parser.add_argument("--trust_region_hamming", type=int, default=None)
    parser.add_argument("--aggregate_mode", default=None, choices=["all connected", "last connected"])
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_tokens() -> tuple[float, float]:
    return float(PromptTokens.instance().value), float(CompletionTokens.instance().value)


def token_delta(before: tuple[float, float], after: tuple[float, float]) -> float:
    return float((after[0] - before[0]) + (after[1] - before[1]))


def spec_to_key(spec: TaskSpec) -> str:
    return "|".join(str(getattr(spec, field)) for field in FIELD_ORDER)


def spec_hamming_distance(left: TaskSpec, right: TaskSpec) -> int:
    return sum(int(getattr(left, field) != getattr(right, field)) for field in FIELD_ORDER)


def spec_rng(spec: TaskSpec, seed: int) -> random.Random:
    digest = hashlib.sha256(f"{seed}|{spec_to_key(spec)}".encode("utf-8")).hexdigest()
    return random.Random(int(digest[:16], 16))


INFORMATIVE_SPEC_WEIGHTS: dict[str, dict[str, float]] = {
    "task_family": {
        "percent_base": 1.2,
        "average_inverse": 1.4,
        "capacity_ceiling": 1.1,
        "reverse_fraction": 1.5,
        "defective_rate": 1.3,
        "overlap_count": 1.5,
        "coupon_tax": 1.4,
        "lcm_meeting": 1.3,
        "growth_difference": 0.8,
    },
    "reasoning_depth": {"1": 0.4, "2": 1.1, "3": 1.7},
    "distractor_type": {
        "none": 0.15,
        "wrong_base": 1.2,
        "wrong_denominator": 1.4,
        "off_by_one": 1.3,
        "irrelevant_extra": 0.8,
    },
    "numeric_scale": {"small": 0.35, "medium": 1.0, "large": 1.5},
    "difficulty_band": {"easy": 0.2, "medium": 1.0, "hard": 1.8},
    "answer_magnitude": {"one_digit": 0.45, "two_digit": 1.1, "three_digit": 1.4},
}


def weighted_choice(rng: random.Random, values: list[str], weights_by_value: dict[str, float]) -> str:
    weights = [float(weights_by_value.get(value, 1.0)) for value in values]
    return rng.choices(values, weights=weights, k=1)[0]


def sample_spec(rng: random.Random, sampling_mode: str = "uniform") -> TaskSpec:
    if sampling_mode == "uniform":
        return TaskSpec(**{field: rng.choice(values) for field, values in SPEC_SPACE.items()})
    if sampling_mode == "informative":
        return TaskSpec(
            **{
                field: weighted_choice(rng, values, INFORMATIVE_SPEC_WEIGHTS[field])
                for field, values in SPEC_SPACE.items()
            }
        )
    raise ValueError(f"unknown spec sampling mode: {sampling_mode}")


def all_specs() -> list[TaskSpec]:
    specs = []
    for values in product(*[SPEC_SPACE[field] for field in FIELD_ORDER]):
        specs.append(TaskSpec(**dict(zip(FIELD_ORDER, values))))
    return specs


def unique_specs(specs: list[TaskSpec]) -> list[TaskSpec]:
    seen = set()
    unique = []
    for spec in specs:
        key = spec_to_key(spec)
        if key in seen:
            continue
        seen.add(key)
        unique.append(spec)
    return unique


def encode_spec(spec: TaskSpec) -> np.ndarray:
    parts = []
    for field in FIELD_ORDER:
        values = SPEC_SPACE[field]
        one = np.zeros(len(values), dtype=np.float32)
        one[values.index(getattr(spec, field))] = 1.0
        parts.append(one)
    return np.concatenate(parts, axis=0)


def decode_onehot(vec: np.ndarray) -> TaskSpec:
    offset = 0
    kwargs = {}
    for field in FIELD_ORDER:
        values = SPEC_SPACE[field]
        chunk = vec[offset : offset + len(values)]
        kwargs[field] = values[int(np.argmax(chunk))]
        offset += len(values)
    return TaskSpec(**kwargs)


def input_dim() -> int:
    return sum(len(values) for values in SPEC_SPACE.values())


def resolve_torch_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("surrogate_device=cuda requested but CUDA is not available")
    return torch.device(device_arg)


def scale_value(spec: TaskSpec, small: int, medium: int, large: int) -> int:
    return {"small": small, "medium": medium, "large": large}[spec.numeric_scale]


def format_numeric_answer(answer: Any) -> str:
    if isinstance(answer, Fraction):
        if answer.denominator == 1:
            return str(answer.numerator)
        value = answer.numerator / answer.denominator
    else:
        value = float(answer)
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.10f}".rstrip("0").rstrip(".")


def add_distractor_sentence(task: str, spec: TaskSpec) -> str:
    extra = {
        "none": "",
        "wrong_base": " Be careful: the percentage or rate refers to the explicitly named base, not the later total.",
        "wrong_denominator": " Be careful: any fraction is taken from the amount remaining at that step.",
        "off_by_one": " Count endpoints only if the story explicitly says to count the starting event.",
        "irrelevant_extra": " One extra detail is included but does not change the requested quantity.",
    }[spec.distractor_type]
    return task + extra


def render_spec(spec: TaskSpec, seed: int, idx: int) -> dict[str, Any]:
    rng = spec_rng(spec, seed + idx * 997)
    family = spec.task_family
    if family == "percent_base":
        purchase = scale_value(spec, 80, 160, 280)
        pct = {"easy": 50, "medium": 75, "hard": 125}[spec.difficulty_band]
        repair = {"one_digit": 5, "two_digit": 20, "three_digit": 40}[spec.answer_magnitude]
        market = purchase + purchase * pct // 100
        answer = market - purchase - repair
        task = (
            f"A shop bought a used machine for ${purchase} and spent ${repair} repairing it. "
            f"After repairs, the machine's value was {pct}% higher than the original purchase price. "
            f"The shop sold it for that value. After subtracting both purchase and repair costs, "
            f"how many dollars of profit did it make?"
        )
    elif family == "average_inverse":
        n = {"1": 3, "2": 5, "3": 7}[spec.reasoning_depth]
        old_avg = scale_value(spec, 60, 70, 80)
        bump = {"easy": 3, "medium": 5, "hard": 8}[spec.difficulty_band]
        new_avg = old_avg + bump
        answer = (n + 1) * new_avg - n * old_avg
        task = (
            f"The average score of {n} quizzes is {old_avg}. After one more quiz is added, "
            f"the average becomes {new_avg}. What was the score on the added quiz?"
        )
    elif family == "capacity_ceiling":
        total = scale_value(spec, 95, 240, 520) + rng.choice([0, 5, 10])
        rate = {"easy": 5, "medium": 4, "hard": 3}[spec.difficulty_band]
        day_limit = {"one_digit": 40, "two_digit": 70, "three_digit": 95}[spec.answer_magnitude]
        setup = {"1": 5, "2": 10, "3": 15}[spec.reasoning_depth]
        capacity = max(1, (day_limit - setup) * rate)
        answer = math.ceil(total / capacity)
        task = (
            f"A team must inspect {total} boxes. Each workday they spend {setup} minutes setting up, "
            f"then inspect {rate} boxes per minute. They can work at most {day_limit} minutes per day including setup. "
            f"What is the minimum number of workdays needed?"
        )
    elif family == "reverse_fraction":
        denom = {"easy": 2, "medium": 3, "hard": 4}[spec.difficulty_band]
        start = scale_value(spec, 24, 72, 180)
        if start % denom != 0:
            start += denom - (start % denom)
        added = {"one_digit": 4, "two_digit": 10, "three_digit": 25}[spec.answer_magnitude]
        if spec.reasoning_depth == "1":
            final = start * (denom - 1) // denom
            task = f"Jia had some stickers and gave away 1/{denom} of them. She then had {final}. How many did she have at first?"
        else:
            final = start * (denom - 1) // denom + added
            task = (
                f"Jia had some stickers. She gave away 1/{denom} of them, then bought {added} more. "
                f"She then had {final} stickers. How many stickers did Jia have at the start?"
            )
        answer = start
    elif family == "defective_rate":
        a_rate = scale_value(spec, 8, 12, 18)
        b_delta = {"easy": 2, "medium": 4, "hard": 6}[spec.difficulty_band]
        b_rate = a_rate + b_delta
        a_hours = {"1": 2, "2": 3, "3": 4}[spec.reasoning_depth]
        b_hours = a_hours + 1
        pct = {"one_digit": 10, "two_digit": 20, "three_digit": 25}[spec.answer_magnitude]
        chosen = None
        for da in range(25):
            for db in range(25):
                cand_a_rate = a_rate + da
                cand_b_rate = cand_a_rate + b_delta + db
                total = cand_a_rate * a_hours + cand_b_rate * b_hours
                if (total * (100 - pct)) % 100 == 0:
                    chosen = (cand_a_rate, cand_b_rate, total)
                    break
            if chosen is not None:
                break
        if chosen is None:
            raise RuntimeError(f"could not render exact defective_rate task for {spec_to_key(spec)}")
        a_rate, b_rate, total = chosen
        answer = total * (100 - pct) // 100
        task = (
            f"Machine A makes {a_rate} parts per hour for {a_hours} hours. Machine B makes {b_rate} parts per hour "
            f"for {b_hours} hours. After production, {pct}% of all parts are discarded. How many good parts remain?"
        )
    elif family == "overlap_count":
        both = {"easy": 4, "medium": 8, "hard": 13}[spec.difficulty_band]
        soccer_only = scale_value(spec, 7, 15, 35)
        chess_only = soccer_only + {"1": 2, "2": 5, "3": 9}[spec.reasoning_depth]
        neither = {"one_digit": 3, "two_digit": 10, "three_digit": 22}[spec.answer_magnitude]
        soccer = soccer_only + both
        chess = chess_only + both
        answer = soccer + chess - both + neither
        task = (
            f"In a club, {soccer} students play soccer, {chess} students play chess, and {both} students do both. "
            f"There are also {neither} students who do neither activity. How many students are in the club altogether?"
        )
    elif family == "coupon_tax":
        price = scale_value(spec, 80, 180, 360)
        coupon = {"easy": 10, "medium": 20, "hard": 25}[spec.difficulty_band]
        tax = {"1": 5, "2": 10, "3": 20}[spec.reasoning_depth]
        gift = {"one_digit": 5, "two_digit": 20, "three_digit": 60}[spec.answer_magnitude]
        while True:
            discounted = Fraction(price * (100 - coupon), 100)
            taxed = discounted * Fraction(100 + tax, 100)
            answer = taxed - gift
            if answer.denominator == 1:
                break
            price += 5
        task = (
            f"A jacket costs ${price}. A {coupon}% coupon is applied first. Then a {tax}% sales tax is charged "
            f"on the discounted price. Finally, Mina uses a ${gift} gift card. How many dollars does she still pay?"
        )
    elif family == "lcm_meeting":
        a = {"easy": 4, "medium": 6, "hard": 8}[spec.difficulty_band]
        b = {"1": 6, "2": 9, "3": 10}[spec.reasoning_depth]
        horizon = scale_value(spec, 24, 60, 120)
        lcm = abs(a * b) // math.gcd(a, b)
        answer = horizon // lcm + 1
        task = (
            f"Two reminder lights flash together at noon today. One flashes every {a} days and the other every {b} days. "
            f"Counting flash events from day 0 through day {horizon} inclusive, how many times will they flash together?"
        )
    elif family == "growth_difference":
        a0 = scale_value(spec, 40, 100, 220)
        offset = {"one_digit": 5, "two_digit": 20, "three_digit": 60}[spec.answer_magnitude]
        while a0 <= offset:
            a0 += 40
        b0 = a0 - offset
        a_pct = {"easy": 25, "medium": 50, "hard": 75}[spec.difficulty_band]
        b_pct = {"1": 20, "2": 40, "3": 60}[spec.reasoning_depth]
        a_final = a0 + a0 * a_pct // 100
        b_final = b0 + b0 * b_pct // 100
        answer = abs(a_final - b_final)
        task = (
            f"Store A started with {a0} orders and increased by {a_pct}%. Store B started with {b0} orders "
            f"and increased by {b_pct}%. How many more orders did the busier store have?"
        )
    else:
        raise ValueError(f"unknown family: {family}")

    task = add_distractor_sentence(task, spec)
    return {
        "task": task,
        "answer": format_numeric_answer(answer),
        "family": family,
        "spec": asdict(spec),
        "validation_status": "valid",
        "renderer": "template",
    }


def task_family_guidance(task_family: str) -> str:
    return {
        "percent_base": "A percent-change problem where the key base quantity must be interpreted correctly.",
        "average_inverse": "An inverse-average problem where the missing value is recovered from old and new averages.",
        "capacity_ceiling": "A capacity/scheduling problem where a minimum number of days, trips, or batches is needed.",
        "reverse_fraction": "A reverse-fraction problem where the original amount is inferred after losing or gaining a fraction.",
        "defective_rate": "A production/defect-rate problem combining two production sources and a discarded percentage.",
        "overlap_count": "A set-overlap/inclusion-exclusion counting problem.",
        "coupon_tax": "A coupon/tax/payment-order problem where the order of operations matters.",
        "lcm_meeting": "A least-common-multiple or periodic-meeting counting problem.",
        "growth_difference": "A two-entity percentage-growth comparison problem.",
    }[task_family]


def task_distractor_guidance(distractor_type: str) -> str:
    return {
        "none": "Do not include any extra distractor sentence.",
        "wrong_base": "Include one natural detail that tempts using the wrong base quantity.",
        "wrong_denominator": "Include one natural detail that tempts using the wrong denominator or remaining amount.",
        "off_by_one": "Include one natural detail that tempts an endpoint/counting mistake.",
        "irrelevant_extra": "Include one irrelevant but plausible detail that does not affect the answer.",
    }[distractor_type]


def spec_generation_prompt(
    spec: TaskSpec,
    seed: int,
    idx: int,
    conditioning: str = "freeform",
    canonical_record: dict[str, Any] | None = None,
) -> str:
    family_guidance = task_family_guidance(spec.task_family)
    distractor_guidance = task_distractor_guidance(spec.distractor_type)
    canonical_block = ""
    if conditioning == "canonical":
        if canonical_record is None:
            raise ValueError("canonical_record is required for canonical conditioning")
        canonical_task = str(canonical_record["task"]).strip()
        canonical_answer = normalize_numeric_answer(canonical_record["answer"])
        canonical_block = f"""

Canonical arithmetic item:
{canonical_task}

Canonical answer: {canonical_answer}

Canonical conditioning requirements:
- Rewrite the canonical item into a fresh word problem; preserve the arithmetic exactly.
- Keep every mathematical relationship and the final answer unchanged.
- You may change names and surface wording, but do not add, remove, or alter quantities in a way that changes the computation.
- The answer field must be exactly {canonical_answer}.
""".rstrip()
    elif conditioning != "freeform":
        raise ValueError(f"unknown generator conditioning: {conditioning}")
    return f"""
Generate one original GSM8K-style grade-school math word problem from this structured task specification.

Specification:
- task_family: {spec.task_family}
- reasoning_depth: {spec.reasoning_depth}
- distractor_type: {spec.distractor_type}
- numeric_scale: {spec.numeric_scale}
- difficulty_band: {spec.difficulty_band}
- answer_magnitude: {spec.answer_magnitude}
- seed_hint: {seed}-{idx}

Family guidance: {family_guidance}
Distractor guidance: {distractor_guidance}
{canonical_block}

Requirements:
- Return JSON only, no Markdown fences.
- The JSON keys must be exactly: task, answer, solution, validation_status.
- task must be a single self-contained English word problem.
- task must not reveal the answer or solution.
- answer must be a single numeric final answer, preferably an integer. No units in answer.
- solution must be a concise arithmetic derivation proving the answer.
- validation_status must be "valid" only if the answer follows from the task.
- Choose numbers by working backward from a clean exact answer, then write the problem.
- Do not round. Do not ask for approximate answers. If tax, percentages, or discounts are used, choose values that give an exact integer or terminating decimal answer.
- The answer_magnitude field is a soft target for choosing numbers, not permission to change the correct answer.
- If any spec target conflicts with arithmetic validity, ignore that target and prioritize validity.
- A distractor must be harmless: it may tempt a mistake but must not change the correct computation.
- Avoid calendar weekdays, business days, rounding conventions, or real-world constraints unless explicitly necessary.
- Make the arithmetic clean enough that the final numeric answer is unambiguous.
- The solution and answer must exactly agree.
- Avoid copying known benchmark text.
- Avoid ambiguous wording.
""".strip()


def extract_json_object(text: str) -> dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def normalize_numeric_answer(answer: Any) -> str:
    text = str(answer).strip()
    match = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
    if not match:
        raise ValueError(f"answer is not numeric: {answer!r}")
    value = float(match.group(0))
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.6f}".rstrip("0").rstrip(".")


async def direct_chat_completion(
    *,
    api_key: str,
    base_url: str | None,
    payload: dict[str, Any],
    timeout: float,
) -> dict[str, Any]:
    base = (base_url or "https://api.openai.com/v1").rstrip("/")
    url = f"{base}/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    request_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=request_timeout) as session:
        async with session.post(url, headers=headers, json=payload) as response:
            text = await response.text()
            if response.status >= 400:
                raise RuntimeError(f"HTTP {response.status}: {text[:500]}")
            try:
                return json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"non-JSON chat completion response: {text[:500]}") from exc


def response_content(response: Any) -> str:
    if isinstance(response, dict):
        return str(response["choices"][0]["message"].get("content") or "")
    return response.choices[0].message.content or ""


async def llm_render_one(
    client: AsyncOpenAI,
    spec: TaskSpec,
    *,
    api_key: str,
    base_url: str | None,
    model: str,
    conditioning: str,
    validation_model: str,
    temperature: float,
    max_tokens: int,
    request_timeout: float,
    seed: int,
    idx: int,
    retries: int,
    validate: bool,
    allow_answer_correction: bool,
) -> dict[str, Any]:
    canonical_record = render_spec(spec, seed, idx) if conditioning == "canonical" else None
    canonical_answer = (
        normalize_numeric_answer(canonical_record["answer"]) if canonical_record is not None else None
    )
    prompt = spec_generation_prompt(
        spec,
        seed,
        idx,
        conditioning=conditioning,
        canonical_record=canonical_record,
    )
    messages = [
        {
            "role": "system",
            "content": (
                "You generate mathematically valid GSM8K-style arithmetic word problems. "
                "You must return strict JSON and keep the final answer numeric."
            ),
        },
        {"role": "user", "content": prompt},
    ]
    last_error: str | None = None
    for attempt in range(max(1, retries)):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            }
            try:
                response = await direct_chat_completion(
                    api_key=api_key,
                    base_url=base_url,
                    payload=kwargs,
                    timeout=request_timeout,
                )
            except (TypeError, RuntimeError) as exc:
                if "response_format" not in kwargs:
                    raise
                kwargs.pop("response_format", None)
                response = await direct_chat_completion(
                    api_key=api_key,
                    base_url=base_url,
                    payload=kwargs,
                    timeout=request_timeout,
                )

            prompt_toks, completion_toks, _ = _extract_usage_tokens(response)
            if prompt_toks is not None:
                PromptTokens.instance().value += float(prompt_toks)
            if completion_toks is not None:
                CompletionTokens.instance().value += float(completion_toks)

            content = response_content(response)
            data = extract_json_object(content)
            task = str(data.get("task", "")).strip()
            solution = str(data.get("solution", "")).strip()
            answer = normalize_numeric_answer(data.get("answer", ""))
            if canonical_answer is not None and answer != canonical_answer:
                raise ValueError(
                    f"generated answer {answer} did not match canonical answer {canonical_answer}"
                )
            if len(task) < 30:
                raise ValueError("generated task is too short")
            if "answer" in task.lower()[-80:]:
                raise ValueError("generated task appears to reveal the answer")
            validation = None
            if validate:
                validation = await llm_validate_generated_task(
                    client,
                    spec=spec,
                    task=task,
                    proposed_answer=answer,
                    model=validation_model,
                    max_tokens=max(300, max_tokens // 2),
                )
                if validation["valid"]:
                    answer = validation["corrected_answer"]
                elif allow_answer_correction and validation.get("corrected_answer"):
                    answer = validation["corrected_answer"]
                    validation["original_valid"] = False
                    validation["valid"] = True
                    validation["corrected"] = True
                    validation["reason"] = "accepted validator-corrected answer"
                else:
                    raise ValueError(f"validator rejected task: {validation.get('reason')}")
            validation_status = "valid_corrected" if validation and validation.get("corrected") else "valid"
            return {
                "task": task,
                "answer": answer,
                "family": spec.task_family,
                "spec": asdict(spec),
                "solution": solution,
                "validation_status": validation_status,
                "renderer": "llm",
                "generator_model": model,
                "generator_conditioning": conditioning,
                "canonical_task": canonical_record["task"] if canonical_record is not None else None,
                "canonical_answer": canonical_answer,
                "generator_attempt": attempt,
                "generator_validation": validation,
            }
        except Exception as exc:
            last_error = str(exc)
            print(
                f"[generator] retry spec={spec_to_key(spec)} attempt={attempt + 1}/{max(1, retries)} error={last_error}",
                flush=True,
            )
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous output was invalid. Return valid JSON only with keys "
                        "task, answer, solution, validation_status. Regenerate from scratch. "
                        f"The validator error was: {last_error}. Ensure answer is numeric and exactly correct."
                    ),
                }
            )
    raise RuntimeError(f"LLM renderer failed for {spec_to_key(spec)} after {retries} attempts: {last_error}")


async def llm_validate_generated_task(
    client: AsyncOpenAI,
    *,
    spec: TaskSpec,
    task: str,
    proposed_answer: str,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    messages = [
        {
            "role": "system",
            "content": (
                "You independently validate generated grade-school arithmetic word problems. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": validation_prompt(spec, task=task, proposed_answer=proposed_answer),
        },
    ]
    kwargs = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    try:
        response = await client.chat.completions.create(**kwargs)
    except TypeError:
        kwargs.pop("response_format", None)
        response = await client.chat.completions.create(**kwargs)

    prompt_toks, completion_toks, _ = _extract_usage_tokens(response)
    if prompt_toks is not None:
        PromptTokens.instance().value += float(prompt_toks)
    if completion_toks is not None:
        CompletionTokens.instance().value += float(completion_toks)

    content = response.choices[0].message.content or ""
    data = extract_json_object(content)
    return parse_validation_payload(data, proposed_answer)


def validation_prompt(spec: TaskSpec, *, task: str, proposed_answer: str) -> str:
    return (
        "Solve the problem independently and check whether it is a valid training task.\n\n"
        "Structured specification:\n"
        f"- task_family: {spec.task_family}\n"
        f"- reasoning_depth: {spec.reasoning_depth}\n"
        f"- distractor_type: {spec.distractor_type}\n"
        f"- numeric_scale: {spec.numeric_scale}\n"
        f"- difficulty_band: {spec.difficulty_band}\n"
        f"- answer_magnitude: {spec.answer_magnitude}\n"
        f"Family guidance: {task_family_guidance(spec.task_family)}\n"
        f"Distractor guidance: {task_distractor_guidance(spec.distractor_type)}\n"
        "numeric_scale, difficulty_band, and answer_magnitude are soft generation controls. "
        "Never reject solely because the final answer magnitude, numeric scale, or difficulty differs from these soft controls.\n\n"
        "distractor_type is also a soft control. Never reject solely because a distractor is weak or missing; "
        "reject only if the distractor makes the computation ambiguous, contradictory, or changes the correct answer.\n\n"
        f"Problem:\n{task}\n\n"
        f"Proposed answer: {proposed_answer}\n\n"
        "Return JSON with exactly these keys: valid, corrected_answer, reason.\n"
        "valid must be true only if all of these hold: the problem is unambiguous; "
        "the proposed answer matches the correct numeric answer; the problem matches the task_family; "
        "no required quantity is already trivially revealed by the wording; and any included distractor is harmless.\n"
        "If proposed answer is wrong but the problem is unambiguous, put the correct numeric answer in corrected_answer.\n"
        "If the problem is ambiguous, impossible, contradictory, asks for a value already directly stated, "
        "or does not match the structured specification, set valid false and corrected_answer to an empty string.\n"
        "Do not include units in corrected_answer."
    )


def parse_validation_payload(data: dict[str, Any], proposed_answer: str) -> dict[str, Any]:
    corrected = ""
    if data.get("corrected_answer", "") not in ("", None):
        corrected = normalize_numeric_answer(data.get("corrected_answer"))
    elif data.get("valid"):
        corrected = normalize_numeric_answer(proposed_answer)
    proposed = normalize_numeric_answer(proposed_answer)
    valid = bool(data.get("valid"))
    coerced = False
    coerced_from_reason = False
    if not valid and corrected and corrected == proposed:
        valid = True
        coerced = True
    reason = str(data.get("reason", ""))
    reason_lower = reason.lower()
    explicit_valid_conclusion = (
        "therefore, the problem is valid" in reason_lower
        or "problem is valid and the answer is correct" in reason_lower
        or "valid should be true" in reason_lower
    )
    if not valid and not corrected and explicit_valid_conclusion:
        valid = True
        corrected = proposed
        coerced_from_reason = True
    return {
        "valid": valid,
        "corrected_answer": corrected,
        "reason": reason,
        "coerced_valid_from_matching_correction": coerced,
        "coerced_valid_from_reason": coerced_from_reason,
    }


async def render_records(
    specs: list[TaskSpec],
    *,
    renderer: str,
    seed: int,
    start_idx: int,
    generator_model: str,
    generator_conditioning: str,
    generator_temperature: float,
    generator_max_tokens: int,
    generator_timeout: float,
    generator_concurrency: int,
    generator_retries: int,
    generator_validation_model: str | None,
    validate_generator: bool,
    allow_generator_answer_correction: bool,
) -> list[dict[str, Any]]:
    if renderer == "template":
        return [render_spec(spec, seed, start_idx + i) for i, spec in enumerate(specs)]

    api_key = os.getenv("API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY or API_KEY is required for --renderer llm")
    base_url = os.getenv("BASE_URL") or os.getenv("OPENAI_BASE_URL") or None
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=generator_timeout, max_retries=0)
    sem = asyncio.Semaphore(generator_concurrency)

    async def run_one(i: int, spec: TaskSpec) -> dict[str, Any]:
        async with sem:
            print(
                f"[generator] start {i + 1}/{len(specs)} spec={spec_to_key(spec)} conditioning={generator_conditioning}",
                flush=True,
            )
            record = await llm_render_one(
                client,
                spec,
                api_key=api_key,
                base_url=base_url,
                model=generator_model,
                conditioning=generator_conditioning,
                validation_model=generator_validation_model or generator_model,
                temperature=generator_temperature,
                max_tokens=generator_max_tokens,
                request_timeout=generator_timeout,
                seed=seed,
                idx=start_idx + i,
                retries=generator_retries,
                validate=validate_generator,
                allow_answer_correction=allow_generator_answer_correction,
            )
            print(
                f"[generator] done {i + 1}/{len(specs)} spec={spec_to_key(spec)} answer={record['answer']}",
                flush=True,
            )
            return record

    return await asyncio.gather(*[run_one(i, spec) for i, spec in enumerate(specs)])


def load_saved_graph(metadata_path: Path, llm_name_override: str | None, variant_name: str) -> tuple[Graph, dict[str, Any]]:
    meta = json.loads(metadata_path.read_text(encoding="utf-8"))
    graph_args = SimpleNamespace(
        llm_name=llm_name_override or meta["llm_name"],
        seed=int(meta.get("seed", 0)),
        mode=meta.get("mode", "FullConnected"),
        decision_method=meta.get("decision_method", "FinalRefer"),
        num_agents=int(meta["num_agents"]),
        num_rounds=int(meta["num_rounds"]),
    )
    graph = build_graph(graph_args)
    variant = next(row for row in meta["variants"] if row["name"] == variant_name)
    graph.spatial_masks = torch.nn.Parameter(torch.tensor(variant["spatial_masks"], dtype=torch.float32), requires_grad=False)
    graph.temporal_masks = torch.nn.Parameter(torch.tensor(variant["temporal_masks"], dtype=torch.float32), requires_grad=False)
    graph.spatial_logits = torch.nn.Parameter(torch.tensor(variant["spatial_logits"], dtype=torch.float32), requires_grad=True)
    graph.temporal_logits = torch.nn.Parameter(torch.tensor(variant["temporal_logits"], dtype=torch.float32), requires_grad=graph_args.num_rounds > 1)
    graph.optimized_spatial = True
    graph.optimized_temporal = graph_args.num_rounds > 1
    meta["graph_source"] = f"metadata:{variant_name}"
    return graph, meta


def load_experiment_graph(args: argparse.Namespace) -> tuple[Graph, dict[str, Any]]:
    if args.fresh_graph:
        graph_args = SimpleNamespace(
            llm_name=args.llm_name or "gpt-4.1-nano",
            seed=args.seed,
            mode=args.mode,
            decision_method=args.decision_method,
            num_agents=args.num_agents,
            num_rounds=args.num_rounds,
        )
        graph = build_graph(graph_args)
        meta = {
            "graph_source": "fresh",
            "llm_name": graph_args.llm_name,
            "seed": args.seed,
            "mode": args.mode,
            "decision_method": args.decision_method,
            "num_agents": args.num_agents,
            "num_rounds": args.num_rounds,
            "aggregate_mode": args.aggregate_mode or "last connected",
        }
        return graph, meta
    return load_saved_graph(Path(args.metadata), args.llm_name, args.variant_name)


async def score_real_ig(
    graph: Graph,
    records: list[dict[str, Any]],
    *,
    seed: int,
    num_rounds: int,
    ensemble_size: int,
    prior_std: float,
    obs_noise: float,
    damping: float,
    max_concurrency: int,
    aggregate_mode: str,
    observation_mode: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    torch.manual_seed(seed)
    np.random.seed(seed % (2**32 - 1))
    random.seed(seed)
    return await eki_score_tasks(
        graph,
        records,
        num_rounds=num_rounds,
        ensemble_size=ensemble_size,
        prior_std=prior_std,
        obs_noise=obs_noise,
        damping=damping,
        max_concurrency=max_concurrency,
        aggregate_mode=aggregate_mode,
        observation_mode=observation_mode,
    )


async def score_real_ig_repeated(
    graph: Graph,
    records: list[dict[str, Any]],
    *,
    seed: int,
    repeats: int,
    num_rounds: int,
    ensemble_size: int,
    prior_std: float,
    obs_noise: float,
    damping: float,
    max_concurrency: int,
    aggregate_mode: str,
    observation_mode: str,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    repeat_scores = []
    repeat_details = []
    for repeat_id in range(max(1, repeats)):
        scores, details = await score_real_ig(
            graph,
            records,
            seed=seed + repeat_id * 10007,
            num_rounds=num_rounds,
            ensemble_size=ensemble_size,
            prior_std=prior_std,
            obs_noise=obs_noise,
            damping=damping,
            max_concurrency=max_concurrency,
            aggregate_mode=aggregate_mode,
            observation_mode=observation_mode,
        )
        repeat_scores.append(scores)
        repeat_details.append(details)

    stacked = np.stack(repeat_scores, axis=0)
    means = stacked.mean(axis=0)
    stds = stacked.std(axis=0, ddof=1) if stacked.shape[0] > 1 else np.zeros(stacked.shape[1], dtype=np.float64)
    details_by_record = []
    for record_id in range(len(records)):
        per_repeat = [repeat_details[r][record_id] for r in range(stacked.shape[0])]
        details_by_record.append(
            {
                "local_task_id": record_id,
                "ig_mean": float(means[record_id]),
                "ig_std": float(stds[record_id]),
                "ig_values": [float(stacked[r, record_id]) for r in range(stacked.shape[0])],
                "nonzero_repeat_fraction": float(np.mean(stacked[:, record_id] > 0.0)),
                "repeat_details": per_repeat,
            }
        )
    return means.astype(np.float64), details_by_record


def train_surrogate(
    specs: list[TaskSpec],
    ig_values: np.ndarray,
    *,
    seed: int,
    epochs: int,
    lr: float,
    device: torch.device,
) -> tuple[IGSurrogate, list[dict[str, Any]], dict[str, float]]:
    set_seed(seed)
    X = torch.tensor(np.stack([encode_spec(spec) for spec in specs]), dtype=torch.float32, device=device)
    y = torch.tensor(np.log1p(np.asarray(ig_values, dtype=np.float32)), dtype=torch.float32, device=device)
    model = IGSurrogate(X.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    history = []
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)
    test_n = max(2, int(0.2 * n)) if n >= 10 else 0
    test_idx_np = idx[:test_n]
    train_idx_np = idx[test_n:] if test_n else idx
    test_idx = torch.tensor(test_idx_np, dtype=torch.long, device=device) if test_n else None
    train_idx = torch.tensor(train_idx_np, dtype=torch.long, device=device)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        model.train()
        pred = model(X[train_idx])
        loss = loss_fn(pred, y[train_idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if epoch % 25 == 0 or epoch == epochs - 1:
            model.eval()
            with torch.no_grad():
                row = {"epoch": epoch, "train_loss": float(loss.item())}
                if test_idx is not None and len(test_idx):
                    test_loss = loss_fn(model(X[test_idx]), y[test_idx])
                    row["test_loss"] = float(test_loss.item())
                    row["val_loss"] = float(test_loss.item())
                else:
                    row["test_loss"] = None
                    row["val_loss"] = None
                history.append(row)
    model.eval()
    with torch.no_grad():
        pred_np = model(X).detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()

    def corr_for(indices: np.ndarray) -> float:
        if len(indices) <= 1:
            return 0.0
        pred_part = pred_np[indices]
        y_part = y_np[indices]
        if np.std(pred_part) <= 1e-8 or np.std(y_part) <= 1e-8:
            return 0.0
        return float(np.corrcoef(pred_part, y_part)[0, 1])

    def mae_for(indices: np.ndarray) -> float:
        if len(indices) == 0:
            return 0.0
        return float(np.mean(np.abs(pred_np[indices] - y_np[indices])))

    def r2_for(indices: np.ndarray) -> float:
        if len(indices) <= 1:
            return 0.0
        y_part = y_np[indices]
        ss_res = float(np.sum((pred_np[indices] - y_part) ** 2))
        ss_tot = float(np.sum((y_part - np.mean(y_part)) ** 2))
        return 0.0 if ss_tot <= 1e-12 else float(1.0 - ss_res / ss_tot)

    all_idx = np.arange(n)
    test_indices_for_metrics = test_idx_np if test_n else np.asarray([], dtype=np.int64)
    metrics = {
        "train_loss_final": history[-1]["train_loss"],
        "test_loss_final": history[-1].get("test_loss"),
        "val_loss_final": history[-1].get("val_loss"),
        "pred_real_log1p_corr": corr_for(all_idx),
        "train_pred_real_log1p_corr": corr_for(train_idx_np),
        "test_pred_real_log1p_corr": corr_for(test_indices_for_metrics),
        "train_mae_log1p": mae_for(train_idx_np),
        "test_mae_log1p": mae_for(test_indices_for_metrics),
        "train_r2_log1p": r2_for(train_idx_np),
        "test_r2_log1p": r2_for(test_indices_for_metrics),
        "train_count": int(len(train_idx_np)),
        "test_count": int(len(test_indices_for_metrics)),
        "train_indices": [int(i) for i in train_idx_np.tolist()],
        "test_indices": [int(i) for i in test_indices_for_metrics.tolist()],
        "device": str(device),
    }
    return model, history, metrics


def surrogate_predict(model: IGSurrogate, specs: list[TaskSpec]) -> np.ndarray:
    if not specs:
        return np.asarray([], dtype=np.float64)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        X = torch.tensor(np.stack([encode_spec(spec) for spec in specs]), dtype=torch.float32, device=device)
        return model(X).detach().cpu().numpy().astype(np.float64)


def soft_encoding_from_logits(logits_by_field: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([torch.softmax(logits_by_field[field], dim=0) for field in FIELD_ORDER], dim=0)


def optimize_spec(
    model: IGSurrogate,
    initial: TaskSpec,
    *,
    steps: int,
    lr: float,
    drift_penalty: float,
) -> tuple[TaskSpec, dict[str, Any]]:
    model.eval()
    device = next(model.parameters()).device
    logits = {}
    initial_onehot = torch.tensor(encode_spec(initial), dtype=torch.float32, device=device)
    offset = 0
    for field in FIELD_ORDER:
        values = SPEC_SPACE[field]
        init = torch.full((len(values),), -2.0, device=device)
        init[values.index(getattr(initial, field))] = 2.0
        logits[field] = nn.Parameter(init)
        offset += len(values)
    opt = torch.optim.Adam(list(logits.values()), lr=lr)
    best_score = -1e30
    best_vec = None
    trace = []
    for step in range(steps):
        vec = soft_encoding_from_logits(logits)
        pred = model(vec.unsqueeze(0)).squeeze(0)
        drift = torch.mean((vec - initial_onehot) ** 2)
        objective = pred - drift_penalty * drift
        opt.zero_grad()
        (-objective).backward()
        opt.step()
        score = float(pred.detach().cpu().item())
        if score > best_score:
            best_score = score
            best_vec = vec.detach().cpu().numpy()
        if step % 20 == 0 or step == steps - 1:
            trace.append(
                {
                    "step": step,
                    "pred_log1p_ig": score,
                    "objective": float(objective.detach().cpu().item()),
                    "drift": float(drift.detach().cpu().item()),
                }
            )
    assert best_vec is not None
    projected = decode_onehot(best_vec)
    return projected, {
        "initial_spec": asdict(initial),
        "optimized_spec": asdict(projected),
        "surrogate_pred_log1p_ig_best": best_score,
        "trace": trace,
    }


def choose_surrogate_guided_specs(
    model: IGSurrogate,
    training_specs: list[TaskSpec],
    *,
    rng: random.Random,
    count: int,
    optimize_starts: int,
    discrete_topk: int,
    steps: int,
    lr: float,
    drift_penalty: float,
    trust_region_hamming: int | None,
) -> tuple[list[TaskSpec], list[dict[str, Any]]]:
    universe = all_specs()
    universe_pred = surrogate_predict(model, universe)
    order = np.argsort(universe_pred)[::-1].tolist()

    if trust_region_hamming is not None:
        train_pred = surrogate_predict(model, training_specs)
        train_order = np.argsort(train_pred)[::-1].tolist()
        seed_specs = [training_specs[i] for i in train_order[: max(1, min(optimize_starts, len(train_order)))]]
        allowed = {
            spec_to_key(spec)
            for spec in universe
            if any(spec_hamming_distance(spec, seed) <= trust_region_hamming for seed in seed_specs)
        }
        order = [idx for idx in order if spec_to_key(universe[idx]) in allowed]

    starts = [universe[i] for i in order[: max(1, min(optimize_starts // 2, len(order)))]]
    while len(starts) < optimize_starts:
        starts.append(rng.choice(training_specs if training_specs else universe))

    traces = []
    candidates = [universe[i] for i in order[: max(count, discrete_topk)]]
    for start_id, start_spec in enumerate(starts):
        optimized, trace = optimize_spec(
            model,
            start_spec,
            steps=steps,
            lr=lr,
            drift_penalty=drift_penalty,
        )
        trace["start_id"] = start_id
        trace["proposal_source"] = "gradient_ascent"
        candidates.append(optimized)
        traces.append(trace)

    candidates = unique_specs(candidates)
    candidate_pred = surrogate_predict(model, candidates)
    ranked = np.argsort(candidate_pred)[::-1].tolist()
    selected = []
    selected_traces = []
    trace_by_spec = {spec_to_key(TaskSpec(**trace["optimized_spec"])): trace for trace in traces}
    for idx in ranked:
        spec = candidates[idx]
        key = spec_to_key(spec)
        trace = trace_by_spec.get(
            key,
            {
                "proposal_source": "surrogate_discrete_scan",
                "optimized_spec": asdict(spec),
                "surrogate_pred_log1p_ig_best": float(candidate_pred[idx]),
                "trace": [],
            },
        )
        trace["surrogate_pred_log1p_selected"] = float(candidate_pred[idx])
        trace["trust_region_hamming"] = trust_region_hamming
        selected.append(spec)
        selected_traces.append(trace)
        if len(selected) >= count:
            break
    return selected, selected_traces


def sample_random_baseline_specs(
    rng: random.Random,
    count: int,
    *,
    exclude: list[TaskSpec],
    sampling_mode: str,
) -> list[TaskSpec]:
    excluded = {spec_to_key(spec) for spec in exclude}
    specs = []
    attempts = 0
    while len(specs) < count and attempts < count * 100:
        attempts += 1
        spec = sample_spec(rng, sampling_mode)
        key = spec_to_key(spec)
        if key in excluded:
            continue
        specs.append(spec)
        excluded.add(key)
    if len(specs) < count:
        for spec in all_specs():
            key = spec_to_key(spec)
            if key in excluded:
                continue
            specs.append(spec)
            excluded.add(key)
            if len(specs) >= count:
                break
    return specs


def bootstrap_prob_mean_greater(a: np.ndarray, b: np.ndarray, *, seed: int, draws: int = 5000) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    rng = np.random.default_rng(seed)
    wins = 0
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    for _ in range(draws):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        wins += float(a_s.mean() > b_s.mean())
    return wins / draws


def bootstrap_mean_diff_interval(
    a: np.ndarray,
    b: np.ndarray,
    *,
    seed: int,
    draws: int = 5000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    if len(a) == 0 or len(b) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diffs = []
    for _ in range(draws):
        a_s = rng.choice(a, size=len(a), replace=True)
        b_s = rng.choice(b, size=len(b), replace=True)
        diffs.append(float(a_s.mean() - b_s.mean()))
    lo = float(np.quantile(diffs, alpha / 2))
    hi = float(np.quantile(diffs, 1.0 - alpha / 2))
    return lo, hi


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL row {line_no} in {path}: {exc}") from exc
    return rows


def task_spec_from_row(row: dict[str, Any], *, row_id: int) -> TaskSpec:
    spec_data = row.get("spec")
    if not isinstance(spec_data, dict):
        raise ValueError(f"row {row_id} is missing object field 'spec'")
    kwargs = {}
    for field in FIELD_ORDER:
        value = str(spec_data.get(field))
        if value not in SPEC_SPACE[field]:
            raise ValueError(f"row {row_id} has invalid {field}={value!r}")
        kwargs[field] = value
    return TaskSpec(**kwargs)


def load_labeled_initial_pairs(path: Path) -> tuple[list[TaskSpec], np.ndarray, list[dict[str, Any]]]:
    source_path = path / "specs_initial.jsonl" if path.is_dir() else path
    rows = read_jsonl(source_path)
    if not rows:
        raise ValueError(f"no labeled rows found in {source_path}")
    specs = []
    ig_values = []
    for row_id, row in enumerate(rows):
        if "real_ig" not in row:
            raise ValueError(f"row {row_id} is missing 'real_ig'")
        specs.append(task_spec_from_row(row, row_id=row_id))
        ig_values.append(float(row["real_ig"]))
    return specs, np.asarray(ig_values, dtype=np.float64), rows


def build_surrogate_prediction_rows(
    *,
    initial_rows: list[dict[str, Any]],
    initial_specs: list[TaskSpec],
    initial_ig: Any,
    pred_log1p: Any,
    test_indices: set[int],
) -> list[dict[str, Any]]:
    ig_values = np.asarray(initial_ig, dtype=np.float64)
    pred_values = np.asarray(pred_log1p, dtype=np.float64)
    rows = []
    for i, spec in enumerate(initial_specs):
        source_row = initial_rows[i] if i < len(initial_rows) else {}
        rows.append(
            {
                "id": source_row.get("id", i),
                "split": "test" if i in test_indices else "train",
                "real_ig": float(ig_values[i]),
                "real_log1p_ig": float(np.log1p(ig_values[i])),
                "surrogate_pred_log1p": float(pred_values[i]),
                "spec": json.dumps(asdict(spec)),
                "task": source_row.get("task"),
                "answer": source_row.get("answer"),
            }
        )
    return rows


def write_svg(path: Path, pairs: list[dict[str, Any]]) -> None:
    width, height = 760, 420
    left, bottom, top = 70, 70, 40
    plot_w, plot_h = width - left - 30, height - top - bottom
    values = [float(row["real_ig_initial"]) for row in pairs] + [float(row["real_ig_optimized"]) for row in pairs]
    max_v = max(values) if values else 1.0
    if max_v <= 0:
        max_v = 1.0
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-family="Arial" font-size="18">Real EKI/KL IG Before vs After Surrogate Optimization</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    n = max(1, len(pairs))
    step_x = plot_w / n
    bar_w = min(26, step_x / 4)
    for i, row in enumerate(pairs):
        x = left + i * step_x + step_x / 2
        for dx, key, color in [(-bar_w * 0.65, "real_ig_initial", "#4C78A8"), (bar_w * 0.65, "real_ig_optimized", "#F58518")]:
            val = float(row[key])
            h = (val / max_v) * plot_h
            y = top + plot_h - h
            parts.append(f'<rect x="{x + dx - bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 42}" text-anchor="middle" font-family="Arial" font-size="10">{i}</text>')
    parts.extend(
        [
            f'<rect x="{width - 170}" y="46" width="12" height="12" fill="#4C78A8"/><text x="{width - 152}" y="57" font-family="Arial" font-size="12">Initial real IG</text>',
            f'<rect x="{width - 170}" y="66" width="12" height="12" fill="#F58518"/><text x="{width - 152}" y="77" font-family="Arial" font-size="12">Optimized real IG</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def write_comparison_svg(path: Path, optimized_ig: np.ndarray, random_ig: np.ndarray) -> None:
    width, height = 760, 420
    left, bottom, top = 70, 70, 40
    plot_w, plot_h = width - left - 30, height - top - bottom
    values = [float(x) for x in optimized_ig.tolist()] + [float(x) for x in random_ig.tolist()]
    max_v = max(values) if values else 1.0
    if max_v <= 0:
        max_v = 1.0
    n = max(len(optimized_ig), len(random_ig), 1)
    step_x = plot_w / n
    bar_w = min(24, step_x / 4)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="24" text-anchor="middle" font-family="Arial" font-size="18">Verified Real IG: Surrogate-Optimized vs Random Specs</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for i in range(n):
        x = left + i * step_x + step_x / 2
        for dx, val, color in [
            (-bar_w * 0.65, float(optimized_ig[i]) if i < len(optimized_ig) else 0.0, "#54A24B"),
            (bar_w * 0.65, float(random_ig[i]) if i < len(random_ig) else 0.0, "#E45756"),
        ]:
            h = (val / max_v) * plot_h
            y = top + plot_h - h
            parts.append(f'<rect x="{x + dx - bar_w/2:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{h:.1f}" fill="{color}"/>')
        parts.append(f'<text x="{x:.1f}" y="{height - 42}" text-anchor="middle" font-family="Arial" font-size="10">{i}</text>')
    parts.extend(
        [
            f'<rect x="{width - 205}" y="46" width="12" height="12" fill="#54A24B"/><text x="{width - 187}" y="57" font-family="Arial" font-size="12">Surrogate-optimized</text>',
            f'<rect x="{width - 205}" y="66" width="12" height="12" fill="#E45756"/><text x="{width - 187}" y="77" font-family="Arial" font-size="12">Random baseline</text>',
            "</svg>",
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def render_report(
    *,
    graph_source: str,
    run_dir: Path,
    initial_count: int,
    pairs: list[dict[str, Any]],
    comparison_metrics: dict[str, float],
    initial_stats: dict[str, float],
    surrogate_metrics: dict[str, Any],
    token_counts: dict[str, float],
) -> str:
    deltas = np.asarray([float(row["delta_ig"]) for row in pairs], dtype=np.float64)
    before = np.asarray([float(row["real_ig_initial"]) for row in pairs], dtype=np.float64)
    after = np.asarray([float(row["real_ig_optimized"]) for row in pairs], dtype=np.float64)
    mean_before = float(before.mean()) if len(before) else 0.0
    mean_after = float(after.mean()) if len(after) else 0.0
    mean_delta = float(deltas.mean()) if len(deltas) else 0.0
    frac_pos = float((deltas > 0).mean()) if len(deltas) else 0.0
    rows = [
        "# Surrogate-Guided High-IG Task Generation PoC",
        "",
        "## Question",
        "",
        "Can a differentiable surrogate trained on EKI/KL graph-parameter IG guide task-spec optimization toward higher verified real IG?",
        "",
        "## Setup",
        "",
        f"- Graph source: `{graph_source}`",
        f"- Initial random specs scored: {initial_count}",
        f"- Initial -> optimized pairs verified: {len(pairs)}",
        f"- Output directory: `{run_dir}`",
        f"- Training-label nonzero fraction: {initial_stats['nonzero_fraction']:.4f}",
        f"- Training-label mean IG: {initial_stats['mean_ig']:.6f}",
        f"- Training-label max IG: {initial_stats['max_ig']:.6f}",
        "",
        "## Surrogate Metrics",
        "",
        f"- Train/test split: {surrogate_metrics.get('train_count', 0)} / {surrogate_metrics.get('test_count', 0)}",
        f"- Final train loss: {surrogate_metrics['train_loss_final']:.6f}",
        f"- Final test loss: {surrogate_metrics.get('test_loss_final')}",
        f"- Train predicted-vs-real log1p IG correlation: {surrogate_metrics.get('train_pred_real_log1p_corr', 0.0):.4f}",
        f"- Test predicted-vs-real log1p IG correlation: {surrogate_metrics.get('test_pred_real_log1p_corr', 0.0):.4f}",
        f"- Test MAE log1p IG: {surrogate_metrics.get('test_mae_log1p', 0.0):.4f}",
        f"- Test R2 log1p IG: {surrogate_metrics.get('test_r2_log1p', 0.0):.4f}",
        "",
        "## Real EKI/KL Verification",
        "",
        f"- Mean real IG before: {mean_before:.6f}",
        f"- Mean real IG after: {mean_after:.6f}",
        f"- Mean delta IG: {mean_delta:.6f}",
        f"- Fraction positive delta: {frac_pos:.4f}",
        "",
        "## Optimized vs Random Generated Specs",
        "",
        f"- Optimized mean verified IG: {comparison_metrics['optimized_mean_ig']:.6f}",
        f"- Random mean verified IG: {comparison_metrics['random_mean_ig']:.6f}",
        f"- Mean difference: {comparison_metrics['optimized_minus_random_mean']:.6f}",
        f"- Mean-difference bootstrap 95% CI: [{comparison_metrics['bootstrap_mean_diff_ci_low']:.6f}, {comparison_metrics['bootstrap_mean_diff_ci_high']:.6f}]",
        f"- Optimized nonzero IG fraction: {comparison_metrics['optimized_nonzero_fraction']:.4f}",
        f"- Random nonzero IG fraction: {comparison_metrics['random_nonzero_fraction']:.4f}",
        f"- Fraction paired optimized > random: {comparison_metrics['paired_fraction_optimized_gt_random']:.4f}",
        f"- Bootstrap P(mean optimized > mean random): {comparison_metrics['bootstrap_prob_optimized_gt_random']:.4f}",
        "",
        "## Pair-Level Results",
        "",
        "| Pair | Initial Family | Optimized Family | Real IG Before | Real IG After | Delta |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for row in pairs:
        rows.append(
            f"| {row['pair_id']} | {row['initial_task_family']} | {row['optimized_task_family']} | "
            f"{row['real_ig_initial']:.6f} | {row['real_ig_optimized']:.6f} | {row['delta_ig']:.6f} |"
        )
    rows.extend(
        [
            "",
            "## Token Accounting",
            "",
            f"- Initial generation tokens: {token_counts.get('initial_generation_tokens', 0.0):.0f}",
            f"- Initial IG scoring tokens: {token_counts['initial_ig_tokens']:.0f}",
            f"- Verification generation tokens: {token_counts.get('verification_generation_tokens', 0.0):.0f}",
            f"- Verification IG scoring tokens: {token_counts['verification_ig_tokens']:.0f}",
            f"- Total run tokens: {token_counts['total_tokens']:.0f}",
            "",
            "## Interpretation",
            "",
        ]
    )
    if (
        comparison_metrics["optimized_minus_random_mean"] > 0
        and comparison_metrics["bootstrap_prob_optimized_gt_random"] >= 0.95
    ):
        rows.append(
            "The verified real EKI/KL results support the PoC claim: surrogate-guided specs achieved higher verified IG than random generated specs."
        )
    else:
        rows.append(
            "The verified real EKI/KL results do not yet support the target optimization claim. Treat this as a diagnostic run and inspect estimator noise/search exploitation."
        )
    rows.append(
        "This result is only about graph-parameter IG. It does not claim final graph accuracy improves."
    )
    return "\n".join(rows) + "\n"


async def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    graph, meta = load_experiment_graph(args)
    aggregate_mode = args.aggregate_mode or meta.get("aggregate_mode", "all connected")
    num_rounds = int(meta.get("num_rounds", 1))
    verification_ensemble_size = args.verification_ensemble_size or args.ensemble_size
    surrogate_device = resolve_torch_device(args.surrogate_device)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[phase] run_dir={run_dir}", flush=True)

    rng = random.Random(args.seed)
    random_spec_sampling = args.random_spec_sampling or args.spec_sampling
    token_start = read_tokens()
    initial_records = []
    initial_rows = []
    generated_rows = []
    initial_ig_values = []
    initial_details = []
    initial_generation_tokens = 0.0
    initial_ig_tokens = 0.0

    if args.initial_data_dir:
        source_path = Path(args.initial_data_dir)
        initial_specs, initial_ig, initial_rows = load_labeled_initial_pairs(source_path)
        initial_ig_values = [float(x) for x in initial_ig.tolist()]
        generated_source = (source_path / "specs_initial_generated.jsonl") if source_path.is_dir() else None
        if generated_source is not None and generated_source.exists():
            generated_rows = read_jsonl(generated_source)
        else:
            generated_rows = [
                {
                    "id": row.get("id", i),
                    "spec": row["spec"],
                    "task": row.get("task"),
                    "answer": row.get("answer"),
                    "validation_status": row.get("validation_status"),
                    "renderer": row.get("renderer"),
                    "generator_model": row.get("generator_model"),
                    "generator_conditioning": row.get("generator_conditioning"),
                    "canonical_task": row.get("canonical_task"),
                    "canonical_answer": row.get("canonical_answer"),
                    "generator_validation": row.get("generator_validation"),
                    "solution": row.get("solution"),
                }
                for i, row in enumerate(initial_rows)
            ]
        write_jsonl(run_dir / "specs_initial.jsonl", initial_rows)
        write_jsonl(run_dir / "specs_initial_generated.jsonl", generated_rows)
        batch_size = 0
        print(
            f"[phase] loaded initial labeled pairs count={len(initial_specs)} source={source_path}",
            flush=True,
        )
    else:
        initial_specs = []
        seen_initial = set()
        while len(initial_specs) < args.initial_specs:
            spec = sample_spec(rng, args.spec_sampling)
            key = spec_to_key(spec)
            if key in seen_initial:
                continue
            seen_initial.add(key)
            initial_specs.append(spec)
        batch_size = args.initial_batch_size if args.initial_batch_size and args.initial_batch_size > 0 else len(initial_specs)
        total_batches = math.ceil(len(initial_specs) / batch_size) if batch_size else 1
        print(
            f"[phase] rendering/scoring initial records count={len(initial_specs)} renderer={args.renderer} batch_size={batch_size}",
            flush=True,
        )
        for batch_id, batch_start in enumerate(range(0, len(initial_specs), batch_size), start=1):
            batch_specs = initial_specs[batch_start : batch_start + batch_size]
            print(f"[phase] initial batch {batch_id}/{total_batches} render count={len(batch_specs)}", flush=True)
            token_before_generation = read_tokens()
            batch_records = await render_records(
                batch_specs,
                renderer=args.renderer,
                seed=args.seed,
                start_idx=batch_start,
                generator_model=args.generator_model,
                generator_conditioning=args.generator_conditioning,
                generator_temperature=args.generator_temperature,
                generator_max_tokens=args.generator_max_tokens,
                generator_timeout=args.generator_timeout,
                generator_concurrency=args.generator_concurrency,
                generator_retries=args.generator_retries,
                generator_validation_model=args.generator_validation_model,
                validate_generator=not args.disable_generator_validation,
                allow_generator_answer_correction=args.allow_generator_answer_correction,
            )
            token_after_generation = read_tokens()
            initial_generation_tokens += token_delta(token_before_generation, token_after_generation)
            initial_records.extend(batch_records)
            for j, (spec, record) in enumerate(zip(batch_specs, batch_records)):
                global_id = batch_start + j
                generated_rows.append(
                    {
                        "id": global_id,
                        "spec": asdict(spec),
                        "task": record["task"],
                        "answer": record["answer"],
                        "validation_status": record.get("validation_status"),
                        "renderer": record.get("renderer"),
                        "generator_model": record.get("generator_model"),
                        "generator_conditioning": record.get("generator_conditioning"),
                        "canonical_task": record.get("canonical_task"),
                        "canonical_answer": record.get("canonical_answer"),
                        "generator_validation": record.get("generator_validation"),
                        "solution": record.get("solution"),
                    }
                )
            write_jsonl(run_dir / "specs_initial_generated.jsonl", generated_rows)

            print(
                f"[phase] initial batch {batch_id}/{total_batches} scoring IG count={len(batch_records)} repeats={args.ig_repeats} ensemble={args.ensemble_size}",
                flush=True,
            )
            token_before_ig = read_tokens()
            batch_ig, batch_details = await score_real_ig_repeated(
                graph,
                batch_records,
                seed=args.seed + 1000 + batch_start,
                repeats=args.ig_repeats,
                num_rounds=num_rounds,
                ensemble_size=args.ensemble_size,
                prior_std=args.prior_std,
                obs_noise=args.obs_noise,
                damping=args.damping,
                max_concurrency=args.max_concurrency,
                aggregate_mode=aggregate_mode,
                observation_mode=args.ig_observation,
            )
            token_after_ig = read_tokens()
            initial_ig_tokens += token_delta(token_before_ig, token_after_ig)
            for j, (spec, record, ig, detail) in enumerate(zip(batch_specs, batch_records, batch_ig.tolist(), batch_details)):
                global_id = batch_start + j
                detail["global_task_id"] = global_id
                initial_rows.append(
                    {
                        "id": global_id,
                        "spec": asdict(spec),
                        "task": record["task"],
                        "answer": record["answer"],
                        "validation_status": record["validation_status"],
                        "renderer": record.get("renderer"),
                        "generator_model": record.get("generator_model"),
                        "generator_conditioning": record.get("generator_conditioning"),
                        "canonical_task": record.get("canonical_task"),
                        "canonical_answer": record.get("canonical_answer"),
                        "generator_validation": record.get("generator_validation"),
                        "solution": record.get("solution"),
                        "real_ig": float(ig),
                        "eki_detail": detail,
                    }
                )
            initial_ig_values.extend(float(x) for x in batch_ig.tolist())
            initial_details.extend(batch_details)
            write_jsonl(run_dir / "specs_initial.jsonl", initial_rows)
            nonzero_so_far = int(np.sum(np.asarray(initial_ig_values, dtype=np.float64) > 0.0))
            print(
                f"[phase] initial batch {batch_id}/{total_batches} complete cumulative_nonzero={nonzero_so_far}/{len(initial_ig_values)}",
                flush=True,
            )
    token_after_initial = read_tokens()
    initial_ig = np.asarray(initial_ig_values, dtype=np.float64)
    print("[phase] initial IG scoring complete", flush=True)
    initial_stats = {
        "mean_ig": float(np.mean(initial_ig)) if len(initial_ig) else 0.0,
        "max_ig": float(np.max(initial_ig)) if len(initial_ig) else 0.0,
        "nonzero_fraction": float(np.mean(initial_ig > 0.0)) if len(initial_ig) else 0.0,
        "nonzero_count": int(np.sum(initial_ig > 0.0)) if len(initial_ig) else 0,
        "count": int(len(initial_ig)),
    }

    if args.initial_only:
        token_counts = {
            "initial_generation_tokens": initial_generation_tokens,
            "initial_ig_tokens": initial_ig_tokens,
            "total_tokens": token_delta(token_start, token_after_initial),
        }
        metadata = {
            "created_at": stamp,
            "metadata_path": str(Path(args.metadata)) if not args.fresh_graph else None,
            "graph_source": meta.get("graph_source"),
            "llm_name": args.llm_name or meta["llm_name"],
            "seed": args.seed,
            "renderer": args.renderer,
            "generator_model": args.generator_model if args.renderer == "llm" else None,
            "generator_conditioning": args.generator_conditioning if args.renderer == "llm" else None,
            "generator_timeout": args.generator_timeout if args.renderer == "llm" else None,
            "generator_validation_model": (args.generator_validation_model or args.generator_model) if args.renderer == "llm" else None,
            "generator_validation": bool(args.renderer == "llm" and not args.disable_generator_validation),
            "allow_generator_answer_correction": bool(args.allow_generator_answer_correction),
            "spec_sampling": args.spec_sampling,
            "random_spec_sampling": random_spec_sampling,
            "trust_region_hamming": args.trust_region_hamming,
            "ig_observation": args.ig_observation,
            "initial_only": True,
            "initial_specs": int(len(initial_specs)),
            "initial_data_dir": args.initial_data_dir,
            "initial_batch_size": batch_size,
            "ensemble_size": args.ensemble_size,
            "ig_repeats": args.ig_repeats,
            "aggregate_mode": aggregate_mode,
            "num_rounds": num_rounds,
            "num_agents": int(meta.get("num_agents", args.num_agents)),
            "initial_stats": initial_stats,
            "token_counts": token_counts,
        }
        (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"[phase] initial-only checkpoint complete nonzero={initial_stats['nonzero_count']}/{initial_stats['count']}", flush=True)
        print(f"Wrote initial-only surrogate IG artifacts to {run_dir}")
        return

    surrogate, training_history, surrogate_metrics = train_surrogate(
        initial_specs,
        initial_ig,
        seed=args.seed + 2000,
        epochs=args.surrogate_epochs,
        lr=args.surrogate_lr,
        device=surrogate_device,
    )
    pd.DataFrame(training_history).to_csv(run_dir / "surrogate_training.csv", index=False)
    torch.save({k: v.detach().cpu() for k, v in surrogate.state_dict().items()}, run_dir / "surrogate_mlp.pt")
    initial_pred_log1p = surrogate_predict(surrogate, initial_specs)
    test_index_set = set(int(i) for i in surrogate_metrics.get("test_indices", []))
    prediction_rows = build_surrogate_prediction_rows(
        initial_rows=initial_rows,
        initial_specs=initial_specs,
        initial_ig=initial_ig,
        pred_log1p=initial_pred_log1p,
        test_indices=test_index_set,
    )
    pd.DataFrame(prediction_rows).to_csv(run_dir / "surrogate_predictions.csv", index=False)
    print(f"[phase] surrogate trained device={surrogate_metrics['device']}", flush=True)

    pair_count = min(args.pair_count, len(all_specs()))
    optimized_specs, opt_traces = choose_surrogate_guided_specs(
        surrogate,
        initial_specs,
        rng=rng,
        count=pair_count,
        optimize_starts=args.optimize_starts,
        discrete_topk=args.discrete_topk,
        steps=args.opt_steps,
        lr=args.opt_lr,
        drift_penalty=args.drift_penalty,
        trust_region_hamming=args.trust_region_hamming,
    )
    random_baseline_count = args.random_baseline_count or pair_count
    random_baseline_specs = sample_random_baseline_specs(
        rng,
        random_baseline_count,
        exclude=optimized_specs,
        sampling_mode=random_spec_sampling,
    )
    print(f"[phase] selected optimized={len(optimized_specs)} random={len(random_baseline_specs)}", flush=True)

    pair_initial_specs = []
    selected_ids = []
    for pair_id, trace in enumerate(opt_traces):
        if "initial_spec" in trace:
            pair_initial_specs.append(TaskSpec(**trace["initial_spec"]))
            selected_ids.append(None)
        else:
            ranked_initial = np.argsort(initial_ig)[::-1].tolist()
            fallback_id = int(ranked_initial[min(pair_id, len(ranked_initial) - 1)])
            pair_initial_specs.append(initial_specs[fallback_id])
            selected_ids.append(fallback_id)
    pair_initial_records = await render_records(
        pair_initial_specs,
        renderer=args.renderer,
        seed=args.seed + 3000,
        start_idx=0,
        generator_model=args.generator_model,
        generator_conditioning=args.generator_conditioning,
        generator_temperature=args.generator_temperature,
        generator_max_tokens=args.generator_max_tokens,
        generator_timeout=args.generator_timeout,
        generator_concurrency=args.generator_concurrency,
        generator_retries=args.generator_retries,
        generator_validation_model=args.generator_validation_model,
        validate_generator=not args.disable_generator_validation,
        allow_generator_answer_correction=args.allow_generator_answer_correction,
    )
    print("[phase] rendered pair-initial records", flush=True)
    optimized_records = await render_records(
        optimized_specs,
        renderer=args.renderer,
        seed=args.seed + 4000,
        start_idx=0,
        generator_model=args.generator_model,
        generator_conditioning=args.generator_conditioning,
        generator_temperature=args.generator_temperature,
        generator_max_tokens=args.generator_max_tokens,
        generator_timeout=args.generator_timeout,
        generator_concurrency=args.generator_concurrency,
        generator_retries=args.generator_retries,
        generator_validation_model=args.generator_validation_model,
        validate_generator=not args.disable_generator_validation,
        allow_generator_answer_correction=args.allow_generator_answer_correction,
    )
    print("[phase] rendered optimized records", flush=True)
    random_records = await render_records(
        random_baseline_specs,
        renderer=args.renderer,
        seed=args.seed + 7000,
        start_idx=0,
        generator_model=args.generator_model,
        generator_conditioning=args.generator_conditioning,
        generator_temperature=args.generator_temperature,
        generator_max_tokens=args.generator_max_tokens,
        generator_timeout=args.generator_timeout,
        generator_concurrency=args.generator_concurrency,
        generator_retries=args.generator_retries,
        generator_validation_model=args.generator_validation_model,
        validate_generator=not args.disable_generator_validation,
        allow_generator_answer_correction=args.allow_generator_answer_correction,
    )
    token_after_verification_generation = read_tokens()
    print(f"[phase] scoring verification IG count={len(pair_initial_records) + len(optimized_records) + len(random_records)} repeats={args.verification_repeats} ensemble={verification_ensemble_size}", flush=True)
    verification_records = pair_initial_records + optimized_records + random_records
    verification_ig, verification_details = await score_real_ig_repeated(
        graph,
        verification_records,
        seed=args.seed + 5000,
        repeats=args.verification_repeats,
        num_rounds=num_rounds,
        ensemble_size=verification_ensemble_size,
        prior_std=args.prior_std,
        obs_noise=args.obs_noise,
        damping=args.damping,
        max_concurrency=args.max_concurrency,
        aggregate_mode=aggregate_mode,
        observation_mode=args.ig_observation,
    )
    token_end = read_tokens()
    before_ig = verification_ig[:pair_count]
    after_ig = verification_ig[pair_count : 2 * pair_count]
    optimized_ig = after_ig
    random_ig = verification_ig[2 * pair_count :]

    pred_initial = surrogate_predict(surrogate, pair_initial_specs)
    pred_opt = surrogate_predict(surrogate, optimized_specs)
    pred_random = surrogate_predict(surrogate, random_baseline_specs)

    paired_n = min(len(optimized_ig), len(random_ig))
    ci_low, ci_high = bootstrap_mean_diff_interval(optimized_ig, random_ig, seed=args.seed + 9100)
    comparison_metrics = {
        "optimized_mean_ig": float(np.mean(optimized_ig)) if len(optimized_ig) else 0.0,
        "random_mean_ig": float(np.mean(random_ig)) if len(random_ig) else 0.0,
        "optimized_minus_random_mean": float(np.mean(optimized_ig) - np.mean(random_ig)) if len(optimized_ig) and len(random_ig) else 0.0,
        "bootstrap_mean_diff_ci_low": ci_low,
        "bootstrap_mean_diff_ci_high": ci_high,
        "optimized_nonzero_fraction": float(np.mean(optimized_ig > 0.0)) if len(optimized_ig) else 0.0,
        "random_nonzero_fraction": float(np.mean(random_ig > 0.0)) if len(random_ig) else 0.0,
        "paired_fraction_optimized_gt_random": float(np.mean(optimized_ig[:paired_n] > random_ig[:paired_n])) if paired_n else 0.0,
        "bootstrap_prob_optimized_gt_random": bootstrap_prob_mean_greater(optimized_ig, random_ig, seed=args.seed + 9000),
    }

    pair_rows = []
    optimized_jsonl = []
    for pair_id, initial_id in enumerate(selected_ids[:pair_count]):
        row = {
            "pair_id": pair_id,
            "initial_id": None if initial_id is None else int(initial_id),
            "proposal_source": opt_traces[pair_id].get("proposal_source"),
            "initial_task_family": pair_initial_specs[pair_id].task_family,
            "optimized_task_family": optimized_specs[pair_id].task_family,
            "real_ig_initial": float(before_ig[pair_id]),
            "real_ig_optimized": float(after_ig[pair_id]),
            "delta_ig": float(after_ig[pair_id] - before_ig[pair_id]),
            "surrogate_pred_log1p_initial": float(pred_initial[pair_id]),
            "surrogate_pred_log1p_optimized": float(pred_opt[pair_id]),
            "initial_spec": asdict(pair_initial_specs[pair_id]),
            "optimized_spec": asdict(optimized_specs[pair_id]),
            "initial_task": pair_initial_records[pair_id]["task"],
            "optimized_task": optimized_records[pair_id]["task"],
            "initial_answer": pair_initial_records[pair_id]["answer"],
            "optimized_answer": optimized_records[pair_id]["answer"],
        }
        pair_rows.append(row)
        optimized_jsonl.append(
            {
                "pair_id": pair_id,
                "initial_id": None if initial_id is None else int(initial_id),
                "proposal_source": opt_traces[pair_id].get("proposal_source"),
                "spec": asdict(optimized_specs[pair_id]),
                "task": optimized_records[pair_id]["task"],
                "answer": optimized_records[pair_id]["answer"],
                "validation_status": optimized_records[pair_id]["validation_status"],
                "renderer": optimized_records[pair_id].get("renderer"),
                "generator_model": optimized_records[pair_id].get("generator_model"),
                "generator_conditioning": optimized_records[pair_id].get("generator_conditioning"),
                "canonical_task": optimized_records[pair_id].get("canonical_task"),
                "canonical_answer": optimized_records[pair_id].get("canonical_answer"),
                "generator_validation": optimized_records[pair_id].get("generator_validation"),
                "solution": optimized_records[pair_id].get("solution"),
                "real_ig": float(after_ig[pair_id]),
                "eki_detail": verification_details[pair_count + pair_id],
                "optimization_trace": opt_traces[pair_id],
            }
        )

    write_jsonl(run_dir / "specs_optimized.jsonl", optimized_jsonl)
    random_jsonl = []
    for i, spec in enumerate(random_baseline_specs):
        random_jsonl.append(
            {
                "id": i,
                "spec": asdict(spec),
                "task": random_records[i]["task"],
                "answer": random_records[i]["answer"],
                "validation_status": random_records[i]["validation_status"],
                "renderer": random_records[i].get("renderer"),
                "generator_model": random_records[i].get("generator_model"),
                "generator_conditioning": random_records[i].get("generator_conditioning"),
                "canonical_task": random_records[i].get("canonical_task"),
                "canonical_answer": random_records[i].get("canonical_answer"),
                "generator_validation": random_records[i].get("generator_validation"),
                "solution": random_records[i].get("solution"),
                "real_ig": float(random_ig[i]),
                "surrogate_pred_log1p": float(pred_random[i]),
                "eki_detail": verification_details[2 * pair_count + i],
            }
        )
    write_jsonl(run_dir / "specs_random_baseline.jsonl", random_jsonl)

    pair_csv_rows = []
    for row in pair_rows:
        flat = dict(row)
        flat["initial_spec"] = json.dumps(flat["initial_spec"])
        flat["optimized_spec"] = json.dumps(flat["optimized_spec"])
        pair_csv_rows.append(flat)
    pd.DataFrame(pair_csv_rows).to_csv(run_dir / "ig_pairs.csv", index=False)

    comparison_rows = []
    for i, row in enumerate(optimized_jsonl):
        comparison_rows.append(
            {
                "group": "surrogate_optimized",
                "id": i,
                "task_family": row["spec"]["task_family"],
                "real_ig": row["real_ig"],
                "surrogate_pred_log1p": float(pred_opt[i]),
                "spec": json.dumps(row["spec"]),
                "task": row["task"],
                "answer": row["answer"],
            }
        )
    for i, row in enumerate(random_jsonl):
        comparison_rows.append(
            {
                "group": "random_baseline",
                "id": i,
                "task_family": row["spec"]["task_family"],
                "real_ig": row["real_ig"],
                "surrogate_pred_log1p": row["surrogate_pred_log1p"],
                "spec": json.dumps(row["spec"]),
                "task": row["task"],
                "answer": row["answer"],
            }
        )
    pd.DataFrame(comparison_rows).to_csv(run_dir / "ig_comparison.csv", index=False)
    write_svg(run_dir / "before_after_ig.svg", pair_rows)
    write_comparison_svg(run_dir / "optimized_vs_random_ig.svg", optimized_ig, random_ig)

    token_counts = {
        "initial_generation_tokens": initial_generation_tokens,
        "initial_ig_tokens": initial_ig_tokens,
        "verification_generation_tokens": token_delta(token_after_initial, token_after_verification_generation),
        "verification_ig_tokens": token_delta(token_after_verification_generation, token_end),
        "total_tokens": token_delta(token_start, token_end),
    }
    metadata = {
        "created_at": stamp,
        "metadata_path": str(Path(args.metadata)) if not args.fresh_graph else None,
        "graph_source": meta.get("graph_source"),
        "llm_name": args.llm_name or meta["llm_name"],
        "seed": args.seed,
        "renderer": args.renderer,
        "generator_model": args.generator_model if args.renderer == "llm" else None,
        "generator_conditioning": args.generator_conditioning if args.renderer == "llm" else None,
        "generator_timeout": args.generator_timeout if args.renderer == "llm" else None,
        "generator_validation_model": (args.generator_validation_model or args.generator_model) if args.renderer == "llm" else None,
        "generator_validation": bool(args.renderer == "llm" and not args.disable_generator_validation),
        "allow_generator_answer_correction": bool(args.allow_generator_answer_correction),
        "spec_sampling": args.spec_sampling,
        "random_spec_sampling": random_spec_sampling,
        "trust_region_hamming": args.trust_region_hamming,
        "ig_observation": args.ig_observation,
        "initial_specs": int(len(initial_specs)),
        "initial_data_dir": args.initial_data_dir,
        "initial_batch_size": batch_size,
        "pair_count": pair_count,
        "random_baseline_count": random_baseline_count,
        "ensemble_size": args.ensemble_size,
        "verification_ensemble_size": verification_ensemble_size,
        "ig_repeats": args.ig_repeats,
        "verification_repeats": args.verification_repeats,
        "aggregate_mode": aggregate_mode,
        "num_rounds": num_rounds,
        "num_agents": int(meta.get("num_agents", args.num_agents)),
        "initial_stats": initial_stats,
        "comparison_metrics": comparison_metrics,
        "surrogate_metrics": surrogate_metrics,
        "token_counts": token_counts,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    report = render_report(
        graph_source=str(meta.get("graph_source")),
        run_dir=run_dir,
        initial_count=len(initial_specs),
        pairs=pair_rows,
        comparison_metrics=comparison_metrics,
        initial_stats=initial_stats,
        surrogate_metrics=surrogate_metrics,
        token_counts=token_counts,
    )
    (run_dir / "report.md").write_text(report, encoding="utf-8")

    deltas = np.asarray([row["delta_ig"] for row in pair_rows], dtype=np.float64)
    print(f"Wrote surrogate IG PoC artifacts to {run_dir}")
    print(f"Mean IG before: {float(np.mean(before_ig)):.6f}")
    print(f"Mean IG after: {float(np.mean(after_ig)):.6f}")
    print(f"Mean delta IG: {float(np.mean(deltas)):.6f}")
    print(f"Fraction positive delta: {float(np.mean(deltas > 0)):.4f}")
    print(f"Optimized mean verified IG: {comparison_metrics['optimized_mean_ig']:.6f}")
    print(f"Random mean verified IG: {comparison_metrics['random_mean_ig']:.6f}")
    print(f"Optimized - random mean IG: {comparison_metrics['optimized_minus_random_mean']:.6f}")
    print(f"Bootstrap P(opt > random): {comparison_metrics['bootstrap_prob_optimized_gt_random']:.4f}")


if __name__ == "__main__":
    asyncio.run(main())
