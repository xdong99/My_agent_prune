import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from experiments.run_surrogate_ig_optimization_poc import (
    TaskSpec,
    build_surrogate_prediction_rows,
    load_labeled_initial_pairs,
    parse_validation_payload,
    render_spec,
    sample_spec,
    spec_generation_prompt,
    spec_hamming_distance,
    validation_prompt,
)


def test_canonical_llm_prompt_conditions_on_known_answer():
    spec = TaskSpec(
        task_family="coupon_tax",
        reasoning_depth="2",
        distractor_type="wrong_base",
        numeric_scale="medium",
        difficulty_band="hard",
        answer_magnitude="two_digit",
    )
    canonical = render_spec(spec, seed=7, idx=3)

    prompt = spec_generation_prompt(
        spec,
        seed=7,
        idx=3,
        conditioning="canonical",
        canonical_record=canonical,
    )

    assert canonical["task"] in prompt
    assert f"Canonical answer: {canonical['answer']}" in prompt
    assert "preserve the arithmetic" in prompt


def test_coupon_tax_canonical_answer_is_integer_currency_amount():
    spec = TaskSpec(
        task_family="coupon_tax",
        reasoning_depth="3",
        distractor_type="off_by_one",
        numeric_scale="small",
        difficulty_band="medium",
        answer_magnitude="one_digit",
    )

    canonical = render_spec(spec, seed=101, idx=0)

    assert canonical["answer"] == "91"


def test_defective_rate_canonical_answer_is_integer_count():
    spec = TaskSpec(
        task_family="defective_rate",
        reasoning_depth="2",
        distractor_type="off_by_one",
        numeric_scale="small",
        difficulty_band="medium",
        answer_magnitude="one_digit",
    )

    canonical = render_spec(spec, seed=101, idx=0)

    assert canonical["answer"] == "72"


def test_defective_rate_large_three_digit_does_not_hang():
    spec = TaskSpec(
        task_family="defective_rate",
        reasoning_depth="2",
        distractor_type="none",
        numeric_scale="large",
        difficulty_band="easy",
        answer_magnitude="three_digit",
    )

    canonical = render_spec(spec, seed=101, idx=6)

    assert canonical["answer"].isdigit()
    assert "discarded" in canonical["task"]


def test_growth_difference_canonical_counts_are_positive():
    spec = TaskSpec(
        task_family="growth_difference",
        reasoning_depth="1",
        distractor_type="wrong_denominator",
        numeric_scale="small",
        difficulty_band="medium",
        answer_magnitude="three_digit",
    )

    canonical = render_spec(spec, seed=101, idx=0)

    assert "started with -" not in canonical["task"]
    assert canonical["answer"] == "96"


def test_percent_base_profit_wording_mentions_repair_cost():
    spec = TaskSpec(
        task_family="percent_base",
        reasoning_depth="1",
        distractor_type="none",
        numeric_scale="small",
        difficulty_band="hard",
        answer_magnitude="two_digit",
    )

    canonical = render_spec(spec, seed=101, idx=0)

    assert "purchase and repair costs" in canonical["task"]


def test_lcm_meeting_wording_makes_endpoint_count_explicit():
    spec = TaskSpec(
        task_family="lcm_meeting",
        reasoning_depth="2",
        distractor_type="wrong_base",
        numeric_scale="large",
        difficulty_band="medium",
        answer_magnitude="two_digit",
    )

    canonical = render_spec(spec, seed=101, idx=0)

    assert "from day 0 through day 120 inclusive" in canonical["task"]


def test_spec_hamming_distance_counts_changed_fields():
    left = TaskSpec(
        task_family="percent_base",
        reasoning_depth="1",
        distractor_type="none",
        numeric_scale="small",
        difficulty_band="easy",
        answer_magnitude="one_digit",
    )
    right = TaskSpec(
        task_family="percent_base",
        reasoning_depth="2",
        distractor_type="none",
        numeric_scale="large",
        difficulty_band="easy",
        answer_magnitude="two_digit",
    )

    assert spec_hamming_distance(left, right) == 3


def test_informative_spec_sampling_biases_toward_nontrivial_controls():
    import random

    rng = random.Random(123)
    specs = [sample_spec(rng, "informative") for _ in range(300)]

    deep_fraction = sum(spec.reasoning_depth == "3" for spec in specs) / len(specs)
    hard_fraction = sum(spec.difficulty_band == "hard" for spec in specs) / len(specs)
    no_distractor_fraction = sum(spec.distractor_type == "none" for spec in specs) / len(specs)

    assert deep_fraction > 0.4
    assert hard_fraction > 0.45
    assert no_distractor_fraction < 0.1


def test_validation_payload_accepts_matching_corrected_answer():
    parsed = parse_validation_payload(
        {
            "valid": False,
            "corrected_answer": "4",
            "reason": "The proposed answer is correct and the problem is valid.",
        },
        proposed_answer="4",
    )

    assert parsed["valid"] is True
    assert parsed["corrected_answer"] == "4"
    assert parsed["coerced_valid_from_matching_correction"] is True


def test_validation_payload_accepts_explicit_valid_reason_conclusion():
    parsed = parse_validation_payload(
        {
            "valid": False,
            "corrected_answer": "",
            "reason": "The proposed answer is correct. Therefore, the problem is valid and the answer is correct.",
        },
        proposed_answer="4",
    )

    assert parsed["valid"] is True
    assert parsed["corrected_answer"] == "4"
    assert parsed["coerced_valid_from_reason"] is True


def test_validation_payload_accepts_valid_should_be_true_reason():
    parsed = parse_validation_payload(
        {
            "valid": False,
            "corrected_answer": "",
            "reason": "The problem is otherwise valid, and the answer is correct. Therefore, valid should be true.",
        },
        proposed_answer="184",
    )

    assert parsed["valid"] is True
    assert parsed["corrected_answer"] == "184"
    assert parsed["coerced_valid_from_reason"] is True


def test_validation_prompt_treats_magnitude_as_soft_control():
    spec = TaskSpec(
        task_family="coupon_tax",
        reasoning_depth="3",
        distractor_type="off_by_one",
        numeric_scale="small",
        difficulty_band="medium",
        answer_magnitude="one_digit",
    )

    prompt = validation_prompt(spec, task="A valid exact-payment problem.", proposed_answer="71.8")

    assert "soft generation controls" in prompt
    assert "distractor_type is also a soft control" in prompt
    assert "Never reject solely because" in prompt


def test_load_labeled_initial_pairs_reads_specs_and_ig(tmp_path):
    rows = [
        {
            "id": 7,
            "spec": {
                "task_family": "percent_base",
                "reasoning_depth": "2",
                "distractor_type": "wrong_base",
                "numeric_scale": "large",
                "difficulty_band": "hard",
                "answer_magnitude": "three_digit",
            },
            "task": "A valid generated task.",
            "answer": "125",
            "real_ig": 1.25,
        },
        {
            "id": 8,
            "spec": {
                "task_family": "overlap_count",
                "reasoning_depth": "3",
                "distractor_type": "irrelevant_extra",
                "numeric_scale": "medium",
                "difficulty_band": "medium",
                "answer_magnitude": "two_digit",
            },
            "task": "Another valid generated task.",
            "answer": "42",
            "real_ig": 0.0,
        },
    ]
    path = tmp_path / "specs_initial.jsonl"
    path.write_text("\n".join(__import__("json").dumps(row) for row in rows), encoding="utf-8")

    specs, ig_values, loaded_rows = load_labeled_initial_pairs(path)

    assert specs == [
        TaskSpec("percent_base", "2", "wrong_base", "large", "hard", "three_digit"),
        TaskSpec("overlap_count", "3", "irrelevant_extra", "medium", "medium", "two_digit"),
    ]
    assert ig_values.tolist() == [1.25, 0.0]
    assert loaded_rows == rows


def test_build_surrogate_prediction_rows_marks_test_split():
    specs = [
        TaskSpec("percent_base", "2", "wrong_base", "large", "hard", "three_digit"),
        TaskSpec("overlap_count", "3", "irrelevant_extra", "medium", "medium", "two_digit"),
    ]
    source_rows = [
        {"id": 7, "task": "A valid generated task.", "answer": "125"},
        {"id": 8, "task": "Another valid generated task.", "answer": "42"},
    ]

    rows = build_surrogate_prediction_rows(
        initial_rows=source_rows,
        initial_specs=specs,
        initial_ig=[1.25, 0.0],
        pred_log1p=[0.7, 0.1],
        test_indices={1},
    )

    assert rows[0]["id"] == 7
    assert rows[0]["split"] == "train"
    assert rows[1]["id"] == 8
    assert rows[1]["split"] == "test"
    assert rows[1]["task"] == "Another valid generated task."
