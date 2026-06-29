import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from experiments.run_gsm8k_generated_task_poc import gsm_observation_value


def test_binary_correct_observation_matches_existing_behavior():
    assert gsm_observation_value("The answer is 42", "42", mode="binary_correct") == 1.0
    assert gsm_observation_value("The answer is 41", "42", mode="binary_correct") == 0.0


def test_soft_numeric_observation_rewards_near_misses():
    near = gsm_observation_value("The answer is 41", "42", mode="soft_numeric")
    far = gsm_observation_value("The answer is 20", "42", mode="soft_numeric")

    assert near < 1.0
    assert near > far
    assert far > 0.0
