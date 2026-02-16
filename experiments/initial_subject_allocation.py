import argparse
from collections import Counter
import numpy as np
import pandas as pd


def largest_remainder_allocation(counts: dict[str, int], budget: int) -> dict[str, int]:
    """
    Allocate 'budget' items across categories proportionally to 'counts'
    using the Hamilton / Largest Remainder method.

    Returns: alloc dict with same keys, sum(alloc.values()) == budget
    """
    total = sum(counts.values())
    if total == 0:
        return {k: 0 for k in counts}

    # Exact quotas
    quotas = {k: budget * (v / total) for k, v in counts.items()}

    # Floor allocations
    alloc = {k: int(np.floor(q)) for k, q in quotas.items()}
    used = sum(alloc.values())
    remaining = budget - used

    # Distribute remaining by largest fractional parts
    frac = sorted(((k, quotas[k] - alloc[k]) for k in counts), key=lambda x: x[1], reverse=True)
    for i in range(remaining):
        k = frac[i % len(frac)][0]
        alloc[k] += 1

    # Safety
    assert sum(alloc.values()) == budget
    return alloc


def cap_allocation_by_availability(alloc: dict[str, int], available: dict[str, int], budget: int) -> dict[str, int]:
    """
    Ensure alloc[k] <= available[k] for all k, while keeping total == budget.
    If a category is over-allocated, reassign surplus to categories with slack.
    """
    alloc = alloc.copy()

    # First cap
    surplus = 0
    for k in alloc:
        if alloc[k] > available.get(k, 0):
            surplus += alloc[k] - available.get(k, 0)
            alloc[k] = available.get(k, 0)

    if surplus == 0:
        return alloc

    # Reassign surplus to categories with remaining capacity (prefer larger availability)
    slack = {k: available.get(k, 0) - alloc.get(k, 0) for k in alloc}
    slack_items = sorted(slack.items(), key=lambda x: x[1], reverse=True)

    i = 0
    while surplus > 0 and i < len(slack_items):
        k, s = slack_items[i]
        if s > 0:
            take = min(s, surplus)
            alloc[k] += take
            surplus -= take
        i += 1

    if surplus != 0:
        raise RuntimeError(
            f"Not enough available items in sampled set to fill budget={budget}. "
            f"Try increasing sample_n."
        )

    assert sum(alloc.values()) == budget
    return alloc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="val", choices=["dev", "val", "test"])
    parser.add_argument("--sample_n", type=int, default=1000)
    parser.add_argument("--budget", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Import your dataset class (adjust this import path to match your repo)
    from dataset.mmlu_dataset import MMLUDataset

    ds = MMLUDataset(split=args.split)

    if not hasattr(ds, "_total_df_with_subject"):
        raise RuntimeError(
            "Dataset does not have _total_df_with_subject. "
            "Make sure you added _load_data_with_subject and initialized it in __init__."
        )

    n_total = len(ds)
    sample_n = min(args.sample_n, n_total)
    if args.budget > sample_n:
        raise ValueError(f"budget ({args.budget}) cannot exceed sample_n ({sample_n}).")

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(n_total, size=sample_n, replace=False)

    # Count subjects inside the sampled set
    sampled_subjects = ds._total_df_with_subject.iloc[idxs]["subject"].tolist()
    counts = Counter(sampled_subjects)

    # Build summary table
    total = sum(counts.values())
    df = pd.DataFrame(
        [{"subject": k, "count": v, "ratio": v / total} for k, v in counts.items()]
    ).sort_values(["count", "subject"], ascending=[False, True])

    print(f"\nSplit: {args.split}")
    print(f"Total questions in dataset: {n_total}")
    print(f"Sampled questions: {sample_n}")
    print(f"Budget to allocate: {args.budget}\n")

    print("Top 15 subjects in the sampled set:")
    print(df.head(15).to_string(index=False))

    # Allocate budget proportionally to sampled distribution
    alloc = largest_remainder_allocation(dict(counts), args.budget)

    # In rare cases (very small sample_n) rounding could allocate more than available for some small subjects.
    # Cap it by availability inside the sampled set and reassign.
    alloc = cap_allocation_by_availability(alloc, dict(counts), args.budget)

    df_alloc = pd.DataFrame(
        [{"subject": k, "sample_count": counts[k], "allocated": alloc[k]} for k in alloc]
    ).sort_values(["allocated", "sample_count", "subject"], ascending=[False, False, True])

    print("\nAllocation result (how many items to pick per subject out of the 30):")
    print(df_alloc.to_string(index=False))

    print(f"\nCheck: sum(allocated) = {df_alloc['allocated'].sum()}")


if __name__ == "__main__":
    main()
