import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

# pip install sentence-transformers umap-learn
from sentence_transformers import SentenceTransformer


from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, f1_score

from typing import Dict, List

from functools import lru_cache
from typing import Any, Sequence

def allocate_budget(subjects: List[str], budget: int) -> Dict[str, int]:
    """
    Largest-remainder (Hamilton) allocation:
    k_s = floor(budget * p_s) + distribute remainders.
    """
    cnt = Counter(subjects)
    subjs = list(cnt.keys())
    counts = np.array([cnt[s] for s in subjs], dtype=np.float64)
    probs = counts / counts.sum()

    raw = probs * budget
    base = np.floor(raw).astype(int)
    rem = raw - base

    left = budget - base.sum()
    if left > 0:
        order = np.argsort(-rem)  # descending remainder
        for i in range(left):
            base[order[i % len(order)]] += 1

    alloc = {s: int(k) for s, k in zip(subjs, base)}
    return alloc

def select_farthest_first(emb_sub: np.ndarray, k: int) -> List[int]:
    """
    emb_sub: (m, d) normalized embeddings for one subject
    Return indices in [0, m) of selected points:
      1) medoid (min avg cosine distance)
      2) farthest-first: each time pick point maximizing min-dist to selected
    """
    m = emb_sub.shape[0]
    k = min(k, m)
    if k <= 0:
        return []

    # cosine distance = 1 - dot, since normalized
    sim = emb_sub @ emb_sub.T                   # (m, m)
    dist = 1.0 - sim                            # (m, m)

    # medoid: minimize mean distance to others
    mean_dist = dist.mean(axis=1)
    medoid = int(np.argmin(mean_dist))

    selected = [medoid]
    if k == 1:
        return selected

    # farthest-first based on min distance to selected set
    min_d = dist[medoid].copy()
    min_d[medoid] = -np.inf  # prevent re-select

    while len(selected) < k:
        nxt = int(np.argmax(min_d))
        selected.append(nxt)
        # update min distance to selected set
        min_d = np.minimum(min_d, dist[nxt])
        for j in selected:
            min_d[j] = -np.inf
    return selected

def build_text_from_record(record) -> str:
    # record 可能是 pandas Series 或 dict
    q = record["question"]
    A = record["A"]
    B = record["B"]
    C = record["C"]
    D = record["D"]
    text = (
        f"Question: {q}\n"
        f"(A) {A}\n"
        f"(B) {B}\n"
        f"(C) {C}\n"
        f"(D) {D}\n"
    )
    return text

def select_mmlu_representatives(
    split: str = "test",
    sample_n: int = 1000,
    seed: int = 0,
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    budget: int = 30,
    *,
    return_debug: bool = False,
    return_selected_embeddings: bool = False,
):
    """
    Sample sample_n questions from the full pool, embed them, then select `budget`
    representatives via per-subject quota + farthest-first selection.

    Returns:
        selected_original: List[int]  # indices in the FULL pool (ds._total_df_with_subject)
        (optional debug dict if return_debug=True)
    """
    from dataset.mmlu_dataset import MMLUDataset  # 按你工程实际路径，如需改就在这里改
    from sentence_transformers import SentenceTransformer

    ds = MMLUDataset(split)

    if not hasattr(ds, "_total_df_with_subject"):
        raise RuntimeError("Your dataset must have _total_df_with_subject initialized.")

    n_total = len(ds)
    sample_n = min(sample_n, n_total)

    rng = np.random.default_rng(seed)
    idxs = rng.choice(n_total, size=sample_n, replace=False)  # full-pool indices

    df = ds._total_df_with_subject.iloc[idxs].reset_index(drop=True)  # sampled df
    texts = [build_text_from_record(df.iloc[i]) for i in range(len(df))]
    subjects = df["subject"].tolist()

    enc = SentenceTransformer(model)
    emb = enc.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype=np.float32)

    # 选择：quota + farthest-first（在 sampled df 内做）
    alloc = allocate_budget(subjects, budget)

    selected_global = []
    subj_arr = np.array(subjects)
    for s, k in sorted(alloc.items(), key=lambda kv: -kv[1]):
        if k <= 0:
            continue
        idx_sub = np.where(subj_arr == s)[0]  # indices in [0, sample_n)
        emb_sub = emb[idx_sub]
        picked_local = select_farthest_first(emb_sub, k)  # indices in [0, len(idx_sub))
        picked_global = idx_sub[picked_local].tolist()    # indices in [0, sample_n)
        selected_global.extend(picked_global)

    # 映射回 full pool index（只保留这一处赋值，避免重复/类型覆盖）
    sel = np.array(selected_global, dtype=int)
    selected_original = idxs[sel].tolist()  # List[int]
    selected_emb = emb[sel]  

    if not return_debug and not return_selected_embeddings:
        return selected_original
    if not return_debug and return_selected_embeddings:
        return selected_original, selected_emb
    debug = {
        "idxs_sampled_full": idxs.tolist(),
        "selected_global_in_sampled_df": selected_global,
        "alloc": alloc,
        "subjects_sampled": subjects,
    }
    if return_selected_embeddings:
        return selected_original, selected_emb, debug
    return selected_original, debug


@lru_cache(maxsize=8)
def _get_encoder(model: str):
    # 缓存 encoder，避免每次都重新加载
    return SentenceTransformer(model)

def encode_texts(
    texts: Sequence[str],
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    enc = _get_encoder(model)
    emb = enc.encode(
        list(texts),
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=normalize,
    )
    return np.asarray(emb, dtype=np.float32)

def build_text_from_mmlu_item(item: Dict[str, Any]) -> str:
    # 让新任务的文本拼接风格与 build_text_from_record 尽量一致
    # 你原来用的是 "(A) ..." 这种格式 :contentReference[oaicite:1]{index=1}
    q = item["question"]
    A, B, C, D = item["A"], item["B"], item["C"], item["D"]
    return (
        f"Question: {q}\n"
        f"(A) {A}\n"
        f"(B) {B}\n"
        f"(C) {C}\n"
        f"(D) {D}\n"
    )

def encode_mmlu_items(
    items: Sequence[Dict[str, Any]],
    model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    normalize: bool = True,
) -> np.ndarray:
    texts = [build_text_from_mmlu_item(it) for it in items]
    return encode_texts(texts, model=model, batch_size=batch_size, normalize=normalize)

    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["dev", "val", "test"])
    parser.add_argument("--sample_n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--budget", type=int, default=30, help="Total number of representatives to select by quota.")

    args = parser.parse_args()

    selected_original = select_mmlu_representatives(
        split=args.split,
        sample_n=args.sample_n,
        seed=args.seed,
        model=args.model,
        budget=args.budget,
        return_debug=False,
    )

    print(f"[Selection] split={args.split} sample_n={args.sample_n} budget={args.budget}")
    print(f"Selected {len(selected_original)} indices (within full pool).")
    print("First 20 selected indices (within full pool):", selected_original[:20])


if __name__ == "__main__":
    main()
