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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=str, default="test", choices=["dev","val","test"])
    parser.add_argument("--sample_n", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--topk", type=int, default=15, help="Top-K subjects to highlight; others shown in gray")
    parser.add_argument("--out", type=str, default="mmlu_embed_umap.png")
    parser.add_argument("--budget", type=int, default=30, help="Total number of representatives to select by quota.")

    args = parser.parse_args()

    from dataset.mmlu_dataset import MMLUDataset  # 按你工程实际路径，如需改就在这里改

    ds = MMLUDataset(args.split)

    if not hasattr(ds, "_total_df_with_subject"):
        raise RuntimeError("Your dataset must have _total_df_with_subject initialized.")

    n_total = len(ds)
    sample_n = min(args.sample_n, n_total)

    rng = np.random.default_rng(args.seed)
    idxs = rng.choice(n_total, size=sample_n, replace=False)

    df = ds._total_df_with_subject.iloc[idxs].reset_index(drop=True)

    texts = [build_text_from_record(df.iloc[i]) for i in range(len(df))]
    subjects = df["subject"].tolist()

    # 1) Encode
    enc = SentenceTransformer(args.model)
    # 建议 normalize_embeddings=True，后续距离/投影更稳定
    emb = enc.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    emb = np.asarray(emb, dtype=np.float32)

    # 线性测试
    min_count = 5  # 因为你要 5-fold
    cnt = Counter(subjects)
    keep_mask = np.array([cnt[s] >= min_count for s in subjects], dtype=bool)

    subjects_f = [s for s, m in zip(subjects, keep_mask) if m]
    emb_f = emb[keep_mask]

    print(f"[Filter] min_count={min_count}: kept {len(subjects_f)}/{len(subjects)} samples, "
          f"kept {len(set(subjects_f))}/{len(set(subjects))} classes")

    le = LabelEncoder()
    y = le.fit_transform(subjects_f)  # 0..C-1

    # 2) linear probe: multinomial logistic regression
    clf = LogisticRegression(
        max_iter=3000,
        solver="lbfgs",
        n_jobs=1,            # lbfgs不一定用n_jobs，但写上无害
    )

    # 3) 5-fold stratified CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    scoring = {
        "acc": "accuracy",
        "macro_f1": make_scorer(f1_score, average="macro"),
    }

    res = cross_validate(clf, emb_f, y, cv=cv, scoring=scoring, n_jobs=1)

    print("\n[Linear probe on embeddings]")
    print(f"Num classes: {len(le.classes_)}  (random baseline ~ {1/len(le.classes_):.4f})")
    print(f"Accuracy:  mean={res['test_acc'].mean():.4f}, std={res['test_acc'].std():.4f}")
    print(f"Macro-F1:  mean={res['test_macro_f1'].mean():.4f}, std={res['test_macro_f1'].std():.4f}\n")

    # 2) UMAP -> 2D
    try:
        import umap
        reducer = umap.UMAP(
            n_neighbors=15,
            min_dist=0.1,
            metric="cosine",
            random_state=args.seed,
        )
        xy = reducer.fit_transform(emb)
        method = "UMAP"
    except Exception as e:
        # fallback to t-SNE if UMAP not installed
        from sklearn.manifold import TSNE
        tsne = TSNE(
            n_components=2,
            perplexity=30,
            init="pca",
            learning_rate="auto",
            random_state=args.seed,
        )
        xy = tsne.fit_transform(emb)
        method = "t-SNE"

    x = xy[:, 0]
    y = xy[:, 1]


    # 4) Select representatives by quota within each subject
    alloc = allocate_budget(subjects, args.budget)

    selected_global = []
    for s, k in sorted(alloc.items(), key=lambda kv: -kv[1]):
        if k <= 0:
            continue
        idx_sub = np.where(np.array(subjects) == s)[0]   # indices in [0, sample_n)
        emb_sub = emb[idx_sub]
        picked_local = select_farthest_first(emb_sub, k)
        picked_global = idx_sub[picked_local].tolist()
        selected_global.extend(picked_global)
    
    print("\n[Selection by quota + farthest-first]")
    print(f"Budget: {args.budget}, selected: {len(selected_global)}")
    # 可选：打印前20个
    print("First 20 selected indices (within sampled df):", selected_global[:20])

    selected_original = idxs[np.array(selected_global, dtype=int)].tolist()
    print("First 20 selected indices (within full pool):", selected_original[:20])

    def q20(x):
        return str(x).replace("\n", " ").strip()[:20]
    
    # 1) selected_global: indices within sampled df (0..sample_n-1)
    print("\n[Check] Questions for selected_global[:20] (within sampled df):")
    for k, sidx in enumerate(selected_global[:20]):
        q = df.iloc[int(sidx)]["question"]
        print(f"  #{k:02d} sampled_idx={int(sidx):4d}  q[:20]={q20(q)}")
    
    # 2) map to full-pool indices: selected_original
    sel = np.array(selected_global, dtype=int)
    selected_original = idxs[sel]  # full indices into ds._total_df_with_subject  :contentReference[oaicite:1]{index=1}
    
    print("\n[Check] Questions for selected_original[:20] (within full pool):")
    for k, gidx in enumerate(selected_original[:20]):
        q = ds._total_df_with_subject.iloc[int(gidx)]["question"]
        print(f"  #{k:02d} full_idx={int(gidx):5d}  q[:20]={q20(q)}")

    breakpoint()

    # 3) Color by subject (topK highlighted)
    cnt = Counter(subjects)
    top_subjects = [s for s, _ in cnt.most_common(args.topk)]
    top_set = set(top_subjects)

    # 给 topK subject 分配离散颜色
    # 57 个全上色会很乱，所以这里只对 topK 上色
    cmap = plt.get_cmap("tab20")
    subject_to_color = {s: cmap(i % 20) for i, s in enumerate(top_subjects)}

    plt.figure(figsize=(10, 8), dpi=180)

    # 先画“其他”（灰）
    other_mask = np.array([s not in top_set for s in subjects], dtype=bool)
    plt.scatter(x[other_mask], y[other_mask], s=10, alpha=0.25, label="Other subjects")

    # 再画 topK
    for s in top_subjects:
        mask = np.array([ss == s for ss in subjects], dtype=bool)
        plt.scatter(x[mask], y[mask], s=14, alpha=0.8, label=s, color=subject_to_color[s])

    plt.title(f"MMLU {args.split}: {method} projection of embeddings (n={sample_n})")
    plt.xlabel("dim-1")
    plt.ylabel("dim-2")

    # legend 可能很长，但 topK 可控
    plt.legend(loc="best", fontsize=7, frameon=True)
    plt.tight_layout()


    out_dir = os.path.dirname(__file__)
    base_out = args.out  # e.g. "mmlu_embed_umap.png"

    # 1) 先保存“未画圈版”
    root, ext = os.path.splitext(base_out)
    out_no_circle = os.path.join(out_dir, f"{root}_no_circle{ext}")
    plt.savefig(out_no_circle)
    print(f"Saved plot (no circle) to: {out_no_circle}")

    # 2) 再叠加红圈
    sel = np.array(selected_global, dtype=int)
    plt.scatter(
        x[sel], y[sel],
        s=140, facecolors="none", edgecolors="red",
        linewidths=2.0, alpha=0.95,
        label="Selected reps",
    )

    # 3) 保存“画圈版”
    out_with_circle = os.path.join(out_dir, f"{root}_with_circle{ext}")
    plt.savefig(out_with_circle)
    print(f"Saved plot (with circle) to: {out_with_circle}")

if __name__ == "__main__":
    main()
