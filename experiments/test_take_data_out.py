import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')
from experiments.metric_cache import MetricCache, eki_metrics_per_task_one_step_cached

cache = MetricCache.from_npz("outputs/metric_cache/selected50_es6.npz")

idx_all = cache.metrics["idx"].astype(int)     # (50,)
kl_all  = cache.metrics["kl"].astype(float)    # (50,)

topk = 10
order = np.argsort(kl_all)[::-1]               # KL 从大到小
top_pos = order[:topk]
top_idx = idx_all[top_pos].tolist()
top_kl  = kl_all[top_pos].tolist()

print("Top-10 idx:", top_idx)
print("Top-10 kl :", top_kl)
