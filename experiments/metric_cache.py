# utils/metric_cache.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Sequence, Optional, Any, List
import numpy as np

@dataclass
class MetricCache:
    npz_path: str
    metrics: Dict[str, np.ndarray]
    idx_to_pos: Dict[int, int]

    @classmethod
    def from_npz(cls, npz_path: str) -> "MetricCache":
        npz_path = str(npz_path)
        data = np.load(npz_path, allow_pickle=False)
        if "idx" not in data.files:
            raise ValueError(f"Cache {npz_path} missing required key 'idx'")
        metrics = {k: data[k] for k in data.files}
        idx_arr = metrics["idx"].astype(np.int64)
        idx_to_pos = {int(idx): int(i) for i, idx in enumerate(idx_arr.tolist())}
        n = len(idx_arr)
        for k, v in metrics.items():
            if v.shape[0] != n:
                raise ValueError(f"Key '{k}' has shape {v.shape}, expected first dim {n}")
        return cls(npz_path=npz_path, metrics=metrics, idx_to_pos=idx_to_pos)

    def query(
        self,
        task_indices: Sequence[int],
        keys: Optional[Sequence[str]] = None,
        *,
        strict: bool = True,
        return_idx: bool = False,
    ) -> Dict[str, np.ndarray]:
        task_indices = [int(i) for i in task_indices]
        if keys is None:
            keys = [k for k in self.metrics.keys() if k != "idx"]
        else:
            keys = list(keys)
        for k in keys:
            if k not in self.metrics:
                raise KeyError(f"Metric key '{k}' not found. Available: {sorted(self.metrics.keys())}")

        pos, missing = [], []
        for idx in task_indices:
            p = self.idx_to_pos.get(idx, None)
            if p is None:
                missing.append(idx)
                pos.append(-1)
            else:
                pos.append(p)

        if missing and strict:
            raise KeyError(f"Missing {len(missing)} task idx: {missing[:10]}{'...' if len(missing)>10 else ''}")

        pos = np.asarray(pos, dtype=np.int64)

        out: Dict[str, np.ndarray] = {}
        if return_idx:
            out["idx"] = np.asarray(task_indices, dtype=np.int64)

        for k in keys:
            arr = self.metrics[k]
            y = np.empty((len(task_indices),) + arr.shape[1:], dtype=np.float64)
            ok = pos >= 0
            y[ok] = arr[pos[ok]]
            if (~ok).any():
                y[~ok] = np.nan
            out[k] = y
        return out


async def eki_metrics_per_task_one_step_cached(
    cache: MetricCache,
    graph: Any,
    dataset_train: Any,
    task_indices: Sequence[int],
    ensemble_size: int = 6,
    keys: Optional[Sequence[str]] = None,
) -> Dict[str, np.ndarray]:
    # graph/dataset_train/ensemble_size 只是为了兼容原接口，不使用
    return cache.query(task_indices, keys=keys, strict=True, return_idx=False)


def metrics_to_per_task(task_indices: Sequence[int], metrics: Dict[str, np.ndarray]) -> List[dict]:
    out = []
    for i, idx in enumerate(task_indices):
        row = {"idx": int(idx)}
        for k, v in metrics.items():
            row[k] = float(v[i]) if np.ndim(v) == 1 else v[i]
        out.append(row)
    return out
