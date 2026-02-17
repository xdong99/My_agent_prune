import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

import asyncio
from typing import Union, Literal, List
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

from experiments.train_mmlu_mini import train
from experiments.train_mmlu_eki import train_eki # 最早的测试函数。可能是4步训练，每步用2个随机任务多次训练
from experiments.train_mmlu_eki import train_eki_final # 新的训练函数。沿用上一步的逻辑，但是可以指定任务。
from experiments.train_mmlu_eki import train_eki_final2 # 新的训练函数。沿用上一步的逻辑，但是可以指定任务。
from experiments.train_mmlu_eki import eki_metrics_per_task_one_step
from experiments.embedding_mmlu import select_mmlu_representatives
from experiments.metric_cache import MetricCache, eki_metrics_per_task_one_step_cached

from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, WhiteKernel
import json
from pathlib import Path


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

    Z_obs = Z_all[obs_pos]
    d = Z_obs.shape[1]

    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e3)) \
             + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-8, 1e0))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,
        normalize_y=True,
        n_restarts_optimizer=5,
        random_state=seed,
    )
    gp.fit(Z_obs, y_s)

    mu, std = gp.predict(Z_all, return_std=True)
    return Z_all, gp, mu, std


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
    selected_idx, selected_emb = select_mmlu_representatives(split="test", sample_n=1000, seed=0, budget=50, return_selected_embeddings=True)
    idx_all = np.array(selected_idx)              # (50,)
    X_all = np.asarray(selected_emb)#, dtype=np.float64)  # (50,384)
    
    # 测试：计算全部的数据
    #df50, metrics50 = await eval_and_cache_metrics(
    #    graph, dataset_train, selected_idx, ensemble_size=6,
    #    out_dir="outputs/metric_cache",
    #    tag="selected50_es6"
    #)

    # TS部分
    cache = MetricCache.from_npz("outputs/metric_cache/selected50_es6.npz")
    metrics = await eki_metrics_per_task_one_step_cached(cache, graph, dataset_train, selected_idx, ensemble_size=6, keys=["kl"])



    # 随机选10个
    rng = np.random.default_rng(0)
    init_pos = rng.choice(len(idx_all), size=10, replace=False)
    init_task_indices = idx_all[init_pos].tolist()

    #metrics = await eki_metrics_per_task_one_step(graph, dataset_train, init_task_indices, ensemble_size=6)
    metrics = await eki_metrics_per_task_one_step_cached(cache, graph, dataset_train, init_task_indices, ensemble_size=6, keys=["kl"])
    y_init = np.asarray(metrics["kl"], dtype=np.float64)  # (10,)
    X_init = X_all[init_pos]  # (10,384)

    # 进行PLS投影
    x_scaler = StandardScaler()
    y_scaler = StandardScaler()

    X_init_s = x_scaler.fit_transform(X_init)                 # (10,384)
    y_init_t = np.arcsinh(y_init)
    y_init_s = y_scaler.fit_transform(y_init_t.reshape(-1,1)).ravel()

    n_components = 3  # 先用3，后面可调 2~5
    pls = PLSRegression(n_components=n_components)
    pls.fit(X_init_s, y_init_s)

    X_all_s = x_scaler.transform(X_all)   # 注意：用 init 的 scaler
    Z_all = pls.transform(X_all_s)        # (50, n_components)

    Z_init = Z_all[init_pos]

    # GP
    d = Z_init.shape[1]

    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=np.ones(d), length_scale_bounds=(1e-2, 1e2)) \
             + WhiteKernel(noise_level=1e-3, noise_level_bounds=(1e-6, 1e0))

    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=0.0,            # 有 WhiteKernel 时通常设 0
        normalize_y=True,     # 再做一次 y normalization（可与 y_scaler 二选一；两者都开也能跑但略重复）
        n_restarts_optimizer=5,
        random_state=0
    )

    gp.fit(Z_init, y_init_s)

    # 再次采样
    observed_mask = np.zeros(len(idx_all), dtype=bool)
    observed_mask[init_pos] = True

    # 从后验采样：返回 shape (n_points, n_samples)
    samples = gp.sample_y(Z_all, n_samples=1, random_state=0).ravel()  # 有一个种子问题 # 是所有的sample

    # 只在未观测的里选最大
    candidates = np.where(~observed_mask)[0]
    next_pos = candidates[np.argmax(samples[candidates])]
    next_idx = int(idx_all[next_pos])

    breakpoint()


    metrics = await eki_metrics_per_task_one_step(graph, dataset_train, task_indices, ensemble_size=6)
    per_task = [
        {"idx": idx, "kl": kl, "pred_var": pv, "cosine": cs}
        for idx, kl, pv, cs in zip(task_indices, metrics["kl"], metrics["pred_var"], metrics["cosine"])
    ]
    print('information matrix', per_task)
    p0, c0 = read_tokens()
    task_prompt = p0 - pn1
    task_completion = c0 - cn1
    print("=== Task evaluation tokens ===", task_prompt, task_completion, task_prompt + task_completion)
    df = pd.DataFrame(per_task)
    print(df.to_string(index=False))

    df_sorted = df.sort_values("kl", ascending=False)
    k = 10
    top_idx_list = df_sorted["idx"].head(k).tolist()
    print("top_idx_list:", top_idx_list)

    breakpoint()

    # ===== train
    print("Before train: graph")
    print(graph.spatial_logits.detach())
    print(graph.temporal_logits.detach())


    task_indices = [160, 473, 27, 629, 808, 460, 10, 16, 1000, 2000]
    #task_indices = [327, 1846, 2590, 4738, 6912, 8127, 10354, 13789] # random
    p0, c0 = read_tokens()
    if args.optimized_spatial or args.optimized_temporal:
        await train_eki_final2(graph=graph,dataset=dataset_train, task_indices=task_indices,num_iters=5,batch_size=2, imp_per_iters=1,pruning_rate=args.pruning_rate,ensemble_size=8)
    #    #await train(graph=graph,dataset=dataset_train,num_iters=4,
    #    #            num_rounds=args.num_rounds,
    #    #            lr=args.lr,batch_size=args.batch_size,
    #    #            imp_per_iters=args.imp_per_iterations,pruning_rate=args.pruning_rate)

    print("After train: graph")
    print(graph.spatial_logits.detach())
    print(graph.temporal_logits.detach())

    p1, c1 = read_tokens()
    train_prompt = p1 - p0
    train_completion = c1 - c0

    p2, c2 = read_tokens()

    score = await evaluate(graph=graph,dataset=dataset_val,num_rounds=args.num_rounds,limit_questions=limit_questions,eval_batch_size=args.batch_size)
    print(f"Score: {score}")

    p3, c3 = read_tokens()
    eval_prompt = p3 - p2
    eval_completion = c3 - c2

    print("=== Train tokens ===", train_prompt, train_completion, train_prompt + train_completion)
    print("=== Eval tokens  ===", eval_prompt, eval_completion, eval_prompt + eval_completion)
    print("=== Total tokens ===", p3, c3, p3 + c3)
    #print(f"Score: {score}")


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
