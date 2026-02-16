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
from experiments.train_mmlu import train, train2, train3
from experiments.evaluate_mmlu import evaluate
from AgentPrune.utils.const import AgentPrune_ROOT
from AgentPrune.utils.globals import PromptTokens, CompletionTokens

from experiments.train_mmlu_reinforce_combination import (
    compute_combo_gradients_from_indices,
    compute_task_rollouts_and_grads,
    select_best_combo_streaming_topk,
    _grad_of_single_rollout_minimal
)

import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Process some parameters.")

    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered','Star', 'Mesh',
                                 'FakeFullConnected','FakeRandom','FakeChain','FakeStar','FakeMesh','FakeAGRandom','FakeAGFull'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1,
                        help="learning rate")
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
    parser.add_argument('--pruning_rate', type=float, default=0.25,
                        help="The Rate of Pruning. Default 0.05.")
    #parser.add_argument('--llm_name', type=str, default="gpt-4-1106-preview",
    #                    help="Model name, None runs the default ChatGPT4")
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

import json
from pathlib import Path

def save_combo_infos_with_grads(combo_infos, out_dir: str, grads_key: str = "grad_flat"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 保存元信息（不含大 tensor）
    meta_path = out_dir / "combo_meta.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for item in combo_infos:
            rec = {
                "combo": item["combo"],
                "avg_reward": item["avg_reward"],
                "grad_norm": item["grad_norm"],
            }
            f.write(json.dumps(rec) + "\n")

    # 2) 保存梯度矩阵（大头）
    grads = torch.stack([item[grads_key].detach().cpu() for item in combo_infos], dim=0)
    torch.save(grads, out_dir / "combo_grads.pt")

    # 3) 额外保存一个索引，便于对齐检查
    index_path = out_dir / "combo_index.pt"
    combos = [item["combo"] for item in combo_infos]
    torch.save(combos, index_path)

    return str(meta_path), str(out_dir / "combo_grads.pt"), str(index_path)


def select_top10_by_l2_then_grad_sum(grads_path: str, combos_path: str, topk: int = 10, device: str = "cpu"):
    G = torch.load(grads_path, map_location=device)          # (N, D)
    combos = torch.load(combos_path)                         # list[list[int]], len=N

    # 1) L2 norm
    l2 = torch.linalg.norm(G, dim=1)                         # (N,)

    # 2) grad_flat.sum()
    gsum = G.sum(dim=1)                                      # (N,)

    # 3) 取 L2 最大的 topk
    top_vals, top_idx = torch.topk(l2, k=min(topk, l2.numel()), largest=True)

    # 4) 打印 top10 信息（combo, l2, gsum）
    print(f"Top-{len(top_idx)} by L2 norm:")
    for rank, i in enumerate(top_idx.tolist(), start=1):
        print(f"[{rank:02d}] combo={combos[i]}  l2={l2[i].item():.6e}  grad_sum={gsum[i].item():.6e}")

    # 5) 在 top10 里选 grad_sum 最大的那个
    best_in_top = top_idx[torch.argmax(gsum[top_idx])].item()

    print("\nSelected (max grad_sum among top-L2):")
    print(f"combo={combos[best_in_top]}  l2={l2[best_in_top].item():.6e}  grad_sum={gsum[best_in_top].item():.6e}")

    return best_in_top, combos[best_in_top], l2[best_in_top].item(), gsum[best_in_top].item()

def seed_everything(seed: int = 0):
    import os, random
    import numpy as np
    import torch

    # 1) python
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # 2) numpy
    np.random.seed(seed)

    # 3) torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

async def main():
    args = parse_args()
    
    mode = args.mode
    decision_method = args.decision_method
    agent_names = [name for name,num in zip(args.agent_names,args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(mode,len(agent_names))
    limit_questions = 153
    #if args.optimized_spatial or args.optimized_temporal:
    #    limit_questions = limit_questions + args.num_iterations

    seed_everything(42)

    graph = Graph(domain=args.domain,
                  llm_name=args.llm_name,
                  agent_names=agent_names,
                  decision_method=decision_method,
                  optimized_spatial=args.optimized_spatial,
                  optimized_temporal=args.optimized_temporal,
                  **kwargs)
    download()
    dataset_train = MMLUDataset('test')#dev')
    dataset_val = MMLUDataset('val')

    assert graph.spatial_logits.requires_grad, "graph.spatial_logits must require grad"
    assert graph.temporal_logits.requires_grad, "graph.temporal_logits must require grad"

    task_indices = [7816, 13422, 9658, 13025, 4173, 8510, 11980, 1521, 2052, 11223, 4, 2101, 11810, 2459, 1729, 9450, 9937, 321, 10269, 13551]
    
    #task_indices = [13551, 9450, 7816, 4, 11223, 10269, 1729, 11980, 2459, 4173]

    pn1, cn1 = read_tokens()

    #pack1 = await compute_task_rollouts_and_grads(
    #    graph=graph,
    #    dataset=dataset_train,
    #    task_indices=task_indices,
    #    num_rounds=args.num_rounds,
    #    max_concurrency=8,
    #    return_rollouts=True,
    #    return_raw_out=False,
    #    grad_fn=_grad_of_single_rollout_minimal,  # 或你已有的 _grad_of_single_rollout
    #)
    #
    #rollouts = pack1["rollouts"]
    #per_task = pack1["per_task"]
    #idx2grad_flat = pack1["idx2grad_flat"]
    #
    ## ========== 模块2：全组合遍历 + 维护topK + 最终按sum选best ==========
    #sel = select_best_combo_streaming_topk(
    #    task_indices=task_indices,
    #    idx2grad_flat=idx2grad_flat,
    #    combo_size=10,
    #    topk=10,
    #)
    #
    #best_combo = sel["best_combo"]
    #best_l2 = sel["best_l2"]
    #best_gsum = sel["best_gsum"]
    #
    #print("best_combo:", best_combo)
    #print("best_l2:", best_l2, "best_gsum:", best_gsum)


    # ============== 第一版的代码
    #combo_pack = await compute_combo_gradients_from_indices(
    #                graph=graph,
    #                dataset=dataset_train,
    #                task_indices=task_indices,
    #                num_rounds=args.num_rounds,
    #                max_concurrency=8,
    #                combo_size=10,
    #                combo_mode="all",
    #                max_combos=None,   # 全组合：184756 个
    #                return_rollouts=True,
    #            )
    #
    #rollouts = combo_pack["rollouts"]
    #per_task = combo_pack["per_task"]
    #combo_infos = combo_pack["combos"] 
    #
    #meta_path, grads_path, idx_path = save_combo_infos_with_grads(
    #    combo_infos,
    #    out_dir="outputs/combo_dump"
    #)
    #print("saved:", meta_path, grads_path, idx_path)
    #
    #best_row, best_combo, best_l2, best_gsum = select_top10_by_l2_then_grad_sum(
    #    grads_path="outputs/combo_dump/combo_grads.pt",
    #    combos_path="outputs/combo_dump/combo_index.pt",
    #    topk=10,
    #    device="cpu",
    #)
    best_combo = [7816, 13025, 11980, 2052, 11223, 1729, 9450, 9937, 321, 10269] 
    best_combo = interleave_ends(best_combo)
    #breakpoint()

    print("Before train: graph")
    print(graph.spatial_logits.detach())
    print(graph.temporal_logits.detach())

    p0, c0 = read_tokens()
    task_prompt = p0 - pn1
    task_completion = c0 - cn1
    print("=== Task evaluation tokens ===", task_prompt, task_completion, task_prompt + task_completion)


    if args.optimized_spatial or args.optimized_temporal:
        # 随机任务训练
        #await train(graph=graph,dataset=dataset_train,num_iters=args.num_iterations,num_rounds=args.num_rounds,
        #            lr=args.lr,batch_size=args.batch_size_train,imp_per_iters=args.imp_per_iterations,pruning_rate=args.pruning_rate)
        
        # 检查任务的index
        #await train3(graph=graph,dataset=dataset_train,num_iters=args.num_iterations,num_rounds=args.num_rounds,
        #            lr=args.lr,batch_size=args.batch_size_train,imp_per_iters=args.imp_per_iterations,pruning_rate=args.pruning_rate)
        
        # 指定任务训练
        #task_indices = [160, 473, 27, 629, 808, 460, 10, 16, 1000, 2000] # no attack
        #task_indices = [12, 986 ,160,719,473,32,808,460,629,27] # under attack no ag
        task_indices = [13551, 9450, 7816, 4, 11223, 10269, 1729, 11980, 2459, 4173]
        #task_indices = [11690, 10268, 13551, 7355, 10101, 4061, 4683, 11810, 8731, 10614, 7816, 8510, 7994, 10157, 12103, 2391, 10324, 9076, 11260, 2171, 10824, 8180
        #                , 12073, 10103, 11980, 6685, 7873, 11223, 11895, 8358, 8226, 8608, 12353, 13025, 889, 2801, 2600, 4646, 12221, 74] # 40个打乱
        
        #task_indices = [13247, 418, 9876, 2561, 7340, 11985, 602, 1401, 8659, 3921,
        #                 11047, 274, 9563, 6842, 1289, 13755, 5016, 820, 10432, 2997,
        #                 741, 12168, 8773, 3459, 136, 9184, 6640, 2095, 11302, 487,
        #                 7771, 5908, 12541, 1684, 999, 13120, 4526, 883, 10675, 3189] # 随机40个
        #task_indices = [ 889,13025, 12353,
        #                    8608, 8226, 8358, 11895, 11223, 11980,
        #                    8180, 2171, 10324, 12103, 7994,
        #                    7816, 8731, 4683, 10101,
        #                    13551, 11690
        #                ] # 从小到大
        #random.shuffle(task_indices)
        #task_indices = [ 889, 11690,13025, 13551, 12353,10101,8608, 4683, 8226, 8731, 8358, 
        #                7816, 11895, 7994,11223, 12103, 11980,10324, 8180, 2171] # 大小掺杂
        #task_indices = [10268, 7355,
        #                    9076, 4061, 10614, 11810, 10103, 8510, 10157, 2391,
        #                    12073, 11260, 10824,
        #                    6685, 74, 12221, 4646, 2600, 2801, 889,
        #                    13025, 12353,
        #                    8608, 8226, 8358, 11895, 11223, 11980,
        #                    8180, 2171, 10324, 12103, 7994,
        #                    7816, 8731, 4683, 10101,
        #                    13551, 11690
        #                ]
        await train2(graph=graph,dataset=dataset_train,
                    task_indices=best_combo,#task_indices,
                    num_iters=args.num_iterations,num_rounds=args.num_rounds,
                    lr=args.lr,batch_size=args.batch_size_train,imp_per_iters=args.imp_per_iterations,pruning_rate=args.pruning_rate)
    

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

if __name__ == "__main__":
    asyncio.run(main())
