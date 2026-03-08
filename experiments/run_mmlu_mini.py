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

from experiments.train_mmlu_mini import train
from experiments.train_mmlu_eki import train_eki # 最早的测试函数。可能是4步训练，每步用2个随机任务多次训练
from experiments.train_mmlu_eki import train_eki_final # 新的训练函数。沿用上一步的逻辑，但是可以指定任务。
from experiments.train_mmlu_eki import train_eki_final2 # 新的训练函数。沿用上一步的逻辑，但是可以指定任务。
from experiments.train_mmlu_eki import eki_metrics_per_task_one_step

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
    #task_indices = [27, 160, 719, 443, 883, 887, 32, 629, 808, 945, 473, 157, 986, 113, 12, 59, 464, 100, 460, 221]#, 30, 40, 50] # 这个是错的
    #task_indices = [7816, 13422, 9658, 13025, 4173, 8510, 11980, 1521, 2052, 11223, 4, 2101, 11810, 2459, 1729, 9450, 9937, 321, 10269, 13551]
    # 选20个
    task_indices = [13466]

    # 选40个
    #task_indices = [ 7816, 11690,  8731,  4683, 10101,    74, 12221, 11980,  4646,
    #    8226,  9076, 11223,  8358,  4061, 10614, 12353,  2600,  2801,
    #    6685, 12103,   889,  7994, 10324, 11810, 10268, 10103,  8608,
    #    8510, 10157,  2391, 13025, 12073,  8180,  7355,  2171, 11895,
    #   11260, 13551, 10824,  7873]

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
