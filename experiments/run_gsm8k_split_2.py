import sys
import os
import argparse
import yaml
import json
import time
import asyncio
from pathlib import Path
import torch
import copy
from typing import List, Union, Literal, Tuple
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')

from AgentPrune.utils.const import AgentPrune_ROOT
from AgentPrune.graph.graph import Graph
from AgentPrune.tools.reader.readers import JSONLReader
from AgentPrune.utils.globals import Time
from dataset.gsm8k_dataset import gsm_data_process, gsm_get_predict
from AgentPrune.utils.globals import Cost, PromptTokens, CompletionTokens


def save_json(path: Path, data):
    with open(path, 'w', encoding='utf-8') as file:
        json.dump(data, file, indent=4)


def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch * batch_size: i_batch * batch_size + batch_size]


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def parse_args():
    parser = argparse.ArgumentParser(description="AgentPrune Experiments on gsm8k with explicit train/test split")

    parser.add_argument("--train_json", type=str, default="dataset/gsm8k/train.jsonl",
                        help="Path to training set json file.")
    parser.add_argument("--test_json", type=str, default="dataset/gsm8k/test.jsonl",
                        help="Path to test set json file.")
    parser.add_argument("--result_file", type=str, default=None)


    parser.add_argument("--llm_name", type=str, default="gpt-4.1-nano")
    parser.add_argument('--mode', type=str, default='FullConnected',
                        choices=['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered', 'Star'],
                        help="Mode of operation. Default is 'FullConnected'.")
    parser.add_argument('--lr', type=float, default=0.1, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=4, help="batch size")
    parser.add_argument('--imp_per_iterations', type=int, default=5, help="Prune every few iterations. Default 5.")
    parser.add_argument('--num_rounds', type=int, default=1, help="Number of optimization/inference rounds for one query")
    parser.add_argument('--pruning_rate', type=float, default=0.25, help="The Rate of Pruning. Default 0.25.")
    parser.add_argument('--train_num_iterations', type=int, default=10, help="The number of training batches to run.")
    parser.add_argument('--domain', type=str, default="gsm8k", help="Domain (the same as dataset name), default 'gsm8k'")
    parser.add_argument('--agent_names', nargs='+', type=str, default=['MathSolver'],
                        help='Specify agent names as a list of strings')
    parser.add_argument('--agent_nums', nargs='+', type=int, default=[4],
                        help='Specify the number of agents for each name in agent_names')
    parser.add_argument('--decision_method', type=str, default='FinalRefer',
                        help='The decision method of AgentPrune')
    parser.add_argument('--optimized_spatial', action='store_true')
    parser.add_argument('--optimized_temporal', action='store_true')

    parser.add_argument('--shuffle_train', action='store_true',
                        help='Shuffle training set before training.')
    parser.add_argument('--eval_batch_size', type=int, default=None,
                        help='Batch size used during test evaluation. Defaults to batch_size.')    
    parser.add_argument('--train_seed', type=int, default=None,
                        help='Local random seed used only to shuffle the training set.')
    parser.add_argument('--test_num_iterations', type=int, default=None,
                        help='Number of test batches to run. Default None means evaluate the full test set.')

    parser.add_argument('--test_ratio', type=float, default=0.35,
                        help='Fraction of test_json kept as the actual evaluation pool.')
    parser.add_argument('--test_split_seed', type=int, default=0,
                        help='Random seed used for splitting test_json if test_shuffle is enabled.')
    parser.add_argument('--test_shuffle', action='store_true',
                        help='Shuffle test_json before taking the tail portion as evaluation pool.')

    args = parser.parse_args()
    result_path = AgentPrune_ROOT / "result"
    os.makedirs(result_path, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args


def perturb_train_set(train_set, shuffle_train: bool, train_seed: int):
    train_set = list(train_set)
    if shuffle_train:
        rng = random.Random(train_seed)
        rng.shuffle(train_set)
    return train_set

def read_tokens():
    p = int(PromptTokens.instance().value)
    c = int(CompletionTokens.instance().value)
    return p, c

def build_eval_test_set(full_test_set, test_ratio: float, test_split_seed: int, test_shuffle: bool):
    full_test_set = list(full_test_set)
    if test_shuffle:
        rng = random.Random(test_split_seed)
        rng.shuffle(full_test_set)

    split_idx = int(len(full_test_set) * (1 - test_ratio))
    eval_test_set = full_test_set[split_idx:]
    return eval_test_set

async def run_batches(
    graph: Graph,
    dataset,
    batch_size: int,
    num_rounds: int,
    result_file: Path,
    phase: str,
    optimizer=None,
    prune_every: int = None,
    pruning_rate: float = None,
    max_batches: int = None,
    allow_optimization: bool = False,
):
    num_batches = int(len(dataset) / batch_size)
    if max_batches is not None:
        num_batches = min(num_batches, max_batches)

    total_solved = 0
    total_executed = 0
    all_records = []
    batch_metrics = []

    for i_batch in range(num_batches):
        print(f"[{phase}] Batch {i_batch}", 80 * '-')
        start_ts = time.time()
        answer_log_probs = []
        answers = []
        steps = []
        current_batch = dataloader(dataset, batch_size, i_batch)
        if not current_batch:
            break

        for record in current_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits

            task = record["task"]
            step = record["step"]
            answer = record["answer"]
            answers.append(answer)
            steps.append(step)
            input_dict = {"task": task}
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds)))

        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)

        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        batch_solved = 0

        for record, step, answer, log_prob, true_answer in zip(current_batch, steps, raw_answers, log_probs, answers):
            predict_answer = gsm_get_predict(answer[0])
            try:
                is_solved = float(predict_answer) == float(true_answer)
            except Exception:
                is_solved = False

            total_solved += int(is_solved)
            total_executed += 1
            batch_solved += int(is_solved)
            cumulative_accuracy = total_solved / total_executed
            utility = float(is_solved)
            utilities.append(utility)

            updated_item = {
                "Phase": phase,
                "Question": record,
                "Answer": true_answer,
                "Step": step,
                "Response": answer,
                "Attempt answer": predict_answer,
                "Solved": bool(is_solved),
                "Total solved": total_solved,
                "Total executed": total_executed,
                "Cumulative accuracy": cumulative_accuracy,
            }
            all_records.append(updated_item)

            if allow_optimization:
                single_loss = -log_prob * utility
                loss_list.append(single_loss)

        batch_accuracy = batch_solved / len(current_batch)
        batch_info = {
            "phase": phase,
            "batch_index": i_batch,
            "batch_size": len(current_batch),
            "batch_accuracy": batch_accuracy,
            "cumulative_accuracy": total_solved / total_executed,
            "solved_in_batch": batch_solved,
            "total_solved": total_solved,
            "total_executed": total_executed,
        }
        batch_metrics.append(batch_info)

        if allow_optimization and loss_list:
            total_loss = torch.mean(torch.stack(loss_list))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if prune_every is not None and prune_every > 0 and (i_batch + 1) % prune_every == 0:
                graph.update_masks(pruning_rate)
        else:
            total_loss = None

        print(f"[{phase}] Batch time {time.time() - start_ts:.3f}")
        print(f"[{phase}] Batch accuracy: {batch_accuracy:.4f}")
        print(f"[{phase}] Cumulative accuracy: {total_solved / total_executed:.4f}")
        print(f"[{phase}] utilities: {utilities}")
        if total_loss is not None:
            print(f"[{phase}] Loss: {float(total_loss.detach().cpu()):.6f}")

    summary = {
        "phase": phase,
        "num_examples_used": total_executed,
        "num_batches_used": len(batch_metrics),
        "solved": total_solved,
        "accuracy": (total_solved / total_executed) if total_executed > 0 else None,
        "batch_metrics": batch_metrics,
        "records": all_records,
    }
    save_json(result_file, summary)
    return summary


async def main():
    args = parse_args()
    train_set = JSONLReader.parse_file(args.train_json)
    train_set = gsm_data_process(train_set)

    full_test_set = JSONLReader.parse_file(args.test_json)
    full_test_set = gsm_data_process(full_test_set)

    train_set = perturb_train_set(train_set, args.shuffle_train, args.train_seed)
    test_set = build_eval_test_set(full_test_set, args.test_ratio, args.test_split_seed, args.test_shuffle,)

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_dir = Path(f"{AgentPrune_ROOT}/result/gsm8k")
    result_dir.mkdir(parents=True, exist_ok=True)

    if args.result_file is not None:
        base_path = Path(args.result_file)
        if base_path.suffix:
            stem = base_path.stem
            parent = base_path.parent
            suffix = base_path.suffix
        else:
            stem = base_path.name
            parent = base_path.parent
            suffix = ".json"
        parent.mkdir(parents=True, exist_ok=True)
        train_result_file = parent / f"{stem}_train{suffix}"
        test_result_file = parent / f"{stem}_test{suffix}"
        summary_file = parent / f"{stem}_summary{suffix}"
    else:
        prefix = f"{args.domain}_{args.llm_name}_{current_time}"
        train_result_file = result_dir / f"{prefix}_train.json"
        test_result_file = result_dir / f"{prefix}_test.json"
        summary_file = result_dir / f"{prefix}_summary.json"

    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(args.mode, len(agent_names))
    graph = Graph(
        domain=args.domain,
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=args.decision_method,
        optimized_spatial=args.optimized_spatial,
        optimized_temporal=args.optimized_temporal,
        **kwargs,
    )

    do_train = args.optimized_spatial or args.optimized_temporal
    optimizer = None

    print(f"Train set size: {len(train_set)}")
    print(f"Test set size: {len(test_set)}")
    print(f"Training batches requested: {args.train_num_iterations}")
    print(f"Testing batches requested: {args.test_num_iterations}")

    p0, c0 = read_tokens()
    if do_train:
        optimizer = torch.optim.Adam([graph.spatial_logits, graph.temporal_logits], lr=args.lr)
        train_summary = await run_batches(
            graph=graph,
            dataset=train_set,
            batch_size=args.batch_size,
            num_rounds=args.num_rounds,
            result_file=train_result_file,
            phase="train",
            optimizer=optimizer,
            prune_every=args.imp_per_iterations,
            pruning_rate=args.pruning_rate,
            max_batches=args.train_num_iterations,
            allow_optimization=do_train,
        )

    p1, c1 = read_tokens()
    train_prompt = p1 - p0
    train_completion = c1 - c0

    eval_batch_size = args.eval_batch_size or args.batch_size
    test_summary = await run_batches(
        graph=graph,
        dataset=test_set,
        batch_size=eval_batch_size,
        num_rounds=args.num_rounds,
        result_file=test_result_file,
        phase="test",
        optimizer=None,
        prune_every=None,
        pruning_rate=None,
        max_batches=args.test_num_iterations,
        allow_optimization=False,
    )
    p2, c2 = read_tokens()

    final_summary = {
        "train_json": args.train_json,
        "test_json": args.test_json,
        "mode": args.mode,
        "llm_name": args.llm_name,
        "domain": args.domain,
        "agent_names": args.agent_names,
        "agent_nums": args.agent_nums,
        "expanded_agent_names": agent_names,
        "optimized_spatial": args.optimized_spatial,
        "optimized_temporal": args.optimized_temporal,
        "train_seed": args.train_seed,
        "shuffle_train": args.shuffle_train,
        "batch_size": args.batch_size,
        "eval_batch_size": eval_batch_size,
        "num_rounds": args.num_rounds,
        "train_num_iterations": args.train_num_iterations,
        "test_num_iterations": args.test_num_iterations,
        "test_ratio": args.test_ratio,
        "test_split_seed": args.test_split_seed,
        "test_shuffle": args.test_shuffle,
        "full_test_size": len(full_test_set),
        "eval_test_size": len(test_set),
        "train_size": len(train_set),
        "test_size": len(test_set),
        "train_summary": {
            "num_examples_used": train_summary["num_examples_used"] if do_train else 0,
            "num_batches_used": train_summary["num_batches_used"] if do_train else 0,
            "accuracy": train_summary["accuracy"] if do_train else None,
        },
        "test_summary": {
            "num_examples_used": test_summary["num_examples_used"],
            "num_batches_used": test_summary["num_batches_used"],
            "accuracy": test_summary["accuracy"],
        },
        "train_result_file": str(train_result_file),
        "test_result_file": str(test_result_file),
    }
    save_json(summary_file, final_summary)

    print("=" * 80)
    if do_train:
        print(f"Train accuracy: {train_summary['accuracy']}")

    print(f"Train set size: {len(train_set)}")
    print(f"Full test set size: {len(full_test_set)}")
    print(f"Evaluation test pool size: {len(test_set)}")
    print(f"Training batches requested: {args.train_num_iterations}")
    print(f"Testing batches requested: {args.test_num_iterations}")
    print(f"Saved test results to: {test_result_file}")
    print(f"Saved summary to: {summary_file}")

    
    train_prompt = p1 - p0
    train_completion = c1 - c0
    eval_prompt = p2 - p1
    eval_completion = c2 - c1
    print("=== Train tokens ===", train_prompt, train_completion, train_prompt + train_completion)
    print("=== Eval tokens  ===", eval_prompt, eval_completion, eval_prompt + eval_completion)



def get_kwargs(mode: Union[Literal['DirectAnswer'], Literal['FullConnected'], Literal['Random'], Literal['Chain'], Literal['Debate'], Literal['Layered'], Literal['Star']],
               N: int):
    initial_spatial_probability: float = 0.5
    fixed_spatial_masks: List[List[int]] = None
    initial_temporal_probability: float = 0.5
    fixed_temporal_masks: List[List[int]] = None
    node_kwargs = None

    def generate_layered_graph(N, layer_num=2):
        adj_matrix = [[0 for _ in range(N)] for _ in range(N)]
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

    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        for i in range(0, n):
            for j in range(i + 1, n):
                matrix[i][j] = 1
        return matrix

    if mode == 'DirectAnswer':
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{'role': 'Programming Expert'}]
    elif mode == 'FullConnected':
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == 'Random':
        fixed_spatial_masks = [[random.randint(0, 1) if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode == 'Chain':
        fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i == 0 and j == N - 1 else 0 for i in range(N)] for j in range(N)]
    elif mode == 'Debate':
        fixed_spatial_masks = [[0 for _ in range(N)] for _ in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]

    return {
        "initial_spatial_probability": initial_spatial_probability,
        "fixed_spatial_masks": fixed_spatial_masks,
        "initial_temporal_probability": initial_temporal_probability,
        "fixed_temporal_masks": fixed_temporal_masks,
        "node_kwargs": node_kwargs,
    }


if __name__ == '__main__':
    asyncio.run(main())
