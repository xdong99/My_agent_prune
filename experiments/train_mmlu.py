import torch
from typing import Iterator, List, Optional
import pandas as pd
import numpy as np
import time
import asyncio
from typing import List
import copy

from AgentPrune.graph.graph import Graph
from experiments.accuracy import Accuracy
from AgentPrune.utils.globals import Cost, PromptTokens, CompletionTokens

async def train(graph:Graph,
            dataset,
            num_iters:int=100,
            num_rounds:int=1,
            lr:float=0.1,
            batch_size:int = 4,
            imp_per_iters: int = 1,
            pruning_rate: float = 0.05,
          ) -> None:
    
    def infinite_data_loader() -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    record = dataset[idx.item()]
                    yield record
    
    loader = infinite_data_loader()
    
    optimizer = torch.optim.Adam([graph.spatial_logits,graph.temporal_logits], lr=lr)    
    
    for i_iter in range(num_iters):
        print(f"Iter {i_iter}", 80*'-')
        start_ts = time.time()
        correct_answers = []
        answer_log_probs = []

        for i_record, record in zip(range(batch_size), loader):
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits
            input_dict = dataset.record_to_input(record)
            #print(input_dict)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict,num_rounds)))
            correct_answer = dataset.record_to_target_answer(record)
            correct_answers.append(correct_answer)
        
        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)
        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        answers: List[str] = []
        
        for raw_answer, log_prob, correct_answer in zip(raw_answers, log_probs, correct_answers):
            answer = dataset.postprocess_answer(raw_answer)
            answers.append(answer)
            assert isinstance(correct_answer, str), \
                    f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"
            accuracy = Accuracy()
            accuracy.update(answer, correct_answer)
            utility = accuracy.get()
            utilities.append(utility)
            single_loss = - log_prob * utility
            loss_list.append(single_loss)
            #print(f"correct answer:{correct_answer}")
    
        total_loss = torch.mean(torch.stack(loss_list))
        optimizer.zero_grad() 
        total_loss.backward()
        optimizer.step()
        spatial_probs = torch.sigmoid(graph.spatial_logits) # Parameter containing:tensor([-0.0501...,501,  0.0501], requires_grad=True)
        temporal_probs = torch.sigmoid(graph.temporal_logits) # Parameter containing:tensor([-0.0501...,501,  0.0501], requires_grad=True)
        
        print("raw_answers:",raw_answers)
        print("answers:",answers)
        print(f"Batch time {time.time() - start_ts:.3f}")
        print("utilities:", utilities) # [0.0, 0.0, 0.0, 1.0]
        
        if (i_iter+1)%imp_per_iters == 0:
            spatial_masks, temporal_masks = graph.update_masks(pruning_rate)



async def train2(
    graph: Graph,
    dataset,
    task_indices: Optional[List[int]] = None,   # NEW
    num_iters: int = 100,
    num_rounds: int = 1,
    lr: float = 0.1,
    batch_size: int = 4,
    imp_per_iters: int = 1,
    pruning_rate: float = 0.05,
) -> None:

    # --------------------------
    # NEW: deterministic index loader (with wrap-around)
    # --------------------------
    def index_data_loader(task_indices: List[int]) -> Iterator[pd.DataFrame]:
        if task_indices is None or len(task_indices) == 0:
            raise ValueError("task_indices is empty.")

        n_data = len(dataset)
        for idx in task_indices:
            if idx < 0 or idx >= n_data:
                raise IndexError(
                    f"task_indices contains out-of-range index {idx} (dataset size={n_data})."
                )

        ptr = 0
        n_idx = len(task_indices)
        while True:
            idx = task_indices[ptr % n_idx]
            ptr += 1
            yield dataset[int(idx)]

    # (optional) keep old random behavior when task_indices is None
    def random_data_loader() -> Iterator[pd.DataFrame]:
        while True:
            perm = np.random.permutation(len(dataset))   # NOTE: regenerate each cycle
            for idx in perm:
                yield dataset[int(idx)]

    loader = index_data_loader(task_indices) if task_indices is not None else random_data_loader()

    optimizer = torch.optim.Adam([graph.spatial_logits, graph.temporal_logits], lr=lr)

    for i_iter in range(num_iters):
        print(f"Iter {i_iter}", 80 * "-")
        start_ts = time.time()
        correct_answers = []
        answer_log_probs = []

        # batch fetch: now deterministic if task_indices is provided
        for i_record, record in zip(range(batch_size), loader):
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits

            input_dict = dataset.record_to_input(record)
            answer_log_probs.append(asyncio.create_task(realized_graph.arun(input_dict, num_rounds)))

            correct_answer = dataset.record_to_target_answer(record)
            correct_answers.append(correct_answer)

        raw_results = await asyncio.gather(*answer_log_probs)
        raw_answers, log_probs = zip(*raw_results)

        loss_list: List[torch.Tensor] = []
        utilities: List[float] = []
        answers: List[str] = []

        for raw_answer, log_prob, correct_answer in zip(raw_answers, log_probs, correct_answers):
            answer = dataset.postprocess_answer(raw_answer)
            answers.append(answer)

            assert isinstance(correct_answer, str), \
                f"String expected but got {correct_answer} of type {type(correct_answer)} (1)"

            accuracy = Accuracy()
            accuracy.update(answer, correct_answer)
            utility = accuracy.get()
            utilities.append(utility)

            single_loss = -log_prob * utility
            loss_list.append(single_loss)

        total_loss = torch.mean(torch.stack(loss_list))
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        spatial_probs = torch.sigmoid(graph.spatial_logits)
        temporal_probs = torch.sigmoid(graph.temporal_logits)

        print("raw_answers:", raw_answers)
        print("answers:", answers)
        print(f"Batch time {time.time() - start_ts:.3f}")
        print("utilities:", utilities)

        if (i_iter + 1) % imp_per_iters == 0:
            graph.update_masks(pruning_rate)

async def train3(graph:Graph,
            dataset,
            num_iters:int=100,
            num_rounds:int=1,
            lr:float=0.1,
            batch_size:int = 4,
            imp_per_iters: int = 1,
            pruning_rate: float = 0.05,
          ) -> None:
    
    def infinite_data_loader() -> Iterator[pd.DataFrame]:
            perm = np.random.permutation(len(dataset))
            while True:
                for idx in perm:
                    yield int(idx)
    
    loader = infinite_data_loader()

    for i_iter in range(num_iters):
        batch_idxs = [next(loader) for _ in range(batch_size)]
        print(f"Iter {i_iter} batch idxs:", batch_idxs)
        # 不调用 LLM，直接 continue
        continue