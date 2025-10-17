import argparse
import asyncio
import copy
import json
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')
import random
import time
from pathlib import Path
from typing import List, Literal, Optional, Union

import numpy as np
import torch

from KVCOMM.graph.graph import Graph
from KVCOMM.llm.config import KVCommConfig
from KVCOMM.tools.reader.readers import JSONLReader
from KVCOMM.utils.globals import Time
from KVCOMM.utils.log import configure_logging, logger
from KVCOMM.utils.metrics import metrics_recorder
from datasets.gsm8k_dataset import (
    get_pred_value,
    gsm_data_process
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEED = int(os.getenv("SEED", 42))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def load_result(result_file: Path) -> list:
    if not result_file.exists():
        os.makedirs(result_file.parent, exist_ok=True)
        with open(result_file, "w", encoding="utf-8") as file:
            json.dump([], file)
    with open(result_file, "r", encoding="utf-8") as file:
        return json.load(file)


def dataloader(data_list, batch_size, i_batch):
    return data_list[i_batch * batch_size : i_batch * batch_size + batch_size]


def parse_args():
    parser = argparse.ArgumentParser(description="KVCOMM Experiments on GSM8K")
    parser.add_argument("--dataset_json", type=str, default="datasets/gsm8k/gsm8k.jsonl")
    parser.add_argument("--llm_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument(
        "--mode",
        type=str,
        default="FullConnected",
        choices=["DirectAnswer", "FullConnected", "Random", "Chain", "Debate", "Layered", "Star"], help="The communication topology among agents."
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size.")
    parser.add_argument("--domain", type=str, default="gsm8k")
    parser.add_argument(
        "--agent_names",
        nargs="+",
        type=str,
        default=["MathSolver"],
        help="List of agent names in the graph.",
    )
    parser.add_argument(
        "--agent_nums",
        nargs="+",
        type=int,
        default=[3],
        help="List of agent counts corresponding to agent names.",
    )
    parser.add_argument(
        "--decision_method",
        type=str,
        default="FinalRefer",
        help="Decision method for the graph.",
    )
    parser.add_argument(
        "--execution_mode",
        type=str,
        default="default",
        choices=["default", "allow_kv_reuse"],
        help="Execution strategy for the graph.",
    )
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "result" / "gsm8k"), help="Directory to save the output results.")
    parser.add_argument("--prefix", type=str, default="Q:\n", help="The prefix text for the input query, kept the same as the default dense prefill mode.")
    parser.add_argument("--kv-threshold", type=float, default=None, help="Threshold for key-value memory usage.")
    parser.add_argument("--kv-max-anchor-num", type=int, default=None, help="Maximum number of anchors for key-value memory.")
    parser.add_argument("--kv-window-size", type=int, default=None, help="Window size for key-value memory update.")
    parser.add_argument("--kv-thread-workers", type=int, default=None, help="Number of thread workers for key-value memory processing.")
    parser.add_argument("--kv-worker-timeout", type=float, default=None, help="Timeout for key-value memory workers processing.")
    args = parser.parse_args()

    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args


async def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_path=output_dir / "logs/log.txt")
    dataset_raw = JSONLReader.parse_file(args.dataset_json)
    dataset = gsm_data_process(dataset_raw)

    current_time = Time.instance().value or time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
    Time.instance().value = current_time
    result_file = output_dir / f"{args.domain}_{args.llm_name}_{current_time}.json"
    latency_target = str(output_dir)

    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(args.mode, len(agent_names))

    kv_config: Optional[KVCommConfig] = None
    if args.execution_mode == "allow_kv_reuse":
        kv_config = KVCommConfig.from_env().apply_overrides(
            threshold=args.kv_threshold,
            max_anchor_num=args.kv_max_anchor_num,
            window_size=args.kv_window_size,
            thread_pool_workers=args.kv_thread_workers,
            worker_timeout=args.kv_worker_timeout,
        )
    else:
        kv_config = KVCommConfig.from_env()

    graph = Graph(
        domain=args.domain,
        llm_name=args.llm_name,
        agent_names=agent_names,
        decision_method=args.decision_method,
        kv_config=kv_config,
        **kwargs,
    )

    num_batches = int(len(dataset) / args.batch_size)
    total_solved, total_executed = 0, 0

    for i_batch in range(num_batches):
        logger.opt(colors=True).info(f"<blue>[BATCH]</blue> {i_batch} {'-' * 40}")
        start_ts = time.time()
        current_batch = dataloader(dataset, args.batch_size, i_batch)
        if not current_batch:
            logger.warning("No more data available.")
            break

        tasks = []
        meta_info = []
        for record in current_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits
            task = record["task"]
            input_dict = {"task": task, "_batch_index": i_batch}

            mode_kwargs = {}
            if args.execution_mode == "allow_kv_reuse":
                mode_kwargs = {
                    "prefix": args.prefix,
                    "output_dir": latency_target,
                }

            tasks.append(
                asyncio.create_task(
                    realized_graph.arun(
                        input_dict,
                        1,
                        mode=args.execution_mode,
                        **mode_kwargs,
                    )
                )
            )
            meta_info.append(
                {
                    "task": task,
                    "step": record["step"],
                    "answer": record["answer"],
                }
            )

        batch_results = await asyncio.gather(*tasks)
        results_by_task = {result.get("task"): result.get("answers", []) for result in batch_results}
        data = load_result(result_file)

        for info in meta_info:
            task = info["task"]
            true_answer = info["answer"]
            answers = results_by_task.get(task, [])
            response = answers if isinstance(answers, list) else [answers]

            if not response:
                predicted = "0"
            else:
                predicted = await get_pred_value(response[0])

            try:
                is_solved = float(predicted) == float(true_answer)
            except Exception:
                is_solved = False

            total_solved += is_solved
            total_executed += 1
            accuracy = total_solved / total_executed

            updated_item = {
                "Question": task,
                "Answer": true_answer,
                "Step": info["step"],
                "Response": response,
                "Attempt answer": predicted,
                "Solved": bool(is_solved),
                "Total solved": total_solved,
                "Total executed": total_executed,
                "Accuracy": accuracy,
            }
            data.append(updated_item)

        with open(result_file, "w", encoding="utf-8") as file:
            json.dump(data, file, indent=4)

        logger.opt(colors=True).info(
            f"<blue>[BATCH TIME]</blue> {time.time() - start_ts:.3f}s"
        )
        logger.opt(colors=True).info(f"<blue>[ACCURACY]</blue> {accuracy:.4f}")
        metrics_recorder.log_cumulative(batch_index=i_batch)


def get_kwargs(
    mode: Union[
        Literal["DirectAnswer"],
        Literal["FullConnected"],
        Literal["Random"],
        Literal["Chain"],
        Literal["Debate"],
        Literal["Layered"],
        Literal["Star"],
    ],
    N: int,
):
    fixed_spatial_masks: List[List[int]] = None                
    fixed_temporal_masks: List[List[int]] = None                
    node_kwargs = None

    def generate_layered_graph(n, layer_num=2):
        adj_matrix = [[0 for _ in range(n)] for _ in range(n)]
        base_size = n // layer_num
        remainder = n % layer_num
        layers: List[int] = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(n):
            current_layer = layers[i]
            for j in range(n):
                if layers[j] == current_layer + 1:
                    adj_matrix[i][j] = 1
        return adj_matrix

    def generate_star_graph(n):
        matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j] = 1
        return matrix

    if mode == "DirectAnswer":
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{"role": "Programming Expert"}]
    elif mode == "FullConnected":
        fixed_spatial_masks = [[1 if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Random":
        fixed_spatial_masks = [[random.randint(0, 1) if i != j else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[random.randint(0, 1) for _ in range(N)] for _ in range(N)]
    elif mode == "Chain":
        fixed_spatial_masks = [[1 if i == j + 1 else 0 for i in range(N)] for j in range(N)]
        fixed_temporal_masks = [[1 if i == 0 and j == N - 1 else 0 for i in range(N)] for j in range(N)]
    elif mode == "Debate":
        fixed_spatial_masks = [[0 for _ in range(N)] for _ in range(N)]
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Layered":
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    elif mode == "Star":
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = [[1 for _ in range(N)] for _ in range(N)]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return {
        "fixed_spatial_masks": fixed_spatial_masks,
        "fixed_temporal_masks": fixed_temporal_masks,
        "node_kwargs": node_kwargs,
    }


if __name__ == "__main__":
    asyncio.run(main())
