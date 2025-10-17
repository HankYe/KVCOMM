import argparse
import asyncio
import sys, os
# Enforce 512-token system prefix and 512-token outputs
os.environ["IN_LENGTH"] = "512"
os.environ["OUT_LENGTH"] = "512"
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.stdout.reconfigure(encoding='utf-8')
import random
import json
import time
from pathlib import Path
from typing import List, Literal, Union, Dict, Any

import numpy as np
import torch
import copy
from KVCOMM.graph.graph import Graph
from KVCOMM.llm.config import KVCommConfig
from KVCOMM.utils.log import configure_logging, logger

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SEED = int(os.getenv("SEED", 42))
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


def parse_args():
    parser = argparse.ArgumentParser(description="KVCOMM Experiments on TTFT Benchmark")
    parser.add_argument(
        "--mode",
        type=str,
        default="FullConnected",
        choices=["DirectAnswer", "FullConnected", "Random", "Chain", "Debate", "Layered", "Star", "Mesh"],
        help="The communication topology among agents.",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--agent_names",
        nargs="+",
        type=str,
        default=["CopyMachine"],
        help="List of agent names to include in the graph."
    )
    parser.add_argument(
        "--agent_nums",
        nargs="+",
        type=int,
        default=[5],
        help="List of counts corresponding to each agent name."
    )
    parser.add_argument("--llm_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--domain", type=str, default="COPY")
    parser.add_argument("--decision_method", type=str, default=None)
    parser.add_argument(
        "--execution_mode",
        type=str,
        default="allow_kv_reuse",
        choices=["default", "allow_kv_reuse"],
        help="Execution strategy for the graph.",
    )
    parser.add_argument("--output_dir", type=str, default=str(PROJECT_ROOT / "result" / "TTFT_Benchmark"), help="Directory to save the output results.")
    parser.add_argument("--prefix", type=str, default="The task is:\n\n", help="The prefix text for the input query, kept the same as the default dense prefill mode.")
    parser.add_argument("--samples", type=int, default=100, help="Number of 1K-token samples")
    parser.add_argument("--kv-threshold", type=float, default=1.0, help="Threshold for key-value memory usage.")
    parser.add_argument("--kv-max-anchor-num", type=int, default=20, help="Maximum number of anchors for key-value memory.")
    parser.add_argument("--kv-window-size", type=int, default=5, help="Window size for key-value memory update.")
    parser.add_argument("--kv-thread-workers", type=int, default=None, help="Number of thread workers for key-value memory processing.")
    parser.add_argument("--kv-worker-timeout", type=float, default=None, help="Timeout for key-value memory workers processing.")

    args = parser.parse_args()
    result_path = Path(args.output_dir)
    result_path.mkdir(parents=True, exist_ok=True)
    if len(args.agent_names) != len(args.agent_nums):
        parser.error("The number of agent names must match the number of agent counts.")
    return args

def _make_random_token_sequence(length: int) -> str:
    symbols = random.choices(["Δ", "Ω"], k=length)
    return " ".join(symbols)


async def evaluate(
        graph: Graph,
        *,
        samples: int,
        output_dir: str,
        **kwargs
        ) -> List[Dict[str, Any]]:

    graph.spatial_logits.requires_grad_ = False
    graph.temporal_logits.requires_grad_ = False

    data = [{"task": _make_random_token_sequence(1000)} for _ in range(samples)]

    all_results: List[Dict[str, Any]] = []
    for i, input_dict in enumerate(data):
        print(80*'-')

        realized_graph = copy.deepcopy(graph)
        realized_graph.spatial_logits = graph.spatial_logits
        realized_graph.temporal_logits = graph.temporal_logits
        tasks = [asyncio.create_task(realized_graph.arun(input_dict, **kwargs))]
        raw_results = await asyncio.gather(*tasks)
        all_results.extend(raw_results)
    print("Done!")

    try:
        _write_per_agent_latency(output_dir)
    except Exception as e:
        logger.warning("Failed to write per-agent latency JSONs: {}", e)

    return all_results


def _write_per_agent_latency(output_dir: str) -> None:
    latency_path = Path(output_dir) / "Latency.json"
    if not latency_path.exists():
        logger.warning("Latency.json not found at {}", str(latency_path))
        return
    try:
        with open(latency_path, "r", encoding="utf-8") as f:
            records = json.load(f)
    except Exception as e:
        logger.warning("Could not read Latency.json: {}", e)
        return
    by_agent: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records if isinstance(records, list) else []:
        agent_id = rec.get("agent_id") or "unknown"
        by_agent.setdefault(agent_id, []).append(rec)
    out_dir = Path(output_dir) / "agent_latency"
    out_dir.mkdir(parents=True, exist_ok=True)
    for agent_id, items in by_agent.items():
        agent_file = out_dir / f"agent_{agent_id}.json"
        with open(agent_file, "w", encoding="utf-8") as f:
            json.dump(items, f, ensure_ascii=False, indent=2)
    combined = out_dir / "PerAgentLatency.json"
    with open(combined, "w", encoding="utf-8") as f:
        json.dump(by_agent, f, ensure_ascii=False, indent=2)

async def main():
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    agent_names = [name for name, num in zip(args.agent_names, args.agent_nums) for _ in range(num)]
    kwargs = get_kwargs(args.mode, len(agent_names))
    kv_config = KVCommConfig.from_env().apply_overrides(
        threshold=args.kv_threshold,
        max_anchor_num=args.kv_max_anchor_num,
        window_size=args.kv_window_size,
        thread_pool_workers=args.kv_thread_workers,
        worker_timeout=args.kv_worker_timeout,
    )

    graph = Graph(
        domain=args.domain,
        llm_name=args.llm_name,
        agent_names=agent_names,
        kv_config=kv_config,
        **kwargs,
    )

    eval_kwargs = {
        "prefix": args.prefix,
        "output_dir": str(output_dir),
    }

    configure_logging(log_path=output_dir / "logs/log.txt")
    _ = await evaluate(
        graph=graph,
        mode=args.execution_mode,
        samples=args.samples,
        **eval_kwargs,
    )


def get_kwargs(
    mode: Union[
        Literal["DirectAnswer"],
        Literal["FullConnected"],
        Literal["Random"],
        Literal["Chain"],
        Literal["Debate"],
        Literal["Layered"],
        Literal["Star"],
        Literal["Mesh"],
    ],
    N: int,
):
    fixed_spatial_masks: List[List[int]] = None                
    fixed_temporal_masks: List[List[int]] = None                
    node_kwargs = None

    def generate_layered_graph(n, layer_num=2):
        adj_matrix = [[0] * n for _ in range(n)]
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

    def generate_mesh_graph(n):
        adj_matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                adj_matrix[i][j] = 1
        return adj_matrix

    def generate_star_graph(n):
        adj_matrix = [[0] * n for _ in range(n)]
        for i in range(1, n):
            adj_matrix[0][i] = 1
        return adj_matrix

    if mode == "DirectAnswer":
        fixed_spatial_masks = [[0]]
        fixed_temporal_masks = [[0]]
        node_kwargs = [{"role": "Normal"}]
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
    elif mode == "Mesh":
        fixed_spatial_masks = generate_mesh_graph(N)
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
