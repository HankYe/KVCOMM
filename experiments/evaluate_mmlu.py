import asyncio
import copy
import json
import math
import os
import time
from typing import Any, Dict, Iterator, List, Optional

from tqdm import tqdm

from KVCOMM.graph.graph import Graph
from experiments.accuracy import Accuracy
from KVCOMM.utils.log import logger
from KVCOMM.utils.metrics import metrics_recorder


async def evaluate(
    graph: Graph,
    dataset,
    limit_questions: Optional[int] = None,
    eval_batch_size: int = 1,
    *,
    mode: str = "default",
    **kwargs,
) -> float:
    """Evaluate a graph on the provided dataset."""
    logger.info(
        "Evaluating KVCOMM on {} split {}",
        dataset.__class__.__name__,
        dataset.split,
    )

    accuracy = Accuracy()

    def eval_loader(batch_size: int) -> Iterator[List[Any]]:
        records: List[Any] = []
        for i_record, record in enumerate(dataset):
            if limit_questions is not None and i_record >= limit_questions:
                break
            records.append(record)
            if len(records) >= batch_size:
                yield records
                records = []
        if records:
            yield records

    data_len = min(len(dataset), limit_questions) if limit_questions is not None else len(dataset)
    num_batches = int(math.ceil(data_len / eval_batch_size))

    for i_batch, record_batch in tqdm(
        enumerate(eval_loader(batch_size=eval_batch_size)), total=num_batches
    ):
        logger.info("{}", "-" * 80)
        start_ts = time.time()


        tasks: List[asyncio.Task[Dict[str, Any]]] = []
        for record in record_batch:
            realized_graph = copy.deepcopy(graph)
            realized_graph.spatial_logits = graph.spatial_logits
            realized_graph.temporal_logits = graph.temporal_logits
            input_dict = dataset.record_to_input(record)
            input_dict["_batch_index"] = i_batch
            mode_kwargs = kwargs if mode == "allow_kv_reuse" else {}
            tasks.append(
                asyncio.create_task(
                    realized_graph.arun(
                        input_dict,
                        1,
                        mode=mode,
                        **mode_kwargs,
                    )
                )
            )

        batch_results = await asyncio.gather(*tasks)
        logger.opt(colors=True).info(
            f"<blue>[BATCH TIME]</blue> {time.time() - start_ts:.3f}s"
        )

        results_by_task = {
            result.get("task"): result.get("answers", [])
            for result in batch_results
        }

        for record in record_batch:
            input_dict = dataset.record_to_input(record)
            task_key = input_dict.get("task")
            raw_answer = results_by_task.get(task_key, [])
            logger.debug("Raw answer: {}", raw_answer)
            answer = await dataset.postprocess_answer(raw_answer)
            logger.debug("Postprocessed answer: {}", answer)
            correct_answer = dataset.record_to_target_answer(record)
            logger.debug("Correct answer: {}", correct_answer)
            accuracy.update(answer, correct_answer)
            accuracy.print()

        metrics_recorder.log_cumulative(batch_index=i_batch)

    accuracy.print()
    logger.info("Evaluation complete")
    return accuracy.get()


def dump_eval_results(self, dct: Dict[str, Any]) -> None:
    if self._art_dir_name is not None:
        eval_json_name = os.path.join(self._art_dir_name, "evaluation.json")
        with open(eval_json_name, "w") as f:
            json.dump(dct, f)
