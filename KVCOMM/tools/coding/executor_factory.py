from KVCOMM.utils.log import logger
from KVCOMM.tools.coding.executor_types import Executor
from KVCOMM.tools.coding.python_executor import PyExecutor

EXECUTOR_MAPPING = {
    "py": PyExecutor,
    "python": PyExecutor,
}

def executor_factory(lang: str) -> Executor:

    if lang not in EXECUTOR_MAPPING:
        raise ValueError(f"Invalid language for executor: {lang}")

    executor_class = EXECUTOR_MAPPING[lang]
    return executor_class()