from typing import Union, Dict, Any, List
import itertools
import torch
import copy
import os
import threading
import gc
import asyncio
from tenacity import retry, wait_random_exponential, stop_after_attempt
from transformers import AutoModelForCausalLM, AutoTokenizer
from KVCOMM.prompt.prompt_set import PromptSet
from KVCOMM.prompt.prompt_set_registry import PromptSetRegistry
from KVCOMM.prompt.common import get_combine_materials

roles = itertools.cycle(['Copy Machine',
                         'Copy Machine',
                         'Copy Machine',
                         'Copy Machine',
                         'Copy Machine',
                         'Copy Machine',
                         'Copy Machine'])

IN_LENGTH = int(os.getenv('IN_LENGTH', '128'))
OUT_LENGTH = int(os.getenv('OUT_LENGTH', '128'))
ROLE_DESCRIPTION = {
"Copy Machine":
" Ω" * IN_LENGTH + f"\n Please randomly output character Ω or Δ by {OUT_LENGTH} times to achieve the output length of {OUT_LENGTH}.",
}


@PromptSetRegistry.register('COPY')
class COPYpromptSet(PromptSet):

    @staticmethod
    def get_role():
        return next(roles)

    @staticmethod
    def get_decision_role():
        return ""

    @staticmethod
    def get_constraint():
        return ""
    
    @staticmethod
    def get_analyze_constraint(role):
        return ROLE_DESCRIPTION[role]
    
    @staticmethod
    def get_decision_constraint():
        return ""
    
    @staticmethod
    def get_format():
        return NotImplementedError

    @staticmethod
    def get_answer_prompt(question):
        return f"""{question}"""

    @staticmethod
    def get_query_prompt(question):
        raise NotImplementedError

    @staticmethod
    def get_file_analysis_prompt(query, file):
        raise NotImplementedError

    @staticmethod
    def get_websearch_prompt(query):
        raise NotImplementedError
    
    @staticmethod
    def get_distill_websearch_prompt(query, results):
        raise NotImplementedError

    @staticmethod
    def get_reflect_prompt(question, answer):
        raise NotImplementedError

    @staticmethod
    def get_combine_materials(materials: Dict[str, Any]) -> str:
        return get_combine_materials(materials)
    
    @staticmethod
    def get_decision_few_shot():
        return ""
    