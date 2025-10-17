import os
from typing import List,Any,Dict
import re
import asyncio

import torch
from KVCOMM.graph.node import Node
from KVCOMM.agents.agent_registry import AgentRegistry
from KVCOMM.llm.llm_registry import LLMRegistry
from KVCOMM.llm.gpt_chat import LLMChat
from KVCOMM.prompt.prompt_set_registry import PromptSetRegistry
from KVCOMM.llm.config import KVCommConfig
from time import perf_counter
from KVCOMM.utils.log import logger

IN_LENGTH = int(os.getenv('IN_LENGTH', '128'))
OUT_LENGTH = int(os.getenv('OUT_LENGTH', '128'))


@AgentRegistry.register('CopyMachine')
class CopyMachine(Node):
    def __init__(
        self,
        id: str | None = None,
        role: str = None,
        domain: str = "",
        llm_name: str = "",
        llm_config: KVCommConfig | None = None,
    ):
        super().__init__(id, "CopyMachine" ,domain, llm_name)
        prefix = ""

        self.llm = LLMRegistry.get(llm_name, prefix=prefix, llm_config=llm_config)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.llm.set_id(self.id, self.role)
        self.constraint = self.prompt_set.get_analyze_constraint(self.role)

    async def _process_inputs(
        self,
        raw_inputs:Dict[str,str],
        spatial_info:Dict[str,Dict],
        temporal_info:Dict[str,Dict],
        mode: str = "allow_kv_reuse",
        **kwargs,
    ) -> Dict[str, Any]:
        """Prepare prompts, optionally populate anchors, and return mode hints."""
        if mode == "allow_kv_reuse":
            request_uid = raw_inputs.get("_request_uid") or kwargs.get("request_uid")
            if request_uid is None:
                raise ValueError("request_uid is required for request-scoped anchor updates.")

            preferred_mode = "kv_reuse"
            agent_memory = self.llm._ensure_agent_memory(self.id)

            if self.llm.has_prefix_initialized(self.id) and "placeholder_info" in agent_memory:
                
                prefix_text = kwargs.get('prefix', "The task is:")
                user_content = prefix_text + raw_inputs['task']
                preferred_mode = self.llm.update_input_anchor(
                    request_uid=request_uid,
                    agent_id=self.id,
                    message=raw_inputs['task'],
                    user_content=user_content,
                    prefix_text=prefix_text,
                    test_time=True,
                )
                logger.opt(colors=True).info(
                    "<green>[MODE]</green> Task: {} Agent {} ({}) mode: {}",
                    raw_inputs["task"],
                    self.id,
                    self.role,
                    preferred_mode,
                )
                return {"preferred_mode": preferred_mode, "early_response": None}

            system_prompt = f"{self.constraint}"
            user_input = "{user_question}"
            user_prompt = f"The task is: {user_input}\n"
            spatial_str = ""
            temporal_str = ""
            for agent_id, info in spatial_info.items():
                agent_role = info['role']
                agent_output = info['output'] if len(info['output']) > 0 else "{agent_" + agent_id + "_current}"
                spatial_str += f"Agent {agent_id}, role is {agent_role}, output is:\n\n {agent_output}\n\n"
            for agent_id, info in temporal_info.items():
                agent_role = info['role']
                agent_output = info['output'] if len(info['output']) > 0 else "{agent_" + agent_id + "_history}"
                temporal_str += f"Agent {agent_id}, role is {agent_role}, output is:\n\n {agent_output}\n\n"

            if spatial_str:
                user_prompt += (
                    "At the same time, the outputs of other agents are as follows:\n\n"
                    f"{spatial_str} \n\n"
                )
            if temporal_str:
                user_prompt += (
                    "In the last round of dialogue, the outputs of other agents were: \n\n"
                    f"{temporal_str}"
                )
            await self.llm.prepare_prefix_kv_segments(self.id, system_prompt, user_prompt)
            return {"preferred_mode": preferred_mode, "early_response": None}


        system_prompt = f"{self.constraint}"
        user_prompt = f"The task is: {raw_inputs['task']}\n"
        spatial_str = ""
        temporal_str = ""
        for id, info in spatial_info.items():
            spatial_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"
        for id, info in temporal_info.items():
            temporal_str += f"Agent {id}, role is {info['role']}, output is:\n\n {info['output']}\n\n"

        user_prompt += f"At the same time, the outputs of other agents are as follows:\n\n{spatial_str} \n\n" if len(spatial_str) else ""
        user_prompt += f"In the last round of dialogue, the outputs of other agents were: \n\n{temporal_str}" if len(temporal_str) else ""
        return {"system_prompt": system_prompt, "user_prompt": user_prompt}

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

        inputs = asyncio.run(
            self._process_inputs(
                input,
                spatial_info,
                temporal_info,
                mode="default",
                **kwargs,
            )
        )
        system_prompt = inputs["system_prompt"]
        user_prompt = inputs["user_prompt"]
        message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
        response = self.llm.gen(message)
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Dict], temporal_info:Dict[str,Dict], mode: str = "default", **kwargs):
        """Handle asynchronous execution across different KV cache modes."""
        if mode == "default":
            request_uid = input.get("_request_uid")
            inputs = await self._process_inputs(
                input,
                spatial_info,
                temporal_info,
                mode=mode,
                **kwargs,
            )
            system_prompt = inputs["system_prompt"]
            user_prompt = inputs["user_prompt"]
            message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
            result = await self.llm.agen(
                message,
                max_tokens=self.llm.DEFAULT_MAX_TOKENS,
                request_uid=request_uid,
                agent_id=self.id,
                agent_name=self.agent_name,
                agent_role=self.role,
            )
            return result

        assert mode == "allow_kv_reuse", f"Unsupported async execution mode: {mode}"
        request_uid = input.get("_request_uid") or kwargs.get("request_uid")
        if request_uid is None:
            raise ValueError("request_uid is required for request-scoped anchor updates.")

        mode_data = await self._process_inputs(
            input,
            spatial_info,
            temporal_info,
            mode=mode,
            **kwargs,
        )
        preferred_mode = mode_data["preferred_mode"]
        result = await self.llm.generate_for_agent(
            request_uid=request_uid,
            message=input['task'],
            preferred_mode=preferred_mode,
            output_dir=kwargs.get("output_dir"),
            agent_id=self.id,
            agent_name=self.agent_name,
            agent_role=self.role,
            test_time=True,
            max_tokens=OUT_LENGTH
        )
        return input['task'], result
