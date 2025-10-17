import asyncio
from typing import Any,Dict
from KVCOMM.graph.node import Node
from KVCOMM.agents.agent_registry import AgentRegistry
from KVCOMM.llm.llm_registry import LLMRegistry
from KVCOMM.prompt.prompt_set_registry import PromptSetRegistry
from KVCOMM.tools.coding.python_executor import execute_code_get_return
from KVCOMM.llm.config import KVCommConfig
from KVCOMM.utils.log import logger

@AgentRegistry.register('MathSolver')
class MathSolver(Node):
    """Math agent that aggregates peer signals and computes final answers."""
    def __init__(
        self,
        id: str | None = None,
        role: str = None,
        domain: str = "",
        llm_name: str = "",
        llm_config: KVCommConfig | None = None,
    ):
        super().__init__(id, "MathSolver" ,domain, llm_name)
        prefix = "A: "
        self.llm = LLMRegistry.get(llm_name, prefix=prefix)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.llm.set_id(self.id, self.role)
        self.constraint = self.prompt_set.get_constraint(self.role)

    async def _process_inputs(
        self,
        raw_inputs:Dict[str,str],
        spatial_info:Dict[str,Dict],
        temporal_info:Dict[str,Dict],
        mode: str = "default",
        **kwargs,
    )->Dict[str, Any]:
        """Prepare prompts, possibly set anchors, and return mode hints."""
        if mode == "allow_kv_reuse":
            request_uid = raw_inputs.get("_request_uid") or kwargs.get("request_uid")
            if request_uid is None:
                raise ValueError("request_uid is required for request-scoped anchor updates.")

            preferred_mode = "kv_reuse"
            agent_memory = self.llm._ensure_agent_memory(self.id)

            if self.llm.has_prefix_initialized(self.id) and "placeholder_info" in agent_memory:
                for agent_id, info in spatial_info.items():
                    if info["role"] != "Programming Expert":
                        continue
                    answer = execute_code_get_return(
                        info["output"].lstrip("```python\n").rstrip("\n```")
                    )
                    self.llm.update_condition_anchor(
                        request_uid=request_uid,
                        owner_agent_id=agent_id,
                        message=raw_inputs["task"],
                        content=f"The answer is:\n{answer}",
                        prefix_text="The answer is:\n",
                    )

                prefix_text = kwargs.get("prefix", "Q:")
                user_content = prefix_text + raw_inputs["task"]
                preferred_mode = self.llm.update_input_anchor(
                    request_uid=request_uid,
                    agent_id=self.id,
                    message=raw_inputs["task"],
                    user_content=user_content,
                    prefix_text=prefix_text,
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
            user_prompt = self.prompt_set.get_answer_prompt(question=user_input, role=self.role)
            spatial_str = ""
            temporal_str = ""
            for id, info in spatial_info.items():
                agent_output = info["output"] if len(info["output"]) > 0 else "{agent_" + id + "_current}"
                if info["role"] == "Programming Expert":
                    agent_output += "\n the result is {condition_" + id + "_current}"
                spatial_str += (
                    f"Agent {id} as a {info['role']} his answer to this question is:\n\n"
                    f"{agent_output}\n\n"
                )
            for id, info in temporal_info.items():
                agent_output = info["output"] if len(info["output"]) > 0 else "{agent_" + id + "_history}"
                temporal_str += (
                    f"Agent {id} as a {info['role']} his answer to this question was:\n\n"
                    f"{agent_output}\n\n"
                )
            user_prompt += (
                f"At the same time, there are the following responses to the same question for your reference:\n\n"
                f"{spatial_str} \n\n" if len(spatial_str)
                else ""
            )
            user_prompt += (
                "In the last round of dialogue, there were the following responses to the same question for your reference: \n\n"
                f"{temporal_str}" if len(temporal_str)
                else ""
            )
            await self.llm.prepare_prefix_kv_segments(self.id, system_prompt, user_prompt)
            return {"preferred_mode": preferred_mode, "early_response": None}

        system_prompt = self.constraint
        spatial_str = ""
        temporal_str = ""
        user_prompt = self.prompt_set.get_answer_prompt(question=raw_inputs["task"],role=self.role)
        for id, info in spatial_info.items():
            spatial_str += f"Agent {id} as a {info['role']} his answer to this question is:\n\n{info['output']}\n\n"
        for id, info in temporal_info.items():
            temporal_str += f"Agent {id} as a {info['role']} his answer to this question was:\n\n{info['output']}\n\n"
        user_prompt += f"At the same time, there are the following responses to the same question for your reference:\n\n{spatial_str} \n\n" if len(spatial_str) else ""
        user_prompt += f"In the last round of dialogue, there were the following responses to the same question for your reference: \n\n{temporal_str}" if len(temporal_str) else ""
        return {"system_prompt": system_prompt, "user_prompt": user_prompt}

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
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

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], mode: str = "default", **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        """ The input type of this node is Dict """
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
                request_uid=request_uid,
                agent_id=self.id,
                agent_name=self.agent_name,
                agent_role=self.role,
            )
            if self.role == "Programming Expert":
                answer = execute_code_get_return(result.text.lstrip("```python\n").rstrip("\n```"))
                result.text += f"\nthe answer is {answer}"
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
            max_tokens=self.llm.DEFAULT_MAX_TOKENS,
            output_dir=kwargs.get('output_dir'),
            agent_id=self.id,
            agent_name=self.agent_name,
            agent_role=self.role,
        )

        if self.role == "Programming Expert":
            answer = execute_code_get_return(result.text.split("```python\n")[-1].split("\n```")[0])
            result.text += f"\nthe answer is {answer}"
        return input['task'], result
