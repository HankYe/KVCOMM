import asyncio
from typing import Any,Dict
from KVCOMM.graph.node import Node
from KVCOMM.agents.agent_registry import AgentRegistry
from KVCOMM.llm.llm_registry import LLMRegistry
from KVCOMM.prompt.prompt_set_registry import PromptSetRegistry
from KVCOMM.tools.coding.python_executor import PyExecutor
from KVCOMM.llm.config import KVCommConfig
from KVCOMM.utils.metrics import GenerationResult
from KVCOMM.utils.log import logger

@AgentRegistry.register('CodeWriting')
class CodeWriting(Node):
    """Programming agent that validates peers with internal tests."""
    def __init__(
        self,
        id: str | None = None,
        role: str = None,
        domain: str = "",
        llm_name: str = "",
        llm_config: KVCommConfig | None = None,
    ):
        super().__init__(id, "CodeWriting" ,domain, llm_name)
        prefix = ""
        self.llm = LLMRegistry.get(llm_name, prefix=prefix, llm_config=llm_config)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_role() if role is None else role
        self.constraint = self.prompt_set.get_constraint(self.role) 
        self.llm.set_id(self.id, self.role)

    async def _process_inputs(
        self,
        raw_inputs:Dict[str,str],
        spatial_info:Dict[str,Dict],
        temporal_info:Dict[str,Dict],
        mode: str = "default",
        **kwargs,
    )->Dict[str, Any]:
        """Prepare prompts, run quick internal checks, and return mode hints."""
        if mode == "allow_kv_reuse":
            request_uid = raw_inputs.get("_request_uid") or kwargs.get("request_uid")
            if request_uid is None:
                raise ValueError("request_uid is required for request-scoped anchor updates.")

            preferred_mode = "kv_reuse"
            shared_memory = self.llm._ensure_agent_memory(self.id)

            if self.llm.has_prefix_initialized(self.id) and "placeholder_info" in shared_memory:
                for agent_id, info in spatial_info.items():
                    if (
                        self.role != 'Normal Programmer'
                        and self.role != 'Stupid Programmer'
                        and info['role'] != 'Algorithm Designer'
                    ):
                        agent_mem = self.llm._ensure_agent_memory(agent_id)
                        if raw_inputs['task'] in agent_mem.get('condition', {}):
                            continue
                        code = info['output'].split("```python\n")[-1].split("\n```")[0]
                        is_solved, feedback, _ = PyExecutor().execute(code, self.internal_tests, timeout=1)
                        condition_text = (
                            "Whether it passes internal testing?\n"
                            f"{is_solved}.\n\nThe feedback is:\n\n {feedback}."
                        )
                        self.llm.update_condition_anchor(
                            request_uid=request_uid,
                            owner_agent_id=agent_id,
                            message=raw_inputs['task'],
                            content=condition_text,
                            prefix_text="Whether it passes internal testing?\n",
                        )

                prefix_text = kwargs.get('prefix', "The task is:\n\n")
                user_content = prefix_text + raw_inputs['task']
                preferred_mode = self.llm.update_input_anchor(
                    request_uid=request_uid,
                    agent_id=self.id,
                    message=raw_inputs['task'],
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
            spatial_str = ""
            temporal_str = ""
            for agent_id, info in spatial_info.items():
                agent_output = info['output'] if len(info['output']) > 0 else "{agent_" + agent_id + "_current}"
                if (
                    self.role != 'Normal Programmer'
                    and self.role != 'Stupid Programmer'
                    and info['role'] != 'Algorithm Designer'
                ):
                    condition = '{condition_' + agent_id + '_current}'
                    spatial_str += (
                        f"Agent {agent_id} as a {info['role']}:\n\nThe code written by the agent is:\n\n{agent_output}\n\n Whether it passes internal testing?\n{condition}\n\n"
                    )
                else:
                    spatial_str += f"Agent {agent_id} as a {info['role']} provides the following info: {agent_output}\n\n"
            for agent_id, info in temporal_info.items():
                agent_output = info['output'] if len(info['output']) > 0 else "{agent_" + agent_id + "_history}"
                if (
                    self.role != 'Normal Programmer'
                    and self.role != 'Stupid Programmer'
                    and info['role'] != 'Algorithm Designer'
                ):
                    condition = '{condition_' + agent_id + '_history}'
                    temporal_str += (
                        f"Agent {agent_id} as a {info['role']}:\n\nThe code written by the agent is:\n\n{agent_output}\n\n Whether it passes internal testing?\n{condition}\n\n"
                    )
                else:
                    temporal_str += f"Agent {agent_id} as a {info['role']} provides the following info: {agent_output}\n\n"
            user_prompt = f"The task is:\n\n{user_input}\n"
            if spatial_str:
                user_prompt += (
                    "At the same time, the outputs and feedbacks of other agents are as follows:\n\n"
                    f"{spatial_str} \n\n"
                )
            if temporal_str:
                user_prompt += (
                    "In the last round of dialogue, the outputs and feedbacks of some agents were: \n\n"
                    f"{temporal_str}"
                )
            await self.llm.prepare_prefix_kv_segments(self.id, system_prompt, user_prompt)
            return {"preferred_mode": preferred_mode, "early_response": None}

        system_prompt = self.constraint
        spatial_str = ""
        temporal_str = ""
        for id, info in spatial_info.items():
            if info['output'].startswith("```python") and info['output'].endswith("```") and self.role != 'Normal Programmer' and self.role != 'Stupid Programmer':
                output = info['output'].split("```python\n")[-1].split("\n```")[0]
                is_solved, feedback, state = PyExecutor().execute(output, self.internal_tests, timeout=10)
                spatial_str += f"Agent {id} as a {info['role']}:\n\nThe code written by the agent is:\n\n{info['output']}\n\n Whether it passes internal testing?\n{is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
            else:
                spatial_str += f"Agent {id} as a {info['role']} provides the following info: {info['output']}\n\n"
        for id, info in temporal_info.items():
            if info['output'].startswith("```python") and info['output'].endswith("```") and self.role != 'Normal Programmer' and self.role != 'Stupid Programmer':
                output = info['output'].split("```python\n")[-1].split("\n```")[0]
                is_solved, feedback, state = PyExecutor().execute(output, self.internal_tests, timeout=10)
                temporal_str += f"Agent {id} as a {info['role']}:\n\nThe code written by the agent is:\n\n{info['output']}\n\n Whether it passes internal testing? {is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
            else:
                temporal_str += f"Agent {id} as a {info['role']} provides the following info: {info['output']}\n\n"
        user_prompt = f"The task is:\n\n{raw_inputs['task']}\n"
        user_prompt += f"At the same time, the outputs and feedbacks of other agents are as follows:\n\n{spatial_str} \n\n" if len(spatial_str) else ""
        user_prompt += f"In the last round of dialogue, the outputs and feedbacks of some agents were: \n\n{temporal_str}" if len(temporal_str) else ""
        return {"system_prompt": system_prompt, "user_prompt": user_prompt}

    def extract_example(self, prompt: str) -> list:
        prompt = prompt['task']
        lines = (line.strip() for line in prompt.split('\n') if line.strip())

        results = []
        lines_iter = iter(lines)
        for line in lines_iter:
            if line.startswith('>>>'):
                function_call = line[4:]
                expected_output = next(lines_iter, None)
                if expected_output:
                    results.append(f"assert {function_call} == {expected_output}")

        return results

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        self.internal_tests = self.extract_example(input)
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
            self.internal_tests = self.extract_example(input)
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
            if system_prompt == "is_solved":
                return GenerationResult(
                    text=user_prompt,
                    mode="default",
                    ttft=0.0,
                )
            message = [{'role':'system','content':system_prompt},{'role':'user','content':user_prompt}]
            result = await self.llm.agen(
                message,
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
        self.internal_tests = self.extract_example(input)

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
        )
        return input['task'], result
