import asyncio
from typing import Any, Dict, List
from KVCOMM.graph.node import Node
from KVCOMM.agents.agent_registry import AgentRegistry
from KVCOMM.llm.llm_registry import LLMRegistry
from KVCOMM.prompt.prompt_set_registry import PromptSetRegistry
from KVCOMM.tools.coding.python_executor import execute_code_get_return, PyExecutor
from KVCOMM.llm.config import KVCommConfig
from KVCOMM.utils.metrics import GenerationResult
from KVCOMM.utils.log import logger

@AgentRegistry.register('FinalWriteCode')
class FinalWriteCode(Node):
    """Final code synthesis agent that integrates peers and executes tests."""
    def __init__(
        self,
        id: str | None = None,
        domain: str = "",
        llm_name: str = "",
        llm_config: KVCommConfig | None = None,
    ):
        super().__init__(id, "FinalWriteCode" ,domain, llm_name)
        prefix = ""
        self.llm = LLMRegistry.get(llm_name, prefix=prefix, llm_config=llm_config)
        self.role = 'FinalWriteCode'
        self.llm.set_id(self.id, 'FinalWriteCode')
        self.prompt_set = PromptSetRegistry.get(domain)
        self._executor = PyExecutor()

    @staticmethod
    def extract_example(prompt: Dict[str, Any] | str) -> List[str]:
        """Return doctest-style assertions extracted from the task description."""
        if isinstance(prompt, dict):
            prompt_text = str(prompt.get("task", ""))
        else:
            prompt_text = str(prompt)

        lines = [line.strip() for line in prompt_text.splitlines() if line.strip()]
        results: List[str] = []
        iterator = iter(lines)
        for line in iterator:
            if not line.startswith(">>>"):
                continue
            function_call = line[4:].strip()
            expected_output = next(iterator, "").strip()
            if not function_call or not expected_output:
                continue
            results.append(f"assert {function_call} == {expected_output}")
        return results

    @staticmethod
    def _is_python_code_block(text: str) -> bool:
        text = text.strip()
        return text.startswith("```python") and text.endswith("```")

    @staticmethod
    def _extract_python_code(text: str) -> str:
        """Extract pure python code from a fenced block."""
        if not FinalWriteCode._is_python_code_block(text):
            return text
        content = text.strip()[len("```python") :].strip()
        if content.endswith("```"):
            content = content[:-3].strip()
        return content

    def _summarize_agent_outputs(
        self,
        raw_inputs: Dict[str, str],
        spatial_info: Dict[str, Any],
        internal_tests: List[str],
    ) -> str:
        """Summarize peer outputs, running tests on code blocks when present."""
        paragraphs: List[str] = []
        for agent_id, info in spatial_info.items():
            role = info.get("role", "agent")
            output = info.get("output")
            if not isinstance(output, str):
                paragraphs.append(f"Agent {agent_id} as a {role} returned invalid output.\n\n")
                continue
            if self._is_python_code_block(output):
                code = self._extract_python_code(output)
                is_solved, feedback, _ = self._executor.execute(code, internal_tests, timeout=10)
                paragraphs.append(
                    f"Agent {agent_id} as a {role}:\n\n"
                    f"The code written by the agent is:\n\n{output}\n\n"
                    f"Whether it passes internal testing? {is_solved}.\n\n"
                    f"The feedback is:\n\n{feedback}.\n\n"
                )
                continue
            paragraphs.append(
                f"Agent {agent_id} as a {role} provides the following info: {output}\n\n"
            )
        return "".join(paragraphs)

    async def _process_inputs(
        self,
        raw_inputs:Dict[str,str],
        spatial_info:Dict[str,Any],
        temporal_info:Dict[str,Any],
        mode: str = "default",
        **kwargs,
    )->Dict[str, Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """

        if mode == "allow_kv_reuse":
            request_uid = raw_inputs.get("_request_uid") or kwargs.get("request_uid")
            if request_uid is None:
                raise ValueError("request_uid is required for request-scoped anchor updates.")

            preferred_mode = "kv_reuse"
            agent_memory = self.llm._ensure_agent_memory(self.id)
            prefix_text = kwargs.get("prefix", "")

            has_shared_prefix = (
                self.llm.has_prefix_initialized(self.id)
                and "placeholder_info" in agent_memory
            )
            early_response: str | None = None

            if has_shared_prefix:
                task = raw_inputs["task"]
                for agent_id, info in spatial_info.items():
                    cond_text: str | None = None
                    cond_prefix: str | None = None

                    if self.domain == "gsm8k" and info["role"] == "Programming Expert":
                        answer = execute_code_get_return(
                            info["output"].split("```python\n")[-1].split("\n```")[0]
                        )
                        if answer is None:
                            answer = "No variable is named answer."
                        cond_text = f"the answer is {answer}"
                        cond_prefix = "the answer is "
                    elif (
                        self.domain == "humaneval"
                        and self.role not in {"Normal Programmer", "Stupid Programmer"}
                        and info["role"] != "Algorithm Designer"
                    ):
                        code = info["output"].split("```python\n")[-1].split("\n```")[0]
                        is_solved, feedback, _ = PyExecutor().execute(
                            code, getattr(self, "internal_tests", []), timeout=10
                        )
                        cond_text = (
                            "Whether it passes internal testing?\n"
                            f"{is_solved}.\n\nThe feedback is:\n\n {feedback}."
                        )
                        cond_prefix = "Whether it passes internal testing?\n"

                    if cond_text and cond_prefix:
                        self.llm.update_condition_anchor(
                            request_uid=request_uid,
                            owner_agent_id=agent_id,
                            message=task,
                            content=cond_text,
                            prefix_text=cond_prefix,
                        )

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
                return {
                    "preferred_mode": preferred_mode,
                    "early_response": early_response,
                }

            system_prompt = self.prompt_set.get_decision_role()
            self.constraint = self.prompt_set.get_decision_constraint()
            system_prompt = f"{system_prompt}.\n {self.constraint}"
            user_input = "{user_question}"
            spatial_str = ""
            decision_few_shot = self.prompt_set.get_decision_few_shot()

            spatial_str = ""
            if self.domain in {"gsm8k", "mmlu"}:
                for agent_id, info in spatial_info.items():
                    agent_output = info["output"] if info["output"] else "{agent_" + agent_id + "_current}"
                    if info["role"] == "Programming Expert":
                        agent_output += "\n the result is {condition_" + agent_id + "_current}"
                    spatial_str += (
                        f"Agent {agent_id}, role is {info['role']}, output is:\n\n {agent_output}\n\n"
                    )
            elif self.domain == "humaneval":
                for agent_id, info in spatial_info.items():
                    agent_output = info["output"] if info["output"] else "{agent_" + agent_id + "_current}"
                    if (
                        self.role not in {"Normal Programmer", "Stupid Programmer"}
                        and info["role"] != "Algorithm Designer"
                    ):
                        condition = "{condition_" + agent_id + "_current}"
                        spatial_str += (
                            f"Agent {agent_id} as a {info['role']}:\n\nThe code written by the agent is:\n\n"
                            f"{agent_output}\n\n Whether it passes internal testing?\n{condition}\n\n"
                        )
                    else:
                        spatial_str += (
                            f"Agent {agent_id} as a {info['role']} provides the following info: {agent_output}\n\n"
                        )

            decision_few_shot = self.prompt_set.get_decision_few_shot()
            user_prompt = (
                f"{decision_few_shot} {prefix_text} {user_input}\n At the same time, the output of other agents is as follows:\n\n"
                f"{spatial_str}\n\n"
            )
            await self.llm.prepare_prefix_kv_segments(self.id, system_prompt, user_prompt)
            return {
                "preferred_mode": preferred_mode,
                "early_response": early_response,
            }

        system_prompt = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()
        system_prompt = f"{system_prompt}.\n {self.constraint}"
        prefix_text = kwargs.get("prefix", "")
        spatial_str = ""
        if self.domain in {"gsm8k", "mmlu"}:
            for agent_id, info in spatial_info.items():
                agent_output = info["output"]
                if info["role"] == "Programming Expert":
                    answer = execute_code_get_return(
                        info["output"].split("```python\n")[-1].split("\n```")[0]
                    )
                    agent_output += f"\n the result is {answer}"
                spatial_str += (
                    f"Agent {agent_id}, role is {info['role']}, output is:\n\n {agent_output}\n\n"
                )
        elif self.domain == "humaneval":
            for agent_id, info in spatial_info.items():
                agent_output = info["output"]
                if (
                    self.role not in {"Normal Programmer", "Stupid Programmer"}
                    and info["role"] != "Algorithm Designer"
                ):
                    code = info["output"].split("```python\n")[-1].split("\n```")[0]
                    is_solved, feedback, _ = PyExecutor().execute(
                        code, getattr(self, "internal_tests", []), timeout=10
                    )
                    spatial_str += (
                        f"Agent {agent_id} as a {info['role']}:\n\nThe code written by the agent is:\n\n"
                        f"{agent_output}\n\n Whether it passes internal testing?\n{is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
                    )
                else:
                    spatial_str += (
                        f"Agent {agent_id} as a {info['role']} provides the following info: {agent_output}\n\n"
                    )

        decision_few_shot = self.prompt_set.get_decision_few_shot()
        user_prompt = (
            f"{decision_few_shot} {prefix_text} {raw_inputs['task']}\n At the same time, the output of other agents is as follows:\n\n"
            f"{spatial_str}\n\n"
        )
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "early_response": early_response,
        }

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
        message = [
            {"role": "system", "content": inputs["system_prompt"]},
            {"role": "user", "content": inputs["user_prompt"]},
        ]
        response = self.llm.gen(message)
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], mode: str = "default", **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        if mode == "default":
            request_uid = input.get("_request_uid")
            inputs = await self._process_inputs(
                input,
                spatial_info,
                temporal_info,
                mode=mode,
                **kwargs,
            )
            message = [
                {"role": "system", "content": inputs["system_prompt"]},
                {"role": "user", "content": inputs["user_prompt"]},
            ]
            result = await self.llm.agen(
                message,
                request_uid=request_uid,
                agent_id=self.id,
                agent_name=self.agent_name,
                agent_role=self.role,
            )
            return result

        mode_data = await self._process_inputs(
            input,
            spatial_info,
            temporal_info,
            mode=mode,
            **kwargs,
        )
        request_uid = input.get("_request_uid") or kwargs.get("request_uid")
        if request_uid is None:
            raise ValueError("request_uid is required for request-scoped anchor updates.")
        result = await self.llm.generate_for_agent(
            request_uid=request_uid,
            message=input["task"],
            preferred_mode=mode_data["preferred_mode"],
            output_dir=kwargs.get("output_dir"),
            agent_id=self.id,
            agent_name=self.agent_name,
            agent_role=self.role,
        )
        return input["task"], result


@AgentRegistry.register('FinalRefer')
class FinalRefer(Node):
    """Final referencing/answer agent assembling the final response."""
    def __init__(
        self,
        id: str | None = None,
        domain: str = "",
        llm_name: str = "",
        llm_config: KVCommConfig | None = None,
    ):
        super().__init__(id, "FinalRefer" ,domain, llm_name)
        prefix = ""
        self.llm = LLMRegistry.get(llm_name, prefix=prefix, llm_config=llm_config)
        self.role = 'FinalRefer'
        self.llm.set_id(self.id, 'FinalRefer')
        self.prompt_set = PromptSetRegistry.get(domain)

    async def _process_inputs(
        self,
        raw_inputs:Dict[str,str],
        spatial_info:Dict[str,Any],
        temporal_info:Dict[str,Any],
        mode: str = "default",
        **kwargs,
    )->Dict[str, Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        if mode == "allow_kv_reuse":
            request_uid = raw_inputs.get("_request_uid") or kwargs.get("request_uid")
            if request_uid is None:
                raise ValueError("request_uid is required for request-scoped anchor updates.")

            preferred_mode = "kv_reuse"
            agent_memory = self.llm._ensure_agent_memory(self.id)
            prefix_text = kwargs.get("prefix", "")

            has_shared_prefix = (
                self.llm.has_prefix_initialized(self.id)
                and "placeholder_info" in agent_memory
            )
            early_response: str | None = None

            if has_shared_prefix:
                task = raw_inputs["task"]
                for agent_id, info in spatial_info.items():
                    cond_text: str | None = None
                    cond_prefix: str | None = None

                    if self.domain == "gsm8k" and info["role"] == "Programming Expert":
                        answer = execute_code_get_return(
                            info["output"].split("```python\n")[-1].split("\n```")[0]
                        )
                        if answer is None:
                            answer = "No variable is named answer."
                        cond_text = f"the answer is {answer}"
                        cond_prefix = "the answer is "
                    elif (
                        self.domain == "humaneval"
                        and self.role not in {"Normal Programmer", "Stupid Programmer"}
                        and info["role"] != "Algorithm Designer"
                    ):
                        code = info["output"].split("```python\n")[-1].split("\n```")[0]
                        is_solved, feedback, _ = PyExecutor().execute(
                            code, getattr(self, "internal_tests", []), timeout=10
                        )
                        cond_text = (
                            "Whether it passes internal testing?\n"
                            f"{is_solved}.\n\nThe feedback is:\n\n {feedback}."
                        )
                        cond_prefix = "Whether it passes internal testing?\n"

                    if cond_text and cond_prefix:
                        self.llm.update_condition_anchor(
                            request_uid=request_uid,
                            owner_agent_id=agent_id,
                            message=task,
                            content=cond_text,
                            prefix_text=cond_prefix,
                        )

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
                return {
                    "preferred_mode": preferred_mode,
                    "early_response": early_response,
                }

            system_prompt = self.prompt_set.get_decision_role()
            self.constraint = self.prompt_set.get_decision_constraint()
            system_prompt = f"{system_prompt}.\n {self.constraint}"
            user_input = "{user_question}"
            spatial_str = ""
            decision_few_shot = self.prompt_set.get_decision_few_shot()

            spatial_str = ""
            if self.domain in {"gsm8k", "mmlu"}:
                for agent_id, info in spatial_info.items():
                    agent_output = info["output"] if info["output"] else "{agent_" + agent_id + "_current}"
                    if info["role"] == "Programming Expert":
                        agent_output += "\n the result is {condition_" + agent_id + "_current}"
                    spatial_str += (
                        f"Agent {agent_id}, role is {info['role']}, output is:\n\n {agent_output}\n\n"
                    )
            elif self.domain == "humaneval":
                for agent_id, info in spatial_info.items():
                    agent_output = info["output"] if info["output"] else "{agent_" + agent_id + "_current}"
                    if (
                        self.role not in {"Normal Programmer", "Stupid Programmer"}
                        and info["role"] != "Algorithm Designer"
                    ):
                        condition = "{condition_" + agent_id + "_current}"
                        spatial_str += (
                            f"Agent {agent_id} as a {info['role']}:\n\nThe code written by the agent is:\n\n"
                            f"{agent_output}\n\n Whether it passes internal testing?\n{condition}\n\n"
                        )
                    else:
                        spatial_str += (
                            f"Agent {agent_id} as a {info['role']} provides the following info: {agent_output}\n\n"
                        )

            decision_few_shot = self.prompt_set.get_decision_few_shot()
            user_prompt = (
                f"{decision_few_shot} {prefix_text} {user_input}\n At the same time, the output of other agents is as follows:\n\n"
                f"{spatial_str}\n\n"
            )
            await self.llm.prepare_prefix_kv_segments(self.id, system_prompt, user_prompt)
            return {
                "preferred_mode": preferred_mode,
                "early_response": early_response,
            }

        system_prompt = self.prompt_set.get_decision_role()
        self.constraint = self.prompt_set.get_decision_constraint()
        system_prompt = f"{system_prompt}.\n {self.constraint}"
        prefix_text = kwargs.get("prefix", "")
        spatial_str = ""
        if self.domain in {"gsm8k", "mmlu"}:
            for agent_id, info in spatial_info.items():
                agent_output = info["output"]
                if info["role"] == "Programming Expert":
                    answer = execute_code_get_return(
                        info["output"].split("```python\n")[-1].split("\n```")[0]
                    )
                    agent_output += f"\n the result is {answer}"
                spatial_str += (
                    f"Agent {agent_id}, role is {info['role']}, output is:\n\n {agent_output}\n\n"
                )
        elif self.domain == "humaneval":
            for agent_id, info in spatial_info.items():
                agent_output = info["output"]
                if (
                    self.role not in {"Normal Programmer", "Stupid Programmer"}
                    and info["role"] != "Algorithm Designer"
                ):
                    code = info["output"].split("```python\n")[-1].split("\n```")[0]
                    is_solved, feedback, _ = PyExecutor().execute(
                        code, getattr(self, "internal_tests", []), timeout=10
                    )
                    spatial_str += (
                        f"Agent {agent_id} as a {info['role']}:\n\nThe code written by the agent is:\n\n"
                        f"{agent_output}\n\n Whether it passes internal testing?\n{is_solved}.\n\nThe feedback is:\n\n {feedback}.\n\n"
                    )
                else:
                    spatial_str += (
                        f"Agent {agent_id} as a {info['role']} provides the following info: {agent_output}\n\n"
                    )

        decision_few_shot = self.prompt_set.get_decision_few_shot()
        user_prompt = (
            f"{decision_few_shot} {prefix_text} {raw_inputs['task']}\n At the same time, the output of other agents is as follows:\n\n"
            f"{spatial_str}\n\n"
        )
        return {
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
        }

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

        inputs = asyncio.run(
            self._process_inputs(
                input,
                spatial_info,
                temporal_info,
                mode="default",
                **kwargs,
            )
        )
        message = [{'role':'system','content':inputs["system_prompt"]},{'role':'user','content':inputs["user_prompt"]}]
        response = self.llm.gen(message)
        return response

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], mode: str = "default", **kwargs):
        """Handle asynchronous execution across different KV cache strategies."""
        if self.domain == 'humaneval':
            self.internal_tests = self.extract_example(input)
        if mode == "default":
            request_uid = input.get("_request_uid")
            inputs = await self._process_inputs(
                input,
                spatial_info,
                temporal_info,
                mode=mode,
                **kwargs,
            )
            message = [{'role':'system','content':inputs["system_prompt"]},{'role':'user','content':inputs["user_prompt"]}]
            result = await self.llm.agen(
                message,
                request_uid=request_uid,
                agent_id=self.id,
                agent_name=self.agent_name,
                agent_role=self.role,
            )
            return result

        request_uid = input.get("_request_uid") or kwargs.get("request_uid")
        if request_uid is None:
            raise ValueError("request_uid is required for request-scoped anchor updates.")

        mode_data = await self._process_inputs(
            input,
            spatial_info,
            temporal_info,
            mode="allow_kv_reuse",
            **kwargs,
        )
        if mode_data["early_response"] is not None:
            early = GenerationResult(
                text=mode_data["early_response"],
                mode="kv_reuse" if mode == "allow_kv_reuse" else mode,
                ttft=0.0,
            )
            return input['task'], early
        result = await self.llm.generate_for_agent(
            request_uid=request_uid,
            message=input['task'],
            preferred_mode=mode_data["preferred_mode"],
            output_dir=kwargs.get("output_dir"),
            agent_id=self.id,
            agent_name=self.agent_name,
            agent_role=self.role,
        )
        return input['task'], result

@AgentRegistry.register('FinalDirect')
class FinalDirect(Node):
    def __init__(self, id: str | None =None,  domain: str = "", llm_name: str = "",):
        """ Used for Directed IO """
        super().__init__(id, "FinalDirect")
        self.prompt_set = PromptSetRegistry.get(domain)

    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        return None

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output = ""
        info_list = []
        for info in spatial_info.values():
            info_list.append(info['output'])
        if len(info_list):
            output = info_list[-1]
        return output

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], mode: str = "default", **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output = ""
        info_list = []
        for info in spatial_info.values():
            info_list.append(info['output'])
        if len(info_list):
            output = info_list[-1]
        if mode == "allow_kv_reuse":
            return input.get("task"), output
        return output


@AgentRegistry.register('FinalMajorVote')
class FinalMajorVote(Node):
    def __init__(self, id: str | None =None,  domain: str = "", llm_name: str = "",):
        """ Used for Directed IO """
        super().__init__(id, "FinalMajorVote")
        self.prompt_set = PromptSetRegistry.get(domain)

    def _process_inputs(self, raw_inputs:Dict[str,str], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
        return None

    def _execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any],**kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output_num = {}
        max_output = ""
        max_output_num = 0
        for info in spatial_info.values():
            processed_output = self.prompt_set.postprocess_answer(info['output'])
            if processed_output in output_num:
                output_num[processed_output] += 1
            else:
                output_num[processed_output] = 1
            if output_num[processed_output] > max_output_num:
                max_output = processed_output
                max_output_num = output_num[processed_output]
        return max_output

    async def _async_execute(self, input:Dict[str,str],  spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], mode: str = "default", **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """
        output_num = {}
        max_output = ""
        max_output_num = 0
        for info in spatial_info.values():
            processed_output = await self.prompt_set.postprocess_answer(info['output'])
            logger.debug("Processed output: {}", processed_output)
            if processed_output in output_num:
                output_num[processed_output] += 1
            else:
                output_num[processed_output] = 1
            if output_num[processed_output] > max_output_num:
                max_output = processed_output
                max_output_num = output_num[processed_output]
        if mode == "allow_kv_reuse":
            return input.get("task"), max_output
        return max_output
