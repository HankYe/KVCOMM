"""Graph orchestration over `Node` instances with optional KV reuse flows."""
import shortuuid
from typing import Any, List, Optional, Dict
from abc import ABC
import numpy as np
import torch
import asyncio
import copy
from KVCOMM.graph.node import Node
from KVCOMM.agents.agent_registry import AgentRegistry
from KVCOMM.llm.config import KVCommConfig
from KVCOMM.llm.gpt_chat import LLMChat
from KVCOMM.utils.metrics import metrics_recorder
from KVCOMM.utils.log import logger

class Graph(ABC):
    """
    A framework for managing and executing a network of nodes using a language model.

    This class enables the creation of a graph structure for processing and analyzing data. Each node
    in the graph can perform specific operations, allowing for complex data processing workflows.
    The graph supports integration with language models, making it suitable for tasks that require
    natural language processing capabilities.

    The communication of the node depends on the node.spatial_predecessors and node.spatial_successors.
    
    Attributes:
        domain (str): The domain for which this graph is used.
        llm_name (str): The name of the llm that used for processing within the nodes.
        nodes (dict): A collection of nodes, each identified by a unique UUID.

    Methods:
        build_graph(): Method to be implemented for constructing the graph structure.
        add_node(node): Adds a new node to the graph with a unique identifier.
        run(inputs, num_rounds=1, max_tries=3, max_time=600): Executes the graph for a specified number of rounds, processing provided inputs.
        arun(input, num_rounds=1, max_tries=3, max_time=600, *, mode="default", **kwargs): Asynchronously executes the graph for a specified number of rounds, processing provided inputs.
        update_memory(message=None): Propagates memory update across nodes; supports multi-request keys.
        check_cycle(new_node, target_nodes): Detects if adding edges would create a cycle starting at `new_node`.
        update_masks(): Returns current spatial and temporal mask parameters.
        __getstate__(): Returns the state of the graph.
        __setstate__(state): Sets the state of the graph.
        __deepcopy__(memo): Creates a deep copy of the graph.
    """

    def __init__(self,
                domain: str,
                llm_name: Optional[str],
                agent_names: List[str],
                decision_method: str = None,
                fixed_spatial_masks:List[List[int]] = None,
                fixed_temporal_masks:List[List[int]] = None,
                node_kwargs:List[Dict] = None,
                kv_config: KVCommConfig | None = None,
                ):

        num_agents = len(agent_names)
        if fixed_spatial_masks is None:
            fixed_spatial_masks = [[1 if i!=j else 0 for j in range(num_agents)] for i in range(num_agents)]
        if fixed_temporal_masks is None:
            fixed_temporal_masks = [[1 for _ in range(num_agents)] for _ in range(num_agents)]
        spatial_mask_tensor = torch.as_tensor(fixed_spatial_masks, dtype=torch.float32).view(num_agents, num_agents)
        temporal_mask_tensor = torch.as_tensor(fixed_temporal_masks, dtype=torch.float32).view(num_agents, num_agents)
        assert spatial_mask_tensor.numel() == num_agents * num_agents, "The fixed_spatial_masks doesn't match the number of agents"
        assert temporal_mask_tensor.numel() == num_agents * num_agents, "The fixed_temporal_masks doesn't match the number of agents"
        self.kv_config = kv_config or KVCommConfig.from_env()
        self.id:str = shortuuid.ShortUUID().random(length=4)
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.agent_names:List[str] = agent_names
        self.decision_node:Node = AgentRegistry.get(
            decision_method,
            **{"domain": self.domain, "llm_name": self.llm_name, "llm_config": self.kv_config},
        ) if decision_method is not None else None
        self.nodes:Dict[str,Node] = {}
        self.potential_spatial_edges:List[List[str, str]] = []
        self.potential_temporal_edges:List[List[str,str]] = []
        self.node_kwargs = node_kwargs if node_kwargs is not None else [{} for _ in agent_names]
        for kwargs in self.node_kwargs:
            kwargs.setdefault("llm_config", self.kv_config)

        self.init_nodes()                              
        self.init_potential_edges()                                                                   


        self._spatial_mask_shape = spatial_mask_tensor.shape
        self._temporal_mask_shape = temporal_mask_tensor.shape
        self.spatial_masks = torch.nn.Parameter(spatial_mask_tensor.view(-1), requires_grad=False)
        self.temporal_masks = torch.nn.Parameter(temporal_mask_tensor.view(-1), requires_grad=False)

        self.spatial_logits = torch.nn.Parameter(self.spatial_masks.data.clone(), requires_grad=False)
        self.temporal_logits = torch.nn.Parameter(self.temporal_masks.data.clone(), requires_grad=False)

    @property
    def spatial_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].spatial_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def temporal_adj_matrix(self):
        matrix = np.zeros((len(self.nodes), len(self.nodes)))
        for i, node1_id in enumerate(self.nodes):
            for j, node2_id in enumerate(self.nodes):
                if self.nodes[node2_id] in self.nodes[node1_id].temporal_successors: 
                    matrix[i, j] = 1
        return matrix

    @property
    def num_edges(self):
        num_edges = 0
        for node in self.nodes.values():
            num_edges += len(node.spatial_successors)
        return num_edges

    @property
    def num_nodes(self):
        return len(self.nodes)

    def find_node(self, id: str):
        if id in self.nodes.keys():
            return self.nodes[id]
        raise Exception(f"Node not found: {id} among "
                        f"{[node.id for node in self.nodes.values()]}")

    def add_node(self, node: Node):
        node_id = node.id if node.id is not None else shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node

    def init_nodes(self):
        """
        Creates and adds new nodes to the graph.
        """
        for agent_name,kwargs in zip(self.agent_names,self.node_kwargs):
            if agent_name in AgentRegistry.registry:
                kwargs["domain"] = self.domain
                kwargs["llm_name"] = self.llm_name
                agent_instance = AgentRegistry.get(agent_name, **kwargs)
                self.add_node(agent_instance)

    def init_potential_edges(self):
        """
        Creates and potential edges to the graph.
        """
        for node1_id in self.nodes.keys():
            for node2_id in self.nodes.keys():
                self.potential_spatial_edges.append([node1_id,node2_id])
                self.potential_temporal_edges.append([node1_id,node2_id])

    def clear_spatial_connection(self):
        """
        Clear all the spatial connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].spatial_predecessors = []
            self.nodes[node_id].spatial_successors = []
        if self.decision_node is not None:
            self.decision_node.spatial_predecessors = []
            self.decision_node.spatial_successors = []

    def clear_temporal_connection(self):
        """
        Clear all the temporal connection of the nodes in the graph.
        """
        for node_id in self.nodes.keys():
            self.nodes[node_id].temporal_predecessors = []
            self.nodes[node_id].temporal_successors = []

    def connect_decision_node(self):
        """Connect every node to the decision node with a spatial edge."""
        for node_id in self.nodes.keys():
            self.nodes[node_id].add_successor(self.decision_node)

    def construct_spatial_connection(
        self,
        temperature: float = 1.0,
        threshold: float = None,
    ):
        """Rebuild spatial edges from masks while preventing cycles."""
        self.clear_spatial_connection()
        for potential_connection, edge_mask in zip(self.potential_spatial_edges, self.spatial_masks.view(-1)):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            if not self.check_cycle(in_node, {out_node}):
                out_node.add_successor(in_node,'spatial')

    def construct_temporal_connection(
        self,
        round:int = 0,
        temperature: float = 1.0,
        threshold: float = None,
    ):
        """Rebuild temporal edges for non-zero rounds from masks while preventing cycles."""
        self.clear_temporal_connection()
        if round == 0:
            return
        for potential_connection, edge_mask in zip(self.potential_temporal_edges, self.temporal_masks.view(-1)):
            out_node:Node = self.find_node(potential_connection[0])
            in_node:Node = self.find_node(potential_connection[1])
            if edge_mask == 0.0:
                continue
            if not self.check_cycle(in_node, {out_node}):
                out_node.add_successor(in_node,'temporal')

    def run(
        self,
        inputs: Any,
        num_rounds:int = 1,
        max_tries: int = 3,
        max_time: int = 600,
    ) -> List[Any]:
        """Execute nodes topologically for `num_rounds` and return final answers."""
        for round in range(num_rounds):
            self.construct_spatial_connection()
            self.construct_temporal_connection(round)

            in_degree = {node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()}
            zero_in_degree_queue = [node_id for node_id, deg in in_degree.items() if deg == 0]

            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                tries = 0
                while tries < max_tries:
                    try:
                        self.nodes[current_node_id].execute(inputs)                                      
                        break
                    except Exception as e:
                        logger.exception(
                            "Error during execution of node {}: {}",
                            current_node_id,
                            e,
                        )
                    tries += 1
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes.keys():
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)

            self.update_memory()
        if self.decision_node:
            self.connect_decision_node()
            self.decision_node.execute(inputs)
            final_answers = self.decision_node.outputs
            if len(final_answers) == 0:
                final_answers.append("No answer of the decision node")
        else:
            final_answers = self.nodes[list(self.nodes.keys())[-1]].outputs

        return final_answers

    async def arun(
        self,
        input: Dict[str, str],
        num_rounds: int = 1,
        max_tries: int = 3,
        max_time: int = 600,
        *,
        mode: str = "default",
        **kwargs,
    ) -> List[Any]:
        """Asynchronous execution entry supporting standard and KV-reuse workflows."""
        if mode == "default":
            request_uid = input.setdefault(
                "_request_uid", shortuuid.ShortUUID().random(length=8)
            )
            metrics_recorder.start_request(
                request_uid=request_uid,
                batch_index=input.get("_batch_index"),
                task=input.get("task"),
                execution_mode=mode,
            )
            for round in range(num_rounds):
                self.construct_spatial_connection()
                self.construct_temporal_connection(round)

                in_degree = {
                    node_id: len(node.spatial_predecessors)
                    for node_id, node in self.nodes.items()
                }
                zero_in_degree_queue = [
                    node_id for node_id, deg in in_degree.items() if deg == 0
                ]

                while zero_in_degree_queue:
                    current_node_id = zero_in_degree_queue.pop(0)
                    tries = 0
                    while tries < max_tries:
                        await asyncio.wait_for(
                            self.nodes[current_node_id].async_execute(input),
                            timeout=max_time,
                        )
                        break
                    for successor in self.nodes[current_node_id].spatial_successors:
                        if successor.id not in self.nodes:
                            continue
                        in_degree[successor.id] -= 1
                        if in_degree[successor.id] == 0:
                            zero_in_degree_queue.append(successor.id)

                self.update_memory()
            if self.decision_node:
                self.connect_decision_node()
                await self.decision_node.async_execute(input)
                final_answers = self.decision_node.outputs
                if len(final_answers) == 0:
                    final_answers.append("No answer of the decision node")
                metrics_recorder.finalize_request(request_uid)
            else:
                final_answers = self.nodes[list(self.nodes.keys())[-1]].outputs
                metrics_recorder.finalize_request(request_uid)
            return {
                "task": input.get("task"),
                "answers": final_answers,
            }

        if mode == "allow_kv_reuse":
            request_uid = input.setdefault(
                "_request_uid", shortuuid.ShortUUID().random(length=8)
            )
            kwargs = dict(kwargs)
            kwargs.setdefault("request_uid", request_uid)
            metrics_recorder.start_request(
                request_uid=request_uid,
                batch_index=input.get("_batch_index"),
                task=input.get("task"),
                execution_mode=mode,
            )
            self.construct_spatial_connection()
            self.construct_temporal_connection(0)
            in_degree = {
                node_id: len(node.spatial_predecessors)
                for node_id, node in self.nodes.items()
            }
            zero_in_degree_queue = [
                node_id for node_id, deg in in_degree.items() if deg == 0
            ]
            while zero_in_degree_queue:
                current_node_id = zero_in_degree_queue.pop(0)
                await asyncio.wait_for(
                    self.nodes[current_node_id].async_execute(
                        input, mode="initialization", **kwargs
                    ),
                    timeout=max_time,
                )
                for successor in self.nodes[current_node_id].spatial_successors:
                    if successor.id not in self.nodes:
                        continue
                    in_degree[successor.id] -= 1
                    if in_degree[successor.id] == 0:
                        zero_in_degree_queue.append(successor.id)
            if self.decision_node:
                self.connect_decision_node()
                await self.decision_node.async_execute(
                    input, mode="initialization", **kwargs
                )

            for round in range(num_rounds):
                self.construct_spatial_connection()
                self.construct_temporal_connection(round)

                in_degree = {
                    node_id: len(node.spatial_predecessors)
                    for node_id, node in self.nodes.items()
                }
                zero_in_degree_queue = [
                    node_id for node_id, deg in in_degree.items() if deg == 0
                ]

                while zero_in_degree_queue:
                    current_node_id = zero_in_degree_queue.pop(0)
                    tries = 0
                    while tries < max_tries:
                        await asyncio.wait_for(
                            self.nodes[current_node_id].async_execute(
                                input, mode="allow_kv_reuse", **kwargs
                            ),
                            timeout=max_time,
                        )
                        break
                    for successor in self.nodes[current_node_id].spatial_successors:
                        if successor.id not in self.nodes:
                            continue
                        in_degree[successor.id] -= 1
                        if in_degree[successor.id] == 0:
                            zero_in_degree_queue.append(successor.id)

                self.update_memory(message=input["task"])

            if self.decision_node:
                self.connect_decision_node()
                await self.decision_node.async_execute(
                    input, mode="allow_kv_reuse", **kwargs
                )
                final_answers = self.decision_node.outputs[input["task"]]
                if len(final_answers) == 0:
                    final_answers.append("No answer of the decision node")
                LLMChat.finalize_request(request_uid)
                metrics_recorder.finalize_request(request_uid)
            else:
                final_answers = self.nodes[list(self.nodes.keys())[-1]].outputs
                LLMChat.finalize_request(request_uid)
                metrics_recorder.finalize_request(request_uid)
            return {
                "task": input["task"],
                "answers": final_answers,
            }

        raise ValueError(f"Unsupported arun mode: {mode}")


    def update_memory(self, message: str = None):
        """Propagate memory update across nodes; supports multi-request keys."""
        if message is None:
            for id,node in self.nodes.items():
                node.update_memory()
        else:
            for id,node in self.nodes.items():
                node.update_memory_multirequest(message)

    def check_cycle(self, new_node, target_nodes):
        """Detect if adding edges would create a cycle starting at `new_node`."""
        if new_node in target_nodes:
            return True
        for successor in new_node.spatial_successors:
            if self.check_cycle(successor, target_nodes):
                return True
        return False

    def update_masks(self) -> torch.Tensor:
        """Return current spatial and temporal mask parameters."""
        return self.spatial_masks, self.temporal_masks

    def __getstate__(self):
        state = self.__dict__.copy()

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

        for node in self.nodes.values():
            if hasattr(node, 'lock'):
                node.lock = asyncio.Lock()

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        state = self.__getstate__()

        copied_state = copy.deepcopy(state, memo)
        result.__setstate__(copied_state)
        return result
