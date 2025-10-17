import shortuuid
from typing import List, Any, Optional,Dict
from abc import ABC, abstractmethod
import asyncio

from KVCOMM.utils.metrics import GenerationResult, metrics_recorder

node_id_ = 0
class Node(ABC):
    """
    Represents a processing unit within a graph-based framework.

    This class encapsulates the functionality for a node in a graph, managing
    connections to other nodes, handling inputs and outputs, and executing
    assigned operations. It supports both individual and aggregated processing modes.

    Attributes:
        id (uuid.UUID): Unique identifier for the node.
        agent_type(str): Associated agent name for node-specific operations.
        spatial_predecessors (List[Node]): Nodes that precede this node in the graph.
        spatial_successors (List[Node]): Nodes that succeed this node in the graph.
        inputs (List[Any]): Inputs to be processed by the node.
        outputs (List[Any]): Results produced after node execution.
        raw_inputs (List[Any]): The original input contains the question or math problem.
        last_memory (Dict[str,List[Any]]): Input and output of the previous timestamp.
        
    Methods:
        add_predecessor(operation): 
            Adds a node as a predecessor of this node, establishing a directed connection.
        add_successor(operation): 
            Adds a node as a successor of this node, establishing a directed connection.
        memory_update():
            Update the last_memory.
        get_spatial_info():
            Get all of the info from spatial spatial_predecessors.
        execute(**kwargs): 
            Processes the inputs through the node's operation, handling each input individually.
        _execute(input, **kwargs): 
            An internal method that defines how a single input is processed by the node. This method should be implemented specifically for each node type.
        _process_inputs(raw_inputs, spatial_info, temporal_info, **kwargs)->List[Any]:
            An internal medthod to process the raw_input, the spatial info and temporal info to get the final inputs.
    """

    def __init__(self,
                 id: Optional[str],
                 agent_name:str="",
                 domain:str="",
                 llm_name:str = "",
                 ):
        """
        Initializes a new Node instance.
        """
        global node_id_
        self.id:str = id if id is not None else str(node_id_)                                        
        node_id_ += 1
        self.agent_name:str = agent_name
        self.domain:str = domain
        self.llm_name:str = llm_name
        self.spatial_predecessors: List[Node] = []
        self.spatial_successors: List[Node] = []
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []
        self.inputs: List[Any] = []
        self.outputs: List[Any] = []
        self.raw_inputs: List[Any] = []
        self.role = ""
        self.last_memory: Dict[str,List[Any]] = {'inputs':[],'outputs':[],'raw_inputs':[]}  

    @property
    def node_name(self):
        return self.__class__.__name__

    def add_predecessor(self, operation: 'Node', st='spatial'):
        if st == 'spatial' and operation not in self.spatial_predecessors:
            self.spatial_predecessors.append(operation)
            operation.spatial_successors.append(self)
        elif st == 'temporal' and operation not in self.temporal_predecessors:
            self.temporal_predecessors.append(operation)
            operation.temporal_successors.append(self)

    def add_successor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation not in self.spatial_successors:
            self.spatial_successors.append(operation)
            operation.spatial_predecessors.append(self)
        elif st == 'temporal' and operation not in self.temporal_successors:
            self.temporal_successors.append(operation)
            operation.temporal_predecessors.append(self)

    def remove_predecessor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation in self.spatial_predecessors:
            self.spatial_predecessors.remove(operation)
            operation.spatial_successors.remove(self)
        elif st =='temporal' and operation in self.temporal_predecessors:
            self.temporal_predecessors.remove(operation)
            operation.temporal_successors.remove(self)

    def remove_successor(self, operation: 'Node', st='spatial'):
        if st =='spatial' and operation in self.spatial_successors:
            self.spatial_successors.remove(operation)
            operation.spatial_predecessors.remove(self)
        elif st =='temporal' and operation in self.temporal_successors:
            self.temporal_successors.remove(operation)
            operation.temporal_predecessors.remove(self)

    def clear_connections(self):
        self.spatial_predecessors: List[Node] = []
        self.spatial_successors: List[Node] = []
        self.temporal_predecessors: List[Node] = []
        self.temporal_successors: List[Node] = []        

    def update_memory(self):
        """Store current inputs/outputs into `last_memory` for temporal edges."""
        self.last_memory['inputs'] = self.inputs
        self.last_memory['outputs'] = self.outputs
        self.last_memory['raw_inputs'] = self.raw_inputs

    def update_memory_multirequest(self, message:str):
        """Update per-message memories when running in multi-request mode."""
        if message not in self.last_memory:
            self.last_memory[message] = {}
        self.last_memory[message]['outputs'] = self.outputs[message]

    def get_spatial_info(self, message=None, prefix=False)->Dict[str,Dict]:
        """ Return a dict that maps id to info. """
        spatial_info = {}
        if self.spatial_predecessors is not None:
            for predecessor in self.spatial_predecessors:
                predecessor_outputs = predecessor.outputs if message is None else predecessor.outputs[message]
                if isinstance(predecessor_outputs, list) and len(predecessor_outputs):
                    predecessor_output = predecessor_outputs[-1]
                elif isinstance(predecessor_outputs, list) and len(predecessor_outputs)==0 and not prefix:
                    continue
                else:
                    predecessor_output = predecessor_outputs
                spatial_info[predecessor.id] = {"role":predecessor.role,"output":predecessor_output}

        return spatial_info

    def get_temporal_info(self, message=None, prefix=False)->Dict[str,Any]:
        temporal_info = {}
        if self.temporal_predecessors is not None:
            for predecessor in self.temporal_predecessors:
                predecessor_outputs = predecessor.last_memory['outputs'] if message is None else predecessor.last_memory[message]['outputs']
                if isinstance(predecessor_outputs, list) and len(predecessor_outputs):
                    predecessor_output = predecessor_outputs[-1]
                elif isinstance(predecessor_outputs, list) and len(predecessor_outputs)==0 and not prefix:
                    continue
                else:
                    predecessor_output = predecessor_outputs
                temporal_info[predecessor.id] = {"role":predecessor.role,"output":predecessor_output}

        return temporal_info

    def execute(self, input:Any, **kwargs):
        """Synchronous execution wrapper that normalizes outputs to a list."""
        self.outputs = []
        spatial_info:Dict[str,Dict] = self.get_spatial_info()
        temporal_info:Dict[str,Dict] = self.get_temporal_info()
        results = [self._execute(input, spatial_info, temporal_info, **kwargs)]

        for result in results:
            if not isinstance(result, list):
                result = [result]
            self.outputs.extend(result)
        return self.outputs

    async def async_execute(self, input:Any, mode: str = "default", **kwargs):
        """Execute a node asynchronously with optional KV cache handling modes."""
        if mode == "default":
            self.outputs = []
            spatial_info:Dict[str,Any] = self.get_spatial_info()
            temporal_info:Dict[str,Any] = self.get_temporal_info()
            tasks = [
                asyncio.create_task(
                    self._async_execute(input, spatial_info, temporal_info, mode=mode, **kwargs)
                )
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            request_uid = input.get("_request_uid") or kwargs.get("request_uid")
            for result in results:
                if not isinstance(result, list):
                    result = [result]
                for item in result:
                    if isinstance(item, GenerationResult):
                        generation = item
                    else:
                        generation = GenerationResult(
                            text=str(item),
                            mode="default",
                            ttft=0.0,
                        )
                    self.outputs.append(generation.text)
                    if request_uid:
                        metrics_recorder.record_agent_output(
                            request_uid=request_uid,
                            agent_id=self.id,
                            agent_name=self.agent_name,
                            agent_role=self.role,
                            generation=generation,
                        )
            return self.outputs

        if mode == "allow_kv_reuse":
            self.outputs = {}
            spatial_info:Dict[str,Any] = self.get_spatial_info(input['task'])
            temporal_info:Dict[str,Any] = self.get_temporal_info(input['task'])
            tasks = [
                asyncio.create_task(
                    self._async_execute(
                        input,
                        spatial_info,
                        temporal_info,
                        mode=mode,
                        **kwargs,
                    )
                )
            ]
            results = await asyncio.gather(*tasks, return_exceptions=False)
            request_uid = input.get("_request_uid") or kwargs.get("request_uid")
            for (message, result) in results:
                if not isinstance(result, list):
                    result = [result]
                if message not in self.outputs:
                    self.outputs[message] = []
                for item in result:
                    if isinstance(item, GenerationResult):
                        generation = item
                    else:
                        generation = GenerationResult(
                            text=str(item),
                            mode="kv_reuse" if mode == "allow_kv_reuse" else mode,
                            ttft=0.0,
                        )
                    self.outputs[message].append(generation.text)
                    if request_uid:
                        metrics_recorder.record_agent_output(
                            request_uid=request_uid,
                            agent_id=self.id,
                            agent_name=self.agent_name,
                            agent_role=self.role,
                            generation=generation,
                        )
            return self.outputs

        if mode == "initialization":
            self.outputs = []
            spatial_info:Dict[str,Any] = self.get_spatial_info(prefix=True)
            temporal_info:Dict[str,Any] = self.get_temporal_info(prefix=True)
            if hasattr(self.llm, "_initialization") and not self.llm._initialization[self.id]:
                await self._process_inputs(
                    input,
                    spatial_info,
                    temporal_info,
                    mode="allow_kv_reuse",
                    **kwargs,
                )
            return self.outputs

        raise ValueError(f"Unsupported async execution mode: {mode}")

    @abstractmethod
    def _execute(self, input:List[Any], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

    @abstractmethod
    async def _async_execute(self, input:List[Any], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], mode: str = "default", **kwargs):
        """ To be overriden by the descendant class """
        """ Use the processed input to get the result """

    @abstractmethod
    def _process_inputs(self, raw_inputs:List[Any], spatial_info:Dict[str,Any], temporal_info:Dict[str,Any], **kwargs)->List[Any]:
        """ To be overriden by the descendant class """
        """ Process the raw_inputs(most of the time is a List[Dict]) """
