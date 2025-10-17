from abc import ABC, abstractmethod
from typing import List, Optional
import os
from KVCOMM.llm.format import Message
from KVCOMM.utils.metrics import GenerationResult


OUT_LENGTH = int(os.getenv('OUT_LENGTH', '0'))
class LLM(ABC):
    """Abstract base class for text-generating language models.

    Subclasses implement synchronous and asynchronous generation. Defaults are
    provided for common decoding parameters and can be overridden by concrete
    implementations.
    """

    DEFAULT_MAX_TOKENS = 512
    DEFAULT_TEMPERATURE = 1.0
    DEFUALT_NUM_COMPLETIONS = 1
    DEFAULT_BANDWIDTH = 3072
    DEFAULT_MIN_TOKENS = OUT_LENGTH

    @abstractmethod
    async def agen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        ) -> GenerationResult:
        """Asynchronously generate a completion.

        Args:
            messages: Conversation history to condition the model.
            max_tokens: Maximum tokens to generate. Uses default if None.
            temperature: Sampling temperature. Uses default if None.

        Returns:
            GenerationResult: Structured output containing text and timing info.
        """
        pass

    @abstractmethod
    def gen(
        self,
        messages: List[Message],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        ) -> GenerationResult:
        """Synchronously generate a completion.

        Args:
            messages: Conversation history to condition the model.
            max_tokens: Maximum tokens to generate. Uses default if None.
            temperature: Sampling temperature. Uses default if None.

        Returns:
            GenerationResult: Structured output containing text and timing info.
        """
        pass
