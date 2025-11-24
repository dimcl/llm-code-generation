"""
Base LLM Client Interface
All LLM clients should inherit from this abstract class
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class GenerationResult:
    """Result of a code generation attempt"""
    code: str
    model: str
    prompt: str
    temperature: float
    tokens_used: int
    time_seconds: float
    cost_usd: float
    error: Optional[str] = None
    metadata: Optional[Dict] = None


class BaseLLMClient(ABC):
    """Abstract base class for all LLM clients"""
    
    def __init__(self, api_key: str, model_name: str, temperature: float = 0.2, max_tokens: int = 1024):
        self.api_key = api_key
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
    
    @abstractmethod
    def generate_code(self, prompt: str, **kwargs) -> GenerationResult:
        """
        Generate code from a prompt
        
        Args:
            prompt: The input prompt
            **kwargs: Additional model-specific parameters
        
        Returns:
            GenerationResult object with code and metadata
        """
        pass
    
    @abstractmethod
    def generate_multiple(self, prompt: str, k: int = 5, **kwargs) -> List[GenerationResult]:
        """
        Generate k code samples from a prompt
        
        Args:
            prompt: The input prompt
            k: Number of generations
            **kwargs: Additional model-specific parameters
        
        Returns:
            List of GenerationResult objects
        """
        pass
    
    def calculate_cost(self, tokens_used: int) -> float:
        """Calculate cost in USD based on tokens used (override for specific pricing)"""
        return 0.0
    
    def __repr__(self):
        return f"{self.__class__.__name__}(model={self.model_name})"
