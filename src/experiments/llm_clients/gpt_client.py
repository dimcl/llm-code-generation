"""
OpenAI GPT Client for code generation
Supports GPT-4o-mini via Azure OpenAI
"""

import time
import os
from typing import List
from openai import AzureOpenAI
from .base_client import BaseLLMClient, GenerationResult


class GPTClient(BaseLLMClient):
    """Client for OpenAI GPT models via Azure"""
    
    # Pricing per 1M tokens for GPT-4o-mini (Azure OpenAI)
    PRICING = {
        "gpt-4o-mini": {"input": 0.15, "output": 0.60},  # per 1M tokens
    }
    
    def __init__(self, api_key: str = None, model_name: str = "gpt-4o-mini", **kwargs):
        super().__init__(api_key, model_name, **kwargs)
        
        # Azure OpenAI configuration
        self.azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        self.azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
        self.api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-08-01-preview")
        
        # Use provided key or get from environment
        azure_key = api_key or os.getenv("AZURE_OPENAI_KEY")
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=azure_key,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint
        )
    
    def generate_code(self, prompt: str, **kwargs) -> GenerationResult:
        """Generate code using GPT-4o-mini via Azure"""
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.azure_deployment,  # Use deployment name for Azure
                messages=[
                    {"role": "system", "content": "You are an expert Python programmer. Generate clean, efficient, and well-documented code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
                n=1
            )
            
            code = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            time_elapsed = time.time() - start_time
            
            # Calculate cost
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            cost = self._calculate_cost(input_tokens, output_tokens)
            
            return GenerationResult(
                code=code,
                model=f"{self.model_name} (Azure)",
                prompt=prompt,
                temperature=self.temperature,
                tokens_used=tokens_used,
                time_seconds=time_elapsed,
                cost_usd=cost,
                metadata={
                    "finish_reason": response.choices[0].finish_reason,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "deployment": self.azure_deployment
                }
            )
        
        except Exception as e:
            return GenerationResult(
                code="",
                model=f"{self.model_name} (Azure)",
                prompt=prompt,
                temperature=self.temperature,
                tokens_used=0,
                time_seconds=time.time() - start_time,
                cost_usd=0.0,
                error=str(e)
            )
    
    def generate_multiple(self, prompt: str, k: int = 5, **kwargs) -> List[GenerationResult]:
        """Generate k code samples"""
        return [self.generate_code(prompt, **kwargs) for _ in range(k)]
    
    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost based on token usage (per 1M tokens)"""
        if self.model_name not in self.PRICING:
            return 0.0
        
        pricing = self.PRICING[self.model_name]
        # Pricing is per 1M tokens
        input_cost = (input_tokens / 1_000_000) * pricing["input"]
        output_cost = (output_tokens / 1_000_000) * pricing["output"]
        
        return input_cost + output_cost



if __name__ == "__main__":
    # Test the Azure OpenAI client
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Azure credentials will be loaded from .env
    client = GPTClient(model_name="gpt-4o-mini")
    
    test_prompt = """
Write a Python function that takes a list of integers and returns the sum of all even numbers.

Function signature:
def sum_even_numbers(numbers: List[int]) -> int:
    pass
"""
    
    print(" Testing GPT-4o-mini (Azure OpenAI)...")
    print(f"Endpoint: {client.azure_endpoint}")
    print(f"Deployment: {client.azure_deployment}\n")
    
    result = client.generate_code(test_prompt)
    
    if result.error:
        print(f" Error: {result.error}")
    else:
        print(f" Success!")
        print(f"Model: {result.model}")
        print(f"Tokens: {result.tokens_used}")
        print(f"Time: {result.time_seconds:.2f}s")
        print(f"Cost: ${result.cost_usd:.6f}")
        print(f"\nGenerated Code:\n{result.code}")

