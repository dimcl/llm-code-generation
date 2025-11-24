"""
Client for Qwen Coder via Groq API
Qwen is Alibaba's open-source LLM, specialized in code generation
"""
import os
import time
from typing import List, Optional
from dataclasses import dataclass

try:
    from groq import Groq
    from dotenv import load_dotenv
except ImportError:
    print(" Installa: pip install groq python-dotenv")
    exit(1)

# Load environment variables
load_dotenv()


@dataclass
class GenerationResult:
    """Result from LLM code generation"""
    code: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    latency_seconds: float
    error: Optional[str] = None


class QwenClient:
    """Client for Qwen Coder via Groq API (fast and free tier available)"""
    
    # Groq pricing for Qwen (FREE tier: 30 req/min, 14400 tokens/min)
    PRICING = {
        "qwen/qwen3-32b": {
            "input_per_1m_tokens": 0.05,  # Estimated
            "output_per_1m_tokens": 0.08,  # Estimated
        },
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "qwen/qwen3-32b",
        temperature: float = 0.2,
    ):
        """
        Initialize Qwen client via Groq
        
        Args:
            api_key: Groq API key (or set GROQ_API_KEY env var)
            model: Model name (qwen/qwen3-32b)
            temperature: Sampling temperature (0.0 - 1.0)
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY env var or pass api_key parameter.\n"
                "Get your free key at: https://console.groq.com/keys"
            )
        
        self.model_name = model
        self.temperature = temperature
        
        # Initialize Groq client
        self.client = Groq(api_key=self.api_key)
        
        print(f" Qwen Coder client initialized: {model}")
    
    def generate_code(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        timeout: int = 30,
    ) -> GenerationResult:
        """
        Generate code from prompt
        
        Args:
            prompt: Problem description
            temperature: Temperature for generation (0.0-1.0), defaults to client default
            max_tokens: Max tokens to generate, defaults to client default
            timeout: Timeout in seconds
            
        Returns:
            GenerationResult with generated code and metadata
        """
        start_time = time.time()
        
        # Use provided parameters or defaults
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else 2048
        
        try:
            # Create chat completion
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "system",
                        "content": "You are Qwen, an expert AI assistant specialized in Python programming. Generate clean, efficient, well-documented Python code with proper error handling."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=temp,
                max_tokens=max_tok,
                top_p=0.95,
            )
            
            # Extract code
            code = response.choices[0].message.content
            
            # Get token usage
            usage = response.usage
            prompt_tokens = usage.prompt_tokens
            completion_tokens = usage.completion_tokens
            total_tokens = usage.total_tokens
            
            # Calculate cost
            cost = self._calculate_cost(prompt_tokens, completion_tokens)
            
            latency = time.time() - start_time
            
            return GenerationResult(
                code=code,
                model=self.model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                latency_seconds=latency,
            )
            
        except Exception as e:
            latency = time.time() - start_time
            return GenerationResult(
                code="",
                model=self.model_name,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                latency_seconds=latency,
                error=str(e),
            )
    
    def generate_multiple(
        self,
        prompt: str,
        k: int = 5,
    ) -> List[GenerationResult]:
        """
        Generate k code samples
        
        Args:
            prompt: Problem description
            k: Number of samples to generate
            
        Returns:
            List of GenerationResults
        """
        results = []
        for i in range(k):
            print(f"  Generating sample {i+1}/{k}...")
            result = self.generate_code(prompt)
            results.append(result)
            
            # Respect rate limits (30 req/min for free tier)
            if i < k - 1:
                time.sleep(2.5)  # ~24 requests per minute
        
        return results
    
    def _calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate cost in USD"""
        pricing = self.PRICING.get(self.model_name, self.PRICING["qwen/qwen3-32b"])
        
        input_cost = (prompt_tokens / 1_000_000) * pricing["input_per_1m_tokens"]
        output_cost = (completion_tokens / 1_000_000) * pricing["output_per_1m_tokens"]
        
        return input_cost + output_cost


# Test script
if __name__ == "__main__":
    print("=" * 60)
    print(" TEST QWEN CODER CLIENT")
    print("=" * 60)
    
    # Test prompt
    test_prompt = """Write a Python function that checks if a number is prime.

def is_prime(n: int) -> bool:
    \"\"\"
    Check if n is a prime number.
    
    Args:
        n: Integer to check
        
    Returns:
        True if n is prime, False otherwise
        
    Examples:
        >>> is_prime(2)
        True
        >>> is_prime(4)
        False
        >>> is_prime(17)
        True
    \"\"\"
"""
    
    try:
        # Initialize client
        client = QwenClient(
            model="qwen/qwen3-32b",  # Alibaba's Qwen 3 32B model
            temperature=0.2
        )
        
        print(f"\n Prompt:\n{test_prompt}\n")
        print(" Generating code...\n")
        
        # Generate code
        result = client.generate_code(test_prompt)
        
        if result.error:
            print(f" Error: {result.error}")
        else:
            print(" Code generated successfully!\n")
            print("=" * 60)
            print("GENERATED CODE:")
            print("=" * 60)
            print(result.code)
            print("=" * 60)
            print(f"\n Metadata:")
            print(f"  - Model: {result.model}")
            print(f"  - Tokens: {result.total_tokens} (prompt: {result.prompt_tokens}, completion: {result.completion_tokens})")
            print(f"  - Cost: ${result.cost_usd:.6f}")
            print(f"  - Latency: {result.latency_seconds:.2f}s")
            
            # Test multiple generations
            print(f"\n Testing k=3 generations...")
            results = client.generate_multiple(test_prompt, k=3)
            
            print(f"\n Generated {len(results)} samples")
            total_cost = sum(r.cost_usd for r in results)
            avg_latency = sum(r.latency_seconds for r in results) / len(results)
            print(f"  - Total cost: ${total_cost:.6f}")
            print(f"  - Avg latency: {avg_latency:.2f}s")
            
    except ValueError as e:
        print(f"\n Configuration Error: {e}")
        print("\n Setup Instructions:")
        print("1. Go to https://console.groq.com/keys")
        print("2. Sign up for free")
        print("3. Create a new API key")
        print("4. Add to .env file:")
        print("   GROQ_API_KEY=gsk-your-key-here")
    
    except Exception as e:
        print(f"\n Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
