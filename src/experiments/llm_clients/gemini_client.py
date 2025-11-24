"""
Client for Google Gemini API
"""
import os
import time
from typing import List, Optional
from dataclasses import dataclass

try:
    import google.generativeai as genai
    from dotenv import load_dotenv
except ImportError:
    print(" Installa: pip install google-generativeai python-dotenv")
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


class GeminiClient:
    """Client for Google Gemini API"""
    
    # Gemini pricing (as of 2025)
    # FREE tier: 60 requests per minute
    # Paid tier: $0.00025 per 1K characters input, $0.0005 per 1K characters output
    PRICING = {
        "gemini-1.5-pro": {
            "input_per_1k_chars": 0.00025,
            "output_per_1k_chars": 0.0005,
        },
        "gemini-1.5-flash": {
            "input_per_1k_chars": 0.000125,
            "output_per_1k_chars": 0.0005,
        },
        "gemini-pro": {
            "input_per_1k_chars": 0.0,  # FREE
            "output_per_1k_chars": 0.0,  # FREE
        }
    }
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-2.0-flash",
        temperature: float = 0.2,
        max_output_tokens: int = 2048,
        top_p: float = 0.95,
        top_k: int = 40,
    ):
        """
        Initialize Gemini client
        
        Args:
            api_key: Google API key (or set GOOGLE_API_KEY env var)
            model: Model name (gemini-1.5-pro, gemini-1.5-flash, gemini-pro)
            temperature: Sampling temperature (0.0 - 1.0)
            max_output_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
        """
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY env var or pass api_key parameter.\n"
                "Get your free key at: https://aistudio.google.com/app/apikey"
            )
        
        self.model_name = model
        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.top_p = top_p
        self.top_k = top_k
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Create model
        self.model = genai.GenerativeModel(
            model_name=model,
            generation_config={
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "max_output_tokens": max_output_tokens,
            }
        )
        
        print(f" Gemini client initialized: {model}")
    
    def generate_code(
        self,
        prompt: str,
        temperature: float = None,
        max_tokens: int = None,
        timeout: int = 30,
    ) -> GenerationResult:
        """
        Generate code from prompt with automatic retry on rate limits
        
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
        max_output = max_tokens if max_tokens is not None else self.max_output_tokens
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Configure generation
                generation_config = genai.types.GenerationConfig(
                    temperature=temp,
                    max_output_tokens=max_output,
                    top_p=self.top_p,
                    top_k=self.top_k,
                )
                
                # Generate response
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                # Extract code
                code = response.text
                
                # Calculate tokens (approximate - Gemini uses characters)
                prompt_chars = len(prompt)
                completion_chars = len(code)
                
                # Estimate tokens (roughly 4 chars per token)
                prompt_tokens = prompt_chars // 4
                completion_tokens = completion_chars // 4
                total_tokens = prompt_tokens + completion_tokens
                
                # Calculate cost
                cost = self._calculate_cost(prompt_chars, completion_chars)
                
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
                error_str = str(e)
                
                # Se è un rate limit error, estrai il retry delay e riprova
                if ("429" in error_str or "ResourceExhausted" in error_str) and attempt < max_retries - 1:
                    # Cerca "Please retry in X.XXXs" nell'errore
                    import re
                    match = re.search(r'retry in (\d+\.?\d*)s', error_str)
                    if match:
                        retry_delay = float(match.group(1))
                        print(f"   Rate limit: attendo {retry_delay:.1f}s (retry {attempt + 1}/{max_retries})...", end='', flush=True)
                        time.sleep(retry_delay + 1)  # +1 secondo di margine
                        print(" ✓")
                        continue
                
                # Altri errori o tentativi esauriti
                latency = time.time() - start_time
                print(f" Warning: Gemini generation error: {type(e).__name__}: {str(e)[:200]}")
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
        
        # Fallback se tutti i retry falliscono (rate limit)
        latency = time.time() - start_time
        return GenerationResult(
            code="",
            model=self.model_name,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            latency_seconds=latency,
            error="Max retries exceeded due to rate limits",
        )
    
    def generate_multiple(
        self,
        prompt: str,
        k: int = 5,
    ) -> List[GenerationResult]:
        """
        Generate k code samples with automatic rate limit handling
        
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
            
            # Small delay between requests (rate limits are handled by retry logic)
            # 4s base delay for 15 RPM limit (60s / 15 = 4s)
            if i < k - 1:
                time.sleep(4.2)
        
        return results
    
    def _calculate_cost(self, input_chars: int, output_chars: int) -> float:
        """Calculate cost in USD"""
        pricing = self.PRICING.get(self.model_name, self.PRICING["gemini-1.5-flash"])
        
        input_cost = (input_chars / 1000) * pricing["input_per_1k_chars"]
        output_cost = (output_chars / 1000) * pricing["output_per_1k_chars"]
        
        return input_cost + output_cost


# Test script
if __name__ == "__main__":
    print("=" * 60)
    print(" TEST GOOGLE GEMINI CLIENT")
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
        client = GeminiClient(
            model="gemini-2.0-flash",  # Latest and fastest (FREE)
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
        print("1. Go to https://aistudio.google.com/app/apikey")
        print("2. Create a new API key (FREE)")
        print("3. Add to .env file:")
        print("   GOOGLE_API_KEY=your-key-here")
    
    except Exception as e:
        print(f"\n Unexpected Error: {e}")
        import traceback
        traceback.print_exc()
