"""
Prompt templates for code generation
"""

from typing import Dict, List


class PromptTemplate:
    """Base class for prompt templates"""
    
    @staticmethod
    def basic(problem_description: str, function_signature: str = None) -> str:
        """
        Basic prompt with just the problem description
        
        Args:
            problem_description: Description of the problem
            function_signature: Function signature (optional)
        
        Returns:
            Formatted prompt
        """
        prompt = f"{problem_description}\n\n"
        
        if function_signature:
            prompt += f"Function signature:\n{function_signature}\n\n"
        
        prompt += "Write a complete, efficient Python solution. Include only the code, without explanations."
        
        return prompt
    
    @staticmethod
    def few_shot(problem_description: str, examples: List[Dict], function_signature: str = None) -> str:
        """
        Few-shot prompt with examples
        
        Args:
            problem_description: Description of the problem
            examples: List of example problems and solutions
            function_signature: Function signature (optional)
        
        Returns:
            Formatted prompt
        """
        prompt = "Here are some examples of similar problems and their solutions:\n\n"
        
        for i, example in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Problem: {example['problem']}\n"
            prompt += f"Solution:\n```python\n{example['solution']}\n```\n\n"
        
        prompt += "Now solve this problem:\n\n"
        prompt += f"{problem_description}\n\n"
        
        if function_signature:
            prompt += f"Function signature:\n{function_signature}\n\n"
        
        prompt += "Write a complete, efficient Python solution following the style of the examples above."
        
        return prompt
    
    @staticmethod
    def chain_of_thought(problem_description: str, function_signature: str = None) -> str:
        """
        Chain of Thought prompt that encourages step-by-step reasoning
        
        Args:
            problem_description: Description of the problem
            function_signature: Function signature (optional)
        
        Returns:
            Formatted prompt
        """
        prompt = f"{problem_description}\n\n"
        
        if function_signature:
            prompt += f"Function signature:\n{function_signature}\n\n"
        
        prompt += """
Before writing the code, let's think step by step:
1. What is the input and expected output?
2. What algorithm or approach should we use?
3. What are the edge cases to handle?
4. What is the time and space complexity?

Now, write a complete, efficient Python solution that addresses all these considerations.
"""
        
        return prompt
    
    @staticmethod
    def with_tests(problem_description: str, test_cases: List[Dict], function_signature: str = None) -> str:
        """
        Prompt that includes test cases
        
        Args:
            problem_description: Description of the problem
            test_cases: List of test cases (input/output pairs)
            function_signature: Function signature (optional)
        
        Returns:
            Formatted prompt
        """
        prompt = f"{problem_description}\n\n"
        
        if function_signature:
            prompt += f"Function signature:\n{function_signature}\n\n"
        
        prompt += "Your solution should pass the following test cases:\n\n"
        
        for i, test in enumerate(test_cases, 1):
            prompt += f"Test {i}:\n"
            prompt += f"Input: {test['input']}\n"
            prompt += f"Expected Output: {test['output']}\n\n"
        
        prompt += "Write a complete, efficient Python solution that passes all test cases."
        
        return prompt
    
    @staticmethod
    def optimized(problem_description: str, function_signature: str = None, constraints: Dict = None) -> str:
        """
        Prompt focusing on optimization and best practices
        
        Args:
            problem_description: Description of the problem
            function_signature: Function signature (optional)
            constraints: Dictionary with constraints (time_complexity, space_complexity, etc.)
        
        Returns:
            Formatted prompt
        """
        prompt = f"{problem_description}\n\n"
        
        if function_signature:
            prompt += f"Function signature:\n{function_signature}\n\n"
        
        prompt += "Requirements:\n"
        prompt += "- Write clean, readable, and well-documented code\n"
        prompt += "- Follow PEP 8 style guidelines\n"
        prompt += "- Use meaningful variable names\n"
        prompt += "- Handle edge cases properly\n"
        
        if constraints:
            if "time_complexity" in constraints:
                prompt += f"- Achieve O({constraints['time_complexity']}) time complexity\n"
            if "space_complexity" in constraints:
                prompt += f"- Achieve O({constraints['space_complexity']}) space complexity\n"
        
        prompt += "\nWrite an optimized Python solution."
        
        return prompt


# Predefined few-shot examples for common problem types
FEW_SHOT_EXAMPLES = {
    "list_manipulation": [
        {
            "problem": "Write a function that removes duplicates from a list while maintaining order.",
            "solution": """def remove_duplicates(lst):
    seen = set()
    result = []
    for item in lst:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result"""
        }
    ],
    "string_manipulation": [
        {
            "problem": "Write a function that reverses words in a string.",
            "solution": """def reverse_words(s):
    return ' '.join(s.split()[::-1])"""
        }
    ],
    "algorithms": [
        {
            "problem": "Write a function that finds the maximum subarray sum.",
            "solution": """def max_subarray_sum(arr):
    max_sum = current_sum = arr[0]
    for num in arr[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum"""
        }
    ]
}


if __name__ == "__main__":
    # Test prompt templates
    
    problem = "Write a function that finds all pairs in a list that sum to a target value."
    signature = "def find_pairs(nums: List[int], target: int) -> List[Tuple[int, int]]:"
    
    print("=== BASIC PROMPT ===")
    print(PromptTemplate.basic(problem, signature))
    print("\n" + "="*50 + "\n")
    
    print("=== CHAIN OF THOUGHT PROMPT ===")
    print(PromptTemplate.chain_of_thought(problem, signature))
    print("\n" + "="*50 + "\n")
    
    print("=== FEW-SHOT PROMPT ===")
    print(PromptTemplate.few_shot(
        problem, 
        FEW_SHOT_EXAMPLES["list_manipulation"],
        signature
    ))
