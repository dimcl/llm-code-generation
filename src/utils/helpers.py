"""
Utility functions and helpers
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv


def setup_logging(log_file: str = None, level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_file: Path to log file (optional)
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
    
    Returns:
        Logger instance
    """
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file) if log_file else logging.NullHandler()
        ]
    )
    
    return logging.getLogger(__name__)


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file
    
    Args:
        config_path: Path to config file
    
    Returns:
        Dictionary with configuration
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_env():
    """Load environment variables from .env file"""
    load_dotenv()


def ensure_dir(path: str) -> Path:
    """
    Create directory if it doesn't exist
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(data: Dict, filepath: str, indent: int = 2):
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    ensure_dir(Path(filepath).parent)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)


def load_json(filepath: str) -> Dict:
    """
    Load data from JSON file
    
    Args:
        filepath: Input file path
    
    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_api_key(service: str) -> str:
    """
    Get API key from environment variables
    
    Args:
        service: Service name (openai, anthropic, google, huggingface)
    
    Returns:
        API key
    
    Raises:
        ValueError if key not found
    """
    load_env()
    
    key_mapping = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "google": "GOOGLE_API_KEY",
        "huggingface": "HUGGINGFACE_TOKEN"
    }
    
    env_var = key_mapping.get(service.lower())
    if not env_var:
        raise ValueError(f"Unknown service: {service}")
    
    api_key = os.getenv(env_var)
    if not api_key:
        raise ValueError(f"{env_var} not found in environment variables. Please check your .env file.")
    
    return api_key


def format_code(code: str) -> str:
    """
    Clean and format generated code
    
    Args:
        code: Raw code string
    
    Returns:
        Cleaned code
    """
    # Remove markdown code blocks if present
    if code.startswith("```"):
        lines = code.split("\n")
        # Remove first line (```python or similar)
        lines = lines[1:]
        # Remove last line (```)
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        code = "\n".join(lines)
    
    return code.strip()


if __name__ == "__main__":
    # Test utilities
    logger = setup_logging(level="DEBUG")
    logger.info("Testing utilities...")
    
    # Test config loading
    try:
        config = load_config("config.yaml")
        logger.info(f"Loaded config with {len(config)} top-level keys")
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
    
    # Test directory creation
    test_dir = ensure_dir("./test_output")
    logger.info(f"Created directory: {test_dir}")
    
    # Test JSON save/load
    test_data = {"test": "data", "number": 42}
    save_json(test_data, "./test_output/test.json")
    loaded = load_json("./test_output/test.json")
    logger.info(f"JSON save/load test: {loaded}")
    
    # Test API key retrieval
    try:
        openai_key = get_api_key("openai")
        logger.info(f"OpenAI key found: {openai_key[:10]}...")
    except ValueError as e:
        logger.warning(f"API key test: {e}")
