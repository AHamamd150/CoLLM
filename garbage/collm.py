# =========================
# Standard Libraries
# =========================
import sys
import subprocess
from pathlib import Path
import re
import logging
from typing import Optional, Dict, Any

# =========================
# Setup Logging
# =========================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =========================
# Minimal Dependency Loader
# =========================
REQUIRED_PACKAGES = {
    "tqdm": "tqdm",
    "matplotlib": "matplotlib",
    "langchain": "langchain",
    "transformers": "transformers",
    "langchain_huggingface": "langchain-huggingface",
    "huggingface_hub": "huggingface_hub",
    "accelerate": "accelerate",
    "torch": "torch",
}

def ensure_packages():
    """Install required packages if not already installed."""
    for module, pip_name in REQUIRED_PACKAGES.items():
        try:
            __import__(module)
            logger.info(f"✓ {module} already installed")
        except ImportError:
            logger.warning(f"Installing {pip_name}...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", pip_name
                ], stdout=subprocess.DEVNULL)
                logger.info(f"✓ Successfully installed {pip_name}")
            except subprocess.CalledProcessError as e:
                logger.error(f"✗ Failed to install {pip_name}: {e}")
                raise

ensure_packages()

# =========================
# Imports (after install)
# =========================
import numpy as np
import streamlit as st
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage

# =========================
# Configuration
# =========================
class Config:
    """Centralized configuration."""
    # FIXED: Changed from system_prompt_f2.txt to system_prompt.txt
    SYSTEM_PROMPT_PATH = Path("system_prompt_generic.txt")
    HUMAN_PROMPT_PATH = Path("human_prompt.txt")
    USER_INPUT_PATH = Path("user_input.txt")
    OUTPUT_CODE_PATH = Path("generated_analysis.py")
    
    # Model Configuration
    MODEL_ID = "bigcode/starcoder2-7b"
    # Alternative models (uncomment to use):
    # MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"
    # MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Generation Parameters
    MAX_NEW_TOKENS = 2048
    TEMPERATURE = 0.2
    TOP_P = 0.95
    TOP_K = 50
    DO_SAMPLE = True

# =========================
# Utility Functions
# =========================
def load_text_file(file_path: Path) -> str:
    """
    Load text from a file with error handling.
    
    Args:
        file_path: Path to the text file
        
    Returns:
        Content of the file as string
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file is empty
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    if not content:
        raise ValueError(f"File is empty: {file_path}")
    
    logger.info(f" Loaded {file_path} ({len(content)} characters)")
    return content


def extract_python_code(text: str) -> str:
    """
    Extract Python code from LLM response.
    
    Handles multiple formats:
    - Code within ```python ``` fences
    - Code within ``` ``` fences
    - Raw code starting with 'import sys'
    - Code after common markers
    
    Args:
        text: Raw LLM response
        
    Returns:
        Extracted Python code
    """
    if not text or not text.strip():
        logger.warning("Empty response received")
        return ""
    
    text = text.strip()
    
    # Remove common markers
    markers = ["## Output Code:", "## Code:", "```python", "Here's the code:", "Here is the code:"]
    for marker in markers:
        if marker in text:
            text = text.split(marker, 1)[1]
            break
    
    # Pattern 1: Extract from ```python ... ``` blocks
    python_block_pattern = r"```python\s*\n?(.*?)```"
    python_matches = re.findall(python_block_pattern, text, re.DOTALL)
    if python_matches:
        code = max(python_matches, key=len).strip()
        logger.info(f"✓ Extracted code from ```python block ({len(code)} characters)")
        return code
    
    # Pattern 2: Extract from ``` ... ``` blocks
    generic_block_pattern = r"```\s*\n?(.*?)```"
    generic_matches = re.findall(generic_block_pattern, text, re.DOTALL)
    if generic_matches:
        # Filter to blocks that look like Python
        python_blocks = [m for m in generic_matches if _is_python_code(m)]
        if python_blocks:
            code = max(python_blocks, key=len).strip()
            logger.info(f"✓ Extracted code from ``` block ({len(code)} characters)")
            return code
    
    # Pattern 3: Raw code starting with import sys
    if text.lstrip().startswith("import sys"):
        code = _extract_raw_python(text)
        logger.info(f"✓ Extracted raw Python code ({len(code)} characters)")
        return code
    
    # Pattern 4: Find where Python code starts
    import_match = re.search(r'^import sys', text, re.MULTILINE)
    if import_match:
        code = _extract_raw_python(text[import_match.start():])
        logger.info(f"✓ Extracted code starting from 'import sys' ({len(code)} characters)")
        return code
    
    # Fallback: return cleaned text
    logger.warning("No code block found, returning raw text")
    return text.strip()



def save_code(code: str, output_path: Path) -> None:
    """
    Save generated code to file.
    
    Args:
        code: Python code to save
        output_path: Path to output file
    """
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        logger.info(f"✓ Saved generated code to {output_path}")
    except IOError as e:
        logger.error(f"✗ Failed to save code: {e}")
        raise

def check_gpu_availability() -> Dict[str, Any]:
    """
    Check GPU availability and memory.
    
    Returns:
        Dictionary with GPU information
    """
    gpu_info = {
        "available": torch.cuda.is_available(),
        "device_count": 0,
        "device_name": None,
        "memory_allocated": 0,
        "memory_reserved": 0,
    }
    
    if gpu_info["available"]:
        gpu_info["device_count"] = torch.cuda.device_count()
        gpu_info["device_name"] = torch.cuda.get_device_name(0)
        gpu_info["memory_allocated"] = torch.cuda.memory_allocated(0) / 1e9  # GB
        gpu_info["memory_reserved"] = torch.cuda.memory_reserved(0) / 1e9  # GB
        
        logger.info(f"✓ GPU: {gpu_info['device_name']}")
        logger.info(f"  Memory: {gpu_info['memory_allocated']:.2f}GB allocated, "
                   f"{gpu_info['memory_reserved']:.2f}GB reserved")
    else:
        logger.warning("⚠ No GPU available, using CPU (this will be slow)")
    
    return gpu_info

# =========================
# Model Loading
# =========================
def load_model_and_tokenizer(model_id: str):
    """
    Load model and tokenizer with error handling.
    
    Args:
        model_id: HuggingFace model identifier
        
    Returns:
        Tuple of (model, tokenizer)
    """
    logger.info(f"Loading model: {model_id}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logger.info(" Tokenizer loaded")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )
        logger.info(" Model loaded")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f" Failed to load model: {e}")
        raise

def create_pipeline(model, tokenizer, config: Config):
    """
    Create HuggingFace pipeline with configuration.
    
    Args:
        model: Loaded model
        tokenizer: Loaded tokenizer
        config: Configuration object
        
    Returns:
        HuggingFace pipeline
    """
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=config.MAX_NEW_TOKENS,
        do_sample=config.DO_SAMPLE,
        temperature=config.TEMPERATURE,
        top_p=config.TOP_P,
        top_k=config.TOP_K,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    logger.info(" Pipeline created")
    return pipe

def parse_user_input(path: Path) -> tuple:
    """
    Parse user input file and extract the three main sections.
    
    Args:
        path: Path to user_input.txt
        
    Returns:
        Tuple of (selection_cuts, plots_for_validation, output_structure)
    """
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    def extract(tag):
        pattern = rf"\[{tag}\](.*?)(?=\n\[|\Z)"
        match = re.search(pattern, text, re.S)
        if match:
            return match.group(1).strip()
        else:
            return ""

    selection_cuts = extract("SELECTION_CUTS")
    plots_for_validation = extract("PLOTS_FOR_VALIDATION")
    output_structure = extract("OUTPUT_STRUCTURE")

    logger.info(f"✓ Parsed user input sections:")
    logger.info(f"  - SELECTION_CUTS: {len(selection_cuts)} chars")
    logger.info(f"  - PLOTS_FOR_VALIDATION: {len(plots_for_validation)} chars")
    logger.info(f"  - OUTPUT_STRUCTURE: {len(output_structure)} chars")

    return selection_cuts, plots_for_validation, output_structure

# =========================
# Main Execution
# =========================
def main():
    """Main execution function."""
    try:
        logger.info("=" * 60)
        logger.info("Delphes Analysis Code Generator")
        logger.info("=" * 60)
        
        # Check GPU
        gpu_info = check_gpu_availability()
        
        # Load prompts
        logger.info("\n[1/5] Loading prompts...")
        system_prompt = load_text_file(Config.SYSTEM_PROMPT_PATH)
        human_prompt = load_text_file(Config.HUMAN_PROMPT_PATH)
        
        # Load user sections
        selection_cuts, plots_for_validation, output_structure = parse_user_input(Config.USER_INPUT_PATH)

        # Load model
        logger.info("\n[2/5] Loading model...")
        model, tokenizer = load_model_and_tokenizer(Config.MODEL_ID)
        
        # Create pipeline
        logger.info("\n[3/5] Creating pipeline...")
        pipe = create_pipeline(model, tokenizer, Config)
        llm = HuggingFacePipeline(pipeline=pipe)
        
        # Build prompt
        logger.info("\n[4/5] Building prompt...")
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
        
        # FIXED: Format with the correct variable names matching human_prompt.txt
        messages = prompt_template.format_messages(
            selection_cuts=selection_cuts,
            plots_for_validation=plots_for_validation,
            output_structure=output_structure,
        )
        
        # Log prompt statistics
        total_chars = sum(len(str(m.content)) for m in messages)
        logger.info(f"  Total prompt length: {total_chars} characters")
        
        # Generate code
        logger.info("\n[5/5] Generating code...")
        logger.info("  (This may take a few minutes...)")
        
        response = llm.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        #print(response)
        logger.info(f"✓ Generation complete ({len(text)} characters)")
        
        # Extract and save code
        code = extract_python_code(text)
        save_code(code, Config.OUTPUT_CODE_PATH)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("GENERATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Model: {Config.MODEL_ID}")
        logger.info(f"GPU Used: {'Yes' if gpu_info['available'] else 'No'}")
        logger.info(f"Generated Code Length: {len(code)} characters")
        logger.info(f"Output File: {Config.OUTPUT_CODE_PATH}")
        logger.info("=" * 60)
        
        # Show code preview
        logger.info("\nCODE PREVIEW (first 500 characters):")
        logger.info("-" * 60)
        print(code[:500] + ("..." if len(code) > 500 else ""))
        logger.info("-" * 60)
        
        return code
        
    except FileNotFoundError as e:
        logger.error(f"✗ File error: {e}")
        logger.error("  Please ensure all required prompt files exist:")
        logger.error(f"    - {Config.SYSTEM_PROMPT_PATH}")
        logger.error(f"    - {Config.HUMAN_PROMPT_PATH}")
        logger.error(f"    - {Config.USER_INPUT_PATH}")
        sys.exit(1)
        
    except Exception as e:
        logger.error(f"✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate Delphes analysis code using LLM"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default=Config.MODEL_ID,
        help="HuggingFace model ID to use"
    )
    
    args = parser.parse_args()
    
    # Override model if specified
    if args.model:
        Config.MODEL_ID = args.model
    
    # Run appropriate mode
    main()
