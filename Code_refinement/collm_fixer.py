# =========================
# collm_fixer.py
# Code Fixer for Delphes Analysis
# =========================
"""
This script takes buggy Python analysis code and fixes it using an LLM.

Usage:
    python collm_fixer.py                                    # Use default files
    python collm_fixer.py --code buggy.py                    # Custom code file
    python collm_fixer.py --error error.txt --report report.txt  # Custom error files
    python collm_fixer.py --model <model_id>                 # Custom model

Input files:
    - original_code.py (or .txt): The buggy Python code to fix
    - terminal_error.txt: The actual error output from running the code (traceback)
    - error_report.txt: Categorized issues (Errors, Warnings, Style)
"""

# =========================
# Standard Libraries
# =========================
import sys
import subprocess
from pathlib import Path
import re
import logging
from typing import Optional, Dict, Any, Tuple

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
            logger.info(f" {module} already installed")
        except ImportError:
            logger.warning(f"Installing {pip_name}...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", pip_name
                ], stdout=subprocess.DEVNULL)
                logger.info(f" Successfully installed {pip_name}")
            except subprocess.CalledProcessError as e:
                logger.error(f" Failed to install {pip_name}: {e}")
                raise

ensure_packages()

# =========================
# Imports after install
# =========================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# =========================
# Configuration
# =========================
class Config:
    """Centralized configuration for code fixer."""
    # Prompt files
    SYSTEM_PROMPT_PATH = Path("system_prompt_fixer.txt")
    HUMAN_PROMPT_PATH = Path("human_prompt_fixer.txt")
    OUTPUT_CODE_PATH = Path("generated_analysis_fixed.py")
    
    # NEW: Separate input files for code, error, and report
    ORIGINAL_CODE_PATH = Path("original_code.py")      # The buggy Python code
    TERMINAL_ERROR_PATH = Path("terminal_error.txt")   # Traceback output
    ERROR_REPORT_PATH = Path("error_report.txt")       # Categorized issues
    
    # Legacy: Combined input file (still supported)
    USER_INPUT_PATH = Path("user_input_fixer.txt")
    
    # Model Configuration
    MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
    # Alternative models:
    # MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"
    # MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
    # MODEL_ID = "codellama/CodeLlama-13b-Instruct-hf"
    
    # Generation Parameters
    MAX_NEW_TOKENS = 4096
    TEMPERATURE = 0.5  # Low temperature for more deterministic fixes
    TOP_P = 0.95
    TOP_K = 50
    DO_SAMPLE = True

# =========================
# Utility Functions
# =========================
def load_text_file(file_path: Path, required: bool = True) -> str:
    """
    Load text from a file with error handling.
    
    Args:
        file_path: Path to the text file
        required: If True, raise error if file not found; if False, return empty string
        
    Returns:
        Content of the file as string
    """
    if not file_path.exists():
        if required:
            raise FileNotFoundError(f"File not found: {file_path}")
        else:
            logger.warning(f"⚠ Optional file not found: {file_path}")
            return ""
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    if not content and required:
        raise ValueError(f"File is empty: {file_path}")
    
    logger.info(f"✓ Loaded {file_path} ({len(content)} characters)")
    return content

def extract_python_code(text: str) -> str:
    """
    Extract Python code from LLM response.
    
    Handles multiple formats:
    - Code after "## Output Code:" marker
    - Code within ```python ``` fences
    - Raw code without markers
    
    Args:
        text: Raw LLM response
        
    Returns:
        Extracted Python code
    """
    # Remove output code marker if present
    if "## Output Code:" in text:
        text = text.split("## Output Code:", 1)[1]
    
    # Try to extract from markdown code blocks
    code_block_pattern = r"```(?:python)?\s*(.*?)```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    
    if matches:
        # Take the largest code block (likely the main code)
        code = max(matches, key=len).strip()
        logger.info(f"✓ Extracted code from markdown block ({len(code)} characters)")
        return code
    
    # If no code blocks, return cleaned text
    code = text.strip()
    logger.info(f"✓ Using raw text as code ({len(code)} characters)")
    return code

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
        logger.info(f"✓ Saved fixed code to {output_path}")
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
        gpu_info["memory_allocated"] = torch.cuda.memory_allocated(0) / 1e9
        gpu_info["memory_reserved"] = torch.cuda.memory_reserved(0) / 1e9
        
        logger.info(f"✓ GPU: {gpu_info['device_name']}")
        logger.info(f"  Memory: {gpu_info['memory_allocated']:.2f}GB allocated, "
                   f"{gpu_info['memory_reserved']:.2f}GB reserved")
    else:
        logger.warning("⚠ No GPU available, using CPU (this will be slow)")
    
    return gpu_info

# =========================
# Input Loading Functions
# =========================
def load_from_separate_files(
    code_path: Path,
    error_path: Path,
    report_path: Path
) -> Tuple[str, str, str]:
    """
    Load original code, terminal error, and error report from separate files.
    
    Args:
        code_path: Path to file containing the buggy Python code
        error_path: Path to file containing the terminal error (traceback)
        report_path: Path to file containing the error report
        
    Returns:
        Tuple of (original_code, terminal_error, error_report)
    """
    logger.info("Loading from separate files...")
    
    # Original code is required
    original_code = load_text_file(code_path, required=True)
    
    # Terminal error and error report - at least one required
    terminal_error = load_text_file(error_path, required=False)
    error_report = load_text_file(report_path, required=False)
    
    if not terminal_error and not error_report:
        raise ValueError(
            f"At least one of terminal error ({error_path}) or "
            f"error report ({report_path}) must exist and contain content."
        )
    
    logger.info(f"✓ Loaded from separate files:")
    logger.info(f"  - ORIGINAL_CODE: {len(original_code)} chars from {code_path}")
    logger.info(f"  - TERMINAL_ERROR: {len(terminal_error)} chars from {error_path}")
    logger.info(f"  - ERROR_REPORT: {len(error_report)} chars from {report_path}")
    
    return original_code, terminal_error, error_report

def parse_combined_input(path: Path) -> Tuple[str, str, str]:
    """
    Parse user input file for the code fixer (legacy combined format).
    
    Expected format:
    [ORIGINAL_CODE]
    ... python code ...
    
    [TERMINAL_ERROR]
    Traceback (most recent call last):
      ...
    ErrorType: message
    
    [ERROR_REPORT]
    ## Errors
    - error 1
    ## Warnings
    - warning 1
    ## Style
    - style issue 1
    
    Args:
        path: Path to the combined user input file
        
    Returns:
        Tuple of (original_code, terminal_error, error_report)
    """
    logger.info(f"Parsing combined input file: {path}")
    
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    def extract(tag):
        pattern = rf"\[{tag}\](.*?)(?=\n\[|\Z)"
        match = re.search(pattern, text, re.S)
        if match:
            return match.group(1).strip()
        else:
            return ""

    original_code = extract("ORIGINAL_CODE")
    terminal_error = extract("TERMINAL_ERROR")
    error_report = extract("ERROR_REPORT")

    # Validate required sections
    if not original_code:
        raise ValueError("Missing [ORIGINAL_CODE] section in input file")
    
    if not terminal_error and not error_report:
        raise ValueError("Missing both [TERMINAL_ERROR] and [ERROR_REPORT] sections. At least one is required.")

    logger.info(f"✓ Parsed combined fixer input:")
    logger.info(f"  - ORIGINAL_CODE: {len(original_code)} chars")
    logger.info(f"  - TERMINAL_ERROR: {len(terminal_error)} chars")
    logger.info(f"  - ERROR_REPORT: {len(error_report)} chars")

    return original_code, terminal_error, error_report

def load_fixer_input(
    use_separate_files: bool,
    code_path: Path,
    error_path: Path,
    report_path: Path,
    combined_path: Path
) -> Tuple[str, str, str]:
    """
    Load fixer input from either separate files or combined file.
    
    Args:
        use_separate_files: If True, load from separate files; if False, use combined file
        code_path: Path to original code file
        error_path: Path to terminal error file
        report_path: Path to error report file
        combined_path: Path to combined input file (legacy format)
        
    Returns:
        Tuple of (original_code, terminal_error, error_report)
    """
    if use_separate_files:
        return load_from_separate_files(code_path, error_path, report_path)
    else:
        return parse_combined_input(combined_path)

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
        logger.info("✓ Tokenizer loaded")
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
        )
        logger.info("✓ Model loaded")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
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
    logger.info("✓ Pipeline created")
    return pipe

# =========================
# Main Execution
# =========================
def main(use_separate_files: bool = True):
    """
    Main execution function for code fixer.
    
    Args:
        use_separate_files: If True, load from separate files; if False, use combined file
    """
    try:
        logger.info("=" * 60)
        logger.info("Delphes Analysis Code Fixer")
        logger.info("=" * 60)
        
        # Check GPU
        gpu_info = check_gpu_availability()
        
        # Load prompts
        logger.info("\n[1/5] Loading prompts and input...")
        system_prompt = load_text_file(Config.SYSTEM_PROMPT_PATH)
        human_prompt = load_text_file(Config.HUMAN_PROMPT_PATH)
        
        # Load fixer input (code + errors)
        original_code, terminal_error, error_report = load_fixer_input(
            use_separate_files=use_separate_files,
            code_path=Config.ORIGINAL_CODE_PATH,
            error_path=Config.TERMINAL_ERROR_PATH,
            report_path=Config.ERROR_REPORT_PATH,
            combined_path=Config.USER_INPUT_PATH
        )
        
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
        
        # Format with fixer-specific variables
        messages = prompt_template.format_messages(
            original_code=original_code,
            terminal_error=terminal_error,
            error_report=error_report,
        )
        
        # Log prompt statistics
        total_chars = sum(len(str(m.content)) for m in messages)
        logger.info(f"  Total prompt length: {total_chars} characters")
        logger.info(f"  Original code length: {len(original_code)} characters")
        
        # Generate fixed code
        logger.info("\n[5/5] Fixing code...")
        logger.info("  (This may take a few minutes...)")
        
        response = llm.invoke(messages)
        text = response.content if hasattr(response, "content") else str(response)
        
        logger.info(f"✓ Generation complete ({len(text)} characters)")
        
        # Extract and save fixed code
        fixed_code = extract_python_code(text)
        save_code(fixed_code, Config.OUTPUT_CODE_PATH)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("FIX SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Model: {Config.MODEL_ID}")
        logger.info(f"GPU Used: {'Yes' if gpu_info['available'] else 'No'}")
        logger.info(f"Original Code Length: {len(original_code)} characters")
        logger.info(f"Fixed Code Length: {len(fixed_code)} characters")
        logger.info(f"Output File: {Config.OUTPUT_CODE_PATH}")
        logger.info("=" * 60)
        
        # Show terminal error that was fixed
        if terminal_error:
            logger.info("\nTERMINAL ERROR FIXED:")
            logger.info("-" * 60)
            # Show just the error type and message (last line of traceback)
            error_lines = terminal_error.strip().split('\n')
            if error_lines:
                print(error_lines[-1])
            logger.info("-" * 60)
        
        # Show code preview
        logger.info("\nFIXED CODE PREVIEW (first 500 characters):")
        logger.info("-" * 60)
        print(fixed_code[:500] + ("..." if len(fixed_code) > 500 else ""))
        logger.info("-" * 60)
        
        return fixed_code
        
    except FileNotFoundError as e:
        logger.error(f"✗ File error: {e}")
        logger.error("  Please ensure all required files exist:")
        logger.error(f"    - {Config.SYSTEM_PROMPT_PATH}")
        logger.error(f"    - {Config.HUMAN_PROMPT_PATH}")
        if use_separate_files:
            logger.error(f"    - {Config.ORIGINAL_CODE_PATH}")
            logger.error(f"    - {Config.TERMINAL_ERROR_PATH} (optional if report exists)")
            logger.error(f"    - {Config.ERROR_REPORT_PATH} (optional if error exists)")
        else:
            logger.error(f"    - {Config.USER_INPUT_PATH}")
        sys.exit(1)
        
    except ValueError as e:
        logger.error(f"✗ Input error: {e}")
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
        description="Fix Delphes analysis code using LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Use separate files (default)
  python collm_fixer.py --code buggy_code.py --error traceback.txt --report issues.txt
  
  # Use combined file (legacy)
  python collm_fixer.py --combined user_input_fixer.txt
  
  # Specify all options
  python collm_fixer.py --code analysis.py --error error.txt --report report.txt \\
                        --output fixed_analysis.py --model Qwen/Qwen2.5-Coder-14B-Instruct
        """
    )
    
    # Input mode group
    input_group = parser.add_argument_group('Input Files (separate mode)')
    input_group.add_argument(
        "--code", "-c",
        type=str,
        default=str(Config.ORIGINAL_CODE_PATH),
        help=f"Path to original buggy code file (default: {Config.ORIGINAL_CODE_PATH})"
    )
    input_group.add_argument(
        "--error", "-e",
        type=str,
        default=str(Config.TERMINAL_ERROR_PATH),
        help=f"Path to terminal error file (default: {Config.TERMINAL_ERROR_PATH})"
    )
    input_group.add_argument(
        "--report", "-r",
        type=str,
        default=str(Config.ERROR_REPORT_PATH),
        help=f"Path to error report file (default: {Config.ERROR_REPORT_PATH})"
    )
    
    # Legacy combined mode
    legacy_group = parser.add_argument_group('Input File (combined mode)')
    legacy_group.add_argument(
        "--combined", "-C",
        type=str,
        default=None,
        help="Path to combined input file with [ORIGINAL_CODE], [TERMINAL_ERROR], [ERROR_REPORT] sections"
    )
    
    # Output and model options
    output_group = parser.add_argument_group('Output and Model Options')
    output_group.add_argument(
        "--output", "-o",
        type=str,
        default=str(Config.OUTPUT_CODE_PATH),
        help=f"Path to output fixed code (default: {Config.OUTPUT_CODE_PATH})"
    )
    output_group.add_argument(
        "--model", "-m",
        type=str,
        default=Config.MODEL_ID,
        help=f"HuggingFace model ID to use (default: {Config.MODEL_ID})"
    )
    output_group.add_argument(
        "--system-prompt", "-s",
        type=str,
        default=str(Config.SYSTEM_PROMPT_PATH),
        help=f"Path to system prompt file (default: {Config.SYSTEM_PROMPT_PATH})"
    )
    output_group.add_argument(
        "--human-prompt", "-p",
        type=str,
        default=str(Config.HUMAN_PROMPT_PATH),
        help=f"Path to human prompt file (default: {Config.HUMAN_PROMPT_PATH})"
    )
    
    args = parser.parse_args()
    
    # Determine mode: separate files or combined
    use_separate_files = args.combined is None
    
    # Override config with command line arguments
    if use_separate_files:
        Config.ORIGINAL_CODE_PATH = Path(args.code)
        Config.TERMINAL_ERROR_PATH = Path(args.error)
        Config.ERROR_REPORT_PATH = Path(args.report)
    else:
        Config.USER_INPUT_PATH = Path(args.combined)
    
    Config.OUTPUT_CODE_PATH = Path(args.output)
    Config.MODEL_ID = args.model
    Config.SYSTEM_PROMPT_PATH = Path(args.system_prompt)
    Config.HUMAN_PROMPT_PATH = Path(args.human_prompt)
    
    # Log which mode is being used
    if use_separate_files:
        logger.info("Using SEPARATE FILES mode:")
        logger.info(f"  Code:   {Config.ORIGINAL_CODE_PATH}")
        logger.info(f"  Error:  {Config.TERMINAL_ERROR_PATH}")
        logger.info(f"  Report: {Config.ERROR_REPORT_PATH}")
    else:
        logger.info(f"Using COMBINED FILE mode: {Config.USER_INPUT_PATH}")
    
    # Run fixer
    main(use_separate_files=use_separate_files)
