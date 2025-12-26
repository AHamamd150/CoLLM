# =========================
# Standard Libraries
# =========================
import sys
import subprocess
from pathlib import Path
import re
import logging
from typing import Optional, Dict, Any, List, Tuple

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
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# =========================
# Configuration
# =========================
class Config:
    """Centralized configuration."""
    SYSTEM_PROMPT_PATH = Path("system_prompt_v3.txt")
    USER_INPUT_PATH = Path("user_input.txt")
    OUTPUT_CODE_PATH = Path("generated_analysis.py")
    
    # Model Configuration
    MODEL_ID = "bigcode/starcoder2-7b"
    # Alternative models:
    # MODEL_ID = "Qwen/Qwen2.5-Coder-14B-Instruct"
    # MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
    
    # Generation Parameters
    MAX_NEW_TOKENS = 4096
    TEMPERATURE = 0.2
    TOP_P = 0.95
    TOP_K = 50
    DO_SAMPLE = True
    
    # Retry Configuration
    MAX_ATTEMPTS = 3

# =========================
# Code Extraction & Validation
# =========================
def extract_python_code(text: str) -> str:
    """
    Extract Python code from LLM response.
    
    Handles multiple formats and rejects placeholder garbage.
    
    Args:
        text: Raw LLM response
        
    Returns:
        Extracted Python code, or empty string if extraction fails
    """
    if not text or not text.strip():
        logger.warning("Empty response received")
        return ""
    
    text = text.strip()
    
    # === EARLY REJECTION: Detect placeholder-only responses ===
    if _is_placeholder_garbage(text):
        logger.error("✗ Response contains only placeholders - no valid code")
        return ""
    
    # === PATTERN 1: Extract from ```python ... ``` blocks ===
    python_block_pattern = r"```python\s*\n?(.*?)```"
    python_matches = re.findall(python_block_pattern, text, re.DOTALL)
    if python_matches:
        valid_blocks = [m for m in python_matches if not _is_placeholder_garbage(m)]
        if valid_blocks:
            code = max(valid_blocks, key=len).strip()
            if _is_valid_python_start(code):
                logger.info(f"✓ Extracted code from ```python block ({len(code)} chars)")
                return code
    
    # === PATTERN 2: Extract from ``` ... ``` blocks ===
    generic_block_pattern = r"```\s*\n?(.*?)```"
    generic_matches = re.findall(generic_block_pattern, text, re.DOTALL)
    if generic_matches:
        valid_blocks = [m for m in generic_matches 
                       if _is_valid_python_start(m) and not _is_placeholder_garbage(m)]
        if valid_blocks:
            code = max(valid_blocks, key=len).strip()
            logger.info(f"✓ Extracted code from ``` block ({len(code)} chars)")
            return code
    
    # === PATTERN 3: Raw code starting with 'import sys' ===
    import_match = re.search(r'^import\s+sys\b', text, re.MULTILINE)
    if import_match:
        code = _extract_raw_python(text[import_match.start():])
        if code and not _is_placeholder_garbage(code):
            logger.info(f"✓ Extracted raw Python code ({len(code)} chars)")
            return code
    
    # === FALLBACK: No valid code found ===
    logger.error("✗ No valid Python code found in response")
    return ""


def _is_placeholder_garbage(text: str) -> bool:
    """
    Detect if text is placeholder garbage instead of real code.
    """
    if not text:
        return True
    
    # Known garbage placeholders
    garbage_indicators = [
        '{output_code}', '{example_input}', '{example_output}', '{example_plot}',
        '{example_statistics}', '{example_cutflow}', '{example_code}',
        '{branch}', '{count}', '{stage}', '{particle}', '{collection}',
        '{cut}', '{plot_name}', '{output_structure}', '{candidate}',
        '{object_cut_code}', '{plotting_code}', '{variable}', '{name}',
        '{selection_cuts}', '{plots_for_validation}', '{input_file}',
    ]
    
    text_lower = text.lower()
    for indicator in garbage_indicators:
        if indicator in text_lower:
            return True
    
    # Check for repetitive template patterns
    if text.count('## Example') > 3:
        return True
    
    # Count placeholder patterns
    placeholder_pattern = r'\{[a-z_]+\}'
    placeholders = re.findall(placeholder_pattern, text, re.IGNORECASE)
    
    if len(placeholders) > 5:
        has_import = 'import sys' in text or 'import uproot' in text
        has_def = 'def main(' in text or 'def find_branch(' in text
        if not has_import and not has_def:
            return True
    
    return False


def _is_valid_python_start(text: str) -> bool:
    """Check if text starts like valid Python code."""
    text = text.strip()
    valid_starts = [
        r'^import\s+sys\b',
        r'^import\s+\w+',
        r'^from\s+\w+\s+import',
        r'^#.*coding',
        r'^#!/usr/bin',
    ]
    for pattern in valid_starts:
        if re.match(pattern, text, re.MULTILINE):
            return True
    return False


def _extract_raw_python(text: str) -> str:
    """Extract Python code from raw text, stopping at non-code sections."""
    lines = text.split('\n')
    code_lines = []
    
    for line in lines:
        stripped = line.strip()
        
        # Stop at markdown headers or obvious non-code
        if stripped.startswith('## '):
            break
        if stripped.startswith('# Example') or stripped.startswith('## Example'):
            break
        if _is_end_marker(stripped):
            break
            
        code_lines.append(line)
    
    return '\n'.join(code_lines).strip()


def _is_end_marker(line: str) -> bool:
    """Check if line marks end of code section."""
    end_patterns = [
        r'^This\s+(code|script|will|should)',
        r'^The\s+(above|code|script)',
        r'^Note[:\s]',
        r'^Output[:\s]',
        r'^Explanation[:\s]',
        r'^Example[:\s]',
        r'^---+$',
        r'^===+$',
        r'^```',
    ]
    for pattern in end_patterns:
        if re.match(pattern, line, re.IGNORECASE):
            return True
    return False


def validate_generated_code(code: str) -> Tuple[bool, List[str]]:
    """
    Validate that generated code meets all requirements.
    
    Args:
        code: Extracted Python code
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # === EMPTY CODE CHECK ===
    if not code or not code.strip():
        return False, ["No code provided"]
    
    # === PLACEHOLDER GARBAGE CHECK ===
    if _is_placeholder_garbage(code):
        return False, ["Code contains template placeholders instead of real Python code"]
    
    # === REQUIRED IMPORTS ===
    required_imports = {
        'sys': r'import\s+sys\b',
        'uproot': r'import\s+uproot\b',
        'awkward': r'import\s+awkward\s+as\s+ak\b',
        'vector': r'import\s+vector\b',
        'numpy': r'import\s+numpy\s+as\s+np\b',
        'matplotlib': r'import\s+matplotlib\.pyplot\s+as\s+plt\b',
    }
    
    for name, pattern in required_imports.items():
        if not re.search(pattern, code):
            errors.append(f"Missing import: {name}")
    
    # === REQUIRED COMPONENTS ===
    required_components = {
        'vector.register_awkward()': r'vector\.register_awkward\s*\(\s*\)',
        'find_branch function': r'def\s+find_branch\s*\(',
        'main function': r'def\s+main\s*\(',
        'sys.argv': r'sys\.argv',
        'tree.arrays()': r'tree\.arrays\s*\(',
        'library="ak"': r'library\s*=\s*["\']ak["\']',
        'if __name__ == "__main__"': r'if\s+__name__\s*==\s*["\']__main__["\']',
        'plt.savefig()': r'plt\.savefig\s*\(',
        'dpi=150': r'dpi\s*=\s*150',
        'plt.close()': r'plt\.close\s*\(\s*\)',
    }
    
    for name, pattern in required_components.items():
        if not re.search(pattern, code):
            errors.append(f"Missing: {name}")
    
    # === FORBIDDEN PLACEHOLDERS ===
    forbidden = [
        r'\{output_code\}',
        r'\{example_\w+\}',
        r'\{branch\}',
        r'\{count\}',
        r'\{stage\}',
        r'\{particle\}',
        r'\{collection\}',
        r'\{cut\}',
        r'\{plot_name\}',
        r'\{variable\}',
        r'\{name\}',
    ]
    
    for pattern in forbidden:
        if re.search(pattern, code, re.IGNORECASE):
            errors.append(f"Forbidden placeholder pattern: {pattern}")
    
    is_valid = len(errors) == 0
    return is_valid, errors


def check_python_syntax(code: str) -> Tuple[bool, Optional[str]]:
    """Check if code has valid Python syntax."""
    if not code:
        return False, "Empty code"
    try:
        compile(code, '<generated>', 'exec')
        return True, None
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def extract_and_validate(text: str) -> Tuple[str, bool, List[str]]:
    """
    Main function: Extract Python code from LLM response and validate it.
    
    Args:
        text: Raw LLM response
        
    Returns:
        Tuple of (extracted_code, is_valid, list_of_errors)
    """
    code = extract_python_code(text)
    
    if not code:
        return "", False, ["Failed to extract valid Python code from response"]
    
    all_errors = []
    
    # Check syntax
    syntax_ok, syntax_error = check_python_syntax(code)
    if not syntax_ok:
        all_errors.append(f"Syntax error: {syntax_error}")
    
    # Validate components
    valid, validation_errors = validate_generated_code(code)
    all_errors.extend(validation_errors)
    
    is_valid = len(all_errors) == 0
    
    if is_valid:
        logger.info("✓ Code extraction and validation passed")
    else:
        logger.warning(f"✗ Validation failed with {len(all_errors)} errors:")
        for err in all_errors:
            logger.warning(f"  - {err}")
    
    return code, is_valid, all_errors


def format_errors_for_retry(errors: List[str]) -> str:
    """Format validation errors as feedback for retry attempt."""
    error_list = "\n".join(f"  • {err}" for err in errors)
    return f"""
═══════════════════════════════════════════════════════════════════════════════
PREVIOUS ATTEMPT FAILED - YOU MUST FIX THESE ERRORS
═══════════════════════════════════════════════════════════════════════════════

{error_list}

CRITICAL REQUIREMENTS:
• Output ONLY executable Python code - NO placeholders like {{variable}}
• First line must be: import sys
• Must include ALL 6 imports: sys, uproot, awkward as ak, vector, numpy as np, matplotlib.pyplot as plt
• Must call: vector.register_awkward()
• Must define: find_branch() and main() functions
• Must use: tree.arrays([...], library="ak")
• Must save plots: plt.savefig("name.png", dpi=150) followed by plt.close()
• Must end with: if __name__ == "__main__": main()

DO NOT output template text or placeholders. Output ONLY real, runnable Python code.
═══════════════════════════════════════════════════════════════════════════════
"""

# =========================
# Utility Functions
# =========================
def load_text_file(file_path: Path) -> str:
    """Load text from a file with error handling."""
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    
    if not content:
        raise ValueError(f"File is empty: {file_path}")
    
    logger.info(f"✓ Loaded {file_path} ({len(content)} characters)")
    return content


def save_code(code: str, output_path: Path) -> None:
    """Save generated code to file."""
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(code)
        logger.info(f"✓ Saved generated code to {output_path}")
    except IOError as e:
        logger.error(f"✗ Failed to save code: {e}")
        raise


def check_gpu_availability() -> Dict[str, Any]:
    """Check GPU availability and memory."""
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


def parse_user_input(path: Path) -> Tuple[str, str, str]:
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
        return match.group(1).strip() if match else ""

    selection_cuts = extract("SELECTION_CUTS")
    plots_for_validation = extract("PLOTS_FOR_VALIDATION")
    output_structure = extract("OUTPUT_STRUCTURE")

    logger.info("✓ Parsed user input sections:")
    logger.info(f"  - SELECTION_CUTS: {len(selection_cuts)} chars")
    logger.info(f"  - PLOTS_FOR_VALIDATION: {len(plots_for_validation)} chars")
    logger.info(f"  - OUTPUT_STRUCTURE: {len(output_structure)} chars")

    return selection_cuts, plots_for_validation, output_structure


def build_user_prompt(selection_cuts: str, plots_for_validation: str, 
                      output_structure: str, attempt: int = 1, 
                      previous_errors: List[str] = None) -> str:
    """
    Build user prompt with analysis request and output contract.
    
    Args:
        selection_cuts: Content from [SELECTION_CUTS]
        plots_for_validation: Content from [PLOTS_FOR_VALIDATION]
        output_structure: Content from [OUTPUT_STRUCTURE]
        attempt: Current attempt number
        previous_errors: Errors from previous attempt (if any)
        
    Returns:
        Complete user prompt string
    """
    # Base prompt with user's analysis specification
    base_prompt = f"""Generate a complete Python analysis script for the following requirements:

[SELECTION_CUTS]
{selection_cuts}

[PLOTS_FOR_VALIDATION]
{plots_for_validation}

[OUTPUT_STRUCTURE]
{output_structure}
"""
    
    if attempt == 1:
        # First attempt: standard output contract
        output_contract = """
═══════════════════════════════════════════════════════════════════════════════
OUTPUT CONTRACT
═══════════════════════════════════════════════════════════════════════════════

You MUST output ONLY executable Python code. Your response must:

• Start with: import sys
• Contain ONLY Python code - no markdown, no explanations
• Use real variable names like n_total, muon_pt, dimuon_mass
• NEVER contain placeholders like {branch}, {count}, {variable}

DO NOT output anything except the Python code.
═══════════════════════════════════════════════════════════════════════════════
"""
        return base_prompt + output_contract
    
    else:
        # Retry attempt: include error feedback
        error_feedback = format_errors_for_retry(previous_errors or [])
        return base_prompt + error_feedback

# =========================
# Model Loading
# =========================
def load_model_and_tokenizer(model_id: str):
    """Load model and tokenizer with error handling."""
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
    """Create HuggingFace pipeline with configuration."""
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
def main():
    """Main execution function with retry logic."""
    try:
        logger.info("=" * 60)
        logger.info("Delphes Analysis Code Generator")
        logger.info("=" * 60)
        
        # Check GPU
        gpu_info = check_gpu_availability()
        
        # Load prompts
        logger.info("\n[1/5] Loading prompts...")
        system_prompt = load_text_file(Config.SYSTEM_PROMPT_PATH)
        
        # Load user sections
        selection_cuts, plots_for_validation, output_structure = parse_user_input(Config.USER_INPUT_PATH)

        # Load model
        logger.info("\n[2/5] Loading model...")
        model, tokenizer = load_model_and_tokenizer(Config.MODEL_ID)
        
        # Create pipeline
        logger.info("\n[3/5] Creating pipeline...")
        pipe = create_pipeline(model, tokenizer, Config)
        llm = HuggingFacePipeline(pipeline=pipe)
        
        # === GENERATION WITH RETRY LOGIC ===
        logger.info("\n[4/5] Generating code with validation...")
        
        code = ""
        is_valid = False
        all_errors = []
        
        for attempt in range(1, Config.MAX_ATTEMPTS + 1):
            logger.info(f"\n{'='*60}")
            logger.info(f"Generation Attempt {attempt}/{Config.MAX_ATTEMPTS}")
            logger.info('='*60)
            
            # Build prompt
            user_prompt = build_user_prompt(
                selection_cuts=selection_cuts,
                plots_for_validation=plots_for_validation,
                output_structure=output_structure,
                attempt=attempt,
                previous_errors=all_errors if attempt > 1 else None
            )
            
            # Create messages
            prompt_template = ChatPromptTemplate.from_messages([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
            
            messages = prompt_template.format_messages()
            
            # Log prompt statistics
            total_chars = sum(len(str(m.content)) for m in messages)
            logger.info(f"  Total prompt length: {total_chars} characters")
            logger.info("  Generating... (this may take a few minutes)")
            
            # Generate
            response = llm.invoke(messages)
            text = response.content if hasattr(response, "content") else str(response)
            logger.info(f"  Response length: {len(text)} characters")
            
            # Extract and validate
            code, is_valid, all_errors = extract_and_validate(text)
            
            if is_valid:
                logger.info(f"\n✓ SUCCESS on attempt {attempt}")
                break
            else:
                logger.warning(f"\n✗ Attempt {attempt} failed with {len(all_errors)} errors")
                if attempt < Config.MAX_ATTEMPTS:
                    logger.info("  Retrying with error feedback...")
        
        # === SAVE RESULTS ===
        logger.info("\n[5/5] Saving results...")
        
        if code:
            save_code(code, Config.OUTPUT_CODE_PATH)
        else:
            logger.error("✗ No valid code generated after all attempts")
            # Save raw response for debugging
            debug_path = Path("debug_raw_response.txt")
            with open(debug_path, "w") as f:
                f.write(text)
            logger.info(f"  Raw response saved to {debug_path} for debugging")
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("GENERATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Model: {Config.MODEL_ID}")
        logger.info(f"GPU Used: {'Yes' if gpu_info['available'] else 'No'}")
        logger.info(f"Attempts: {attempt}/{Config.MAX_ATTEMPTS}")
        logger.info(f"Success: {'Yes' if is_valid else 'No'}")
        if code:
            logger.info(f"Generated Code Length: {len(code)} characters")
            logger.info(f"Output File: {Config.OUTPUT_CODE_PATH}")
        if all_errors and not is_valid:
            logger.info(f"Final Errors: {all_errors}")
        logger.info("=" * 60)
        
        # Show code preview
        if code:
            logger.info("\nCODE PREVIEW (first 500 characters):")
            logger.info("-" * 60)
            print(code[:500] + ("..." if len(code) > 500 else ""))
            logger.info("-" * 60)
        
        return code if is_valid else None
        
    except FileNotFoundError as e:
        logger.error(f"✗ File error: {e}")
        logger.error("  Please ensure all required prompt files exist:")
        logger.error(f"    - {Config.SYSTEM_PROMPT_PATH}")
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
        "--model", "-m",
        type=str,
        default=Config.MODEL_ID,
        help="HuggingFace model ID to use"
    )
    parser.add_argument(
        "--max-attempts", "-a",
        type=int,
        default=Config.MAX_ATTEMPTS,
        help="Maximum generation attempts"
    )
    parser.add_argument(
        "--max-tokens", "-t",
        type=int,
        default=Config.MAX_NEW_TOKENS,
        help="Maximum new tokens to generate"
    )
    
    args = parser.parse_args()
    
    # Override config from command line
    if args.model:
        Config.MODEL_ID = args.model
    if args.max_attempts:
        Config.MAX_ATTEMPTS = args.max_attempts
    if args.max_tokens:
        Config.MAX_NEW_TOKENS = args.max_tokens
    
    main()
