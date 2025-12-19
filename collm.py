# =========================
# Standard Libraries
# =========================
import sys
import subprocess
from pathlib import Path
import re
import logging
import time
import threading
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
# Core packages always required
REQUIRED_PACKAGES = {
    "tqdm": "tqdm",
    "matplotlib": "matplotlib",
    "langchain": "langchain",
    "transformers": "transformers",
    "langchain_huggingface": "langchain-huggingface",
    "huggingface_hub": "huggingface_hub",
    "accelerate": "accelerate",
    "torch": "torch",
    "pydantic": "pydantic",
}

def ensure_packages(extra_packages: Optional[Dict[str, str]] = None):
    """Install required packages if not already installed."""
    packages = {**REQUIRED_PACKAGES, **(extra_packages or {})}
    for module, pip_name in packages.items():
        try:
            __import__(module)
            logger.info(f"✓ {module} already installed")
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

# =========================
# Configuration
# =========================
class Config:
    """Centralized configuration."""
    SYSTEM_PROMPT_PATH = Path("system_prompt.txt")
    HUMAN_PROMPT_PATH = Path("human_prompt.txt")
    USER_INPUT_PATH = Path("user_input.txt")
    OUTPUT_CODE_PATH = Path("generated_analysis.py")
    
    # Model Configuration
    MODEL_ID = "microsoft/Phi-3.5-mini-instruct"
    
    # Generation Parameters
    MAX_NEW_TOKENS = 2048  # Increased for complete code generation
    TEMPERATURE = 0.0
    TOP_P = 0.95
    TOP_K = 50
    DO_SAMPLE = False

    # Required placeholders in human_prompt.txt
    REQUIRED_PLACEHOLDERS = [
        "{selection_cuts}",
        "{plots_for_validation}",
        "{output_structure}",
    ]
    
    # Required section tags in user_input.txt
    REQUIRED_SECTIONS = [
        "SELECTION_CUTS",
        "PLOTS_FOR_VALIDATION",
        "OUTPUT_STRUCTURE",
    ]
    
    # Retry configuration
    MAX_ATTEMPTS = 2
    

# =========================
# CLI Argument Parsing (Early)
# =========================
def parse_args():
    """Parse command-line arguments early to conditionally install packages."""
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
    # --- GOAL A: Quantization options ---
    parser.add_argument(
        "--quant",
        type=str,
        choices=["none", "8bit", "4bit"],
        default="none",
        help="Quantization mode: none, 8bit, or 4bit (default: none)"
    )
    parser.add_argument(
        "--bnb_4bit_compute_dtype",
        type=str,
        choices=["float16", "bfloat16"],
        default="float16",
        help="Compute dtype for 4-bit quantization (default: float16)"
    )
    parser.add_argument(
        "--bnb_4bit_quant_type",
        type=str,
        choices=["nf4", "fp4"],
        default="nf4",
        help="Quantization type for 4-bit (default: nf4)"
    )
    parser.add_argument(
        "--max_memory_gib",
        type=float,
        default=None,
        help="Max GPU memory in GiB to avoid VRAM cliff (e.g., 15.0 for 16GB GPU)"
    )
    # --- GOAL B: Direct generation options ---
    parser.add_argument(
        "--use_generate",
        action="store_true",
        help="Use model.generate() directly instead of LangChain pipeline"
    )
    parser.add_argument(
        "--attn",
        type=str,
        choices=["auto", "sdpa", "flash_attention_2"],
        default="auto",
        help="Attention implementation (default: auto)"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help=f"Max new tokens to generate (default: {Config.MAX_NEW_TOKENS})"
    )
    # --- GOAL C: Progress options ---
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bar"
    )
    # --- New options ---
    parser.add_argument(
        "--max_attempts",
        type=int,
        default=Config.MAX_ATTEMPTS,
        help=f"Maximum generation attempts (default: {Config.MAX_ATTEMPTS})"
    )
    parser.add_argument(
        "--save_partial",
        action="store_true",
        help="Save code even if validation fails (with .partial suffix)"
    )
    
    return parser.parse_args()


# Parse args early to determine if we need bitsandbytes
args = parse_args()

# Conditionally add bitsandbytes if quantization requested
extra_pkgs = {}
if args.quant != "none":
    extra_pkgs["bitsandbytes"] = "bitsandbytes"
    logger.info(f"Quantization mode '{args.quant}' requested, adding bitsandbytes...")

ensure_packages(extra_pkgs)

# =========================
# Imports (after install)
# =========================
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextIteratorStreamer
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage

# Local modules for validation
from schemas import AnalysisSpec
from validators import validate_generated_code

# Try to import enhanced validation hints
try:
    from validators import get_validation_hints
except ImportError:
    def get_validation_hints(errors: str) -> str:
        return ""

# Conditional import for quantization
if args.quant != "none":
    from transformers import BitsAndBytesConfig


# =========================
# GOAL D: Safe Speed Tweaks
# =========================
def apply_cuda_optimizations():
    """Apply safe CUDA optimizations for faster inference."""
    if torch.cuda.is_available():
        # Enable TF32 for faster matmul on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        logger.info("✓ Enabled TF32 for CUDA matmul (Ampere+ optimization)")


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
    
    logger.info(f" Loaded {file_path} ({len(content)} characters)")
    return content


def extract_python_code(text: str) -> str:
    """Extract Python code from LLM response."""
    # Remove output code marker if present
    if "## Output Code:" in text:
        text = text.split("## Output Code:", 1)[1]
    if "## OUTPUT CODE" in text.upper():
        parts = re.split(r"##\s*OUTPUT\s*CODE[^:]*:", text, flags=re.IGNORECASE)
        if len(parts) > 1:
            text = parts[1]
    
    # Try to extract from markdown code blocks
    code_block_pattern = r"```(?:python)?\s*(.*?)```"
    matches = re.findall(code_block_pattern, text, re.DOTALL)
    
    if matches:
        code = max(matches, key=len).strip()
        logger.info(f"✓ Extracted code from markdown block ({len(code)} characters)")
        return code

    # Fallback: aggressively strip any non-code preamble and take code from first import/def/shebang
    code = sanitize_to_python(text)
    logger.info(f"✓ Using sanitized text as code ({len(code)} characters)")
    return code


def sanitize_to_python(text: str) -> str:
    """
    Best-effort sanitizer when the model returns non-code preamble.
    """
    # Remove common preamble markers
    for marker in ["## Output Code:", "```python", "```"]:
        if marker in text:
            text = text.split(marker, 1)[-1]

    lines = text.splitlines()
    start_idx = 0
    for i, line in enumerate(lines):
        s = line.lstrip()
        if (
            s.startswith("#!") or s.startswith("import ") or s.startswith("from ")
            or s.startswith("def ") or s.startswith("class ") or s.startswith("#")
        ):
            start_idx = i
            break
    code = "\n".join(lines[start_idx:]).strip()

    # Remove trailing markdown fences
    code = re.sub(r"\n?```+\s*$", "", code).strip()
    return code


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


# =========================
# Input Validation Functions
# =========================
def validate_human_prompt(content: str) -> None:
    """Validate that human_prompt.txt contains all required placeholders."""
    missing = []
    for placeholder in Config.REQUIRED_PLACEHOLDERS:
        if placeholder not in content:
            missing.append(placeholder)
    
    if missing:
        raise ValueError(
            f"human_prompt.txt missing required placeholder(s): {', '.join(missing)}"
        )


def validate_user_input_sections(content: str) -> None:
    """Validate that user_input.txt contains all required section tags."""
    missing = []
    for section in Config.REQUIRED_SECTIONS:
        tag = f"[{section}]"
        if tag not in content:
            missing.append(tag)
    
    if missing:
        raise ValueError(
            f"user_input.txt missing required section(s): {', '.join(missing)}"
        )


def validate_inputs(human_prompt: str, user_input: str) -> None:
    """Validate both human_prompt.txt and user_input.txt before model invocation."""
    logger.info("Validating input prompts...")
    validate_human_prompt(human_prompt)
    logger.info("  ✓ human_prompt.txt placeholders valid")
    
    validate_user_input_sections(user_input)
    logger.info("  ✓ user_input.txt sections valid")


# =========================
# GOAL A: Quantization Config Builder
# =========================
def build_quant_config(args) -> Optional[Any]:
    """Build BitsAndBytesConfig based on CLI arguments."""
    if args.quant == "none":
        return None
    
    compute_dtype = torch.float16 if args.bnb_4bit_compute_dtype == "float16" else torch.bfloat16
    
    if args.quant == "8bit":
        logger.info("Building 8-bit quantization config...")
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    elif args.quant == "4bit":
        logger.info(f"Building 4-bit quantization config (type={args.bnb_4bit_quant_type}, "
                   f"compute_dtype={args.bnb_4bit_compute_dtype})...")
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_quant_type=args.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=True,
        )
    
    return None


def get_attention_implementation(args) -> Optional[str]:
    """Determine attention implementation with graceful fallback."""
    if args.attn == "auto":
        return None
    
    if args.attn == "flash_attention_2":
        try:
            import flash_attn
            logger.info("✓ Flash Attention 2 available")
            return "flash_attention_2"
        except ImportError:
            logger.warning("⚠ Flash Attention 2 not available, falling back to SDPA")
            return "sdpa"
    
    return args.attn


# =========================
# Model Loading (Enhanced)
# =========================
def load_model_and_tokenizer(
    model_id: str,
    quant_config: Optional[Any] = None,
    attn_impl: Optional[str] = None,
    max_memory: Optional[Dict[str, str]] = None,
) -> Tuple[Any, Any]:
    """Load model and tokenizer with optional quantization and attention config."""
    quant_str = "none" if quant_config is None else ("8bit" if getattr(quant_config, 'load_in_8bit', False) else "4bit")
    attn_str = attn_impl or "auto"
    mem_str = f"{max_memory}" if max_memory else "auto"
    
    logger.info(f"Loading model: {model_id}")
    logger.info(f"  quant={quant_str}, attn={attn_str}, max_memory={mem_str}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        logger.info(" Tokenizer loaded")
        
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        
        model_kwargs = {
            "device_map": "auto",
            "torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32,
            "low_cpu_mem_usage": True,
        }
        
        if quant_config is not None:
            model_kwargs["quantization_config"] = quant_config
        
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl
        
        if max_memory is not None:
            model_kwargs["max_memory"] = max_memory
        
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        model.eval()
        logger.info(" Model loaded and set to eval mode")
        
        if hasattr(model, 'hf_device_map'):
            device_map = model.hf_device_map
            if 'cpu' in str(device_map.values()):
                logger.warning("⚠ Some model layers offloaded to CPU (may impact performance)")
            logger.info(f"  Device map: {device_map}")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f" Failed to load model: {e}")
        raise


def create_pipeline(model, tokenizer, max_new_tokens: int):
    """Create HuggingFace pipeline with configuration."""
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=Config.DO_SAMPLE,
        temperature=Config.TEMPERATURE,
        top_p=Config.TOP_P,
        top_k=Config.TOP_K,
        return_full_text=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    logger.info(" Pipeline created")
    return pipe


def parse_user_input(path: Path) -> Tuple[str, str, str]:
    """Parse user input file and extract the three main sections."""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()

    def extract(tag: str) -> str:
        pattern = rf"\[{tag}\](.*?)(?=\n\[|\Z)"
        match = re.search(pattern, text, re.S)
        return match.group(1).strip() if match else ""

    selection_cuts = extract("SELECTION_CUTS")
    plots_for_validation = extract("PLOTS_FOR_VALIDATION")
    output_structure = extract("OUTPUT_STRUCTURE")

    empty_sections = []
    if not selection_cuts:
        empty_sections.append("SELECTION_CUTS")
    if not plots_for_validation:
        empty_sections.append("PLOTS_FOR_VALIDATION")
    if not output_structure:
        empty_sections.append("OUTPUT_STRUCTURE")
    
    if empty_sections:
        raise ValueError(
            f"user_input.txt has empty section(s): {', '.join(empty_sections)}"
        )

    logger.info(f"✓ Parsed user input sections:")
    logger.info(f"  - SELECTION_CUTS: {len(selection_cuts)} chars")
    logger.info(f"  - PLOTS_FOR_VALIDATION: {len(plots_for_validation)} chars")
    logger.info(f"  - OUTPUT_STRUCTURE: {len(output_structure)} chars")

    return selection_cuts, plots_for_validation, output_structure


# =========================
# Error Feedback Builder
# =========================
def build_error_feedback(error_msg: str, attempt: int) -> str:
    """
    Build a structured error feedback message to help the LLM fix issues.
    This is the key to making retry loops effective.
    """
    feedback = f"""

═══════════════════════════════════════════════════════════════════════════════
VALIDATION FAILED (Attempt {attempt}) - FIX THESE ERRORS:
═══════════════════════════════════════════════════════════════════════════════

{error_msg}

═══════════════════════════════════════════════════════════════════════════════
SPECIFIC FIXES REQUIRED:
═══════════════════════════════════════════════════════════════════════════════
"""
    
    # Add specific guidance based on error type
    if "placeholder" in error_msg.lower():
        feedback += """
• CRITICAL: You used template syntax like {branch} or {count}
• These are NOT valid Python - replace with REAL variable names
• WRONG: print(f"After {{stage}}: {{count}}")
• CORRECT: print(f"After electron cuts: {n_after_electron_cuts}")
• WRONG: for {{particle}} in particles:
• CORRECT: for electron in electrons:
"""
    
    if "register_awkward" in error_msg.lower():
        feedback += """
• Add this line immediately after your imports:
  vector.register_awkward()
"""
    
    if "library='ak'" in error_msg.lower() or "tree.arrays" in error_msg.lower():
        feedback += """
• Change your tree.arrays() call to include library="ak":
  arrays = tree.arrays(branch_list, library="ak")
"""
    
    if "syntax error" in error_msg.lower():
        feedback += """
• Check for:
  - Missing colons after if/for/def/class
  - Unmatched parentheses or brackets
  - Incomplete statements
  - Wrong indentation (use 4 spaces)
"""
    
    if "savefig" in error_msg.lower():
        feedback += """
• Add plt.savefig() for each plot:
  plt.savefig("plot_name.png", dpi=150)
  plt.close()
"""
    
    if "__main__" in error_msg.lower():
        feedback += """
• Add entry point at the end:
  if __name__ == "__main__":
      main()
"""

    if 'file["Delphes"]' in error_msg or "Delphes" in error_msg:
        feedback += """
• Access the Delphes tree:
  file = uproot.open(sys.argv[1])
  tree = file["Delphes"]
"""

    feedback += """
═══════════════════════════════════════════════════════════════════════════════
NOW OUTPUT THE COMPLETE CORRECTED PYTHON CODE (no explanations, just code):
═══════════════════════════════════════════════════════════════════════════════
"""
    
    return feedback


# =========================
# Generation with Progress
# =========================
def generate_with_progress(
    model,
    tokenizer,
    prompt_text: str,
    max_new_tokens: int,
    show_progress: bool = True,
) -> str:
    """Generate text using model.generate() with streaming progress indicator."""
    logger.info(f"Generating (method=generate, max_new_tokens={max_new_tokens})...")
    
    device = next(model.parameters()).device
    inputs = tokenizer(prompt_text, return_tensors="pt")
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    input_length = input_ids.shape[1]
    
    logger.info(f"  Input tokens: {input_length}")
    
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True,
    )
    
    gen_kwargs = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "max_new_tokens": max_new_tokens,
        "do_sample": Config.DO_SAMPLE,
        "temperature": Config.TEMPERATURE if Config.DO_SAMPLE else None,
        "top_p": Config.TOP_P if Config.DO_SAMPLE else None,
        "top_k": Config.TOP_K if Config.DO_SAMPLE else None,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
        "use_cache": True,
        "streamer": streamer,
    }
    
    gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
    
    start_time = time.time()
    generated_text = []
    token_count = 0
    last_log_time = start_time
    
    def generate_thread():
        with torch.inference_mode():
            model.generate(**gen_kwargs)
    
    thread = threading.Thread(target=generate_thread)
    thread.start()
    
    if show_progress:
        pbar = tqdm(total=max_new_tokens, desc="Generating", unit="tok")
    
    try:
        for chunk in streamer:
            generated_text.append(chunk)
            new_tokens = len(tokenizer.encode(chunk, add_special_tokens=False))
            new_tokens = max(1, new_tokens)
            token_count += new_tokens

            if show_progress:
                remaining = max_new_tokens - pbar.n
                pbar.update(min(new_tokens, max(0, remaining)))

            current_time = time.time()
            if current_time - last_log_time >= 10:
                elapsed = current_time - start_time
                tok_per_sec = token_count / elapsed if elapsed > 0 else 0
                logger.info(f"  Progress: ~{token_count} tokens, {tok_per_sec:.1f} tok/s")
                last_log_time = current_time
    
    finally:
        if show_progress:
            pbar.close()
        thread.join()
    
    elapsed = time.time() - start_time
    final_text = "".join(generated_text)
    actual_tokens = len(tokenizer.encode(final_text))
    tok_per_sec = actual_tokens / elapsed if elapsed > 0 else 0
    
    logger.info(f"✓ Generation complete: {actual_tokens} tokens in {elapsed:.1f}s ({tok_per_sec:.1f} tok/s)")
    
    return final_text


def invoke_llm_langchain(llm, messages, show_progress: bool = True) -> str:
    """Invoke LLM using LangChain pipeline with heartbeat logging."""
    logger.info("Generating (method=langchain)...")
    logger.info("  (This may take a few minutes...)")
    
    stop_heartbeat = threading.Event()
    
    def heartbeat():
        interval = 15
        count = 0
        while not stop_heartbeat.wait(interval):
            count += 1
            logger.info(f"  Still generating... ({count * interval}s elapsed)")
    
    if show_progress:
        heartbeat_thread = threading.Thread(target=heartbeat, daemon=True)
        heartbeat_thread.start()
    
    try:
        start_time = time.time()
        response = llm.invoke(messages)
        elapsed = time.time() - start_time
        
        text = response.content if hasattr(response, "content") else str(response)
        logger.info(f"✓ Generation complete ({len(text)} chars in {elapsed:.1f}s)")
        
        return text
    finally:
        stop_heartbeat.set()


# =========================
# Main Execution
# =========================
def main(args):
    """Main execution function."""
    try:
        logger.info("=" * 60)
        logger.info("Delphes Analysis Code Generator (Enhanced)")
        logger.info("=" * 60)
        
        apply_cuda_optimizations()
        gpu_info = check_gpu_availability()
        
        max_new_tokens = args.max_new_tokens if args.max_new_tokens else Config.MAX_NEW_TOKENS
        max_attempts = args.max_attempts
        show_progress = not args.no_progress
        
        logger.info(f"\nConfiguration:")
        logger.info(f"  Model: {args.model}")
        logger.info(f"  Quantization: {args.quant}")
        logger.info(f"  Attention: {args.attn}")
        logger.info(f"  Use generate(): {args.use_generate}")
        logger.info(f"  Max new tokens: {max_new_tokens}")
        logger.info(f"  Max attempts: {max_attempts}")
        logger.info(f"  Show progress: {show_progress}")
        
        # Load prompts
        logger.info("\n[1/6] Loading prompts...")
        system_prompt = load_text_file(Config.SYSTEM_PROMPT_PATH)
        human_prompt = load_text_file(Config.HUMAN_PROMPT_PATH)
        user_input_raw = load_text_file(Config.USER_INPUT_PATH)
        
        # Validate prompts and user input structure
        logger.info("\n[2/6] Validating inputs...")
        validate_inputs(human_prompt, user_input_raw)
        
        # Parse and validate user input with Pydantic
        logger.info("Validating user input schema...")
        selection_cuts, plots_for_validation, output_structure = parse_user_input(
            Config.USER_INPUT_PATH
        )
        spec = AnalysisSpec(
            selection_cuts=selection_cuts,
            plots_for_validation=plots_for_validation,
            output_structure=output_structure,
        )
        logger.info("  ✓ AnalysisSpec validated successfully")

        # Build quantization config
        quant_config = build_quant_config(args)
        attn_impl = get_attention_implementation(args)
        
        max_memory = None
        if args.max_memory_gib is not None:
            max_memory = {0: f"{args.max_memory_gib}GiB", "cpu": "64GiB"}
            logger.info(f"  Max memory limit: {args.max_memory_gib} GiB GPU + 64 GiB CPU")

        # Load model
        logger.info("\n[3/6] Loading model...")
        model, tokenizer = load_model_and_tokenizer(
            args.model,
            quant_config=quant_config,
            attn_impl=attn_impl,
            max_memory=max_memory,
        )
        
        # Build prompt using validated spec
        logger.info("\n[4/6] Building prompt...")
        prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt),
        ])
        
        messages = prompt_template.format_messages(
            selection_cuts=spec.selection_cuts,
            plots_for_validation=spec.plots_for_validation,
            output_structure=spec.output_structure,
        )
        
        total_chars = sum(len(str(m.content)) for m in messages)
        logger.info(f"  Total prompt length: {total_chars} characters")
        
        # Generate code (with retries)
        logger.info("\n[5/6] Generating code...")
        
        last_error = None
        last_code = None
        current_human_prompt = human_prompt
        accumulated_errors = []

        for attempt in range(1, max_attempts + 1):
            logger.info(f"\n[5/6] Generation attempt {attempt}/{max_attempts}...")

            if args.use_generate:
                # Build messages for this attempt
                prompt_template_retry = ChatPromptTemplate.from_messages([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=current_human_prompt),
                ])
                messages_retry = prompt_template_retry.format_messages(
                    selection_cuts=spec.selection_cuts,
                    plots_for_validation=spec.plots_for_validation,
                    output_structure=spec.output_structure,
                )
                
                system_content = messages_retry[0].content
                user_content = messages_retry[1].content
                
                # Add strict output contract on first attempt
                if attempt == 1:
                    user_content = (
                        user_content
                        + "\n\n═══════════════════════════════════════════════════════════════════════════════\n"
                          "STRICT OUTPUT CONTRACT:\n"
                          "═══════════════════════════════════════════════════════════════════════════════\n"
                          "• Output ONLY valid Python code - no markdown, no explanations\n"
                          "• First line MUST be: import sys\n"
                          "• NEVER use placeholder syntax like {branch} or {count}\n"
                          "• You MUST include: vector.register_awkward()\n"
                          "• You MUST include: tree.arrays(..., library='ak')\n"
                          "• You MUST save plots with: plt.savefig(..., dpi=150) and plt.close()\n"
                          "• You MUST have: if __name__ == '__main__': main()\n"
                          "═══════════════════════════════════════════════════════════════════════════════\n"
                    )
                
                chat = [
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ]
                prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)

                text = generate_with_progress(
                    model,
                    tokenizer,
                    prompt_text,
                    max_new_tokens,
                    show_progress=show_progress,
                )
            else:
                # LangChain path
                prompt_template_retry = ChatPromptTemplate.from_messages([
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=current_human_prompt),
                ])
                messages_retry = prompt_template_retry.format_messages(
                    selection_cuts=spec.selection_cuts,
                    plots_for_validation=spec.plots_for_validation,
                    output_structure=spec.output_structure,
                )
                
                logger.info("\n[4.5/6] Creating pipeline...")
                pipe = create_pipeline(model, tokenizer, max_new_tokens)
                llm = HuggingFacePipeline(pipeline=pipe)

                text = invoke_llm_langchain(llm, messages_retry, show_progress=show_progress)

            # Extract & validate
            code = extract_python_code(text)
            last_code = code

            logger.info("\n[6/6] Validating generated code (AST + structure)...")
            try:
                validate_generated_code(code)
                logger.info("  ✓ Generated code passed all validation checks")
                break  # success
            except ValueError as e:
                last_error = str(e)
                accumulated_errors.append(f"Attempt {attempt}: {last_error}")
                logger.error(f"✗ Code validation failed (attempt {attempt}):\n{e}")

                # Build structured error feedback for next attempt
                if attempt < max_attempts:
                    error_feedback = build_error_feedback(last_error, attempt)
                    current_human_prompt = human_prompt + error_feedback
                    logger.info("  → Feeding error details back to model for next attempt...")
                else:
                    logger.error("✗ Maximum attempts reached.")
                    
                    # Optionally save partial code
                    if args.save_partial and last_code:
                        partial_path = Config.OUTPUT_CODE_PATH.with_suffix('.partial.py')
                        save_code(last_code, partial_path)
                        logger.warning(f"  → Saved partial (invalid) code to {partial_path}")
                    
                    logger.error("\nAll validation errors:")
                    for err in accumulated_errors:
                        logger.error(f"  {err}")
                    
                    sys.exit(1)

        # Save code (only if validation passed)
        save_code(code, Config.OUTPUT_CODE_PATH)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("GENERATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Model: {args.model}")
        logger.info(f"Quantization: {args.quant}")
        logger.info(f"GPU Used: {'Yes' if gpu_info['available'] else 'No'}")
        logger.info(f"Generation Method: {'generate()' if args.use_generate else 'LangChain'}")
        logger.info(f"Attempts: {attempt}/{max_attempts}")
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
    
    except ValueError as e:
        logger.error(f"✗ Validation error: {e}")
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
    if args.model:
        Config.MODEL_ID = args.model
    
    main(args)