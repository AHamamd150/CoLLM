# =========================
# Standard Libraries
# =========================
import sys
import subprocess
from pathlib import Path
import re
import logging
from typing import Dict, Any, Tuple

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
    "transformers": "transformers",
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
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================
# Configuration
# =========================
class Config:
    """Centralized configuration."""
    SYSTEM_PROMPT_PATH = Path("system_prompt.txt")
    USER_INPUT_PATH = Path("user_input.txt")
    OUTPUT_CODE_PATH = Path("generated_analysis.py")
    
    # Model Configuration
    # For code generation, use instruction-tuned models:
    MODEL_ID = "Qwen/Qwen2.5-Coder-7B-Instruct"
    # Alternatives:
    # MODEL_ID = "codellama/CodeLlama-7b-Instruct-hf"
    # MODEL_ID = "deepseek-ai/deepseek-coder-6.7b-instruct"
    
    # NOTE: StarCoder2-7b is NOT instruction-tuned, it's a code completion model
    # It won't follow instructions properly. Use instruction-tuned models instead.
    
    # Generation Parameters
    MAX_NEW_TOKENS = 4096
    TEMPERATURE = 0.2
    TOP_P = 0.95
    TOP_K = 50
    DO_SAMPLE = True

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


def extract_python_code(text: str) -> str:
    """
    Extract Python code from LLM response.
    """
    if not text or not text.strip():
        logger.warning("Empty response received")
        return ""
    
    text = text.strip()
    
    # Pattern 1: Extract from ```python ... ``` blocks
    python_block_pattern = r"```python\s*\n?(.*?)```"
    python_matches = re.findall(python_block_pattern, text, re.DOTALL)
    if python_matches:
        code = max(python_matches, key=len).strip()
        logger.info(f"✓ Extracted code from ```python block ({len(code)} chars)")
        return code
    
    # Pattern 2: Extract from ``` ... ``` blocks
    generic_block_pattern = r"```\s*\n?(.*?)```"
    generic_matches = re.findall(generic_block_pattern, text, re.DOTALL)
    if generic_matches:
        code = max(generic_matches, key=len).strip()
        logger.info(f"✓ Extracted code from ``` block ({len(code)} chars)")
        return code
    
    # Pattern 3: Find code starting with 'import sys'
    import_match = re.search(r'^import\s+sys\b', text, re.MULTILINE)
    if import_match:
        # Extract from import sys to end, but stop at obvious non-code
        code_start = text[import_match.start():]
        lines = code_start.split('\n')
        code_lines = []
        for line in lines:
            # Stop if we hit conversation markers
            if line.strip().startswith(('Human:', 'H:', 'Assistant:', 'A:', '###')):
                break
            code_lines.append(line)
        code = '\n'.join(code_lines).strip()
        logger.info(f"✓ Extracted raw Python code ({len(code)} chars)")
        return code
    
    # Check if response is garbage (conversation continuation)
    if 'Human:' in text or text.startswith('H:') or '[SELECTION_CUTS]' in text[:100]:
        logger.error("✗ Model generated conversation instead of code")
        return ""
    
    # Fallback
    logger.warning("No code block found, returning raw text")
    return text


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
    }
    
    if gpu_info["available"]:
        gpu_info["device_count"] = torch.cuda.device_count()
        gpu_info["device_name"] = torch.cuda.get_device_name(0)
        logger.info(f"✓ GPU: {gpu_info['device_name']}")
    else:
        logger.warning("⚠ No GPU available, using CPU (this will be slow)")
    
    return gpu_info


def parse_user_input(path: Path) -> Tuple[str, str, str]:
    """Parse user input file and extract the three main sections."""
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

# =========================
# Model Loading
# =========================
def load_model_and_tokenizer(model_id: str):
    """Load model and tokenizer."""
    logger.info(f"Loading model: {model_id}")
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        logger.info("✓ Tokenizer loaded")
        
        # Set pad token if not set
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        logger.info("✓ Model loaded")
        
        return model, tokenizer
        
    except Exception as e:
        logger.error(f"✗ Failed to load model: {e}")
        raise

# =========================
# Prompt Building
# =========================
def build_prompt(system_prompt: str, selection_cuts: str, plots_for_validation: str, 
                 output_structure: str, tokenizer) -> str:
    """
    Build the prompt using the model's chat template.
    
    This is the KEY fix - we use the tokenizer's built-in chat template
    which properly formats the prompt for instruction-following.
    """
    
    user_message = f"""Generate a complete, executable Python script for the following analysis:

[SELECTION_CUTS]
{selection_cuts}

[PLOTS_FOR_VALIDATION]
{plots_for_validation}

[OUTPUT_STRUCTURE]
{output_structure}

Output ONLY the Python code, starting with 'import sys'. No explanations."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    # Use the tokenizer's chat template - this is crucial!
    try:
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        logger.info("✓ Applied chat template")
    except Exception as e:
        # Fallback for models without chat template
        logger.warning(f"Chat template failed ({e}), using fallback format")
        prompt = f"""### System:
{system_prompt}

### User:
{user_message}

### Assistant:
```python
import sys
"""
    
    return prompt

# =========================
# Generation
# =========================
def generate_code(model, tokenizer, prompt: str, config: Config) -> str:
    """Generate code using the model."""
    
    logger.info("Tokenizing prompt...")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    logger.info(f"  Input tokens: {inputs['input_ids'].shape[1]}")
    logger.info("Generating... (this may take a few minutes)")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=config.MAX_NEW_TOKENS,
            do_sample=config.DO_SAMPLE,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            top_k=config.TOP_K,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    # Decode only the new tokens
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    logger.info(f"✓ Generated {len(response)} characters")
    return response

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
        logger.info("\n[1/4] Loading prompts...")
        system_prompt = load_text_file(Config.SYSTEM_PROMPT_PATH)
        selection_cuts, plots_for_validation, output_structure = parse_user_input(Config.USER_INPUT_PATH)

        # Load model
        logger.info("\n[2/4] Loading model...")
        model, tokenizer = load_model_and_tokenizer(Config.MODEL_ID)
        
        # Build prompt
        logger.info("\n[3/4] Building prompt...")
        prompt = build_prompt(
            system_prompt=system_prompt,
            selection_cuts=selection_cuts,
            plots_for_validation=plots_for_validation,
            output_structure=output_structure,
            tokenizer=tokenizer
        )
        
        logger.info(f"  Total prompt length: {len(prompt)} characters")
        
        # Generate
        logger.info("\n[4/4] Generating code...")
        response = generate_code(model, tokenizer, prompt, Config)
        
        # Extract and save
        code = extract_python_code(response)
        
        if code and code.startswith("import"):
            save_code(code, Config.OUTPUT_CODE_PATH)
            success = True
        else:
            logger.error("✗ Failed to generate valid Python code")
            # Save raw response for debugging
            debug_path = Path("debug_raw_response.txt")
            with open(debug_path, "w") as f:
                f.write(response)
            logger.info(f"  Raw response saved to {debug_path}")
            success = False
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("GENERATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Model: {Config.MODEL_ID}")
        logger.info(f"GPU Used: {'Yes' if gpu_info['available'] else 'No'}")
        logger.info(f"Success: {'Yes' if success else 'No'}")
        if success:
            logger.info(f"Generated Code: {len(code)} characters")
            logger.info(f"Output File: {Config.OUTPUT_CODE_PATH}")
            logger.info("\nCODE PREVIEW:")
            logger.info("-" * 60)
            print(code[:500] + ("..." if len(code) > 500 else ""))
        logger.info("=" * 60)
        
        return code if success else None
        
    except FileNotFoundError as e:
        logger.error(f"✗ File error: {e}")
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
    
    parser = argparse.ArgumentParser(description="Generate Delphes analysis code using LLM")
    parser.add_argument("--model", "-m", type=str, default=Config.MODEL_ID,
                        help="HuggingFace model ID")
    parser.add_argument("--max-tokens", "-t", type=int, default=Config.MAX_NEW_TOKENS,
                        help="Maximum new tokens")
    
    args = parser.parse_args()
    
    if args.model:
        Config.MODEL_ID = args.model
    if args.max_tokens:
        Config.MAX_NEW_TOKENS = args.max_tokens
    
    main()
