# =========================
# LHCO Analysis Code Generator using LLM
# =========================
# Generates Python code to analyze LHCO files
# Does NOT execute the generated code
# =========================

"""
Usage:
    from collm_lhco import generate_lhco_code
    
    # Using local model with file path
    code = generate_lhco_code(
        user_input_path="user_input.txt",
        output_path="generated_analysis.py",
        model_id="Qwen/Qwen2.5-Coder-14B-Instruct"
    )
    
    # Using local model with string input directly
    code = generate_lhco_code(
        user_input_text="[SELECTION_CUTS]\\n...",
        output_path="generated_analysis.py",
        model_id="Qwen/Qwen2.5-Coder-14B-Instruct"
    )
    
    # Using Hugging Face Inference API
    code = generate_lhco_code(
        user_input_path="user_input.txt",
        output_path="generated_analysis.py",
        model_id="Qwen/Qwen2.5-Coder-14B-Instruct",
        use_api=True,
        api_key="your_hf_api_key"
    )
"""

# Standard Libraries
import sys
import subprocess
from pathlib import Path
import re
from typing import Dict, Any, Tuple, Optional

# =========================
# Minimal Dependency Loader
# =========================
REQUIRED_PACKAGES = {
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
        except ImportError:
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", pip_name
                ], stdout=subprocess.DEVNULL)
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Failed to install {pip_name}: {e}")

ensure_packages()

# =========================
# Imports (after install)
# =========================
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from huggingface_hub import InferenceClient  

# =========================
# Configuration
# =========================
class Config:
    """Centralized configuration."""
    SYSTEM_PROMPT_PATH = Path("./source/configs/system_prompt.txt")
    
    # Generation Parameters
    MAX_NEW_TOKENS = 4096
    TEMPERATURE = 0.0
    TOP_P = 1
    TOP_K = 00
    DO_SAMPLE = False

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
    
    return content


def extract_python_code(text: str) -> str:
    """Extract Python code from LLM response."""
    if not text or not text.strip():
        return ""
    
    text = text.strip()
    
    # Pattern 1: Extract from ```python ... ``` blocks
    python_block_pattern = r"```python\s*\n?(.*?)```"
    python_matches = re.findall(python_block_pattern, text, re.DOTALL)
    if python_matches:
        code = max(python_matches, key=len).strip()
        return code
    
    # Pattern 2: Extract from ``` ... ``` blocks
    generic_block_pattern = r"```\s*\n?(.*?)```"
    generic_matches = re.findall(generic_block_pattern, text, re.DOTALL)
    if generic_matches:
        code = max(generic_matches, key=len).strip()
        return code
    
    # Pattern 3: Find code starting with common imports
    import_patterns = [
        r'^#!/usr/bin/env python',
        r'^import\s+sys\b',
        r'^import\s+math\b',
        r'^"""',
        r"^'''",
    ]
    
    for pattern in import_patterns:
        import_match = re.search(pattern, text, re.MULTILINE)
        if import_match:
            code_start = text[import_match.start():]
            lines = code_start.split('\n')
            code_lines = []
            for line in lines:
                if line.strip().startswith(('Human:', 'H:', 'Assistant:', 'A:', '###')):
                    break
                code_lines.append(line)
            code = '\n'.join(code_lines).strip()
            return code
    
    # Check if response is garbage
    if 'Human:' in text or text.startswith('H:') or '[SELECTION_CUTS]' in text[:100]:
        return ""
    
    return text


def save_code(code: str, output_path: Path) -> None:
    """Save generated code to file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(code)


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

    return selection_cuts, plots_for_validation, output_structure


def parse_user_input_text(text: str) -> Tuple[str, str, str]:
    """Parse user input text string and extract the three main sections.
    
    Args:
        text: User input text containing [SELECTION_CUTS], [PLOTS_FOR_VALIDATION], 
              and [OUTPUT_STRUCTURE] sections.
    
    Returns:
        Tuple of (selection_cuts, plots_for_validation, output_structure)
    """
    def extract(tag):
        pattern = rf"\[{tag}\](.*?)(?=\n\[|\Z)"
        match = re.search(pattern, text, re.S)
        return match.group(1).strip() if match else ""

    selection_cuts = extract("SELECTION_CUTS")
    plots_for_validation = extract("PLOTS_FOR_VALIDATION")
    output_structure = extract("OUTPUT_STRUCTURE")

    return selection_cuts, plots_for_validation, output_structure

# =========================
# Device Detection
# =========================
def get_device_and_dtype():
    """Detect the best available device and appropriate dtype."""
    if torch.cuda.is_available():
        # NVIDIA GPU
        return "cuda", torch.float16
    elif torch.backends.mps.is_available():
        # Apple Silicon / Mac GPU
        return "mps", torch.float16
    else:
        # CPU fallback
        return "cpu", torch.float32

# =========================
# Model Loading
# =========================
def load_model_and_tokenizer(model_id: str):
    """Load model and tokenizer."""
    print(f"Loading model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device, dtype = get_device_and_dtype()
    print(f"Using device: {device}")
    
    if device == "cuda":
        # CUDA supports 4-bit quantization with BitsAndBytes
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            device_map="auto",
            trust_remote_code=True,
        )
    elif device == "mps":
        # MPS: Use float16 and CPU offloading for large models
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        ).to(device)
    else:
        # CPU fallback
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
    
    return model, tokenizer

# =========================
# Prompt Building
# =========================
def build_prompt(system_prompt: str, selection_cuts: str, plots_for_validation: str, 
                 output_structure: str, tokenizer) -> str:
    """Build the prompt using the model's chat template."""
    
    user_message = f"""Generate a complete, executable Python script to analyze LHCO files.

[SELECTION_CUTS]
{selection_cuts if selection_cuts and selection_cuts != '-' else 'No specific cuts required'}

[PLOTS_FOR_VALIDATION]
{plots_for_validation if plots_for_validation and plots_for_validation != '-' else 'No specific plots required'}

[OUTPUT_STRUCTURE]
{output_structure if output_structure else 'Print summary statistics'}

REQUIREMENTS:
1. Use ONLY standard Python libraries (math, sys) and optionally numpy/matplotlib
2. Include the complete LHCO parser function
3. The script should accept the LHCO filename as command line argument: sys.argv[1]
4. Make the script complete and executable
5. Include all helper functions needed

Output ONLY the Python code. Start with the shebang or imports. No explanations."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    try:
        prompt = tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    except Exception as e:
        prompt = f"""### System:
{system_prompt}

### User:
{user_message}

### Assistant:
```python
#!/usr/bin/env python3
import sys
import math
"""
    
    return prompt

# =========================
# Generation
# =========================
def generate_code(model, tokenizer, prompt: str) -> str:
    """Generate code using the model."""
    
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
        
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=8192)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    print("Generating code may takes few minutes...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            do_sample=Config.DO_SAMPLE,
            temperature=Config.TEMPERATURE,
            top_p=Config.TOP_P,
            top_k=Config.TOP_K,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    
    new_tokens = outputs[0][inputs['input_ids'].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return response


# =========================
# API Generation
# =========================
def generate_code_api(system_prompt: str, selection_cuts: str, plots_for_validation: str, 
                      output_structure: str, model_id: str, api_key: str) -> str:
    """Generate code using the Hugging Face Inference API."""
    
    client = InferenceClient(api_key=api_key)
    
    user_message = f"""Generate a complete, executable Python script to analyze LHCO files.

[SELECTION_CUTS]
{selection_cuts if selection_cuts and selection_cuts != '-' else 'No specific cuts required'}

[PLOTS_FOR_VALIDATION]
{plots_for_validation if plots_for_validation and plots_for_validation != '-' else 'No specific plots required'}

[OUTPUT_STRUCTURE]
{output_structure if output_structure else 'Print summary statistics'}

REQUIREMENTS:
1. Use ONLY standard Python libraries (math, sys) and optionally numpy/matplotlib
2. Include the complete LHCO parser function
3. The script should accept the LHCO filename as command line argument: sys.argv[1]
4. Make the script complete and executable
5. Include all helper functions needed

Output ONLY the Python code. Start with the shebang or imports. No explanations."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message}
    ]
    
    print("Generating code via API...")
    
    response = client.chat.completions.create(
        model=model_id,
        messages=messages,
        max_tokens=Config.MAX_NEW_TOKENS,
        temperature=Config.TEMPERATURE,
        top_p=Config.TOP_P,
        seed=42, 
    )
    
    return response.choices[0].message.content

# =========================
# Main Function
# =========================
def generate_lhco_code(
    output_path: str,
    model_id: str = "Qwen/Qwen2.5-Coder-14B-Instruct",
    user_input_path: Optional[str] = None,
    user_input_text: Optional[str] = None,
    use_api: bool = False,
    api_key: Optional[str] = None
) -> Optional[str]:
    """
    Generate LHCO analysis code using LLM.
    
    Args:
        output_path: Path to save generated code
        model_id: HuggingFace model ID
        user_input_path: Path to user input file (optional if user_input_text is provided)
        user_input_text: User input as string directly (optional if user_input_path is provided)
        use_api: If True, use Hugging Face Inference API instead of local model
        api_key: Hugging Face API key (required if use_api=True)
    
    Returns:
        Generated code string, or None if failed
    
    Note:
        Either user_input_path or user_input_text must be provided.
        If both are provided, user_input_text takes precedence.
    """
    if use_api and not api_key:
        raise ValueError("api_key is required when use_api=True")
    
    if user_input_text is None and user_input_path is None:
        raise ValueError("Either user_input_path or user_input_text must be provided")
    
    output_path = Path(output_path)
    
    # Load system prompt
    system_prompt = load_text_file(Config.SYSTEM_PROMPT_PATH)
    
    if user_input_text is not None:
        selection_cuts, plots_for_validation, output_structure = parse_user_input_text(user_input_text)
    else:
        user_input_path = Path(user_input_path)
        selection_cuts, plots_for_validation, output_structure = parse_user_input(user_input_path)

    if use_api:
        # Use Hugging Face Inference API
        response = generate_code_api(
            system_prompt=system_prompt,
            selection_cuts=selection_cuts,
            plots_for_validation=plots_for_validation,
            output_structure=output_structure,
            model_id=model_id,
            api_key=api_key
        )
    else:
        # Use local model
        model, tokenizer = load_model_and_tokenizer(model_id)
        
        # Build prompt
        prompt = build_prompt(
            system_prompt=system_prompt,
            selection_cuts=selection_cuts,
            plots_for_validation=plots_for_validation,
            output_structure=output_structure,
            tokenizer=tokenizer
        )
        
        # Generate
        response = generate_code(model, tokenizer, prompt)
    
    # Extract and save
    code = extract_python_code(response)
    
    if code:
        save_code(code, output_path)
        return code
    else:
        print("Failed to generate valid code")
        return None


# =========================
# Backward Compatibility
# =========================
# Keep old signature working for backward compatibility
def generate_lhco_code_legacy(
    user_input_path: str,
    output_path: str,
    model_id: str = "Qwen/Qwen2.5-Coder-14B-Instruct",
    use_api: bool = False,
    api_key: Optional[str] = None
) -> Optional[str]:
    """
    Legacy function signature for backward compatibility.
    
    Deprecated: Use generate_lhco_code() with keyword arguments instead.
    """
    return generate_lhco_code(
        output_path=output_path,
        model_id=model_id,
        user_input_path=user_input_path,
        use_api=use_api,
        api_key=api_key
    )
