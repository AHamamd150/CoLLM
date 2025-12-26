#!/usr/bin/env python3
"""
pyfixer.py - Fix Python code using local LLMs (improved version)

Usage:
    from pyfixer import fix_code
    fixed = fix_code(original_code, terminal_error)
"""

import re
import ast
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig  

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-14B-Instruct"

SYSTEM_PROMPT = """You are an expert Python debugger. Your task is to fix buggy Python code.

## Process:
1. Read the error message carefully - note the error TYPE and LINE NUMBER
2. Understand what the code is trying to do
3. Identify the root cause (not just the symptom)
4. Fix ALL issues, not just the reported one
5. Return COMPLETE working code

## Common fixes:
- SyntaxError: missing colons, parentheses, commas, quotes
- NameError: typos in variable names, undefined variables, missing imports
- TypeError: wrong types, missing/extra arguments, None operations
- IndexError: check list bounds, empty lists
- KeyError: check dict keys exist, use .get()
- AttributeError: wrong method names, None objects
- IndentationError: fix spacing consistency
- ImportError: correct module names, install missing packages

## Rules:
- Return the COMPLETE fixed code, not just the changed parts
- Keep all original functionality
- Add missing imports if needed
- Use ```python``` code block
- NO explanations outside the code block"""

_model = None
_tokenizer = None
_device = None


def _get_device_and_dtype():
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


def _load_model(model_id: str = DEFAULT_MODEL):
    global _model, _tokenizer, _device
    if _model is None:
        print(f"Loading model: {model_id}")
        _device, dtype = _get_device_and_dtype()
        print(f"Using device: {_device}")
        _tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        
        if _device == "cuda":
            # CUDA supports 4-bit quantization with BitsAndBytes
            from transformers import BitsAndBytesConfig
            
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4"
            )
            
            _model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=quantization_config,
                device_map="auto",
                trust_remote_code=True,
            )
        elif _device == "mps":
            # MPS: Use float16 for faster inference
            _model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=torch.float16,
            ).to(_device)
        else:
            # CPU fallback
            _model = AutoModelForCausalLM.from_pretrained(
                model_id,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
    
    return _model, _tokenizer, _device

def _extract_code(response: str) -> str:
    match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def _parse_error(error: str) -> dict:
    """Parse error to extract useful information."""
    info = {
        "type": None,
        "message": None,
        "line": None,
        "file": None,
        "code_snippet": None
    }
    
    # Extract error type and message (last line usually)
    lines = error.strip().split('\n')
    for line in reversed(lines):
        match = re.match(r'^(\w+Error|\w+Exception|SyntaxError|IndentationError): (.+)$', line)
        if match:
            info["type"] = match.group(1)
            info["message"] = match.group(2)
            break
    
    # Extract line number
    match = re.search(r'line (\d+)', error, re.IGNORECASE)
    if match:
        info["line"] = int(match.group(1))
    
    # Extract file name
    match = re.search(r'File "([^"]+)"', error)
    if match:
        info["file"] = match.group(1)
    
    # Extract code snippet (line with ^)
    for i, line in enumerate(lines):
        if '^^^' in line or (line.strip() == '^' or re.match(r'^\s*\^+\s*$', line)):
            if i > 0:
                info["code_snippet"] = lines[i-1].strip()
            break
    
    return info


def _validate_syntax(code: str) -> tuple[bool, str]:
    """Check if code has valid Python syntax."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"SyntaxError at line {e.lineno}: {e.msg}"


def _add_line_numbers(code: str) -> str:
    """Add line numbers to code for reference."""
    lines = code.split('\n')
    return '\n'.join(f"{i+1:3d} | {line}" for i, line in enumerate(lines))


def fix_code(code: str, error: str, model_id: str = DEFAULT_MODEL) -> str:
    """
    Fix Python code given the terminal error.
    
    Args:
        code: Original buggy Python code
        error: Terminal error message
        model_id: Hugging Face model ID
    
    Returns:
        Fixed Python code
    """
    model, tokenizer, device = _load_model(model_id)
    
    # Parse error for better context
    error_info = _parse_error(error)
    
    # Build detailed prompt
    numbered_code = _add_line_numbers(code)
    
    error_summary = []
    if error_info["type"]:
        error_summary.append(f"Error Type: {error_info['type']}")
    if error_info["line"]:
        error_summary.append(f"Error Line: {error_info['line']}")
    if error_info["message"]:
        error_summary.append(f"Message: {error_info['message']}")
    
    user_prompt = f"""Fix this Python code:

## Code (with line numbers):
```python
{numbered_code}
```

## Error Analysis:
{chr(10).join(error_summary) if error_summary else "See traceback below"}

## Full Traceback:
```
{error}
```

## Task:
1. Identify the bug at/near line {error_info['line'] or 'shown in traceback'}
2. Check for related bugs elsewhere
3. Return the COMPLETE fixed code (without line numbers)"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print(f"Fixing {error_info['type'] or 'error'} at line {error_info['line'] or '?'}...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=4096,
            temperature=0.9,
            top_p=0.90,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    fixed_code = _extract_code(response)
    
    # Validate syntax
    is_valid, syntax_error = _validate_syntax(fixed_code)
    if not is_valid:
        print(f"Warning: Generated code has syntax error: {syntax_error}")
    
    return fixed_code
