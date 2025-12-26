#!/usr/bin/env python3
"""
pyfixer.py - Fix Python code using local LLMs

Usage:
    from pyfixer import fix_code
    
    # Without web search (default)
    fixed = fix_code(code, error)
    
    # With web search
    fixed = fix_code(code, error, use_search=True)
"""

import re
import ast
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"

SYSTEM_PROMPT = """You are an expert Python debugger. Fix the buggy code by:
1. Analyzing the error traceback
2. Locating and fixing the bug
3. Returning COMPLETE working code

RULES:
- Return the COMPLETE fixed code, not just the fix
- Keep ALL original functionality
- Add imports if missing
- Use ```python``` code block
- NO explanations outside code block"""

_model = None
_tokenizer = None
_device = None
_tavily = None


def _load_tavily():
    """Load Tavily search."""
    global _tavily
    if _tavily is None:
        from langchain_tavily import TavilySearch
        _tavily = TavilySearch(max_results=3)
        print("✓ Tavily search enabled")
    return _tavily


def _search_solution(error_type: str, error_message: str) -> str:
    """Search for solutions online."""
    try:
        tavily = _load_tavily()
        query = f"python {error_type} {error_message} fix"
        results = tavily.invoke(query)
        if results:
            return f"\n## Web Search Results:\n{results}\n"
    except Exception as e:
        print(f"⚠ Search failed: {e}")
    return ""


def _load_model(model_id: str = DEFAULT_MODEL):
    """Load model (cached)."""
    global _model, _tokenizer, _device
    if _model is None:
        print(f"Loading model: {model_id}")
        _device = "cuda" if torch.cuda.is_available() else "cpu"
        _tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        _model = AutoModelForCausalLM.from_pretrained(
            model_id,
            trust_remote_code=True,
            torch_dtype=torch.float16 if _device == "cuda" else torch.float32,
            device_map="auto" if _device == "cuda" else None
        )
    return _model, _tokenizer, _device


def _extract_code(response: str) -> str:
    """Extract Python code from LLM response."""
    match = re.search(r'```python\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    match = re.search(r'```\s*(.*?)\s*```', response, re.DOTALL)
    if match:
        return match.group(1).strip()
    return response.strip()


def _parse_error(error: str) -> dict:
    """Parse error to extract type, message, and line."""
    info = {"type": None, "message": None, "line": None}
    
    lines = error.strip().split('\n')
    for line in reversed(lines):
        match = re.match(r'^(\w+Error|\w+Exception): (.+)$', line)
        if match:
            info["type"] = match.group(1)
            info["message"] = match.group(2)
            break
    
    match = re.search(r'line (\d+)', error, re.IGNORECASE)
    if match:
        info["line"] = int(match.group(1))
    
    return info


def _validate_syntax(code: str) -> tuple:
    """Check if code has valid Python syntax."""
    try:
        ast.parse(code)
        return True, ""
    except SyntaxError as e:
        return False, f"Line {e.lineno}: {e.msg}"


def fix_code(code: str, error: str, model_id: str = DEFAULT_MODEL, use_search: bool = False) -> str:
    """
    Fix Python code given the terminal error.
    
    Args:
        code: Original buggy Python code
        error: Terminal error message
        model_id: Hugging Face model ID
        use_search: Use Tavily web search for solutions (default: False)
    
    Returns:
        Fixed Python code
    """
    model, tokenizer, device = _load_model(model_id)
    error_info = _parse_error(error)
    
    # Optional web search
    search_results = ""
    if use_search and error_info["type"]:
        print(f"Searching solutions for: {error_info['type']}...")
        search_results = _search_solution(error_info["type"], error_info["message"] or "")
    
    # Build prompt
    user_prompt = f"""## Buggy Code:
```python
{code}
```

## Error:
```
{error}
```
{search_results}
## Task: Fix the bug and return the COMPLETE corrected code."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ]
    
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print(f"Fixing {error_info['type'] or 'error'}" + 
          (f" at line {error_info['line']}" if error_info['line'] else "") + "...")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=8192,
            temperature=0.2,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    fixed_code = _extract_code(response)
    
    is_valid, syntax_err = _validate_syntax(fixed_code)
    if not is_valid:
        print(f"Warning: Output has syntax error: {syntax_err}")
    
    return fixed_code
