# Refinement logic for multiple attempts - matches generic system prompt

def build_output_contract(attempt: int, previous_error: str = None) -> str:
    """Build appropriate output contract based on attempt number and any previous errors."""
    
    if attempt == 1:
        # First attempt: emphasize reading user request and strict output format
        return (
            "\n\n═══════════════════════════════════════════════════════════════════════════════\n"
            "STRICT OUTPUT CONTRACT\n"
            "═══════════════════════════════════════════════════════════════════════════════\n"
            "\n"
            "READ THE USER REQUEST ABOVE CAREFULLY. Generate code that implements EXACTLY\n"
            "what is specified - no more, no less.\n"
            "\n"
            "MANDATORY REQUIREMENTS:\n"
            "• Output ONLY valid Python code - no markdown, no explanations, no prose\n"
            "• First line MUST be: import sys\n"
            "• Include: vector.register_awkward() after imports\n"
            "• Include: tree.arrays(..., library='ak') for loading data\n"
            "• Include: if __name__ == '__main__': main()\n"
            "• Save all plots with: plt.savefig(..., dpi=150) followed by plt.close()\n"
            "\n"
            "FORBIDDEN:\n"
            "• NO placeholder syntax like {branch}, {count}, {particle}, {stage}, etc.\n"
            "• NO markdown code fences (```python)\n"
            "• NO explanatory text before or after the code\n"
            "• NO cuts, plots, or features not explicitly requested by user\n"
            "\n"
            "YOUR CODE MUST:\n"
            "• Load only the physics objects mentioned in the user request\n"
            "• Apply only the selection cuts specified by the user\n"
            "• Compute only the observables the user asked for\n"
            "• Generate only the plots the user requested\n"
            "• Print cutflow with descriptive labels matching user's cuts\n"
            "═══════════════════════════════════════════════════════════════════════════════\n"
        )
    
    elif attempt == 2:
        # Second attempt: include error feedback and reinforce requirements
        error_section = ""
        if previous_error:
            error_section = (
                f"\n"
                f"PREVIOUS ERROR (you must fix this):\n"
                f"{previous_error}\n"
            )
        
        return (
            "\n\n═══════════════════════════════════════════════════════════════════════════════\n"
            "RETRY - FIX PREVIOUS ISSUES\n"
            "═══════════════════════════════════════════════════════════════════════════════\n"
            f"{error_section}"
            "\n"
            "REQUIREMENTS (must follow exactly):\n"
            "• Output ONLY valid Python code starting with: import sys\n"
            "• NO placeholders like {branch}, {name}, {count} - use real variable names\n"
            "• NO markdown formatting - raw Python only\n"
            "• Use vector.register_awkward() after imports\n"
            "• Use tree.arrays([...], library='ak') to load branches\n"
            "• Use find_branch() helper to handle branch name variations\n"
            "• Apply cuts sequentially, updating all relevant arrays after each cut\n"
            "• Save plots: plt.savefig('name.png', dpi=150) then plt.close()\n"
            "• End with: if __name__ == '__main__': main()\n"
            "\n"
            "IMPLEMENT ONLY WHAT THE USER REQUESTED - nothing extra.\n"
            "═══════════════════════════════════════════════════════════════════════════════\n"
        )
    
    else:
        # Third+ attempt: minimal, focused instructions
        error_section = ""
        if previous_error:
            error_section = f"\nFIX THIS ERROR: {previous_error}\n"
        
        return (
            "\n\n═══════════════════════════════════════════════════════════════════════════════\n"
            "FINAL ATTEMPT - STRICT COMPLIANCE REQUIRED\n"
            "═══════════════════════════════════════════════════════════════════════════════\n"
            f"{error_section}"
            "\n"
            "OUTPUT RULES:\n"
            "1. Raw Python code only - no markdown, no explanations\n"
            "2. First line: import sys\n"
            "3. No placeholders: {x} is FORBIDDEN except in f-strings with real variables\n"
            "4. Required: vector.register_awkward(), tree.arrays(..., library='ak')\n"
            "5. Required: plt.savefig(..., dpi=150), plt.close(), main() with CLI\n"
            "6. Implement ONLY what user requested\n"
            "═══════════════════════════════════════════════════════════════════════════════\n"
        )


# Usage in your LangChain code:
# ═══════════════════════════════════════════════════════════════════════════════

# Example integration
def generate_code(system_content: str, user_content: str, tokenizer, attempt: int = 1, previous_error: str = None):
    """Generate code with appropriate output contract based on attempt."""
    
    # Add output contract to user content
    user_content_with_contract = user_content + build_output_contract(attempt, previous_error)
    
    chat = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content_with_contract},
    ]
    
    prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    
    return prompt_text


# ═══════════════════════════════════════════════════════════════════════════════
# Simple inline version (drop-in replacement for your current code):
# ═══════════════════════════════════════════════════════════════════════════════

# Add strict output contract on first attempt
if attempt == 1:
    user_content = (
        user_content
        + "\n\n═══════════════════════════════════════════════════════════════════════════════\n"
          "STRICT OUTPUT CONTRACT\n"
          "═══════════════════════════════════════════════════════════════════════════════\n"
          "\n"
          "READ THE USER REQUEST ABOVE CAREFULLY. Generate code that implements EXACTLY\n"
          "what is specified - no more, no less.\n"
          "\n"
          "MANDATORY REQUIREMENTS:\n"
          "• Output ONLY valid Python code - no markdown, no explanations, no prose\n"
          "• First line MUST be: import sys\n"
          "• Include: vector.register_awkward() after imports\n"
          "• Include: tree.arrays(..., library='ak') for loading data\n"
          "• Include: if __name__ == '__main__': main()\n"
          "• Save all plots with: plt.savefig(..., dpi=150) followed by plt.close()\n"
          "\n"
          "FORBIDDEN:\n"
          "• NO placeholder syntax like {branch}, {count}, {particle}, {stage}, etc.\n"
          "• NO markdown code fences (```python)\n"
          "• NO explanatory text before or after the code\n"
          "• NO cuts, plots, or features not explicitly requested by user\n"
          "\n"
          "YOUR CODE MUST:\n"
          "• Load only the physics objects mentioned in the user request\n"
          "• Apply only the selection cuts specified by the user\n"
          "• Compute only the observables the user asked for\n"
          "• Generate only the plots the user requested\n"
          "• Print cutflow with descriptive labels matching user's cuts\n"
          "═══════════════════════════════════════════════════════════════════════════════\n"
    )

chat = [
    {"role": "system", "content": system_content},
    {"role": "user", "content": user_content},
]
prompt_text = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
