"""
Add this function to collm.py to parse the fixer user input format.

The user_input_fixer.txt should have:
- [ORIGINAL_CODE] section with the buggy Python code
- [TERMINAL_ERROR] section with the actual error output from running the code
- [ERROR_REPORT] section with Errors, Warnings, and Style issues
"""

import re

def parse_fixer_input(path="user_input_fixer.txt"):
    """
    Parse user input file for the code fixer.
    
    Expected format:
    [ORIGINAL_CODE]
    ... python code ...
    
    [TERMINAL_ERROR]
    Traceback (most recent call last):
      File "...", line X, in <module>
        ...
    ErrorType: error message
    
    [ERROR_REPORT]
    ## Errors
    - error 1
    - error 2
    
    ## Warnings
    - warning 1
    
    ## Style
    - style issue 1
    
    Returns:
        Tuple of (original_code, terminal_error, error_report)
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

    original_code = extract("ORIGINAL_CODE")
    terminal_error = extract("TERMINAL_ERROR")
    error_report = extract("ERROR_REPORT")

    return original_code, terminal_error, error_report


# =====================================================
# INTEGRATION WITH collm.py
# =====================================================
# 
# Add this to the Config class:
# 
#     SYSTEM_PROMPT_FIXER_PATH = Path("system_prompt_fixer.txt")
#     HUMAN_PROMPT_FIXER_PATH = Path("human_prompt_fixer.txt")
#     USER_INPUT_FIXER_PATH = Path("user_input_fixer.txt")
#     OUTPUT_CODE_FIXED_PATH = Path("generated_analysis_fixed.py")
#
# Add command line argument for mode:
#
#     parser.add_argument(
#         "--mode", "-M",
#         type=str,
#         choices=["generate", "fix"],
#         default="generate",
#         help="Mode: 'generate' for new code, 'fix' for fixing existing code"
#     )
#
# In the main() function, modify the prompt loading section:
#
#     if args.mode == "fix":
#         system_prompt = load_text_file(Config.SYSTEM_PROMPT_FIXER_PATH)
#         human_prompt = load_text_file(Config.HUMAN_PROMPT_FIXER_PATH)
#         original_code, terminal_error, error_report = parse_fixer_input(Config.USER_INPUT_FIXER_PATH)
#         
#         prompt_template = ChatPromptTemplate.from_messages([
#             SystemMessage(content=system_prompt),
#             HumanMessage(content=human_prompt),
#         ])
#         
#         messages = prompt_template.format_messages(
#             original_code=original_code,
#             terminal_error=terminal_error,
#             error_report=error_report,
#         )
#         
#         Config.OUTPUT_CODE_PATH = Config.OUTPUT_CODE_FIXED_PATH
#     else:
#         # Original generator mode
#         system_prompt = load_text_file(Config.SYSTEM_PROMPT_PATH)
#         human_prompt = load_text_file(Config.HUMAN_PROMPT_PATH)
#         selection_cuts, plots_for_validation, output_structure = parse_user_input(Config.USER_INPUT_PATH)
#         
#         prompt_template = ChatPromptTemplate.from_messages([
#             SystemMessage(content=system_prompt),
#             HumanMessage(content=human_prompt),
#         ])
#         
#         messages = prompt_template.format_messages(
#             selection_cuts=selection_cuts,
#             plots_for_validation=plots_for_validation,
#             output_structure=output_structure,
#         )
#
# Usage:
#     python collm.py --mode generate  # Generate new analysis code
#     python collm.py --mode fix       # Fix existing code with errors
