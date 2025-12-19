# Delphes Analysis Code Generator (LLM + Validators)

![Workflow overview](workflow.jpg)

This project generates **single-file, executable Python analysis scripts** for **Delphes ROOT** outputs (TTree `"Delphes"`), based on a user-specified set of **selection cuts**, **validation plots**, and a requested **cutflow/output structure**. The goal is to automate repetitive “analysis boilerplate” while enforcing **deterministic safety checks** before writing the script to disk.

The overall pipeline is shown in the workflow above: prompt inputs feed an LLM-backed orchestrator, code is generated, optionally repaired, validated, and then written as `generated_analysis.py`. The diagram also includes an optional ML loop; this repository currently focuses on the **analysis generation + validation** path, but the architecture is compatible with downstream extensions.


## What this repository is

### Core idea
1. You describe **physics cuts**, **plots**, and **how you want the cutflow printed** in `user_input.txt`.
2. `collm.py` loads your prompts, builds an LLM request, generates Python code, and validates it.
3. Validation rejects scripts with common failure modes (syntax errors, unresolved placeholders like `{branch}`, missing `plt.savefig(...)`, missing `vector.register_awkward()`, etc.).
4. On success, the script is saved to `generated_analysis.py` and can be executed on Delphes ROOT files.

## Repository layout

- `collm.py` — Orchestrator: loads prompts + user input, runs the HuggingFace LLM (either via LangChain or direct `generate()`), retries/repairs, validates, and saves output.
- `validators.py` — Static validation gate: AST parse + “required patterns” + “forbidden patterns” + placeholder detection + plot-save enforcement.
- `schemas.py` — Pydantic schema for structured user input sections (selection cuts, plots, output structure). :contentReference[oaicite:5]{index=5}  
- `system_prompt.txt` — Global “contract” for what the generated analysis script must do (vectorization rules, required startup code, plotting rules, etc.).  
- `human_prompt.txt` — Short template that injects `{selection_cuts}`, `{plots_for_validation}`, `{output_structure}` into the model request.
- `user_input.txt` — Your analysis specification with the three required sections.
- `generated_analysis.py` — Output script (produced by `collm.py`).

## Requirements

### A) Generator (this repo)
`collm.py` uses:
- `transformers`, `torch`
- `langchain` + HuggingFace integration
- `pydantic`
- `tqdm`, `matplotlib`
- optional: `bitsandbytes` for 4-bit quantization (NF4)

> Note: `collm.py` currently performs dependency checks (and in some setups may attempt to install missing packages). This is separate from the **generated analysis script**, which must not install anything.

### B) Generated analysis script (what gets produced)
The generated analysis script is constrained by the system prompt to use only the Scikit-HEP “columnar” stack + plotting:
- `uproot`, `awkward`, `vector`, `numpy`, `matplotlib`
- plus a small set of standard libs (`sys`, `json`, `pathlib`, `typing`)

It is also required to register vector behavior immediately:
- `vector.register_awkward()`


## Quickstart

### 1) Edit your analysis spec
Modify `user_input.txt` with the three required sections:

- `SELECTION_CUTS` — physics/object/event selections
- `PLOTS_FOR_VALIDATION` — which histograms to save
- `OUTPUT_STRUCTURE` — how to print cutflow + final counts

### 2) Run the generator
Typical usage (works with your current style of invoking from IPython):

```bash
ipython collm.py -- --use_generate --quant 4bit --max_new_token 4096
```

`collm.py` is the orchestration script that:

- loads `system_prompt.txt`, `human_prompt.txt`, `user_input.txt`
- validates `user_input.txt` into an `AnalysisSpec` (Pydantic)
- loads the Hugging Face model + tokenizer
- generates Python code
- validates the generated code
- saves the final output to `generated_analysis.py`


## Run the generated analysis on a ROOT file

Once code generation passes validation:

```bash
python generated_analysis.py path/to/input.root
```

Your script should:

- open the ROOT file
- load required branches via `tree.arrays(..., library="ak")`
- apply cuts in order (object-level → multiplicity → event-level)
- save plots with `plt.savefig(..., dpi=150)` and `plt.close()`
- print a cutflow summary and final requested counts

## How the workflow works

The figure above summarizes this pipeline:

### 1) Inputs
- ROOT files + cross section info + your `user_input.txt`
- optional prompt tuning knobs (edit prompts, change model)

### 2) Orchestrator (LangChain / collm.py)
- Builds prompt messages:
  - `SystemMessage(system_prompt)`
  - `HumanMessage(human_prompt.format(...))`

- Two inference modes:
  - **Direct `generate()` path** (`--use_generate`), applying the model’s chat template (important for instruct models).
  - **LangChain pipeline path** (when not using `--use_generate`).

### 3) Validation gate (`validators.py`)
The validator is intentionally strict to prevent “runs but wrong physics” situations. It checks, among other things:

- Python parses (AST syntax check)
- No unresolved placeholders like `{branch}` / `{candidate}` in the output
- Required calls/patterns exist (e.g., `vector.register_awkward()`)
- At least one plot is saved via `plt.savefig(...)`

### 4) Repair / Regenerate loop
When validation fails, `collm.py` can:

- feed the validation errors back into the prompt for another attempt
- optionally save a `.partial.py` for debugging (if configured)

### 5) Output
- On success, you get a concrete `generated_analysis.py` that follows the startup contract and produces cutflow + plots.

### Prompts and "contracts"
`system_prompt.txt`

This is the hard constraint layer. It specifies mandatory behaviors like:
- required imports + immediate `vector.register_awkward()`
- no Python event loops (must use awkward masks / vector ops)
- plotting rules (save PNGs, dpi=150, close figures)
- branch-name robustness via a `find_branch()` helper

`human_prompt.txt`  
This is the structured injection layer that passes your three sections into the LLM call.

`user_input.txt`  
This is the analysis specification; you modify this file to benchmark or run different analyses.  

## Troubleshooting

### Validation fails with “Syntax error at line 1”
This typically means the model returned non-code content or a malformed header. The generator enforces a strict output contract (e.g., “first line MUST be: `import sys`” in some modes).

**Actions:**
- Increase `--max_new_token` so the model can complete the full script.
- Prefer `--use_generate` for instruct models (uses `apply_chat_template`).

---

### Validation fails with “Unresolved template placeholders found: {branch}, {candidate}”
The model is emitting template-style placeholders instead of concrete Python identifiers. Validators treat these as hard failures.

**Actions:**
- Strengthen the system prompt section that bans placeholders.
- Add a short “NEVER use `{...}` outside f-strings” reminder to the human prompt.
- Enable retry/repair loop (already present in `collm.py`).

---

### Validation fails with “Missing required plot save call: *.savefig(...)”
Your prompt asked for plots, but the model did not save them. Validators enforce that at least one `savefig(...)` exists.

**Actions:**
- Make plot requirements explicit in `PLOTS_FOR_VALIDATION`.
- Add a “must call `plt.savefig(..., dpi=150)` and `plt.close()` for every plot” line (often already included in prompts).