from ..LLM.collm_lhco import generate_lhco_code
from ..LLM.pyfixer import fix_code
from ..utils.requirements_check import ensure_packages 
from ..utils.read_configs import read_conf 
import subprocess
import sys
from pathlib import Path
import os
import shutil
import warnings
warnings.filterwarnings("ignore", message="To copy construct from a tensor")
##=============================##
##=============================##



if len(sys.argv) > 1:
    config_path = sys.argv[1]
    
output_dir, DEFAULT_MODEL,MAX_RETRIES,input_test,input_user = read_conf(config_path)
''' 
This is the main code that runs the LLM analysis to generate the analysis code.

*** Do we need to separate the analysis validation from the actual run over the signal and background?
I think at the moment, we can ask the user to run the given script over  the signal and background events.
'''
'''
QualityNotesQwen/Qwen2.5-Coder-32B-Instruct       32B     ~48GB    ⭐⭐⭐⭐⭐      Best Qwen, needs large GPU 
deepseek-ai/DeepSeek-Coder-V2-Instruct            236B    ~40GB    ⭐⭐⭐⭐⭐      Often best for code
deepseek-ai/deepseek-coder-33b-instruct           33B     ~48GB    ⭐⭐⭐⭐⭐      Excellent for fixing
Qwen/Qwen2.5-Coder-14B-Instruct                   14B     ~20GB    ⭐⭐⭐⭐        Good balance
deepseek-ai/deepseek-coder-6.7b-instruct          6.7B    ~10GB    ⭐⭐⭐⭐        Best small model
Qwen/Qwen2.5-Coder-7B-Instruct                    7B      ~10GB    ⭐⭐⭐          Decentcode
llama/CodeLlama-13b-Instruct-hf                   13B     ~18GB    ⭐⭐⭐          Older, less capable
'''
##=============================
# Configs 
##=============================
#output_dir = "/data1/hammad/CoLLM/output/"
#DEFAULT_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
#MAX_RETRIES = 3
#input_test = "/data1/hammad/collm/collm_lhco/signal_1.lhco"
#input_user = "/data1/hammad/CoLLM/templates/user_input_2.txt"
output_code = output_dir+"generated_lhco_analysis.py"

def recreate_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
recreate_dir(output_dir)    


def run_and_capture_error(script_path: str, *args) -> str:
    """Run a Python script and return any error output."""
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    result = subprocess.run(
        [sys.executable, script_path] + list(args),
        capture_output=True,
        text=True,
        cwd=output_dir,
        env=env
    )
    return result.stderr if result.returncode != 0 else ""

#=======================================
# Ensure the core packages are installed
#=======================================
ensure_packages()
#===========================

def run_LLM(output_dir: str,DEFAULT_MODEL: str, input_test: str,input_user: str, output_code:str,MAX_RETRIES: int ):
    
    ##=======================================================================
   # suppress the plt.show() if exist in the generated code
    ##=======================================================================
   env = os.environ.copy()
   env['MPLBACKEND'] = 'Agg'
    ##================================

   generate_lhco_code(input_user, output_code, DEFAULT_MODEL)
   error = run_and_capture_error(output_code, input_test)
   retries = 0

   while error and retries < MAX_RETRIES:
       print(f"Trial {retries + 1}/{MAX_RETRIES}")
    
       # Step 1: Try to fix the code
       print("  Attempting to fix...")
       code = Path(output_code).read_text()
       fixed = fix_code(code, error, DEFAULT_MODEL)
       Path(output_code).write_text(fixed)
       error = run_and_capture_error(output_code, input_test)
    
       if not error:
           break  # Fixed successfully
           # Step 2: If fix failed, regenerate from scratch
       print("  Fix failed, regenerating...")
       generate_lhco_code(input_user, output_code, DEFAULT_MODEL)
       error = run_and_capture_error(output_code, input_test)
    
       retries += 1


   if error: 
       print(''' Failed to excute the analysis code. Please check the following:  
   1- please make sure the requested plots are ensured in the selection cuts.
   2- Make sure to write the code in an understandable and clear way.
   3- You may increase the number of trials to fix the generated code.
   4- You may consider larger LLM model for complicated analysis.
    ''')
   else:
       print(f"Analysis code is generated susccesfully. Output is stored in {output_dir}")
       subprocess.run([sys.executable,output_code,input_test ],cwd=output_dir, env = env)

run_LLM(output_dir,DEFAULT_MODEL, input_test,input_user, output_code,MAX_RETRIES )
