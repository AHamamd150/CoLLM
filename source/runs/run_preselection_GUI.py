from source.LLM.collm_lhco import generate_lhco_code
from source.LLM.pyfixer import fix_code
from source.utils.requirements_check import ensure_packages 
from source.utils.read_configs import read_conf 
import subprocess
import sys
from pathlib import Path
import os
import shutil
import warnings
warnings.filterwarnings("ignore", message="To copy construct from a tensor")
##=============================
# Configs 
##=============================

config_path = None
if len(sys.argv) > 1:
    config_path = sys.argv[1]

def recreate_dir(path):
    path = Path(path)
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)    


def run_and_capture_error(script_path: str, *args, output_dir: str = None) -> str:
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


def run_LLM(output_dir: str, DEFAULT_MODEL: str, input_test: str, input_user: str, output_code: str, MAX_RETRIES: int, use_api: bool, api_key: str):
    
    ##==========================================
    # suppress the plt.show() if exist in the generated code
    ##==========================================
    env = os.environ.copy()
    env['MPLBACKEND'] = 'Agg'
    ##================================
    ensure_packages()
    
    output_code = output_dir + "generated_lhco_analysis.py"
    recreate_dir(output_dir)
    
    generate_lhco_code(user_input_text = input_user,output_path= output_code, model_id= DEFAULT_MODEL, use_api=use_api, api_key=api_key)
    error = run_and_capture_error(output_code, input_test, output_dir=output_dir)
    retries = 0

    while error and retries < MAX_RETRIES:
        print(f"Trial {retries + 1}/{MAX_RETRIES}")
    
        # Try to fix the code
        print("  Attempting to fix...")
        code = Path(output_code).read_text()
        fixed = fix_code(code, error, DEFAULT_MODEL, use_api=use_api, api_key=api_key)
        Path(output_code).write_text(fixed)
        error = run_and_capture_error(output_code, input_test, output_dir=output_dir)
    
        if not error:
            break  
       
        #  If fix failed, regenerate from scratch
        print("  Fix failed, regenerating...")
        generate_lhco_code(user_input_text = input_user,output_path= output_code, model_id= DEFAULT_MODEL, use_api=use_api, api_key=api_key)
        error = run_and_capture_error(output_code, input_test, output_dir=output_dir)
    
        if not error:
            break  
       
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
        subprocess.run([sys.executable, output_code, input_test], cwd=output_dir, env=env)

