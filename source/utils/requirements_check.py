import sys
import subprocess


import logging

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

#==============================
# Core packages always required
#==============================
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
    "streamlit": "streamlit",
    "yaml" :  "yaml" 
}

def ensure_packages():
    """Install required packages if not already installed."""
    packages = {
        "tqdm": "tqdm",
        "matplotlib": "matplotlib",
        "langchain": "langchain",
        "transformers": "transformers",
        "langchain_huggingface": "langchain-huggingface",
        "huggingface_hub": "huggingface-hub",
        "accelerate": "accelerate",
        "torch": "torch",
        "pydantic": "pydantic",
        "streamlit": "streamlit",
        "yaml" :  "pyyaml"
        }


    for module, pip_name in packages.items():
        try:
            __import__(module)
            logger.info(f" {module} already installed")
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
