import sys
import subprocess
import logging

# Set up logger
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

#==============================
# Core packages always required
#==============================
def _ensure_compatible_numpy():
    """Check NumPy version and downgrade if 2.x is installed.
    
    NumPy 2.x has breaking changes that cause compatibility issues with
    packages compiled against NumPy 1.x (like PyTorch and transformers).
    """
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import numpy; print(numpy.__version__)"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            version = result.stdout.strip()
            major_version = int(version.split(".")[0])
            
            if major_version >= 2:
                logger.warning(f"NumPy {version} detected. Downgrading to NumPy<2 for compatibility...")
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", "numpy<2",
                        "--force-reinstall", "--quiet"
                    ], stdout=subprocess.DEVNULL)
                    logger.info("Successfully downgraded NumPy")
                    logger.warning("=" * 50)
                    logger.warning("NumPy was downgraded. Please RESTART Python!")
                    logger.warning(" Run your script again after restarting.")
                    logger.warning("=" * 50)
                    sys.exit(0)
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to downgrade NumPy: {e}")
                    raise
    except FileNotFoundError:
        pass  # NumPy not installed yet, will be installed later
    except Exception:
        pass  # Ignore other errors, numpy check in packages dict will handle it


def ensure_packages():
    """Install required packages if not already installed."""
    import sys
    import subprocess
    
    # First, check if NumPy 2.x is installed and downgrade if needed
    # This must happen before importing packages that depend on NumPy
    _ensure_compatible_numpy()
    
    packages = {
        # Core dependencies
        # Pin numpy<2 for compatibility with PyTorch/transformers compiled against NumPy 1.x
        "numpy":                 "numpy<2",
        "pandas":                "pandas", 
        "matplotlib":            "matplotlib",
        "tqdm":                  "tqdm",
        "yaml":                  "pyyaml",
        
        # LangChain / LLM stack (pinned for compatibility)
        # IMPORTANT: huggingface-hub must be <1.0.0 for langchain-huggingface
        "huggingface_hub":       "huggingface-hub>=0.33.4,<1.0.0",
        "transformers":          "transformers>=4.45.0,<4.52.0",
        "accelerate":            "accelerate>=0.26.0,<1.0.0",
        "langchain":             "langchain>=0.3.0",
        "langchain_huggingface": "langchain-huggingface>=0.1.0,<1.3.0",
        "pydantic":              "pydantic>=2.0.0",
        
        # UI
        "streamlit":             "streamlit",
    }
    
    # Add typing_extensions for Python < 3.9
    if sys.version_info < (3, 9):
        packages["typing_extensions"] = "typing_extensions"
    
    # Install regular packages first
    for module, pip_name in packages.items():
        # Check if package has version constraints
        has_version_constraint = any(c in pip_name for c in ['<', '>', '=', '!'])
        
        if has_version_constraint:
            # For packages with version constraints, always run pip install with --force-reinstall
            # This ensures the correct version is installed even if an incompatible version exists
            logger.info(f"Installing/checking {pip_name}...")
            try:
                # First try without force to see if already satisfied
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", pip_name],
                    capture_output=True, text=True
                )
                output = result.stdout.lower() + result.stderr.lower()
                
                if "already satisfied" in output and "incompatible" not in output:
                    logger.info(f"{module} already installed with compatible version")
                elif result.returncode == 0:
                    logger.info(f"Successfully installed {pip_name}")
                else:
                    # If regular install fails, try force reinstall
                    logger.warning(f"Forcing reinstall of {pip_name}...")
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", pip_name, "--force-reinstall"
                    ], stdout=subprocess.DEVNULL)
                    logger.info(f"Successfully reinstalled {pip_name}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to install {pip_name}: {e}")
                raise
        else:
            # For packages without version constraints, just check if importable
            try:
                __import__(module)
                logger.info(f"{module} already installed")
            except ImportError:
                logger.warning(f"Installing {pip_name}...")
                try:
                    subprocess.check_call([
                        sys.executable, "-m", "pip", "install", pip_name
                    ], stdout=subprocess.DEVNULL)
                    logger.info(f"Successfully installed {pip_name}")
                except subprocess.CalledProcessError as e:
                    logger.error(f"Failed to install {pip_name}: {e}")
                    raise
    
    # Handle PyTorch separately for MPS support
    _ensure_pytorch()


def _get_macos_version():
    """Get macOS major and minor version numbers."""
    import platform
    
    mac_ver = platform.mac_ver()[0]
    if mac_ver:
        parts = mac_ver.split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return major, minor
    return None, None


def _get_pytorch_version_for_macos():
    """Determine the appropriate PyTorch version based on macOS version."""
    import platform
    
    system = platform.system()
    if system != "Darwin":
        return None  # Not macOS, use latest
    
    major, minor = _get_macos_version()
    if major is None:
        return None  # Can't determine version, use latest
    
    # PyTorch version requirements for MPS:
    # - PyTorch 2.4+ requires macOS 14.0+ (Sonoma)
    # - PyTorch 2.2.x - 2.3.x supports macOS 12.3+ (Monterey/Ventura)
    
    if major >= 14:
        return None  # Use latest PyTorch
    elif major == 13 or (major == 12 and minor >= 3):
        # macOS 12.3 - 13.x: Use PyTorch 2.2.2
        return {
            "torch": "torch==2.2.2",
            "torchvision": "torchvision==0.17.2",
            "torchaudio": "torchaudio==2.2.2"
        }
    elif major == 12 and minor < 3:
        logger.warning("macOS 12.3+ required for MPS support")
        return None
    else:
        logger.warning(f"macOS {major}.{minor} is too old for MPS support")
        return None


def _get_installed_torch_version():
    """Get installed torch version without importing torch."""
    import subprocess
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print(torch.__version__)"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _check_mps_available():
    """Check if MPS is available without keeping torch imported."""
    import subprocess
    
    try:
        result = subprocess.run(
            [sys.executable, "-c", 
             "import torch; print(torch.backends.mps.is_built(), torch.backends.mps.is_available())"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split()
            mps_built = parts[0] == "True"
            mps_available = parts[1] == "True"
            return mps_built, mps_available
    except Exception:
        pass
    return False, False


def _install_pytorch(reason: str, version_dict: dict = None):
    """Install or reinstall PyTorch."""
    import platform
    
    logger.warning(f"Installing PyTorch ({reason})...")
    
    system = platform.system()
    
    try:
        # Uninstall existing torch first to avoid conflicts
        subprocess.check_call([
            sys.executable, "-m", "pip", "uninstall", "-y",
            "torch", "torchvision", "torchaudio"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        pass  # May not be installed
    
    try:
        if system == "Darwin":  # macOS
            if version_dict:
                # Install specific versions for older macOS
                major, minor = _get_macos_version()
                logger.info(f"Installing PyTorch {version_dict['torch']} for macOS {major}.{minor}")
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    version_dict["torch"],
                    version_dict["torchvision"],
                    version_dict["torchaudio"]
                ], stdout=subprocess.DEVNULL)
            else:
                # Install latest for macOS 14+
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install",
                    "torch", "torchvision", "torchaudio"
                ], stdout=subprocess.DEVNULL)
            logger.info("Successfully installed PyTorch for macOS")
            
        elif system == "Linux":
            # For Linux with CUDA support
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio",
                "--index-url", "https://download.pytorch.org/whl/cu121"
            ], stdout=subprocess.DEVNULL)
            logger.info("Successfully installed PyTorch with CUDA support")
            
        else:  # Windows or other
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "torch", "torchvision", "torchaudio"
            ], stdout=subprocess.DEVNULL)
            logger.info("Successfully installed PyTorch")
        
        return True
            
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to install PyTorch: {e}")
        raise


def _ensure_pytorch():
    """Install PyTorch with proper GPU support (CUDA or MPS)."""
    import platform
    
    system = platform.system()
    machine = platform.machine()
    
    # Get appropriate PyTorch version for macOS
    pytorch_versions = _get_pytorch_version_for_macos()
    
    # Check if torch is installed (without importing)
    installed_version = _get_installed_torch_version()
    pytorch_reinstalled = False
    
    if installed_version is None:
        # Not installed
        _install_pytorch("not installed", pytorch_versions)
        pytorch_reinstalled = True
        installed_version = _get_installed_torch_version()
    
    logger.info(f"PyTorch {installed_version} installed")
    
    # Check MPS/CUDA support
    if system == "Darwin" and machine == "arm64":
        # Apple Silicon Mac - should have MPS
        major, minor = _get_macos_version()
        mps_built, mps_available = _check_mps_available()
        
        if not mps_built:
            logger.warning("PyTorch installed without MPS support, reinstalling...")
            _install_pytorch("MPS not built", pytorch_versions)
            pytorch_reinstalled = True
            mps_built, mps_available = _check_mps_available()
            
        elif not mps_available and major is not None and major < 14:
            # MPS is built but not available - likely version mismatch
            if installed_version and _is_pytorch_too_new(installed_version):
                logger.warning(f"PyTorch {installed_version} requires macOS 14+, but you have macOS {major}.{minor}")
                logger.warning("Downgrading to compatible PyTorch version...")
                _install_pytorch("macOS version compatibility", pytorch_versions)
                pytorch_reinstalled = True
                mps_built, mps_available = _check_mps_available()
        
        # Final status check
        if mps_available:
            logger.info("MPS support available")
        elif mps_built:
            logger.warning("MPS built but not available")
            _print_mps_diagnostics()
        else:
            logger.warning("MPS not available")
            _print_mps_diagnostics()
        
        # If we reinstalled, warn user about restart
        if pytorch_reinstalled:
            logger.warning("=" * 50)
            logger.warning("PyTorch was reinstalled. Please RESTART Python!")
            logger.warning(" Run your script again after restarting.")
            logger.warning("=" * 50)
            sys.exit(0)
                
    elif system == "Linux" or (system == "Darwin" and machine == "x86_64"):
        # Linux or Intel Mac - check for CUDA
        try:
            result = subprocess.run(
                [sys.executable, "-c", 
                 "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(maxsplit=1)
                cuda_available = parts[0] == "True"
                if cuda_available:
                    device_name = parts[1] if len(parts) > 1 else "GPU"
                    logger.info(f"CUDA support available ({device_name})")
                else:
                    logger.info("CUDA not available, using CPU")
        except Exception:
            logger.info("CUDA not available, using CPU")


def _is_pytorch_too_new(version: str) -> bool:
    """Check if PyTorch version requires macOS 14+."""
    try:
        parts = version.split(".")
        major = int(parts[0])
        minor = int(parts[1])
        # PyTorch 2.4+ requires macOS 14+
        return major >= 2 and minor >= 4
    except Exception:
        return False


def _print_mps_diagnostics():
    """Print diagnostic information for MPS issues."""
    import platform
    
    installed_version = _get_installed_torch_version()
    mps_built, mps_available = _check_mps_available()
    
    logger.info("=" * 40)
    logger.info("MPS Diagnostics")
    logger.info("=" * 40)
    logger.info(f"macOS version: {platform.mac_ver()[0]}")
    logger.info(f"Chip: {platform.processor()}")
    logger.info(f"Architecture: {platform.machine()}")
    logger.info(f"PyTorch version: {installed_version}")
    logger.info(f"MPS built: {mps_built}")
    logger.info(f"MPS available: {mps_available}")
    logger.info("=" * 40)
    
    major, minor = _get_macos_version()
    
    if major is not None:
        if major < 12 or (major == 12 and minor < 3):
            logger.error("macOS 12.3+ required for MPS. Please upgrade macOS.")
        elif major < 14:
            logger.info(f"macOS {major}.{minor} detected. Requires PyTorch 2.2.x - 2.3.x for MPS.")
            logger.info("For latest PyTorch, upgrade to macOS 14.0+ (Sonoma)")
    
    if platform.machine() != "arm64":
        logger.warning("MPS works best on Apple Silicon (M1/M2/M3/M4)")
