#!/bin/bash

# Default values
INPUT_FILE=""
RUN_TUI=false
RUN_GUI=false
RUN_MLP=false
RUN_GNN=false
RUN_Transformer=false

usage() {
    echo "Usage:"
    echo "  ./run.sh --run_TUI --input <path_to_input_file>"
    echo "  ./run.sh --run_GUI"
    echo "  ./run.sh --run_MLP --input <path_to_config_yaml>"
    echo "  ./run.sh --run_GNN --input <path_to_config_yaml>"
    echo "  ./run.sh --run_Transformer --input <path_to_config_yaml>"
    echo ""
    echo "Options:"
    echo "  --run_TUI       Run the TUI mode"
    echo "  --run_GUI       Run the Streamlit GUI mode"
    echo "  --run_MLP       Run the MLP deep learning mode"
    echo "  --run_GNN       Run the GNN deep learning mode"
    echo "  --run_Transformer       Run the Transformer deep learning mode"
    echo "  --input FILE    Path to the input file (required for TUI, MLP, and GNN modes)"
    echo "  --help          Display this help message"
    exit 1
}

# Function to setup Streamlit config (suppress email prompt but allow browser to open)
setup_streamlit() {
    mkdir -p ~/.streamlit
    
    # Config to disable usage stats but allow browser to open
    cat > ~/.streamlit/config.toml << TOML
[browser]
gatherUsageStats = false

[server]
headless = false
TOML

    # Credentials file to skip email prompt
    cat > ~/.streamlit/credentials.toml << TOML
[general]
email = ""
TOML
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run_TUI)
            RUN_TUI=true
            shift
            ;;
        --run_GUI)
            RUN_GUI=true
            shift
            ;;
        --run_MLP)
            RUN_MLP=true
            shift
            ;;
        --run_GNN)
            RUN_GNN=true
            shift
            ;;
        --run_Transformer)
            RUN_Transformer=true
            shift
            ;;    
        --input)
            INPUT_FILE="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Run GUI mode
if [ "$RUN_GUI" = true ]; then
    echo "Checking requirements..."
    #python -c "from source.utils.requirements_check import ensure_packages; ensure_packages()"
    python source/utils/run_checker.py
    if [ $? -ne 0 ]; then
        echo "Error: Requirements check failed"
        exit 1
    fi
    echo "Starting CoLLM GUI..."
    setup_streamlit
    streamlit run source/GUI/main.py --browser.gatherUsageStats=false
    exit 0
fi

# Run TUI mode
if [ "$RUN_TUI" = true ]; then
    echo "Checking requirements..."
    #python -c "from source.utils.requirements_check import ensure_packages; ensure_packages()"
    python source/utils/run_checker.py
    if [ -z "$INPUT_FILE" ]; then
        echo "Error: --input is required when using --run_TUI"
        usage
    fi
    
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Error: Input file '$INPUT_FILE' not found"
        exit 1
    fi
    
    python -m source.runs.run_preselection "$INPUT_FILE"
    exit 0
fi

# Run MLP mode
if [ "$RUN_MLP" = true ]; then
    echo "Checking requirements..."
    #python -c "from source.utils.requirements_check import ensure_packages; ensure_packages()"
    python source/utils/run_checker.py
    if [ -z "$INPUT_FILE" ]; then
        echo "Error: --input is required when using --run_MLP"
        usage
    fi

    if [ ! -f "$INPUT_FILE" ]; then
        echo "Error: Configuration file '$INPUT_FILE' not found"
        exit 1
    fi

    echo "Starting MLP training with config: $INPUT_FILE"
    python -m source.DL.MLP.main_mlp --config "$INPUT_FILE"
    exit 0
fi

# Run GNN mode
if [ "$RUN_GNN" = true ]; then
    echo "Checking requirements..."
    #python -c "from source.utils.requirements_check import ensure_packages; ensure_packages()"
    python source/utils/run_checker.py
    if [ -z "$INPUT_FILE" ]; then
        echo "Error: --input is required when using --run_GNN"
        usage
    fi

    if [ ! -f "$INPUT_FILE" ]; then
        echo "Error: Configuration file '$INPUT_FILE' not found"
        exit 1
    fi

    echo "Starting GNN training with config: $INPUT_FILE"
    python -m source.DL.GNN.main_gnn --config "$INPUT_FILE"
    exit 0
fi

# Run Transformer mode
if [ "$RUN_Transformer" = true ]; then
    echo "Checking requirements..."
    #python -c "from source.utils.requirements_check import ensure_packages; ensure_packages()"
    python source/utils/run_checker.py
    if [ -z "$INPUT_FILE" ]; then
        echo "Error: --input is required when using --run_Transformer"
        usage
    fi

    if [ ! -f "$INPUT_FILE" ]; then
        echo "Error: Configuration file '$INPUT_FILE' not found"
        exit 1
    fi

    echo "Starting Transformer training with config: $INPUT_FILE"
    python -m source.DL.Transformer.main_transformer --config "$INPUT_FILE"
    exit 0
fi

# No mode selected
echo "Error: --run_TUI, --run_GUI, --run_MLP, or --run_GNN flag is required"
usage
