#!/bin/bash

# Default values
INPUT_FILE=""
RUN_TUI=false
RUN_GUI=false

# Function to display usage
usage() {
    echo "Usage:"
    echo "  ./run.sh --run_TUI --input <path_to_input_file>"
    echo "  ./run.sh --run_GUI"
    echo ""
    echo "Options:"
    echo "  --run_TUI       Run the TUI mode"
    echo "  --run_GUI       Run the Streamlit GUI mode"
    echo "  --input FILE    Path to the input file (required for TUI mode)"
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
    python -c "from source.utils.requirements_check import ensure_packages; ensure_packages()"
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
    if [ -z "$INPUT_FILE" ]; then
        echo "Error: --input is required when using --run_TUI"
        usage
    fi
    
    # Check if input file exists
    if [ ! -f "$INPUT_FILE" ]; then
        echo "Error: Input file '$INPUT_FILE' not found"
        exit 1
    fi
    
    # Execute the command
    python -m source.runs.run_preselection "$INPUT_FILE"
    exit 0
fi

# No mode selected
echo "Error: --run_TUI or --run_GUI flag is required"
usage
