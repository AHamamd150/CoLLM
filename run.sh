#!/bin/bash

# Default values
INPUT_FILE=""
RUN_TUI=false

# Function to display usage
usage() {
    echo "Usage: ./run.sh --run_TUI --input <path_to_input_file>"
    echo ""
    echo "Options:"
    echo "  --run_TUI       Run the TUI mode"
    echo "  --input FILE    Path to the input file"
    echo "  --help          Display this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --run_TUI)
            RUN_TUI=true
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

# Check if required arguments are provided
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
else
    echo "Error: --run_TUI flag is required"
    usage
fi
