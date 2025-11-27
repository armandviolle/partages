#!/bin/sh

if [ $# == 0 ]; then
    echo "Error: No options found." >&2
    echo "Try running 'bash $0 --help' to display usage." >&2
    exit 1
fi

while [[ $# -gt 0 ]]; do
    case $1 in
        -c|--commercial)
            # if [ -z "$2" ]; then
            #     echo "Error: -c or --commercial requires an argument." >&2
            #     exit 1
            # fi
            commercial_compatibility="$2"
            shift 2
            ;;
        -p|--push)
            # if [ -z "$2" ]; then
            #     echo "Error: -p or --push requires an argument." >&2
            #     exit 1
            # fi
            push_to_hub="$2"
            shift 2
            ;;
        -h|--help)
            cat << EOF
Usage: $0 [OPTIONS]

Options:
    -c, --commercial    Input file
    -f, --push          Output file
    -v, --verbose        Enable verbose mode
    -h, --help           Show this help message
EOF
            exit 0
            ;;
        *)
            echo "Error: Unknown option --$1." >&2
            echo "Run 'bash $0 --help' to display usage" >&2
            exit 1
            ;;
    esac
done

if [ -z $commercial_compatibility ]; then 
    echo "Error: Option -c, --commercial requires an argument." >&2
    exit 1
fi 
if [ -z $push_to_hub ]; then 
    echo "Error: Option -p, --push requires an argument." >&2
    exit 1
fi

python main.py \
    --hf_token config/hf-token.txt \
    --use_all_sources True \
    --make_commercial_version $commercial_compatibility \
    --push_to_hub $push_to_hub \
    --log_level INFO \
    # --adaptation_type instruction-tuning \
    # --source