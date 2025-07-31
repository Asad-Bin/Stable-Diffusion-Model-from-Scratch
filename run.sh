#!/bin/bash

# Create log directory if it doesn't exist
mkdir -p output/0_output_logs

# Generate timestamp
timestamp=$(date +"%Y%m%d_%H%M%S")
log_file="output/0_output_logs/log_${timestamp}.out"
export MY_TIMESTAMP="$timestamp"

# Run your model and let it create the output directory
nohup python3 -u -m main.main > "$log_file" 2>&1 &

wait

echo "Log file is: $log_file"