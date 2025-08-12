#!/bin/bash
DATE=$(date +"%Y%m%d_%H%M%S")
export MY_TIMESTAMP=$DATE
mkdir -p output/0_output_logs
nohup python3 -m main.main > output/0_output_logs/log_${DATE}.out 2>&1 &
