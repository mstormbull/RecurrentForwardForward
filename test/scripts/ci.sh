#!/bin/bash
set -e


# Execute the Python command and capture its exit status
export WANDB_MODE=dryrun
python -m RecurrentFF.benchmarks.mnist.mnist --config-file ./test/config-files/ci.toml > log.txt 2>&1 || python_exit_code=$?

# If the Python command failed, display the log and exit
if [ -n "$python_exit_code" ]; then
    cat log.txt
    exit $python_exit_code
fi

cat log.txt

# Extract the accuracy from the log
accuracy=$(grep "Test accuracy" log.txt | awk -F': ' '{print $NF}')
accuracy=${accuracy%\%}  # Remove the % sign from the accuracy
echo "Test Accuracy: $accuracy%"

# Compare the accuracy to the threshold
threshold=55
if (( $(echo "$accuracy >= $threshold" | bc -l) )); then
    echo "Test passed. Accuracy ($accuracy%) is above the threshold ($threshold%)."
else
    echo "Test failed. Accuracy ($accuracy%) is below the threshold ($threshold%)."
    exit 1
fi