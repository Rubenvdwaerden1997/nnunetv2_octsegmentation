#!/bin/bash

# Navigate to the directory containing the Python script
cd "/mnt/c/Users/z923198/Documents/Work/Learning_Rick_to_use_a_PC/Codes/nnunetv2/Pipeline" || exit 1

# Print the hostname (local machine)
echo "Running on machine: $(hostname)"

# Run the Docker container and execute the Python script inside it
docker run --rm \
  -v "/mnt/c/Users/z923198/Documents/Work/Learning_Rick_to_use_a_PC/Codes/nnunetv2:/workspace" \
  -w "/workspace/Pipeline" \
  rubenvdw97/nnunetv2:1.3 \
  python3 Pipeline_withoutinternalsaving.py \
  --data_input value1 \
  --data_output value2 \
  --k 3 \
  --radius 352 \
  --preprocessing \
  --predict