#!/bin/bash

# Loop over every .pt file in the current directory
for file in *.pt; do
    # Check if the file exists
    if [ -f "$file" ]; then
        # Execute the Python script with the current .pt file as an argument
        echo "Processing $file..."
        python fix_checkpoints.py "$file"
    else
        echo "No .pt files found in the current directory."
    fi
done
