#!/bin/bash

#SBATCH --job-name=perf_test
#SBATCH --output=perf_test.out
#SBATCH --error=perf_test.err
#SBATCH --cpus-per-task=64
#SBATCH --partition=cits3402
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<your_student_number>@student.uwa.edu.au

# Generate a 1000000 * 1000000 matrix
make

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

echo "Generating 1,000,000 x 1,000,000 matrix to $OUTPUT_DIR/matrix_1000000.txt"
echo "Estimated file size: ~3.7TB"

./build/conv_test -H 1000000 -W 1000000 -G -o "$OUTPUT_DIR/matrix_1000000.txt" 
