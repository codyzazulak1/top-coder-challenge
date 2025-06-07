#!/bin/bash

# Black Box Challenge - Results Generation Script
# This script runs your implementation against test cases and outputs results to private_results.txt

set -e

echo "🧾 Black Box Challenge - Generating Private Results"
echo "===================================================="
echo

# Check if jq is available
if ! command -v jq &> /dev/null; then
    echo "❌ Error: jq is required but not installed!"
    echo "Please install jq to parse JSON files:"
    echo "  macOS: brew install jq"
    echo "  Ubuntu/Debian: sudo apt-get install jq"
    echo "  CentOS/RHEL: sudo yum install jq"
    exit 1
fi

# Check if run.sh exists
if [ ! -f "run.sh" ]; then
    echo "❌ Error: run.sh not found!"
    echo "Please create a run.sh script that takes three parameters:"
    echo "  ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>"
    echo "  and outputs the reimbursement amount"
    exit 1
fi

# Make run.sh executable
chmod +x run.sh

# Check if private cases exist
if [ ! -f "private_cases.json" ]; then
    echo "❌ Error: private_cases.json not found!"
    echo "Please ensure the private cases file is in the current directory."
    exit 1
fi

echo "📊 Processing test cases and generating results..."
echo "📝 Output will be saved to private_results.txt"
echo

# Extract all test data upfront in a single jq call for better performance
echo "Extracting test data..."
test_data=$(jq -r '.[] | "\(.trip_duration_days):\(.miles_traveled):\(.total_receipts_amount)"' private_cases.json)

# Convert to arrays for faster access (compatible with bash 3.2+)
test_cases=()
while IFS= read -r line; do
    test_cases+=("$line")
done <<< "$test_data"
total_cases=${#test_cases[@]}

# Remove existing results file if it exists
rm -f private_results.txt

echo "Processing $total_cases test cases..." >&2

# Process each test case
for ((i=0; i<total_cases; i++)); do
    if [ $((i % 100)) -eq 0 ] && [ $i -gt 0 ]; then
        echo "Progress: $i/$total_cases cases processed..." >&2
    fi
    
    # Extract test case data from pre-loaded array
    IFS=':' read -r trip_duration miles_traveled receipts_amount <<< "${test_cases[i]}"
    
    # Run the user's implementation
    if script_output=$(./run.sh "$trip_duration" "$miles_traveled" "$receipts_amount" 2>/dev/null); then
        # Check if output is a valid number
        output=$(echo "$script_output" | tr -d '[:space:]')
        if [[ $output =~ ^-?[0-9]+\.?[0-9]*$ ]]; then
            echo "$output" >> private_results.txt
        else
            echo "Error on case $((i+1)): Invalid output format: $output" >&2
            echo "ERROR" >> private_results.txt
        fi
    else
        # Capture stderr for error reporting
        error_msg=$(./run.sh "$trip_duration" "$miles_traveled" "$receipts_amount" 2>&1 >/dev/null | tr -d '\n')
        echo "Error on case $((i+1)): Script failed: $error_msg" >&2
        echo "ERROR" >> private_results.txt
    fi
done

echo
echo "✅ Results generated successfully!" >&2
echo "📄 Output saved to private_results.txt" >&2
echo "📊 Each line contains the result for the corresponding test case in private_cases.json" >&2

echo
echo "🎯 Next steps:"
echo "  1. Check private_results.txt - it should contain one result per line"
echo "  2. Each line corresponds to the same-numbered test case in private_cases.json"
echo "  3. Lines with 'ERROR' indicate cases where your script failed"
echo "  4. Submit your private_results.txt file when ready!"
echo
echo "📈 File format:"
echo "  Line 1: Result for private_cases.json[0]"
echo "  Line 2: Result for private_cases.json[1]" 
echo "  Line 3: Result for private_cases.json[2]"
echo "  ..."
echo "  Line N: Result for private_cases.json[N-1]" 