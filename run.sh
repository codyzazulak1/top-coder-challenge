#!/bin/bash

# Black Box Challenge - kNN baseline implementation
# Usage: ./run.sh <trip_duration_days> <miles_traveled> <total_receipts_amount>

set -euo pipefail

python3 "$(dirname "$0")/predict.py" "$1" "$2" "$3"
