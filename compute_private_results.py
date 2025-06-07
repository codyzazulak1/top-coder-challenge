import json, pathlib, sys, math
from predict import predict

MODULE_DIR = pathlib.Path(__file__).resolve().parent
PRIVATE_PATH = MODULE_DIR / 'private_cases.json'
OUTPUT_PATH = MODULE_DIR / 'private_results.txt'

def main():
    if not PRIVATE_PATH.exists():
        print('private_cases.json not found', file=sys.stderr)
        sys.exit(1)
    with PRIVATE_PATH.open() as f:
        cases = json.load(f)

    with OUTPUT_PATH.open('w') as out_f:
        for case in cases:
            inp = case
            td = inp['trip_duration_days']
            miles = inp['miles_traveled']
            rec = inp['total_receipts_amount']
            result = predict(td, miles, rec)
            out_f.write(f"{result}\n")
    print('Wrote', OUTPUT_PATH)

if __name__ == '__main__':
    main()
