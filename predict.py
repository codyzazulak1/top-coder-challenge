import json, math, os, sys, pathlib
from collections import defaultdict

# Hyper-parameters (residual kNN)
K_NEIGHBOURS = 5
SCALE_DAYS = 0.5      # distance scaling for trip duration
SCALE_MILES = 0.001   # distance scaling for miles travelled
SCALE_RECEIPTS = 0.001  # distance scaling for receipts amount

# Cofactor to avoid division by zero
_EPS = 1e-12

# Pre-compute per-day linear regression on the public data
def _fit_per_day_betas(cases):
    """Return dict: day -> (bias, coef_miles, coef_receipts)"""
    groups = defaultdict(list)
    for c in cases:
        d = c["input"]["trip_duration_days"]
        groups[d].append(c)

    betas = {}
    for d, rows in groups.items():
        # Least-squares for y = b0 + b1*miles + b2*receipts
        # Build normal equations 3x3
        XtX = [[0.0]*3 for _ in range(3)]
        Xty = [0.0]*3
        for c in rows:
            m = c["input"]["miles_traveled"]
            r = c["input"]["total_receipts_amount"]
            y = c["expected_output"]
            vec = [1.0, m, r]
            for i in range(3):
                Xty[i] += vec[i]*y
                for j in range(3):
                    XtX[i][j] += vec[i]*vec[j]

        # Solve XtX * beta = Xty with Gaussian elimination (matrix is 3x3)
        A = [row[:] for row in XtX]
        for i in range(3):
            A[i] += [1.0 if i==j else 0.0 for j in range(3)]  # append identity

        # Forward elimination / Gauss-Jordan
        for i in range(3):
            # pivot
            piv = A[i][i] or _EPS
            factor = 1.0 / piv
            for j in range(6):
                A[i][j] *= factor
            for k in range(3):
                if k == i:
                    continue
                f = A[k][i]
                for j in range(6):
                    A[k][j] -= f * A[i][j]
        inv = [row[3:] for row in A]
        beta = [sum(inv[i][j]*Xty[j] for j in range(3)) for i in range(3)]
        betas[d] = tuple(beta)
    return betas

# Load data & train artifacts
_TRAIN_CASES = []

with (pathlib.Path(__file__).resolve().parent / "public_cases.json").open() as _f:
    _RAW_DATA = json.load(_f)
    _TRAIN_CASES.extend(_RAW_DATA)

_DAY_BETAS = _fit_per_day_betas(_TRAIN_CASES)

# Compute residuals for kNN
_RESIDUAL_SET = []  # list of (input_dict, residual)
for c in _TRAIN_CASES:
    d = c["input"]["trip_duration_days"]
    b0,b1,b2 = _DAY_BETAS[d]
    baseline = b0 + b1 * c["input"]["miles_traveled"] + b2 * c["input"]["total_receipts_amount"]
    residual = c["expected_output"] - baseline
    _RESIDUAL_SET.append((c["input"], residual))

# Helper functions
def _squared_distance(a, b):
    return (
        ((a["trip_duration_days"] - b["trip_duration_days"]) * SCALE_DAYS) ** 2
        + ((a["miles_traveled"] - b["miles_traveled"]) * SCALE_MILES) ** 2
        + ((a["total_receipts_amount"] - b["total_receipts_amount"]) * SCALE_RECEIPTS) ** 2
    )

# Ridge-regression model (global, avoids over-fitting public set)
LAMBDA_RIDGE = 5e-3  # slightly stronger regularisation for even larger feature set

import math

# Helper to capture piecewise penalties/bonuses hinted in interviews
def _features(days: float, miles: float, receipts: float):
    if days <= 0:
        days = 1e-6  # guard
    rec_per_day = receipts / days
    miles_per_day = miles / days

    # Penalty buckets
    high_receipt_pen = max(0.0, rec_per_day - 200.0)
    low_receipt_pen = max(0.0, 20.0 - rec_per_day)

    # Non-linear transforms
    log_r = math.log1p(receipts)

    total_receipts_large = max(0.0, receipts - 1500.0)
    is_five_day = 1.0 if days == 5 else 0.0
    is_short = 1.0 if days <= 3 else 0.0
    is_long = 1.0 if days >= 7 else 0.0
    high_receipts_flag = 1.0 if receipts > 800 else 0.0
    low_receipts_flag  = 1.0 if receipts < 50 else 0.0
    rounding_flag = 1.0 if abs((receipts*100) % 100 - 49) < 1e-6 or abs((receipts*100) % 100 - 99) < 1e-6 else 0.0
    return [
        1.0,
        days,
        miles,
        receipts,
        days * days,
        miles * miles,
        receipts * receipts,
        days * miles,
        days * receipts,
        miles * receipts,
        miles_per_day,
        rec_per_day,
        miles_per_day * miles_per_day,
        rec_per_day * rec_per_day,
        miles_per_day * rec_per_day,
        is_five_day,
        is_short,
        is_long,
        high_receipts_flag,
        low_receipts_flag,
        rounding_flag,
        math.log1p(receipts),
        is_long * max(0.0, rec_per_day - 150.0),
        high_receipt_pen,
        low_receipt_pen,
        high_receipt_pen * high_receipt_pen,
        total_receipts_large,
        total_receipts_large * total_receipts_large,
        high_receipt_pen * miles_per_day,
    ]

_BETA = []  # learned coefficients (length auto-detected)

def _train_ridge():
    """Fit ridge regression on the public sample and populate _BETA."""
    global _BETA

    dim = len(_features(1.0, 1.0, 1.0))
    XtX = [[0.0] * dim for _ in range(dim)]
    Xty = [0.0] * dim

    for case in _RAW_DATA:
        d = case["input"]["trip_duration_days"]
        m = case["input"]["miles_traveled"]
        r = case["input"]["total_receipts_amount"]
        y = case["expected_output"]
        feats = _features(d, m, r)
        for i in range(dim):
            Xty[i] += feats[i] * y
            for j in range(dim):
                XtX[i][j] += feats[i] * feats[j]

    # L2 regularisation
    for i in range(dim):
        XtX[i][i] += LAMBDA_RIDGE

    # Invert XtX via Gauss-Jordan (dim is small – 10 × 10)
    A = [row[:] for row in XtX]
    for i in range(dim):
        A[i] += [1.0 if i == j else 0.0 for j in range(dim)]

    for i in range(dim):
        pivot = A[i][i] or 1e-12
        inv_pivot = 1.0 / pivot
        for j in range(2 * dim):
            A[i][j] *= inv_pivot
        for k in range(dim):
            if k == i:
                continue
            factor = A[k][i]
            for j in range(2 * dim):
                A[k][j] -= factor * A[i][j]

    inv = [row[dim:] for row in A]
    _BETA = [sum(inv[i][j] * Xty[j] for j in range(dim)) for i in range(dim)]

_train_ridge()

# Build bias table for coarse residual correction
def _cell(d, mpd, spd):
    return (min(int(d), 9), int(mpd // 50), int(spd // 50))

_BIAS_TABLE = {}
_TABLE_COUNTS = {}
for _c in _RAW_DATA:
    d = _c["input"]["trip_duration_days"]
    miles = _c["input"]["miles_traveled"]
    receipts = _c["input"]["total_receipts_amount"]
    mpd = miles / max(d,1e-6)
    spd = receipts / max(d,1e-6)
    cell = _cell(d, mpd, spd)
    residual = _c["expected_output"] - sum(b*f for b,f in zip(_BETA, _features(d,miles,receipts)))
    _BIAS_TABLE[cell] = _BIAS_TABLE.get(cell,0.0) + residual
    _TABLE_COUNTS[cell] = _TABLE_COUNTS.get(cell,0) + 1

for k in list(_BIAS_TABLE):
    _BIAS_TABLE[k] /= _TABLE_COUNTS[k]

# weight for bias table adjustment
TABLE_ALPHA = 0.6

# Build residual set relative to ridge baseline for kNN smoothing
_RIDGE_RES_LIST = []  # list of (input_dict, residual)
for _c in _RAW_DATA:
    d = _c["input"]["trip_duration_days"]
    m = _c["input"]["miles_traveled"]
    r = _c["input"]["total_receipts_amount"]
    baseline = sum(b * f for b, f in zip(_BETA, _features(d, m, r)))
    residual = _c["expected_output"] - baseline
    _RIDGE_RES_LIST.append((_c["input"], residual))

# kNN residual hyper-params
# residual kNN weighting dynamically determined
# Function will interpolate 0.3 (low receipts) to 0.8 (very high receipts)
# Increase k to 9 for smoother residual estimate
RES_K = 9  # smoother residual

def _alpha(receipts: float) -> float:
    """Return weight for residual adjustment based on receipts.
    Low receipts use ~0.2 weight, high receipts (~2000) use ~0.5."""
    return 0.2 + 0.3 / (1.0 + math.exp(-(receipts - 1200.0) / 400.0))

# Public API
def predict(trip_duration_days: float, miles_traveled: float, total_receipts_amount: float) -> float:
    """Return a reimbursement prediction rounded to 2 decimals."""
    # Ridge model prediction – smoother and less likely to over-fit
    days = float(trip_duration_days)
    miles = float(miles_traveled)
    receipts = float(total_receipts_amount)

    ridge_val = sum(b * f for b, f in zip(_BETA, _features(days, miles, receipts)))

    # kNN residual prediction
    dists = [(_squared_distance({"trip_duration_days": days, "miles_traveled": miles, "total_receipts_amount": receipts}, inp), res) for inp, res in _RIDGE_RES_LIST]
    dists.sort(key=lambda t: t[0])
    top = dists[:RES_K]
    if top and top[0][0] < _EPS:
        residual_pred = top[0][1]
    else:
        w = [1/(d+_EPS) for d,_ in top]
        residual_pred = sum(wi*ri for wi,(_,ri) in zip(w,top)) / sum(w)

    # bias table adjustment
    mpd = miles / max(days,1e-6)
    spd = receipts / max(days,1e-6)
    table_adj = _BIAS_TABLE.get(_cell(days, mpd, spd), 0.0)

    base_val = ridge_val + _alpha(receipts) * residual_pred + TABLE_ALPHA * table_adj

    # Floor adjustment for trips with very low receipts (legacy per-diem rule)
    rec_per_day = receipts / max(days, 1e-6)
    if rec_per_day < 50.0:
        day_rate = 100.0 if days <= 5 else 80.0
        floor_val = day_rate * days + 0.5 * miles
        base_val = max(base_val, floor_val)

    return round(base_val, 2)

def main(argv):
    if len(argv) != 4:
        print("Usage: predict.py <trip_duration_days> <miles_traveled> <total_receipts_amount>", file=sys.stderr)
        sys.exit(1)
    try:
        td = float(argv[1])
        miles = float(argv[2])
        receipts = float(argv[3])
    except ValueError:
        print("All three arguments must be numeric.", file=sys.stderr)
        sys.exit(1)

    print(predict(td, miles, receipts))

if __name__ == "__main__":
    main(sys.argv)
