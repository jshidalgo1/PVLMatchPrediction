#!/usr/bin/env python3
"""
Validate model metrics for CI/CD pipeline.

This script checks the latest metrics comparison and validates that
metrics haven't regressed beyond acceptable thresholds.
"""

import json
import sys
from pathlib import Path
import glob


def validate_metrics():
    """Validate metrics against regression thresholds."""
    
    # Find latest comparison JSON
    json_files = glob.glob("outputs/metrics_comparison_*.json")
    if not json_files:
        print("::error title=No Metrics Found::No metrics comparison JSON files found")
        sys.exit(1)
    
    latest_json = sorted(json_files)[-1]
    print(f"📊 Validating metrics from: {latest_json}")
    
    # Load metrics data
    with open(latest_json) as f:
        data = json.load(f)
    
    delta_acc = data["delta"]["accuracy"]
    delta_logloss = data["delta"]["logloss"]
    delta_auc = data["delta"]["AUC"]
    
    # Display metrics
    print(f"::notice title=Accuracy Delta::{delta_acc:+.4f}")
    print(f"::notice title=LogLoss Delta::{delta_logloss:+.4f}")
    print(f"::notice title=AUC Delta::{delta_auc:+.4f}")
    
    # Regression thresholds
    ACC_THRESHOLD = -0.02  # Fail if accuracy drops >2%
    LOGLOSS_THRESHOLD = 0.05  # Fail if logloss increases >0.05
    AUC_THRESHOLD = -0.02  # Fail if AUC drops >2%
    
    failed = False
    
    # Check accuracy
    if delta_acc < ACC_THRESHOLD:
        print(
            f"::error title=Accuracy Regression::"
            f"Accuracy dropped by {-delta_acc:.4f} "
            f"(threshold: {-ACC_THRESHOLD:.4f})"
        )
        failed = True
    
    # Check logloss
    if delta_logloss > LOGLOSS_THRESHOLD:
        print(
            f"::error title=LogLoss Regression::"
            f"LogLoss increased by {delta_logloss:.4f} "
            f"(threshold: {LOGLOSS_THRESHOLD:.4f})"
        )
        failed = True
    
    # Check AUC
    if delta_auc < AUC_THRESHOLD:
        print(
            f"::error title=AUC Regression::"
            f"AUC dropped by {-delta_auc:.4f} "
            f"(threshold: {-AUC_THRESHOLD:.4f})"
        )
        failed = True
    
    if failed:
        print("\n❌ Metrics regression detected!")
        sys.exit(1)
    else:
        print("\n✓ All metrics within acceptable thresholds")
        sys.exit(0)


if __name__ == "__main__":
    try:
        validate_metrics()
    except Exception as e:
        print(f"::error title=Validation Error::{e}")
        sys.exit(1)
