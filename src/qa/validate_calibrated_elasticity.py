from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "generated"

CALIBRATED_FILE = DATA_DIR / "calibrated_decision_elasticity.parquet"
DEMAND_PROFILE_FILE = DATA_DIR / "demand_profiles.parquet"


def main():

    # ---------------------------------------------------------
    # Load decision outputs and hidden synthetic truth
    # ---------------------------------------------------------
    decision = pd.read_parquet(CALIBRATED_FILE)
    truth = pd.read_parquet(DEMAND_PROFILE_FILE)

    required_decision = {
        "sku_id",
        "benchmark_elasticity",
        "decision_elasticity",
        "calibrated_elasticity",
        "decision_confidence",
        "model_weight",
    }

    required_truth = {
        "sku_id",
        "true_price_elasticity",
    }

    missing_decision = required_decision - set(decision.columns)
    missing_truth = required_truth - set(truth.columns)

    if missing_decision:
        raise ValueError(
            f"Missing decision columns: {sorted(missing_decision)}"
        )

    if missing_truth:
        raise ValueError(
            f"Missing truth columns: {sorted(missing_truth)}"
        )

    truth = truth[
        ["sku_id", "true_price_elasticity"]
    ].rename(
        columns={"true_price_elasticity": "true_elasticity"}
    )

    df = decision.merge(
        truth,
        on="sku_id",
        how="left",
        validate="one_to_one",
    )

    if df["true_elasticity"].isna().any():
        missing = int(df["true_elasticity"].isna().sum())
        raise ValueError(
            f"{missing} SKUs are missing hidden true elasticity."
        )

    # ---------------------------------------------------------
    # Error calculations
    # ---------------------------------------------------------
    methods = {
        "Benchmark": "benchmark_elasticity",
        "Raw Hierarchy": "decision_elasticity",
        "Calibrated": "calibrated_elasticity",
    }

    rows = []

    for method, col in methods.items():

        error = df[col] - df["true_elasticity"]
        absolute_error = error.abs()

        rows.append(
            {
                "method": method,
                "mae": absolute_error.mean(),
                "median_ae": absolute_error.median(),
                "bias": error.mean(),
                "correlation": df[col].corr(
                    df["true_elasticity"]
                ),
            }
        )

    comparison = pd.DataFrame(rows)

    benchmark_mae = comparison.loc[
        comparison["method"] == "Benchmark", "mae"
    ].iloc[0]

    raw_mae = comparison.loc[
        comparison["method"] == "Raw Hierarchy", "mae"
    ].iloc[0]

    calibrated_mae = comparison.loc[
        comparison["method"] == "Calibrated", "mae"
    ].iloc[0]

    # ---------------------------------------------------------
    # Improvement metrics
    # ---------------------------------------------------------
    improvement_vs_raw = (
        (raw_mae - calibrated_mae) / raw_mae * 100
    )

    improvement_vs_benchmark = (
        (benchmark_mae - calibrated_mae)
        / benchmark_mae
        * 100
    )

    # ---------------------------------------------------------
    # SKU-level diagnostics
    # ---------------------------------------------------------
    df["benchmark_abs_error"] = (
        df["benchmark_elasticity"]
        - df["true_elasticity"]
    ).abs()

    df["raw_abs_error"] = (
        df["decision_elasticity"]
        - df["true_elasticity"]
    ).abs()

    df["calibrated_abs_error"] = (
        df["calibrated_elasticity"]
        - df["true_elasticity"]
    ).abs()

    df["calibration_improved"] = (
        df["calibrated_abs_error"]
        < df["raw_abs_error"]
    )

    improved_pct = (
        df["calibration_improved"].mean() * 100
    )

    # ---------------------------------------------------------
    # Confidence diagnostics
    # ---------------------------------------------------------
    confidence_summary = (
        df.groupby("decision_confidence")
        .agg(
            skus=("sku_id", "count"),
            benchmark_mae=("benchmark_abs_error", "mean"),
            raw_mae=("raw_abs_error", "mean"),
            calibrated_mae=("calibrated_abs_error", "mean"),
            avg_model_weight=("model_weight", "mean"),
        )
        .round(3)
    )

    # ---------------------------------------------------------
    # Department diagnostics
    # ---------------------------------------------------------
    if "department" in df.columns:

        department_summary = (
            df.groupby("department")
            .agg(
                skus=("sku_id", "count"),
                truth=("true_elasticity", "mean"),
                benchmark=("benchmark_elasticity", "mean"),
                raw=("decision_elasticity", "mean"),
                calibrated=("calibrated_elasticity", "mean"),
                benchmark_mae=("benchmark_abs_error", "mean"),
                raw_mae=("raw_abs_error", "mean"),
                calibrated_mae=("calibrated_abs_error", "mean"),
            )
            .round(3)
        )

    else:
        department_summary = None

    # ---------------------------------------------------------
    # Best / worst calibration outcomes
    # ---------------------------------------------------------
    df["calibration_error_change"] = (
        df["raw_abs_error"]
        - df["calibrated_abs_error"]
    )

    best = (
        df.sort_values(
            "calibration_error_change",
            ascending=False,
        )
        .head(15)
    )

    worst = (
        df.sort_values(
            "calibration_error_change",
            ascending=True,
        )
        .head(15)
    )

    display_cols = [
        "sku_id",
        "product_class",
        "decision_confidence",
        "true_elasticity",
        "benchmark_elasticity",
        "decision_elasticity",
        "calibrated_elasticity",
        "model_weight",
        "raw_abs_error",
        "calibrated_abs_error",
        "calibration_error_change",
    ]

    display_cols = [
        c for c in display_cols if c in df.columns
    ]

    # ---------------------------------------------------------
    # Output
    # ---------------------------------------------------------
    print()
    print("CALIBRATED ELASTICITY VALIDATION")
    print("=" * 62)
    print(f"SKUs                    : {len(df):,}")
    print()

    print("MODEL COMPARISON")
    print("-" * 62)

    print(
        comparison.round(3).to_string(
            index=False
        )
    )

    print()
    print(
        f"Calibration improvement vs Raw Hierarchy : "
        f"{improvement_vs_raw:+.1f}%"
    )

    print(
        f"Calibration improvement vs Benchmark     : "
        f"{improvement_vs_benchmark:+.1f}%"
    )

    print(
        f"SKUs improved by calibration             : "
        f"{improved_pct:.1f}%"
    )

    print()
    print("ACCURACY BY CONFIDENCE")
    print("-" * 62)
    print(confidence_summary.to_string())

    if department_summary is not None:
        print()
        print("ACCURACY BY DEPARTMENT")
        print("-" * 62)
        print(department_summary.to_string())

    print()
    print("BEST CALIBRATION IMPROVEMENTS")
    print("-" * 62)

    print(
        best[display_cols]
        .round(3)
        .to_string(index=False)
    )

    print()
    print("WORST CALIBRATION OUTCOMES")
    print("-" * 62)

    print(
        worst[display_cols]
        .round(3)
        .to_string(index=False)
    )

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------
    assert len(df) == 1500
    assert df["true_elasticity"].notna().all()
    assert np.isfinite(
        df["calibrated_elasticity"]
    ).all()

    print()
    print("Validation passed.")


if __name__ == "__main__":
    main()