from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DECISION_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "decision_elasticity.parquet"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "generated"
    / "calibrated_decision_elasticity.parquet"
)


# ---------------------------------------------------------
# Commercial priors
#
# These are intentionally broad behavioural assumptions,
# NOT derived from hidden synthetic truth.
#
# They represent sensible starting expectations:
#
# Furniture       -> moderately elastic
# Office Supplies -> less elastic
# Technology      -> more elastic
# ---------------------------------------------------------

DEPARTMENT_PRIORS = {
    "Furniture": -1.20,
    "Office Supplies": -0.80,
    "Technology": -1.40,
}


# How much of the raw model estimate we trust.
CONFIDENCE_WEIGHTS = {
    "High": 0.45,
    "Medium": 0.30,
    "Low": 0.15,
}


def get_department_prior(
    department: str,
) -> float:

    return DEPARTMENT_PRIORS.get(
        department,
        -1.10,
    )


def get_model_weight(
    confidence: str,
) -> float:

    return CONFIDENCE_WEIGHTS.get(
        confidence,
        0.15,
    )


def calibrate_elasticity(
    df: pd.DataFrame,
) -> pd.DataFrame:

    result = df.copy()

    # -----------------------------------------------------
    # Commercial prior
    # -----------------------------------------------------

    result["commercial_prior"] = (
        result["department"]
        .map(DEPARTMENT_PRIORS)
        .fillna(-1.10)
    )

    # -----------------------------------------------------
    # Base model weight
    # -----------------------------------------------------

    result["model_weight"] = (
        result["decision_confidence"]
        .map(CONFIDENCE_WEIGHTS)
        .fillna(0.15)
    )

    # -----------------------------------------------------
    # Additional SKU evidence
    #
    # Strong SKU-level evidence can slightly increase
    # confidence in the observed signal.
    # -----------------------------------------------------

    sku_evidence_bonus = np.minimum(
        result["sku_weight"] * 0.25,
        0.10,
    )

    result["model_weight"] = (
        result["model_weight"]
        + sku_evidence_bonus
    ).clip(
        upper=0.55
    )

    # -----------------------------------------------------
    # Regularised elasticity
    #
    # Blend model signal with commercial prior.
    # -----------------------------------------------------

    result["calibrated_elasticity"] = (
        result["model_weight"]
        * result["decision_elasticity"]
        +
        (
            1
            - result["model_weight"]
        )
        * result["commercial_prior"]
    )

    # -----------------------------------------------------
    # Commercial guardrails
    #
    # These are safety boundaries for scenario modelling,
    # not replacements for statistical estimation.
    # -----------------------------------------------------

    result["calibrated_elasticity"] = (
        result["calibrated_elasticity"]
        .clip(
            lower=-3.0,
            upper=-0.20,
        )
    )

    # -----------------------------------------------------
    # Shrinkage amount
    # -----------------------------------------------------

    result["calibration_adjustment"] = (
        result["calibrated_elasticity"]
        - result["decision_elasticity"]
    )

    result["calibration_pct"] = np.where(
        result["decision_elasticity"] != 0,
        (
            result["calibration_adjustment"]
            / result["decision_elasticity"].abs()
        ),
        np.nan,
    )

    return result


def validate_result(
    result: pd.DataFrame,
) -> None:

    assert len(result) == 1500

    assert (
        result["sku_id"].nunique()
        == 1500
    )

    assert (
        result["calibrated_elasticity"]
        .notna()
        .all()
    )

    assert (
        result["calibrated_elasticity"]
        < 0
    ).all()

    assert (
        result["calibrated_elasticity"]
        .between(
            -3.0,
            -0.20,
        )
        .all()
    )

    assert (
        result["model_weight"]
        .between(
            0,
            0.55,
        )
        .all()
    )

    print("\nValidation passed.")


def main():

    decision = pd.read_parquet(
        DECISION_PATH
    )

    result = calibrate_elasticity(
        decision
    )

    validate_result(
        result
    )

    result.to_parquet(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\nCALIBRATED DECISION ELASTICITY"
    )
    print("=" * 70)

    print(
        f"SKUs                     : "
        f"{len(result):,}"
    )

    print(
        f"Raw mean elasticity      : "
        f"{result['decision_elasticity'].mean():.3f}"
    )

    print(
        f"Calibrated mean          : "
        f"{result['calibrated_elasticity'].mean():.3f}"
    )

    print(
        f"Average adjustment       : "
        f"{result['calibration_adjustment'].mean():.3f}"
    )

    print("\nCalibrated Distribution:")

    print(
        result[
            "calibrated_elasticity"
        ]
        .describe()
        .round(3)
    )

    print("\nBy Department:")

    department = (
        result.groupby(
            "department"
        )
        .agg(
            skus=(
                "sku_id",
                "count",
            ),
            raw_elasticity=(
                "decision_elasticity",
                "mean",
            ),
            commercial_prior=(
                "commercial_prior",
                "mean",
            ),
            calibrated_elasticity=(
                "calibrated_elasticity",
                "mean",
            ),
            avg_model_weight=(
                "model_weight",
                "mean",
            ),
        )
        .round(3)
    )

    print(department)

    print("\nBy Confidence:")

    confidence = (
        result.groupby(
            "decision_confidence"
        )
        .agg(
            skus=(
                "sku_id",
                "count",
            ),
            raw_elasticity=(
                "decision_elasticity",
                "mean",
            ),
            calibrated_elasticity=(
                "calibrated_elasticity",
                "mean",
            ),
            avg_model_weight=(
                "model_weight",
                "mean",
            ),
        )
        .round(3)
    )

    print(confidence)

    print("\nLargest Calibration Adjustments:")

    largest = (
        result.assign(
            abs_adjustment=lambda x:
                x[
                    "calibration_adjustment"
                ].abs()
        )
        .sort_values(
            "abs_adjustment",
            ascending=False,
        )
        [
            [
                "sku_id",
                "department",
                "product_class",
                "decision_confidence",
                "decision_elasticity",
                "commercial_prior",
                "model_weight",
                "calibrated_elasticity",
                "calibration_adjustment",
            ]
        ]
        .head(20)
    )

    print(
        largest.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print("\nSample:")

    sample = result[
        [
            "sku_id",
            "department",
            "product_class",
            "decision_confidence",
            "decision_elasticity",
            "commercial_prior",
            "model_weight",
            "calibrated_elasticity",
        ]
    ].head(20)

    print(
        sample.to_string(
            index=False,
            float_format=lambda x: f"{x:.3f}",
        )
    )

    print(
        f"\nOutput                   : "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()