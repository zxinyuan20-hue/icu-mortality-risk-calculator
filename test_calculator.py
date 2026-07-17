"""Smoke tests for the deployable calculator; run with `python test_calculator.py`."""
from __future__ import annotations

from app import predict_patient, recalibrate_probability, risk_category


CASES = {
    "low_risk": ({"Age": 40.0, "Lactate": 1.0, "Platelet": 300.0, "Urine Output": 2500.0, "BUN": 10.0, "SOFA": 1.0, "Vasopressor": 0}, "Low risk"),
    "intermediate_risk": ({"Age": 65.0, "Lactate": 2.5, "Platelet": 180.0, "Urine Output": 1200.0, "BUN": 25.0, "SOFA": 6.0, "Vasopressor": 0}, "Intermediate risk"),
    "high_risk": ({"Age": 75.0, "Lactate": 7.0, "Platelet": 60.0, "Urine Output": 300.0, "BUN": 70.0, "SOFA": 15.0, "Vasopressor": 1}, "High risk"),
}


def main() -> None:
    for label, (values, expected_category) in CASES.items():
        original, recalibrated, shap_values, metadata, parameters = predict_patient(values)
        category = risk_category(recalibrated, metadata["risk_category_rules"]["recalibrated_probability"])
        expected_probability = recalibrate_probability(original, parameters["calibration_intercept"], parameters["calibration_slope"])
        assert 0 <= original <= 1 and 0 <= recalibrated <= 1
        assert abs(recalibrated - expected_probability) < 1e-12
        assert len(shap_values) == len(metadata["feature_columns"])
        assert category == expected_category, f"{label}: expected {expected_category}, got {category}"
        print(f"{label}: original={original:.4f}; recalibrated={recalibrated:.4f}; category={category}")
    print("All calculator smoke tests passed.")


if __name__ == "__main__":
    main()
