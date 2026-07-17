# ICU Mortality Risk Calculator Based on Explainable LightGBM Model

## Purpose

This Streamlit application estimates ICU mortality risk for critically ill adults undergoing surgery for gastrointestinal perforation. It packages the final frozen LightGBM model developed with MIMIC-III and MIMIC-IV and externally validated using eICU and NWICU.

The application does not train, tune, update, or select model variables. It applies a previously estimated logistic probability-recalibration equation as a secondary analysis for populations resembling the external validation cohort.

## Inputs

| Input | Unit or coding |
|---|---|
| Age | years |
| SOFA score | points |
| Lactate | mmol/L |
| Platelet count | 10^9/L |
| Blood urea nitrogen (BUN) | mg/dL |
| Urine output | mL/24 h |
| Vasopressor use | Yes or No |

## Prediction flow

1. Input values are passed through the fitted winsorization and scaling pipeline.
2. The frozen LightGBM model produces the original ICU mortality probability.
3. External probability recalibration is applied:

`logit(recalibrated probability) = intercept + slope x logit(original probability)`

4. The calculator reports the recalibrated probability and a low/intermediate/high risk category.

Risk thresholds were derived from the 33.3rd and 66.7th percentiles of development-cohort out-of-fold probabilities, then transformed using the frozen external recalibration equation.

## Local use

1. Create and activate a Python environment (Python 3.10 or later is recommended).
2. Install packages:

```bash
pip install -r requirements.txt
```

3. Run the application from this directory:

```bash
streamlit run app.py
```

## Streamlit Community Cloud deployment

1. Push the `Clinical_AI_Calculator` directory, including the `model/` directory, to a GitHub repository.
2. In Streamlit Community Cloud, create a new app from that repository.
3. Set the main file path to `Clinical_AI_Calculator/app.py` if this folder is stored within a larger repository; otherwise use `app.py`.
4. Streamlit Cloud installs `requirements.txt` and launches the application without model retraining.

## Limitations

- This is a research calculator, not clinical decision-support software.
- It was developed for adults undergoing surgery for gastrointestinal perforation who require ICU care; use outside that population is not validated.
- The recalibrated probability is an apparent secondary analysis fitted in the external validation cohort and requires prospective validation before clinical use.
- Predictions can be affected by measurement timing, unit conventions, missingness, and case mix.
- Do not enter patient-identifiable information into public deployments.
