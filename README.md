# ML Surrogate Optimizer for Distillation Column Design

XGBoost surrogate of a benzene-toluene distillation column built in **DWSIM**, used to find the minimum-energy design (reflux ratio, feed stage) that meets a distillate purity spec.

**Live app:** TODO: paste Streamlit link

> **Data note:** all data is from **DWSIM simulations** (300 runs), not plant measurements.

## Method
1. 300 DWSIM runs varying reflux ratio (1.1 to 5.0) and feed stage (3 to 15); all runs valid.
2. Multi-output XGBoost predicts distillate purity, bottoms purity, and reboiler duty.
3. Validation: 80/20 hold-out, 5-fold shuffled CV, and an extrapolation check (train on low reflux, test on high).
4. Optimizer: differential evolution minimizes predicted reboiler duty subject to purity >= spec (bounds from training data, integer feed stage).
5. Deployed as a Streamlit app.

## Results

| Target | Hold-out R² | 5-fold CV R² | Extrapolation R² |
|---|---|---|---|
| Distillate benzene purity | 0.952 | 0.951 ± 0.003 | 0.686 |
| Reboiler duty (kW) | 1.000 | 1.000 | -3.66 |

| Purity spec | Reflux | Feed stage | Predicted duty (kW) | Predicted purity | DWSIM check |
|---|---|---|---|---|---|
| 0.90 | 1.230 | 8 | 34.87 | 0.906 | TODO |
| 0.92 | 1.402 | 8 | 38.01 | 0.926 | TODO |
| 0.95 | 1.845 | 8 | 44.25 | 0.953 | TODO |

## Limitations
- Simulated data: the model reproduces DWSIM, not a real column.
- Interpolator only: extrapolation scores are much lower, so stay within the training ranges.
- Duty is nearly linear in reflux ratio, so R² = 1.0 reflects simple physics; XGBoost extrapolates it poorly.
- Distillate and bottoms purity are identical in this dataset.
- Tree models are piecewise constant, and the purity margin above spec is smaller than the model error (MAE about 0.01), so verify designs in DWSIM.

## Run
```bash
pip install -r requirements.txt
```
Open `distillation_surrogate.py` in Google Colab (each `# %%` is a cell) and upload the simulation file. App: `streamlit run app.py` (TODO: confirm).

**Stack:** DWSIM, Python, XGBoost, scikit-learn, SciPy, Streamlit

**Author:** Jairus Sharma
