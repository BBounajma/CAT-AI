# CAT-AI
The aim of this project is to use deep-learning to improve catastrophic modeling in case of a seismic event. For now, the focus is on the fragility function. This project is a collaboration between the design and technology team of UNDRR and the ROAP of UNDRR.

This project is lead and developed by Bilal Bounajma (design & tech , UNDRR). It  is supervised Revati Mani (design & tech , UNDRR) and Alper Aras (ROAP, UNDRR).

It is also the basis of a semester paper in pursue of a Master's degree in Applied Mathematics at ETHZ suppervised by Patrick Cheridito and Nino Antulov-Fantulin. The tex file can be viewed here:
https://fr.overleaf.com/read/mtntvjxccvhs#fe27e5

# CAT-AI

Short guide for the CAT-AI repository: a collection of trainings, models and ensembling utilities for seismic damage-grade prediction.

Project highlights
- Purpose: predict building damage grades after seismic events using classical and deep models (XGBoost, Random Forest, CatBoost, SAINT, TabNet, ensembles).
- Main folders: `Data/`, `Models/`, `Trainings/`, `Ensemble/`, `results/`, `Tests/`.

Quick start
1. Create a Python environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Prepare data: place or generate `Data/processed_new_data2.csv` (the code expects this path).

3. Train individual models (examples):

```bash
# Train XGBoost
python Trainings/train_XGboost.py
# Train Random Forest
python Trainings/train_rf.py
# Train SAINT / TabNet (deep models)
python Trainings/train_saint.py
python Trainings/train_tabnet.py
```

4. Run ensembling scripts (examples):

```bash
# Logistic regression stacking
python Ensemble/logreg_stacking_saint_tabnet.py
# MLP stacking (OOF)
python Ensemble/MLP_stacking_oof_saint_tabnet.py
# Weighted and stacking meta-ensembles
python Ensemble/weighted_average_saint_tabnet.py
```

Outputs
- Trained model artifacts are saved under `Models/` (per-model subfolders).
- Ensemble meta-models and their info are saved to `Models/` as `*_model.joblib` and `*_info.joblib`.
- Experiment metrics and diagnostics (including confusion matrices) are saved to `results/` as JSON and CSV files.

Tests
- Unit/integration tests are in `Tests/`. Run them with `pytest` or `python -m pytest`.

Notes
- The code assumes a 5-class target `damage_grade` and uses stratified splits; check `Data/processed_new_data2.csv` for expected columns.
- If you encounter missing model files, follow the printed hints in the ensemble scripts to run the corresponding training script.

Credits
- Project lead: Bilal Bounajma. Supervisors and contributors are listed in the repository.


