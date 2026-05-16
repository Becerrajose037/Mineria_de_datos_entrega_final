# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

All commands run from `RingVoz_CRISP_DM/`:

```bash
# Install dependencies
pip install -r requirements.txt

# Train models and export artifacts (required before running the app)
python train_and_export.py

# Run the Streamlit inference app
streamlit run streamlit_app.py
```

There are no automated tests or linters configured.

## Architecture

The project follows **CRISP-DM** methodology for fraud detection on telecom transactions (RingVoz).

### Data flow

```
datos/mining_data_clean.arff          (raw, 89,890 records, 24 attrs)
        ↓  [fraude_ringvoz.ipynb]
datos/dataset_fraude_ringvoz_class_last.arff   (cleaned, 9 attrs)
        ↓  [train_and_export.py]
artifacts/modelo_riesgo.joblib        (best model: RandomForest pipeline)
artifacts/feature_domains.json        (valid categorical values + amount range)
artifacts/resultados_modelado.json    (metrics for all 3 models)
        ↓  [streamlit_app.py]
Interactive inference UI
```

### Key design decisions

- **Target variable:** `is_fraud` — labeled positive when `Result=Declined` AND `respreasoncode ∈ {27,35,65}` AND `respsubcode ∈ {1,2}` AND not a test request. Only 17 positives out of 89,890 records (0.02%).
- **Evaluation metric: PR-AUC** (not accuracy/ROC-AUC), because the class imbalance makes accuracy trivially 99.98%; PR-AUC properly weights the minority fraud class.
- **Model selection:** Best PR-AUC wins. Currently RandomForest (PR-AUC=0.0122) > XGBoost > MLP.
- **Preprocessing pipeline:** `ColumnTransformer` with `OneHotEncoder` for 5 categorical features and `StandardScaler` for 3 numeric features, all embedded inside the `sklearn.Pipeline` that is serialized to `modelo_riesgo.joblib`. The Streamlit app loads only the joblib — no separate preprocessing step needed at inference.
- **`feature_domains.json`** is the contract between training and UI: it stores valid categorical values and amount min/max so the Streamlit dropdowns always match what the model was trained on. It is regenerated every time `train_and_export.py` runs.

### Files that matter

| File | Role |
|------|------|
| `train_and_export.py` | Single-file training pipeline: loads ARFF → trains 3 models → selects best by PR-AUC → exports all artifacts |
| `streamlit_app.py` | Inference UI: loads artifacts, builds input form from `feature_domains.json`, calls `pipe.predict_proba` |
| `notebooks/fraude_ringvoz.ipynb` | EDA & data preparation (defines fraud label logic) |
| `notebooks/modelado_riesgo_ringvoz.ipynb` | Exploratory model training (mirrors `train_and_export.py`) |
| `datos/dataset_fraude_ringvoz_class_last.arff` | Input to training script (output of notebook EDA) |
