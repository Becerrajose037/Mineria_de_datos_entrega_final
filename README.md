# Minería de Datos - Entrega Final

Proyecto académico CRISP-DM de detección de fraude para RingVoz (proveedor de servicios telecom/pagos).

**Equipo:** José Becerra · Marleny Guiral Marín · Geraldine Suárez

## Descripción

Pipeline completo de machine learning que entrena y compara tres modelos (RandomForest, XGBoost, MLP) sobre 89,890 transacciones con 17 casos de fraude (0.02%). El mejor modelo se expone mediante una aplicación web interactiva en Streamlit.

## Quickstart

```bash
cd RingVoz_CRISP_DM
pip install -r requirements.txt

# 1. Entrenar modelos y exportar artefactos
python train_and_export.py

# 2. Lanzar la interfaz de inferencia
streamlit run streamlit_app.py
```

## Estructura

```
RingVoz_CRISP_DM/
├── train_and_export.py          # Pipeline de entrenamiento completo
├── streamlit_app.py             # Interfaz web de inferencia
├── requirements.txt
├── datos/                       # Datasets en formato ARFF
├── notebooks/                   # EDA y modelado exploratorio
└── artifacts/                   # Modelo serializado y métricas
```

## Modelo

| Modelo | ROC-AUC | PR-AUC | Recall clase 1 |
|--------|---------|--------|----------------|
| **RandomForest** ✓ | 0.9079 | 0.0122 | 0.25 |
| XGBoost | 0.8449 | 0.0120 | 0.25 |
| MLP | 0.5332 | 0.0002 | 0.00 |

Métrica de selección: **PR-AUC** (adecuada para datasets con alta desbalance de clases).
