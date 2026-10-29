# Salary Prediction Project

This project predicts `salary` from the supplied `job_salary_prediction_dataset.csv`.

## Files

- `app.py` — Streamlit application with prediction, data exploration, and model details pages.
- `salary_model.pkl` — trained scikit-learn preprocessing and regression pipeline.
- `Job_Salary_Statistical_Inference_Project.ipynb` — Colab/Jupyter statistical-inference notebook.
- `job_salary_prediction_dataset.csv` — source dataset.
- `model_metrics.json` — model evaluation metrics used by the app.
- `requirements.txt` — Python dependencies.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The dataset does not specify a currency, so the app displays salary in the original dataset units and does not apply an exchange-rate conversion.
