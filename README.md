# RidePulse — Smart Transit Crowding & Fleet Control

RidePulse is a predictive bus passenger crowding and fleet dispatch platform built with **FastAPI**, **scikit-learn**, and **Tailwind CSS**. It operationalizes the leakage-aware chronological regression pipeline from `Bus_Passenger_Count_Model_Evaluation.ipynb`.

## Features
- **Rider App:** Stop/route autocomplete search (`/`), live SVG radial load gauge, vector bus cabin floorplan showing seat occupancy and standing overflow, morning departure demand sparkline, and smart ride recommendations.
- **Dispatch Control:** Dark-mode transit operations console with network health proportion bar, severity filtering (`Critical >100%`, `High 80–100%`, `Normal`), and a split-screen relief fleet simulator.
- **ML Engine (`Bus_Passenger_Count_Model_Evaluation.ipynb`):**
  - **Experiment A:** 6 single-feature models (`Historical_Stop_Mean`) with 5-fold `TimeSeriesSplit` cross-validation (`Simple Linear Regression`, `R² = 0.9406`).
  - **Experiment B:** `Multiple Linear Regression` with 18 pre-trip features (`R² = 0.9397`) powering live rider and dispatch predictions.
  - **Live Retrain:** Upload `.csv` or `.xlsx` datasets directly from the ML Engine drawer.

## Run Locally
```bash
pip install -r requirements.txt
python app.py
```
Open `http://127.0.0.1:8000` in your browser.

## Deploy (Render / Railway)
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Includes `render.yaml` and `Procfile` for one-click deployment.
