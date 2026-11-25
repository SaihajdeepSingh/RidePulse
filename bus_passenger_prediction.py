"""
Bus Passenger Count Prediction – CE-1 Final Implementation
Input: FINAL_CLEANED_ENCODED_DATASET.xlsx
Target: Passenger_Count
Feature: Historical_Stop_Mean
Best baseline: Simple Linear Regression

Run:
    pip install pandas openpyxl scikit-learn matplotlib
    python bus_passenger_prediction.py

The script uses a chronological 80:20 split. The test feature is computed
from training-period stop means only. Do not use Occupancy_Percentage or
Occupancy_Category as predictors because they are derived from the target.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from pathlib import Path

FILE = "FINAL_CLEANED_ENCODED_DATASET.xlsx"
CSV_FILE = "bus_passenger_dataset.csv"
TARGET = "Passenger_Count"

if Path(FILE).exists():
    df = pd.read_excel(FILE)
elif Path(CSV_FILE).exists():
    df = pd.read_csv(CSV_FILE)
else:
    raise FileNotFoundError(f"Neither {FILE} nor {CSV_FILE} was found.")
required = {"Date", "Scheduled_Time", "Trip_ID", "Stop_ID", TARGET}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# 1. Preprocessing: parse date/time and order trips chronologically.
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
df["time"] = pd.to_timedelta(df["Scheduled_Time"], errors="coerce")
df = df.dropna(subset=["Date", "time", "Stop_ID", TARGET]).copy()
df["timestamp"] = df["Date"] + df["time"]
df = df.sort_values(["timestamp", "Trip_ID"]).reset_index(drop=True)

# 2. Chronological holdout: first 80% train, final 20% test.
cut = int(len(df) * 0.80)
train, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()

# 3. Feature engineering. Training rows use only earlier rows at the same stop.
train["Historical_Stop_Mean"] = (
    train.groupby("Stop_ID")[TARGET]
         .transform(lambda s: s.shift(1).expanding().mean())
)
global_mean = train[TARGET].mean()
train["Historical_Stop_Mean"] = train["Historical_Stop_Mean"].fillna(global_mean)

# Test rows use stop-level means from training data only.
stop_means = train.groupby("Stop_ID")[TARGET].mean()
test["Historical_Stop_Mean"] = test["Stop_ID"].map(stop_means).fillna(global_mean)

X_train = train[["Historical_Stop_Mean"]]
X_test = test[["Historical_Stop_Mean"]]
y_train, y_test = train[TARGET], test[TARGET]

# 4. Train and compare regression algorithms on the same feature.
models = {
    "Linear Regression": LinearRegression(),
    "Polynomial Regression (degree 2)": make_pipeline(
        PolynomialFeatures(2, include_bias=False), LinearRegression()
    ),
    "SVR (RBF)": make_pipeline(
        StandardScaler(), SVR(C=10, epsilon=0.1, kernel="rbf")
    ),
    "Decision Tree Regression": DecisionTreeRegressor(
        max_depth=5, random_state=42
    ),
    "Random Forest Regression": RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1
    ),
    "KNN Regression (k=5)": make_pipeline(
        StandardScaler(), KNeighborsRegressor(n_neighbors=5)
    ),
}

rows, predictions = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    predictions[name] = pred
    mse = mean_squared_error(y_test, pred)
    rows.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, pred),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(y_test, pred),
    })

results = pd.DataFrame(rows).sort_values("RMSE")
print("\\nMODEL COMPARISON")
print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
results.to_csv("model_comparison.csv", index=False)

# 5. Final model and prediction output.
best_model = LinearRegression()
best_model.fit(X_train, y_train)
final_pred = best_model.predict(X_test)
prediction_output = test[["Trip_ID", "Stop_ID", "timestamp", TARGET]].copy()
prediction_output["Predicted_Passenger_Count"] = final_pred.round(2)
prediction_output["Residual"] = (
    prediction_output[TARGET] - prediction_output["Predicted_Passenger_Count"]
)
prediction_output.to_csv("test_predictions.csv", index=False)

# 6. Visualizations.
plt.figure(figsize=(9, 5))
plt.bar(results["Model"], results["RMSE"])
plt.ylabel("RMSE (lower is better)")
plt.title("Model Comparison – Test RMSE")
plt.xticks(rotation=25, ha="right")
plt.tight_layout()
plt.savefig("model_comparison_rmse.png", dpi=180)
plt.close()

plt.figure(figsize=(6, 6))
plt.scatter(y_test, final_pred, alpha=0.45)
lo = min(y_test.min(), final_pred.min())
hi = max(y_test.max(), final_pred.max())
plt.plot([lo, hi], [lo, hi], "--")
plt.xlabel("Actual passenger count")
plt.ylabel("Predicted passenger count")
plt.title("Linear Regression: Actual vs Predicted")
plt.tight_layout()
plt.savefig("actual_vs_predicted.png", dpi=180)
plt.close()

residuals = y_test.to_numpy() - final_pred
plt.figure(figsize=(8, 5))
plt.hist(residuals, bins=30)
plt.xlabel("Residual (actual − predicted)")
plt.ylabel("Frequency")
plt.title("Linear Regression: Residual Distribution")
plt.tight_layout()
plt.savefig("residual_distribution.png", dpi=180)
plt.close()

print("\\nBest baseline: Linear Regression")
print(f"MAE={mean_absolute_error(y_test, final_pred):.3f}")
print(f"RMSE={mean_squared_error(y_test, final_pred)**0.5:.3f}")
print(f"R2={r2_score(y_test, final_pred):.4f}")
print("Saved: model_comparison.csv, test_predictions.csv and three PNG plots.")
