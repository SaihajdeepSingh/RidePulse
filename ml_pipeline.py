from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_CSV = BASE_DIR / "bus_passenger_dataset.csv"
DEFAULT_XLSX = BASE_DIR / "FINAL_CLEANED_ENCODED_DATASET.xlsx"
MODEL_JOBLIB = BASE_DIR / "best_passenger_count_model.joblib"
TARGET = "Passenger_Count"
RANDOM_STATE = 42

DAY_COLUMNS = [
    ("Monday", "Day_Monday"),
    ("Tuesday", "Day_Tuesday"),
    ("Wednesday", "Day_Wednesday"),
    ("Thursday", "Day_Thursday"),
    ("Friday", "Day_Friday"),
    ("Saturday", "Day_Saturday"),
    ("Sunday", "Day_Sunday"),
]

DAY_TO_NUM = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}

WEATHER_COLUMNS = [
    ("Clear", "Weather_Clear"),
    ("Cloudy", "Weather_Cloudy"),
    ("Rain", "Weather_Rain"),
]


def resolve_dataset_path(dataset_path: str | Path | None = None) -> Path:
    if dataset_path is not None:
        p = Path(dataset_path)
        if p.exists():
            return p
    if DEFAULT_CSV.exists():
        return DEFAULT_CSV
    if DEFAULT_XLSX.exists():
        return DEFAULT_XLSX
    raise FileNotFoundError(
        f"No dataset found. Expected {DEFAULT_CSV.name} or {DEFAULT_XLSX.name} in {BASE_DIR}."
    )


def load_dataframe(
    path: str | Path | None = None,
    file_bytes: bytes | None = None,
    filename: str | None = None,
) -> tuple[pd.DataFrame, str]:
    if file_bytes is not None:
        fname = (filename or "uploaded_dataset.csv").lower()
        if fname.endswith(".xlsx") or fname.endswith(".xls"):
            df = pd.read_excel(io.BytesIO(file_bytes))
        else:
            df = pd.read_csv(io.BytesIO(file_bytes))
        return df, filename or "uploaded_dataset"

    resolved = resolve_dataset_path(path)
    if resolved.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(resolved)
    else:
        df = pd.read_csv(resolved)
    return df, resolved.name


def _decode_one_hot(df: pd.DataFrame, mapping: list[tuple[str, str]], default: str = "Unknown") -> pd.Series:
    result = pd.Series(default, index=df.index, dtype="object")
    for label, col in mapping:
        if col in df.columns:
            mask = pd.to_numeric(df[col], errors="coerce").fillna(0) > 0
            result = result.where(~mask, label)
    return result


def build_single_feature_models() -> dict[str, Any]:
    return {
        "Simple Linear Regression": LinearRegression(),
        "Polynomial Regression (degree 2)": make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False), LinearRegression()
        ),
        "KNN Regression (k=5)": make_pipeline(
            StandardScaler(), KNeighborsRegressor(n_neighbors=5)
        ),
        "Decision Tree Regression": DecisionTreeRegressor(
            max_depth=5, random_state=RANDOM_STATE
        ),
        "Random Forest Regression": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=1
        ),
        "SVR (RBF)": make_pipeline(
            StandardScaler(), SVR(C=10, epsilon=0.1, kernel="rbf")
        ),
    }


def _build_histogram(residuals: np.ndarray, bins: int = 18) -> dict[str, Any]:
    counts, edges = np.histogram(residuals, bins=bins)
    labels = [f"{edges[i]:.1f} to {edges[i+1]:.1f}" for i in range(len(counts))]
    centers = [round(float((edges[i] + edges[i + 1]) / 2), 2) for i in range(len(counts))]
    return {
        "labels": labels,
        "centers": centers,
        "counts": [int(c) for c in counts],
    }


class BusPassengerPipeline:
    """Implements the leakage-aware chronological ML workflow from Bus_Passenger_Count_Model_Evaluation.ipynb."""

    def __init__(self) -> None:
        self.dataset_name: str = ""
        self.df: pd.DataFrame = pd.DataFrame()
        self.train: pd.DataFrame = pd.DataFrame()
        self.test: pd.DataFrame = pd.DataFrame()
        self.global_mean: float = 0.0
        self.stop_means: pd.Series = pd.Series(dtype=float)
        self.full_stop_means: pd.Series = pd.Series(dtype=float)
        self.trained_models: dict[str, Any] = {}
        self.final_single_model: Any = None
        self.multi_model: Any = None
        self.multi_features: list[str] = []
        self.multi_metrics: dict[str, Any] = {}
        self.metrics: list[dict[str, Any]] = []
        self.best_model_name: str = ""
        self.test_predictions: dict[str, np.ndarray] = {}
        self.stops_catalog: list[dict[str, Any]] = []
        self.overview: dict[str, Any] = {}
        self.analytics: dict[str, Any] = {}
        self.model_diagnostics: dict[str, dict[str, Any]] = {}

    def fit(
        self,
        path: str | Path | None = None,
        file_bytes: bytes | None = None,
        filename: str | None = None,
    ) -> "BusPassengerPipeline":
        raw_df, source_name = load_dataframe(path=path, file_bytes=file_bytes, filename=filename)
        self.dataset_name = source_name

        required = {"Date", "Scheduled_Time", "Stop_ID", TARGET}
        missing = required - set(raw_df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        df = raw_df.copy()
        df["Date_Parsed"] = pd.to_datetime(df["Date"], errors="coerce")
        df["_scheduled_delta"] = pd.to_timedelta(df["Scheduled_Time"].astype(str).str.strip(), errors="coerce")
        df["_timestamp"] = df["Date_Parsed"] + df["_scheduled_delta"]
        df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
        df = df.dropna(subset=["_timestamp", "Stop_ID", TARGET]).copy()

        sort_cols = ["_timestamp"]
        if "Trip_ID" in df.columns:
            sort_cols.append("Trip_ID")
        df = df.sort_values(sort_cols).reset_index(drop=True)

        # Readable labels & pre-trip calendar features (aligned with notebook)
        df["Day_Label"] = _decode_one_hot(df, DAY_COLUMNS, default="Unknown")
        df["Weather_Label"] = _decode_one_hot(df, WEATHER_COLUMNS, default="Unknown")
        df["Date_Str"] = df["Date_Parsed"].dt.strftime("%Y-%m-%d")
        df["Scheduled_Hour"] = df["_timestamp"].dt.hour
        df["Scheduled_Minute"] = df["_timestamp"].dt.minute
        df["Day_Of_Week_Num"] = df["_timestamp"].dt.dayofweek
        df["Month_Num"] = df["_timestamp"].dt.month

        if "Trip_ID" not in df.columns:
            df["Trip_ID"] = [f"TRIP_{i+1}" for i in range(len(df))]
        if "Stop_Name" not in df.columns:
            df["Stop_Name"] = df["Stop_ID"].astype(str)
        if "Route_ID" not in df.columns:
            df["Route_ID"] = "N/A"
        if "Bus_Capacity" not in df.columns:
            df["Bus_Capacity"] = 50
        if "Delay_Minutes" not in df.columns:
            df["Delay_Minutes"] = 0.0
        if "Occupancy_Percentage" not in df.columns:
            df["Occupancy_Percentage"] = (df[TARGET] / df["Bus_Capacity"].replace(0, np.nan)) * 100.0

        self.df = df

        # Chronological 80:20 split (Cell 13)
        split_index = int(len(df) * 0.80)
        train, test = df.iloc[:split_index].copy(), df.iloc[split_index:].copy()

        # Leakage-aware Historical_Stop_Mean (Cell 15)
        train["_stop_prior_mean"] = (
            train.groupby("Stop_ID", sort=False)[TARGET]
            .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        )
        train["_global_prior_mean"] = train[TARGET].shift(1).expanding(min_periods=1).mean()
        train["Historical_Stop_Mean"] = train["_stop_prior_mean"].combine_first(train["_global_prior_mean"])
        train = train.dropna(subset=["Historical_Stop_Mean"]).copy()

        training_stop_means = train.groupby("Stop_ID")[TARGET].mean()
        training_global_mean = float(train[TARGET].mean())
        test["Historical_Stop_Mean"] = test["Stop_ID"].map(training_stop_means).fillna(training_global_mean)

        self.train = train
        self.test = test
        self.global_mean = training_global_mean
        self.stop_means = training_stop_means
        self.full_stop_means = df.groupby("Stop_ID")[TARGET].mean()

        X_train = train[["Historical_Stop_Mean"]]
        X_test = test[["Historical_Stop_Mean"]]
        y_train = train[TARGET].astype(float).to_numpy()
        y_test = test[TARGET].astype(float).to_numpy()

        # Experiment A: 6 single-feature models + TimeSeriesSplit CV (Cells 18 & 23)
        models = build_single_feature_models()
        tscv = TimeSeriesSplit(n_splits=5)

        self.trained_models = {}
        self.test_predictions = {}
        self.model_diagnostics = {}
        rows = []

        for name, estimator in models.items():
            model = clone(estimator)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            residuals = y_test - preds

            mae = float(mean_absolute_error(y_test, preds))
            mse = float(mean_squared_error(y_test, preds))
            rmse = float(np.sqrt(mse))
            r2 = float(r2_score(y_test, preds))

            cv_scores = cross_validate(
                clone(estimator),
                X_train,
                y_train,
                cv=tscv,
                scoring={"mse": "neg_mean_squared_error"},
                n_jobs=1,
            )
            cv_rmse = np.sqrt(-cv_scores["test_mse"])

            self.trained_models[name] = model
            self.test_predictions[name] = preds
            rows.append({
                "Model": name,
                "MAE": round(mae, 4),
                "MSE": round(mse, 4),
                "RMSE": round(rmse, 4),
                "R2": round(r2, 4),
                "CV_RMSE_Mean": round(float(cv_rmse.mean()), 4),
                "CV_RMSE_Std": round(float(cv_rmse.std()), 4),
            })

            scatter_points = [
                {
                    "actual": round(float(a), 2),
                    "predicted": round(float(p), 2),
                    "residual": round(float(a - p), 2),
                    "trip_id": str(tid),
                    "stop_name": str(sname),
                }
                for a, p, tid, sname in zip(
                    y_test,
                    preds,
                    test["Trip_ID"].tolist(),
                    test["Stop_Name"].tolist(),
                )
            ]

            self.model_diagnostics[name] = {
                "model": name,
                "scatter": scatter_points,
                "histogram": _build_histogram(residuals, bins=18),
                "residual_stats": {
                    "mean": round(float(np.mean(residuals)), 4),
                    "std": round(float(np.std(residuals)), 4),
                    "min": round(float(np.min(residuals)), 4),
                    "max": round(float(np.max(residuals)), 4),
                },
            }

        rows.sort(key=lambda r: r["R2"], reverse=True)
        best_name = rows[0]["Model"]
        for r in rows:
            r["is_best"] = (r["Model"] == best_name)

        self.metrics = rows
        self.best_model_name = best_name

        # Cell 33: Refit final single-feature model on full time-ordered dataset
        final_data = df.copy()
        final_data["_stop_prior_mean"] = (
            final_data.groupby("Stop_ID", sort=False)[TARGET]
            .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
        )
        final_data["_global_prior_mean"] = final_data[TARGET].shift(1).expanding(min_periods=1).mean()
        final_data["Historical_Stop_Mean"] = final_data["_stop_prior_mean"].combine_first(
            final_data["_global_prior_mean"]
        )
        final_data = final_data.dropna(subset=["Historical_Stop_Mean"]).copy()
        final_model = clone(models[best_name])
        final_model.fit(final_data[["Historical_Stop_Mean"]], final_data[TARGET].astype(float))
        self.final_single_model = final_model

        model_payload = {
            "model": final_model,
            "model_name": best_name,
            "feature_columns": ["Historical_Stop_Mean"],
            "target": TARGET,
            "evaluation_metrics": rows[0],
            "course_code": "24CSE0316",
        }
        try:
            joblib.dump(model_payload, MODEL_JOBLIB)
        except OSError:
            # Serverless environments (e.g. Vercel) mount the code directory read-only.
            tmp_model = Path("/tmp") / MODEL_JOBLIB.name
            try:
                joblib.dump(model_payload, tmp_model)
            except OSError:
                pass

        # Experiment B: Multiple Linear Regression with legitimate pre-trip features (Cell 26)
        self._fit_multiple_regression(train, test, y_train, y_test)

        self._build_stops_catalog()
        self._build_overview()
        self._build_analytics()
        return self

    def _fit_multiple_regression(
        self,
        train: pd.DataFrame,
        test: pd.DataFrame,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        multi_candidates = [
            "Historical_Stop_Mean",
            "Bus_Capacity",
            "Peak_Hour",
            "Scheduled_Hour",
            "Scheduled_Minute",
            "Day_Of_Week_Num",
            "Month_Num",
        ]
        multi_candidates += [c for c in train.columns if c.startswith("Day_")]
        multi_candidates += [c for c in train.columns if c.startswith("Weather_")]

        forbidden = [
            "Passenger_Count", "Occupancy", "Actual_Time", "Delay",
            "Avg_Passenger", "Max_Passenger", "Passenger_Demand",
            "Avg_Occupancy", "Avg_Delay",
        ]
        multi_features = []
        for c in dict.fromkeys(multi_candidates):
            if c in train.columns and not any(tok.lower() in c.lower() for tok in forbidden):
                if c != TARGET:
                    multi_features.append(c)

        self.multi_features = multi_features
        if len(multi_features) < 2:
            self.multi_model = None
            self.multi_metrics = {}
            return

        X_tr = train[multi_features].copy()
        X_te = test[multi_features].copy()
        for c in multi_features:
            if X_tr[c].dtype == "bool":
                X_tr[c] = X_tr[c].astype(int)
                X_te[c] = X_te[c].astype(int)

        num_cols = X_tr.select_dtypes(include=np.number).columns.tolist()
        cat_cols = [c for c in multi_features if c not in num_cols]

        transformers = []
        if num_cols:
            transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), num_cols))
        if cat_cols:
            transformers.append(("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat_cols))

        pipe = Pipeline([
            ("preprocessor", ColumnTransformer(transformers)),
            ("regressor", LinearRegression()),
        ])
        pipe.fit(X_tr, y_train)
        preds = pipe.predict(X_te)

        mse = float(mean_squared_error(y_test, preds))
        self.multi_model = pipe
        self.multi_metrics = {
            "Model": "Multiple Linear Regression (Pre-Trip Features)",
            "MAE": round(float(mean_absolute_error(y_test, preds)), 4),
            "MSE": round(mse, 4),
            "RMSE": round(float(np.sqrt(mse)), 4),
            "R2": round(float(r2_score(y_test, preds)), 4),
            "feature_count": len(multi_features),
            "features": multi_features,
        }

    def _build_multi_feature_row(
        self,
        hist_mean: float,
        bus_capacity: int,
        hour: int,
        minute: int,
        day: str,
        weather: str,
    ) -> pd.DataFrame:
        row: dict[str, Any] = {}
        day_num = DAY_TO_NUM.get(day, 0)
        for c in self.multi_features:
            if c == "Historical_Stop_Mean":
                row[c] = hist_mean
            elif c == "Bus_Capacity":
                row[c] = bus_capacity
            elif c == "Scheduled_Hour":
                row[c] = hour
            elif c == "Scheduled_Minute":
                row[c] = minute
            elif c == "Day_Of_Week_Num":
                row[c] = day_num
            elif c == "Month_Num":
                row[c] = int(self.df["Month_Num"].mode().iloc[0]) if len(self.df) else 1
            elif c == "Peak_Hour":
                row[c] = 1 if hour in {7, 8, 9, 17, 18, 19} else 0
            elif c.startswith("Day_"):
                row[c] = 1 if c == f"Day_{day}" else 0
            elif c.startswith("Weather_"):
                row[c] = 1 if c == f"Weather_{weather}" else 0
            else:
                row[c] = 0
        return pd.DataFrame([row])[self.multi_features]

    def _build_stops_catalog(self) -> None:
        grouped = (
            self.df.groupby(["Stop_ID", "Stop_Name"], as_index=False)
            .agg(
                Route_ID=("Route_ID", "first"),
                Total_Trips=("Trip_ID", "count"),
                Avg_Passenger_Count=(TARGET, "mean"),
                Max_Passenger_Count=(TARGET, "max"),
                Avg_Occupancy_Percentage=("Occupancy_Percentage", "mean"),
                Avg_Delay_Minutes=("Delay_Minutes", "mean"),
                Avg_Bus_Capacity=("Bus_Capacity", "mean"),
            )
            .sort_values("Avg_Passenger_Count", ascending=False)
        )
        catalog = []
        for row in grouped.to_dict(orient="records"):
            sid = row["Stop_ID"]
            hist_mean = float(self.full_stop_means.get(sid, self.global_mean))
            catalog.append({
                "Stop_ID": str(sid),
                "Stop_Name": str(row["Stop_Name"]),
                "Route_ID": str(row["Route_ID"]),
                "Historical_Stop_Mean": round(hist_mean, 2),
                "Total_Trips": int(row["Total_Trips"]),
                "Avg_Passenger_Count": round(float(row["Avg_Passenger_Count"]), 2),
                "Max_Passenger_Count": int(row["Max_Passenger_Count"]),
                "Avg_Occupancy_Percentage": round(float(row["Avg_Occupancy_Percentage"]), 1),
                "Avg_Delay_Minutes": round(float(row["Avg_Delay_Minutes"]), 2),
                "Avg_Bus_Capacity": round(float(row["Avg_Bus_Capacity"]), 1),
            })
        self.stops_catalog = catalog

    def _build_overview(self) -> None:
        best_row = self.metrics[0]
        self.overview = {
            "dataset_name": self.dataset_name,
            "total_trips": int(len(self.df)),
            "train_size": int(len(self.train)),
            "test_size": int(len(self.test)),
            "split_ratio": "80:20 Chronological",
            "unique_stops": int(self.df["Stop_ID"].nunique()),
            "unique_routes": int(self.df["Route_ID"].nunique()),
            "date_min": str(self.df["Date_Str"].min()),
            "date_max": str(self.df["Date_Str"].max()),
            "avg_passenger_count": round(float(self.df[TARGET].mean()), 2),
            "max_passenger_count": int(self.df[TARGET].max()),
            "avg_delay_minutes": round(float(self.df["Delay_Minutes"].mean()), 2),
            "avg_occupancy_percentage": round(float(self.df["Occupancy_Percentage"].mean()), 1),
            "global_train_mean": round(self.global_mean, 2),
            "best_model": best_row,
            "multi_regression": self.multi_metrics,
        }

    def _build_analytics(self) -> None:
        top_stops = self.stops_catalog[:12]
        weather_order = ["Clear", "Cloudy", "Rain"]
        w_grp = self.df.groupby("Weather_Label", as_index=False).agg(
            trip_count=("Trip_ID", "count"),
            avg_passengers=(TARGET, "mean"),
            avg_occupancy=("Occupancy_Percentage", "mean"),
            avg_delay=("Delay_Minutes", "mean"),
        )
        w_map = {r["Weather_Label"]: r for r in w_grp.to_dict(orient="records")}
        weather_breakdown = [
            {
                "weather": w,
                "trip_count": int(w_map[w]["trip_count"]),
                "avg_passengers": round(float(w_map[w]["avg_passengers"]), 2),
                "avg_occupancy": round(float(w_map[w]["avg_occupancy"]), 1),
                "avg_delay": round(float(w_map[w]["avg_delay"]), 2),
            }
            for w in weather_order if w in w_map
        ]

        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        d_grp = self.df.groupby("Day_Label", as_index=False).agg(
            trip_count=("Trip_ID", "count"),
            avg_passengers=(TARGET, "mean"),
            avg_occupancy=("Occupancy_Percentage", "mean"),
            avg_delay=("Delay_Minutes", "mean"),
        )
        d_map = {r["Day_Label"]: r for r in d_grp.to_dict(orient="records")}
        day_breakdown = [
            {
                "day": d,
                "trip_count": int(d_map[d]["trip_count"]),
                "avg_passengers": round(float(d_map[d]["avg_passengers"]), 2),
                "avg_occupancy": round(float(d_map[d]["avg_occupancy"]), 1),
                "avg_delay": round(float(d_map[d]["avg_delay"]), 2),
            }
            for d in day_order if d in d_map
        ]

        h_grp = (
            self.df.groupby("Scheduled_Hour", as_index=False)
            .agg(
                trip_count=("Trip_ID", "count"),
                avg_passengers=(TARGET, "mean"),
                avg_delay=("Delay_Minutes", "mean"),
            )
            .sort_values("Scheduled_Hour")
        )
        hourly_breakdown = [
            {
                "hour": f"{int(r['Scheduled_Hour']):02d}:00",
                "trip_count": int(r["trip_count"]),
                "avg_passengers": round(float(r["avg_passengers"]), 2),
                "avg_delay": round(float(r["avg_delay"]), 2),
            }
            for r in h_grp.to_dict(orient="records")
        ]

        self.analytics = {
            "top_stops": top_stops,
            "weather_breakdown": weather_breakdown,
            "day_breakdown": day_breakdown,
            "hourly_breakdown": hourly_breakdown,
        }

    def get_predictions_table(
        self,
        model_name: str | None = None,
        search: str = "",
        weather: str = "",
        stop_id: str = "",
        limit: int = 100,
        offset: int = 0,
    ) -> dict[str, Any]:
        selected = model_name if model_name in self.test_predictions else self.best_model_name
        preds = self.test_predictions[selected]

        table = pd.DataFrame({
            "Trip_ID": self.test["Trip_ID"].astype(str).values,
            "Stop_ID": self.test["Stop_ID"].astype(str).values,
            "Stop_Name": self.test["Stop_Name"].astype(str).values,
            "Route_ID": self.test["Route_ID"].astype(str).values,
            "Date": self.test["Date_Str"].astype(str).values,
            "Scheduled_Time": self.test["Scheduled_Time"].astype(str).values,
            "Bus_Capacity": self.test["Bus_Capacity"].astype(int).values,
            "Passenger_Count": self.test[TARGET].astype(int).values,
            "Predicted_Passenger_Count": np.round(preds, 2),
            "Residual": np.round(self.test[TARGET].to_numpy(dtype=float) - preds, 2),
            "Delay_Minutes": np.round(self.test["Delay_Minutes"].to_numpy(dtype=float), 2),
            "Weather": self.test["Weather_Label"].astype(str).values,
            "Day": self.test["Day_Label"].astype(str).values,
        })

        if search:
            q = search.strip().lower()
            mask = (
                table["Trip_ID"].str.lower().str.contains(q, na=False)
                | table["Stop_Name"].str.lower().str.contains(q, na=False)
                | table["Stop_ID"].str.lower().str.contains(q, na=False)
                | table["Route_ID"].str.lower().str.contains(q, na=False)
            )
            table = table[mask]

        if weather and weather.lower() != "all":
            table = table[table["Weather"].str.lower() == weather.strip().lower()]

        if stop_id and stop_id.lower() != "all":
            table = table[table["Stop_ID"] == stop_id.strip()]

        total_filtered = int(len(table))
        sliced = table.iloc[offset : offset + limit]
        return {
            "model": selected,
            "total": total_filtered,
            "offset": offset,
            "limit": limit,
            "rows": sliced.to_dict(orient="records"),
        }

    def predict_live(
        self,
        stop_id: str | None = None,
        custom_historical_mean: float | None = None,
        bus_capacity: int = 50,
        day: str = "Monday",
        weather: str = "Clear",
        selected_model: str | None = None,
        hour: int = 7,
        minute: int = 30,
    ) -> dict[str, Any]:
        stop_info = next((s for s in self.stops_catalog if s["Stop_ID"] == str(stop_id)), None) if stop_id else None

        if custom_historical_mean is not None:
            hist_mean = float(custom_historical_mean)
            mean_source = "Custom Historical_Stop_Mean"
        elif stop_info is not None:
            hist_mean = float(stop_info["Historical_Stop_Mean"])
            mean_source = f"Full Historical Stop Mean ({stop_info['Stop_Name']})"
        else:
            hist_mean = self.global_mean
            mean_source = "Global Training Mean"

        X_input = pd.DataFrame({"Historical_Stop_Mean": [hist_mean]})
        active_model = selected_model if selected_model in self.trained_models else self.best_model_name

        all_predictions = []
        for m in self.metrics:
            mname = m["Model"]
            raw_pred = float(self.trained_models[mname].predict(X_input)[0])
            clamped_pred = max(0.0, raw_pred)
            occ_pct = round((clamped_pred / max(bus_capacity, 1)) * 100.0, 1)
            tier = "Low" if occ_pct < 50 else ("Medium" if occ_pct <= 80 else "High")
            all_predictions.append({
                "model": mname,
                "predicted_passengers": round(clamped_pred, 2),
                "rounded_passengers": int(round(clamped_pred)),
                "occupancy_percentage": occ_pct,
                "crowding_tier": tier,
                "r2": m["R2"],
                "rmse": m["RMSE"],
                "is_selected": (mname == active_model),
            })

        multi_pred = None
        if self.multi_model is not None:
            X_multi = self._build_multi_feature_row(hist_mean, bus_capacity, hour, minute, day, weather)
            multi_raw = max(0.0, float(self.multi_model.predict(X_multi)[0]))
            multi_occ = round((multi_raw / max(bus_capacity, 1)) * 100.0, 1)
            multi_pred = {
                "model": "Multiple Linear Regression (Pre-Trip Features)",
                "predicted_passengers": round(multi_raw, 2),
                "rounded_passengers": int(round(multi_raw)),
                "occupancy_percentage": multi_occ,
                "crowding_tier": "Low" if multi_occ < 50 else ("Medium" if multi_occ <= 80 else "High"),
            }

        primary = next(p for p in all_predictions if p["is_selected"])

        ctx_df = self.df
        if stop_info is not None:
            stop_rows = ctx_df[ctx_df["Stop_ID"].astype(str) == stop_info["Stop_ID"]]
            if len(stop_rows) > 0:
                ctx_df = stop_rows

        weather_rows = ctx_df[ctx_df["Weather_Label"].str.lower() == weather.lower()]
        day_rows = ctx_df[ctx_df["Day_Label"].str.lower() == day.lower()]

        return {
            "input": {
                "stop_id": stop_info["Stop_ID"] if stop_info else None,
                "stop_name": stop_info["Stop_Name"] if stop_info else "Custom Input",
                "historical_stop_mean": round(hist_mean, 2),
                "mean_source": mean_source,
                "bus_capacity": int(bus_capacity),
                "day": day,
                "weather": weather,
                "selected_model": active_model,
            },
            "primary_prediction": primary,
            "multi_prediction": multi_pred,
            "all_model_predictions": all_predictions,
            "stop_baseline": stop_info,
            "context": {
                "weather_avg_passengers": round(float(weather_rows[TARGET].mean()), 2) if len(weather_rows) else None,
                "weather_avg_delay": round(float(weather_rows["Delay_Minutes"].mean()), 2) if len(weather_rows) else None,
                "day_avg_passengers": round(float(day_rows[TARGET].mean()), 2) if len(day_rows) else None,
            },
        }

    def get_commuter_forecast(
        self,
        stop_id: str | None = None,
        bus_capacity: int | None = None,
        day: str = "Monday",
        weather: str = "Clear",
        departure_time: str = "07:30",
    ) -> dict[str, Any]:
        if not stop_id and self.stops_catalog:
            stop_id = self.stops_catalog[0]["Stop_ID"]

        stop_info = next((s for s in self.stops_catalog if s["Stop_ID"] == str(stop_id)), None)
        if stop_info is None:
            raise ValueError(f"Unknown stop_id: {stop_id}")

        effective_cap = int(bus_capacity) if bus_capacity else int(round(stop_info["Avg_Bus_Capacity"]))
        try:
            hh, mm = [int(x) for x in departure_time.split(":")[:2]]
        except Exception:
            hh, mm = 7, 30

        pred = self.predict_live(
            stop_id=stop_info["Stop_ID"],
            bus_capacity=effective_cap,
            day=day,
            weather=weather,
            selected_model=self.best_model_name,
            hour=hh,
            minute=mm,
        )

        # Use Experiment B (Multiple Linear Regression) when available so Day/Weather/Time/Capacity all drive the forecast
        chosen = pred["multi_prediction"] if pred["multi_prediction"] is not None else pred["primary_prediction"]
        primary_occ = chosen["occupancy_percentage"]
        primary_pax = chosen["rounded_passengers"]

        stop_rows = self.df[self.df["Stop_ID"].astype(str) == stop_info["Stop_ID"]].copy()
        sched_times = (
            stop_rows.groupby("Scheduled_Time", as_index=False)
            .agg(
                avg_pax=(TARGET, "mean"),
                avg_delay=("Delay_Minutes", "mean"),
                trips=("Trip_ID", "count"),
            )
            .sort_values("Scheduled_Time")
            .head(8)
        )
        departures = []
        for r in sched_times.to_dict(orient="records"):
            est_pax = round(float(r["avg_pax"]), 1)
            occ = round((est_pax / max(effective_cap, 1)) * 100.0, 1)
            status = (
                "Overcrowded" if occ > 100
                else "Standing Room" if occ > 80
                else "Busy" if occ > 55
                else "Seats Available"
            )
            departures.append({
                "scheduled_time": str(r["Scheduled_Time"])[:5],
                "expected_passengers": est_pax,
                "occupancy_percentage": occ,
                "status": status,
                "avg_delay": round(float(r["avg_delay"]), 1),
                "trips_observed": int(r["trips"]),
            })

        best_departure = min(departures, key=lambda d: d["occupancy_percentage"]) if departures else None

        if primary_occ > 100:
            crowd_label = "Overcrowded"
            advisory = f"Expect ~{primary_pax - effective_cap} standing/waiting passengers on a {effective_cap}-seat bus."
        elif primary_occ > 80:
            crowd_label = "Standing Room Only"
            advisory = "Few or no seats left. Consider an earlier departure if available."
        elif primary_occ > 55:
            crowd_label = "Moderate Crowd"
            advisory = "Seats likely available, though window seats may fill quickly."
        else:
            crowd_label = "Comfortable Ride"
            advisory = "Plenty of seats available on this route."

        return {
            "stop": stop_info,
            "bus_capacity": effective_cap,
            "day": day,
            "weather": weather,
            "departure_time": f"{hh:02d}:{mm:02d}",
            "model_used": chosen["model"],
            "predicted_passengers": primary_pax,
            "occupancy_percentage": primary_occ,
            "crowd_label": crowd_label,
            "advisory": advisory,
            "expected_delay": pred["context"]["weather_avg_delay"] if pred["context"]["weather_avg_delay"] is not None else stop_info["Avg_Delay_Minutes"],
            "departures": departures,
            "best_departure": best_departure,
        }

    def get_operator_dispatch(self, weather: str = "Clear", day: str = "Monday") -> dict[str, Any]:
        alerts = []
        critical_count = 0
        high_count = 0
        normal_count = 0

        for s in self.stops_catalog:
            hist_mean = float(s["Historical_Stop_Mean"])
            cap = int(round(s["Avg_Bus_Capacity"])) or 50
            if self.multi_model is not None:
                X_multi = self._build_multi_feature_row(hist_mean, cap, 7, 30, day, weather)
                pred_pax = max(0.0, float(self.multi_model.predict(X_multi)[0]))
            else:
                pred_pax = max(0.0, float(self.final_single_model.predict(pd.DataFrame({"Historical_Stop_Mean": [hist_mean]}))[0]))

            occ = round((pred_pax / max(cap, 1)) * 100.0, 1)

            if occ > 100:
                severity = "Critical"
                critical_count += 1
            elif occ >= 80:
                severity = "High"
                high_count += 1
            else:
                severity = "Normal"
                normal_count += 1

            recommended_cap = 60
            for c in [45, 50, 55, 60]:
                if (pred_pax / c) * 100.0 <= 85.0:
                    recommended_cap = c
                    break
            buses_needed = max(1, int(np.ceil(pred_pax / (recommended_cap * 0.85))))
            optimized_occ = round((pred_pax / (recommended_cap * buses_needed)) * 100.0, 1)

            alerts.append({
                "Stop_ID": s["Stop_ID"],
                "Stop_Name": s["Stop_Name"],
                "Route_ID": s["Route_ID"],
                "predicted_passengers": int(round(pred_pax)),
                "current_capacity": cap,
                "current_occupancy": occ,
                "severity": severity,
                "avg_delay": s["Avg_Delay_Minutes"],
                "recommended_capacity": recommended_cap,
                "buses_needed": buses_needed,
                "optimized_occupancy": optimized_occ,
                "action": (
                    f"Deploy {buses_needed}× {recommended_cap}-seat buses"
                    if buses_needed > 1
                    else f"Use {recommended_cap}-seat bus"
                ),
            })

        return {
            "summary": {
                "total_routes": len(alerts),
                "critical_routes": critical_count,
                "high_load_routes": high_count,
                "normal_routes": normal_count,
                "avg_network_delay": self.overview["avg_delay_minutes"],
                "weather": weather,
                "day": day,
            },
            "routes": alerts,
        }
