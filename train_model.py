from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import pandas as pd
import numpy as np


# Step 1: Load Dataset

print("Loading dataset...")
california = fetch_california_housing(as_frame=True)
data = california.frame


# Step 2: Feature Engineering
# New feature: Rooms per person
data["RoomsPerPerson"] = data["AveRooms"] / data["AveOccup"]

# Optional: Handle any potential infinities or NaNs
data.replace([np.inf, -np.inf], np.nan, inplace=True)
data.dropna(inplace=True)

# ✅ Inspect the engineered feature
print("\nSample rows showing engineered feature 'RoomsPerPerson':")
print(data[["AveRooms", "AveOccup", "RoomsPerPerson"]].head())

print("\nStatistics for 'RoomsPerPerson':")
print(data["RoomsPerPerson"].describe())

# Step 3: Feature/Target Split

X = data.drop("MedHouseVal", axis=1)
y = data["MedHouseVal"]


# Step 4: Train-Test Split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Step 5: Hyperparameter Tuning

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

print("Starting hyperparameter tuning...")
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)


# Step 6: Model Evaluation

y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"\nEvaluation Metrics:")
print(f"RMSE: {rmse:.2f}")
print(f"R² Score: {r2:.2f}")


# Step 7: Save the Model

os.makedirs("model", exist_ok=True)
joblib.dump(best_model, "model/model.pkl")
print("\nTuned model trained and saved at: model/model.pkl")


# Step 8: Save Feature Names
# Optional — helps if you want to check or debug features later
feature_names = list(X.columns)
joblib.dump(feature_names, "model/feature_names.pkl")
print("Feature names saved.")
loaded_features = joblib.load("model/feature_names.pkl")
print("Features used by model:", loaded_features)
