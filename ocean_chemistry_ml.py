# ocean_chemistry_ml.py
# Synthetic Ocean Chemistry Change Tracking using ML

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

np.random.seed(42)

n = 4000

years = np.random.uniform(1980, 2024, n)
temp = np.random.uniform(0, 30, n)
salinity = np.random.uniform(30, 37, n)
depth = np.random.uniform(1, 300, n)
co2 = 320 + 2.1*(years - 1980) + np.random.normal(0, 2, n)
oxygen = np.random.uniform(2, 9, n)
conductivity = np.random.uniform(40, 60, n)
turbidity = np.random.uniform(0.5, 5, n)

pH = (
    8.10
    - 0.0025*(years - 1980)
    - 0.0006*(co2 - 320)
    - 0.002*(temp - 15)
    + 0.003*(oxygen - 6)
    - 0.0008*(salinity - 34)
    + np.random.normal(0, 0.02, n)
)

df = pd.DataFrame({
    "year": years,
    "temp": temp,
    "salinity": salinity,
    "depth": depth,
    "co2_ppm": co2,
    "oxygen_mgL": oxygen,
    "conductivity": conductivity,
    "turbidity": turbidity,
    "pH": pH
})

features = ["year","temp","salinity","depth","co2_ppm",
            "oxygen_mgL","conductivity","turbidity"]

X = df[features]
y = df["pH"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

model = RandomForestRegressor(n_estimators=250, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

print("RMSE:", rmse)
print("R²:", r2)

future_years = np.arange(2025, 2041)

future_df = pd.DataFrame({
    "year": future_years,
    "temp": np.full_like(future_years, 18),
    "salinity": np.full_like(future_years, 34),
    "depth": np.full_like(future_years, 20),
    "co2_ppm": 320 + 2.1*(future_years - 1980),
    "oxygen_mgL": np.full_like(future_years, 6),
    "conductivity": np.full_like(future_years, 50),
    "turbidity": np.full_like(future_years, 2)
})

future_pred = model.predict(future_df)

plt.figure(figsize=(8,4))
plt.plot(future_years, future_pred, marker="o")
plt.xlabel("Year")
plt.ylabel("Predicted pH")
plt.title("Projected Ocean Chemistry Changes (2030–2040)")
plt.grid(True)
plt.tight_layout()
plt.show()
