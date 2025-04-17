import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# from sklearn.model_selection import cross_val_score
#  from sklearn.preprocessing import StandardScaler

data = pd.read_csv('data/time_domain_features_train.csv')
test_data = pd.read_csv('data/time_domain_features_test.csv')

# Strip column names (in case of spaces)
data.columns = data.columns.str.strip()

# Convert to numeric (handles non-numeric values)
data["MEAN_RR"] = pd.to_numeric(data["MEAN_RR"], errors="coerce")
data["HR"] = pd.to_numeric(data["HR"], errors="coerce")

# Drop missing values
data = data.dropna(subset=["MEAN_RR", "HR"])

# Plot



# logistic_regression = LogisticRegression(
#     solver='lbfgs',
#     penalty='l2'
# )

regressor = LinearRegression()

meanrr_train = data["MEAN_RR"].iloc[:100000].values
hr_train = data[["HR"]].iloc[:100000]

regressor.fit(hr_train, meanrr_train)

# meanrr_test = test_data["MEAN_RR"][:10000]
# hr_test = test_data["HR"][:10000]

y_pred = regressor.predict(hr_train)


r2 = r2_score(meanrr_train, y_pred)

mse = mean_squared_error(meanrr_train, y_pred)

print(f"R2 Score ACCURACY: {r2:.4f}")
print(f"Mean Squared Error (MSE) LOSS: {mse:.4f}")

next_mean_rr = regressor.predict([[95]])
# breakpoint()
#print(f"95 HR is: {next_mean_rr[0]:.4f}")

plt.figure(figsize=(8, 5))  # Optional: Adjust figure size
plt.scatter(data["MEAN_RR"][:1000].values, data["HR"][:1000].values, color="skyblue", label="ECG Data")

# X_pred = np.linspace(meanrr_train.min(), meanrr_train.max(), 10000)

# plt.plot(X_pred, y_pred, color="red", linewidth=2, label="Regression Line")

plt.xlabel("Mean_RR")
plt.ylabel("Heart Rate")

plt.title("Mean RR vs. Heart Rate")
# plt.legend()
plt.grid(True)
plt.show(block=True)

