import statsmodels.api as sm
import numpy as np

# Example time series data
data = np.random.randn(100)  # Replace this with your actual time series data

# Define a simpler Markov Switching Model without switching variances and exog_tvtp
model = sm.tsa.MarkovAutoregression(data, k_regimes=2, order=1, switching_ar=True, switching_variance=False)

# Fit the model to the data
fitted_model = model.fit()

# In-sample prediction
in_sample_preds = fitted_model.predict(start=0, end=len(data)-1, probabilities='smoothed')

# Out-of-sample forecasting (e.g., 20 steps ahead)
out_of_sample_forecast = fitted_model.predict(start=len(data), end=len(data)+19, probabilities='smoothed')

print("In-sample Predictions:", in_sample_preds)
print("Out-of-sample Forecast:", out_of_sample_forecast)