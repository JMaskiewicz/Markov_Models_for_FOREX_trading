import pandas as pd
import numpy as np
import pandas_datareader as pdr
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
import warnings
warnings.filterwarnings("ignore")  # To ignore convergence warnings

# Fetch data
start_date = '2010-01-01'
end_date = '2021-12-31'
df_train = pdr.get_data_fred('DEXUSEU', start=start_date, end=end_date)

# Drop NaN values (if any)
df_train.dropna(inplace=True)

# calculate returns
df_train['Returns'] = df_train['DEXUSEU'].pct_change()

# Calculate log returns
df_train['Log_Returns'] = np.log(df_train['DEXUSEU'] / df_train['DEXUSEU'].shift(1))

# Drop NaN values created by shift operation
df_train.dropna(inplace=True)

# Standardize the log returns
#  df_train['Standardized_Log_Returns'] = (df_train['Log_Returns'] - df_train['Log_Returns'].mean()) / df_train['Log_Returns'].std()

# Initialize variables to track the best model
best_aic = float('inf')
best_model = None
best_k_regimes = None

# Loop over different numbers of regimes
for k_regimes in range(2, 5):
    mod = MarkovRegression(df_train['Log_Returns'], k_regimes=k_regimes, trend='c', switching_variance=True)
    res = mod.fit()
    print(f"Model Summary (Regimes: {k_regimes}): AIC: {res.aic}, BIC: {res.bic}")
    # Check if this model has a better AIC
    if res.aic < best_aic:
        best_aic = res.aic
        best_model = res
        best_k_regimes = k_regimes

# Print summary of the best model
print(f"Best Model Summary (Regimes: {best_k_regimes}):")
print(best_model.summary())

# Print AIC and BIC of the best model
print(f"Best Model AIC: {best_model.aic}, BIC: {best_model.bic}")

# Print Transition Matrix of the best model
print("Transition Matrix of the Best Model:")
print(best_model.regime_transition)

# Extract the parameters from the model
params = best_model.params

# The parameters are ordered as [regime1_mean, regime2_mean, ..., regime1_variance, regime2_variance, ...]
# Split them into means and variances
num_states = best_k_regimes
means = params[:num_states]
variances = params[num_states:2*num_states]

# Print the means and variances for each state
for i in range(num_states):
    print(f"State {i+1} Mean: {means[i]}, Variance: {variances[i]}")

import matplotlib.pyplot as plt

# Create subplots (3 rows now)
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Plot 1: Original Time Series Data
axs[0].plot(df_train.index, df_train['DEXUSEU'], label='EUR/USD Exchange Rate')
axs[0].set_title('Original EUR/USD Time Series Data')
axs[0].set_ylabel('Value')
axs[0].legend()

# Plot 2: Standardized Log Returns
axs[1].plot(df_train.index, df_train['Log_Returns'], label='Standardized Log Returns', color='green')
axs[1].set_title('Standardized Log Returns Over Time')
axs[1].set_ylabel('Standardized Log Returns')
axs[1].legend()

# Extract the smoothed probabilities
smoothed_probs = best_model.smoothed_marginal_probabilities

# Plot 3: Probabilities for Each State
for state in range(best_k_regimes):
    axs[2].plot(df_train.index, smoothed_probs[state], label=f'State {state+1}')
axs[2].set_title('Probability of Being in Each State Over Time')
axs[2].set_xlabel('Date')
axs[2].set_ylabel('Probability')
axs[2].legend()

plt.tight_layout()
plt.show()

# Create subplots (3 plots in total)
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Plot 1: Rolling Mean of Log Returns
# Calculating rolling mean with a window (e.g., 30 days)
rolling_mean = df_train['Log_Returns'].rolling(window=50).mean()
axs[0].plot(df_train.index, rolling_mean, label='Rolling Mean (50 days)')
axs[0].set_title('Rolling Mean of Log Returns')
axs[0].set_ylabel('Mean')
axs[0].legend()

# Plot 2: Rolling Variance of Log Returns
rolling_variance = df_train['Log_Returns'].rolling(window=50).var()
axs[1].plot(df_train.index, rolling_variance, label='Rolling Variance (50 days)')
axs[1].set_title('Rolling Variance of Log Returns')
axs[1].set_ylabel('Variance')
axs[1].legend()

# Plot 3: Probabilities for Each State
for state in range(best_k_regimes):
    axs[2].plot(df_train.index, smoothed_probs[state], label=f'State {state+1}')
axs[2].set_title('Probability of Being in Each State Over Time')
axs[2].set_xlabel('Date')
axs[2].set_ylabel('Probability')
axs[2].legend()

plt.tight_layout()
plt.show()
