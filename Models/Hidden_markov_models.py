import pandas as pd
import numpy as np
import pandas_datareader as pdr
from hmmlearn import hmm

# Function to calculate the number of parameters in an HMM
def calculate_hmm_params(n_states, n_features):
    # Transition probabilities
    trans_params = n_states * (n_states - 1)
    # Emission means and covariances for each state
    emit_params = n_states * n_features + n_states * n_features * (n_features + 1) / 2
    # Initial state probabilities
    start_params = n_states - 1
    return int(trans_params + emit_params + start_params)

# Fetch data
start_date = '2010-01-01'
end_date = '2021-12-31'
df = pdr.get_data_fred('DEXUSEU', start=start_date, end=end_date)

# Drop NaN values (if any)
df.dropna(inplace=True)

# Calculate log returns
df['Log_Returns'] = np.log(df['DEXUSEU'] / df['DEXUSEU'].shift(1))

# Drop NaN values created by shift operation
df.dropna(inplace=True)

# Standardize the log returns
mean = df['Log_Returns'].mean()
std = df['Log_Returns'].std()
df['Standardized_Log_Returns'] = (df['Log_Returns'] - mean) / std

# Prepare the data for the HMM
X = df['Standardized_Log_Returns'].values.reshape(-1, 1)

# Variables to store the best model and its AIC/BIC
best_aic = np.inf
best_bic = np.inf
best_model = None
best_n_states = None

# Loop over the range of hidden states
for n_states in range(2, 10):
    # Define the HMM
    model = hmm.GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000, random_state=42)

    # Train the HMM
    model.fit(X)

    # Calculate the log likelihood
    log_likelihood = model.score(X)

    # Calculate the number of parameters
    n_params = calculate_hmm_params(n_states, X.shape[1])

    # Calculate AIC and BIC
    aic = 2 * n_params - 2 * log_likelihood
    bic = np.log(X.shape[0]) * n_params - 2 * log_likelihood

    # Check if this model has a better AIC/BIC
    if aic < best_aic:
        best_aic = aic
        best_bic = bic
        best_model = model
        best_n_states = n_states

    print(f"Hidden States: {n_states}, Log Likelihood: {log_likelihood}, AIC: {aic}, BIC: {bic}")

# Print the best model's details
print(f"Best Model: {best_n_states} States, AIC: {best_aic}, BIC: {best_bic}")

# Predict the hidden states of the best model
hidden_states = best_model.predict(X)

# Predict the probabilities of each state
state_probabilities = best_model.predict_proba(X)

# Add hidden states and probabilities to the DataFrame
df['Hidden_State'] = hidden_states

for i in range(best_n_states):
    df[f'State_{i}_Prob'] = state_probabilities[:, i]

# Transition matrix of the best model
transition_matrix = best_model.transmat_
print("Transition Matrix (Probability of moving from one state to another):")
print(transition_matrix)

# Means and variances for each state
state_means = best_model.means_
state_covars = best_model.covars_

print("\nMeans and Variances for each state:")

for i in range(best_n_states):
    print(f"State {i}:")
    print(f"  Mean: {state_means[i][0]}")

    # For 'full' covariance, covars_ would be a matrix. We're interested in the variance, so we'll take diagonal elements.
    if best_model.covariance_type == 'full':
        print(f"  Variance: {np.diag(state_covars[i])[0]}")
    else:
        print(f"  Variance: {state_covars[i][0]}")

import matplotlib.pyplot as plt

# Ensure the DataFrame index is a proper DateTimeIndex for plotting
df.index = pd.to_datetime(df.index)

# Create subplots (3 rows now)
fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

# Plot 1: Original Time Series Data
axs[0].plot(df.index, df['DEXUSEU'], label='EUR/USD Exchange Rate')
axs[0].set_title('Original EUR/USD Time Series Data')
axs[0].set_ylabel('Value')
axs[0].legend()

# Plot 2: Standardized Log Returns
axs[1].plot(df.index, df['Standardized_Log_Returns'], label='Standardized Log Returns', color='green')
axs[1].set_title('Standardized Log Returns Over Time')
axs[1].set_ylabel('Standardized Log Returns')
axs[1].legend()

# Extract the smoothed probabilities
state_probabilities = best_model.predict_proba(X)

# Plot 3: Probabilities for Each State
for i in range(best_n_states):
    axs[2].plot(df.index, state_probabilities[:, i], label=f'State {i}', linewidth=1)

axs[2].set_title('Probability of Being in Each State Over Time (HMM)')
axs[2].set_xlabel('Date')
axs[2].set_ylabel('Probability')
axs[2].legend()

plt.tight_layout()
plt.show()

# Plot 2: Rolling Mean of Log Returns
# Create subplots
fig, axs = plt.subplots(3, 1, figsize=(15, 15), sharex=True)

# Plot 2: Rolling Mean of Log Returns (50 days window)
rolling_mean = df['Log_Returns'].rolling(window=50).mean()
axs[0].plot(df.index, rolling_mean, label='Rolling Mean (50 days)', color='blue')
axs[0].set_title('Rolling Mean of Log Returns (50 days window)')
axs[0].set_ylabel('Mean')
axs[0].legend()

# Plot 2: Rolling Variance of Log Returns (50 days window)
rolling_variance = df['Log_Returns'].rolling(window=50).var()
axs[1].plot(df.index, rolling_variance, label='Rolling Variance (50 days)', color='red')
axs[1].set_title('Rolling Variance of Log Returns (50 days window)')
axs[1].set_ylabel('Variance')
axs[1].legend()

# Plot 3: Probabilities for Each State using state_probabilities
for i in range(best_n_states):
    axs[2].plot(df.index, state_probabilities[:, i], label=f'State {i}', linewidth=1)

axs[2].set_title('Probability of Being in Each State Over Time (HMM)')
axs[2].set_xlabel('Date')
axs[2].set_ylabel('Probability')
axs[2].legend()

plt.tight_layout()
plt.show()
