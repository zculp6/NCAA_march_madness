import pandas as pd
import kagglehub
import pymc as pm
import arviz as az
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.special import expit  # Sigmoid function

# Download latest version
path = kagglehub.dataset_download("andrewsundberg/college-basketball-dataset")

# File paths
file_path = f"cbb_past_results.csv"
file_path_new = f"mbb_25.csv"

# Set display options
pd.set_option('display.max_columns', None)

# Load datasets
df = pd.read_csv(file_path)  # Original data
df_new = pd.read_csv(file_path_new)  # 2025 season


# Bayesian Logistic Regression Model
def run_bayesian_logistic_model():
    df['win_percentage'] = df["wins"] / df["games"]

    # Convert categorical conference variable into dummy variables
    df_dummies = pd.get_dummies(df, columns=['conf'], drop_first=True)
    df_new_dummies = pd.get_dummies(df_new, columns=['conf'], drop_first=True)

    # Define predictor and target variables
    x = df_dummies.drop(columns=['barthag', "team", "wins", "games", "def_ft_pct"])
    y = df_dummies['barthag']

    # Standardize predictors
    scaler_x = StandardScaler()
    x_scaled = scaler_x.fit_transform(x)

    with pm.Model() as model:
        # Prior distributions
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        betas = pm.Normal('betas', mu=0, sigma=1, shape=x_scaled.shape[1])

        x_shared = pm.Data("X_shared", x_scaled)
        y_shared = pm.Data("y_shared", y)

        # Logistic link function
        mu = pm.math.sigmoid(alpha + pm.math.dot(x_shared, betas))

        # Likelihood: Use Beta distribution (as BARTHAG is continuous between 0 and 1)
        likelihood = pm.Beta('y_obs', alpha=mu * 1, beta=(1 - mu) * 1, observed=y_shared)

        # Sampling
        trace = pm.sample(5000, tune=2000, cores=4, target_accept=0.99)

    # Posterior analysis
    print(az.summary(trace))

    # Plot trace and posterior distributions
    az.plot_trace(trace)
    az.plot_posterior(trace, var_names=['alpha', 'betas'])
    az.plot_energy(trace)
    print(az.rhat(trace))

    # Prepare new data for prediction
    columns_to_drop = ['barthag', 'team', 'wins', 'games', "def_ft_pct"]
    columns_to_drop = [col for col in columns_to_drop if col in df_new_dummies.columns]
    x_new = df_new_dummies.drop(columns=columns_to_drop)

    # Align new data columns with training data
    x_new_aligned, _ = x_new.align(x, join='right', axis=1, fill_value=0)
    x_new_scaled = scaler_x.transform(x_new_aligned)

    # Extract posterior samples
    alpha_vals = trace.posterior['alpha'].values.mean(axis=0)
    betas_vals = trace.posterior['betas'].values.mean(axis=0)

    # Compute raw predictions (before logistic transformation)
    mu_new_samples = alpha_vals + np.dot(x_new_scaled, betas_vals.T)

    # Apply logistic (sigmoid) transformation to keep values between 0 and 1
    y_pred_mean = expit(mu_new_samples.mean(axis=1))
    y_pred_std = np.std(expit(mu_new_samples), axis=1)

    # Save predictions to CSV
    predictions = pd.DataFrame({
        'team': df_new['team'],
        'Predicted_BARTHAG_Mean': y_pred_mean,
        'Predicted_BARTHAG_Std': y_pred_std
    })

    predictions.to_csv("predicted_barthag_2025.csv", index=False)
    print("\nPredicted team strengths saved to 'predicted_barthag_2025.csv'.")


if __name__ == '__main__':
    run_bayesian_logistic_model()


