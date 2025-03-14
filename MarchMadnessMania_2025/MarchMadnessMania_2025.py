import pandas as pd
import kagglehub
import pymc as pm
import arviz as az
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.special import expit  # Sigmoid function

# Download latest version
path = kagglehub.dataset_download("andrewsundberg/college-basketball-dataset")

# Set display options
pd.set_option('display.max_columns', None)

# Load datasets
men_df = pd.read_csv(f"C:/Users/ASUS/OneDrive/Desktop/Python/NCAA_Project/NCAA_march_madness/mbb_past_results.csv")
men_df_new = pd.read_csv(f"C:/Users/ASUS/OneDrive/Desktop/Python/NCAA_Project/NCAA_march_madness/mbb_25.csv")

women_df = pd.read_csv(f"C:/Users/ASUS/OneDrive/Desktop/Python/NCAA_Project/NCAA_march_madness/wbb_past_results.csv")
women_df_new = pd.read_csv(f"C:/Users/ASUS/OneDrive/Desktop/Python/NCAA_Project/NCAA_march_madness/wbb_25.csv")

# Remove the Ivy League teams that didn't play in 2020-21 season due to COVID
women_df = women_df[women_df["games"] != 0]
# Bayesian Logistic Regression Model
def run_bayesian_logistic_model(df, df_new, file_path):
    df['win_percentage'] = df["wins"] / df["games"]

    # Convert categorical conference variable into dummy variables
    df_dummies = pd.get_dummies(df, columns=['conf'], drop_first=True)
    df_new_dummies = pd.get_dummies(df_new, columns=['conf'], drop_first=True)
    # Define predictor and target variables
    x = df_dummies.drop(columns=['barthag', "team", "wins", "games","def_ft_pct"])
    y = df_dummies['barthag']

    # Standardize predictors
    scaler_x = StandardScaler()
    x_scaled = scaler_x.fit_transform(x)
    #print(pd.DataFrame(x_scaled).isna().sum())

    with pm.Model() as model:
        # Prior distributions
        alpha = pm.Normal('alpha', mu=0, sigma=1)
        betas = pm.Normal('betas', mu=0, sigma=1, shape=x_scaled.shape[1])

        x_shared = pm.Data("X_shared", x_scaled)
        y_shared = pm.Data("y_shared", y)

        # Logistic link function
        mu = pm.math.sigmoid(alpha + pm.math.dot(x_shared, betas))

        # Likelihood: Use Beta distribution (as BARTHAG is continuous between 0 and 1)
        likelihood = pm.Beta('y_obs', alpha=mu * 10, beta=(1 - mu) * 10, observed=y_shared)

        # Sampling
        trace = pm.sample(10000, tune=2000, cores=4, target_accept=0.99)

    # Posterior analysis
    print(az.summary(trace))

    # Plot trace and posterior distributions
    az.plot_trace(trace)
    az.plot_posterior(trace, var_names=['alpha', 'betas'])
    az.plot_energy(trace)
    print(az.rhat(trace))

    # Prepare new data for prediction
    columns_to_drop = ['barthag', 'team', 'wins', 'games',"def_ft_pct"]
    columns_to_drop = [col for col in columns_to_drop if col in df_new_dummies.columns]
    x_new = df_new_dummies.drop(columns=columns_to_drop)

    # Align new data columns with training data
    x_new_aligned, _ = x_new.align(x, join='right', axis=1, fill_value=0)
    x_new_scaled = scaler_x.transform(x_new_aligned)

    # Extract posterior samples
    alpha_vals = trace.posterior['alpha'].values.mean(axis=0)  # (2000 samples,)
    betas_vals = trace.posterior['betas'].values.mean(axis=0)  # (2000 samples, num_features)

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
    # Adjust the predicted standard deviation based on the number of games played
    predictions['Adjusted_Std_Dev'] = predictions['Predicted_BARTHAG_Std'] / np.sqrt(men_df_new['games'])

    predictions.to_csv(file_path, index=False)
    print("\nPredicted team strengths saved to file.")


if __name__ == '__main__':
    #run_bayesian_logistic_model(men_df, men_df_new, "predicted_barthag_men.csv")
    run_bayesian_logistic_model(women_df, women_df_new, "predicted_barthag_women.csv")
