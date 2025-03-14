import numpy as np
import pandas as pd
from scipy.stats import norm

# Load data
men_data = pd.read_csv(
    "C:/Users/ASUS/OneDrive/Desktop/Python/NCAA_Project/NCAA_march_madness/MarchMadnessMania_2025/predicted_barthag_men.csv")
women_data = pd.read_csv(
    "C:/Users/ASUS/OneDrive/Desktop/Python/NCAA_Project/NCAA_march_madness/MarchMadnessMania_2025/predicted_barthag_women.csv")

#test_data = pd.read_csv("C:/Users/ASUS/OneDrive/Desktop/Python/NCAA_Project/NCAA_march_madness/predicted_barthag_2024.csv")

men_team_id = pd.read_csv(
    "C:/Users/ASUS/OneDrive/Desktop/Python/NCAA_Project/NCAA_march_madness/MarchMadnessMania_2025/MTeamSpellings.csv",
    encoding="Windows-1252"
)
women_team_id = pd.read_csv(
    "C:/Users/ASUS/OneDrive/Desktop/Python/NCAA_Project/NCAA_march_madness/MarchMadnessMania_2025/WTeamSpellings.csv",
    encoding="Windows-1252"
)

### Getting Team ID for men's

men_data['team'] = men_data['team'].str.lower()
men_team_id['TeamNameSpelling'] = men_team_id["TeamNameSpelling"].str.lower()
# Merge based on matching team names
men_merged_data = men_data.merge(men_team_id, left_on="team", right_on="TeamNameSpelling", how="left")

# Drop the original TeamNameSpelling column (optional)
men_merged_data.drop(columns=["TeamNameSpelling"], inplace=True)

teams_with_nan_id = men_merged_data[men_merged_data['TeamID'].isna()]['team']
updated_teams = teams_with_nan_id.str.replace(' ', '-', regex=False)
men_merged_data.loc[men_merged_data['team'].isin(teams_with_nan_id), 'team'] = updated_teams
men_merged_data = men_merged_data.merge(men_team_id, left_on="team", right_on="TeamNameSpelling", how="left")
men_merged_data.drop(columns=["TeamNameSpelling", "TeamID_x"], inplace=True)
# Rename TeamID_Y to TeamID because it created a second TeamID
men_merged_data.rename(columns={'TeamID_y': 'TeamID'}, inplace=True)
# Print rows where TeamID_Y is still NaN
#print(men_merged_data[men_merged_data['TeamID'].isna()][['team', 'TeamID']])
#print(men_merged_data['TeamID'].unique())
# Manually changing the last teams because the abbreviations are not on the list
manual_team_ids = {
    'texas-a&m-corpus-chris': 1394,
    'southeast-missouri-st.': 1369,
    'queens': 1474,
    'ut-rio-grande-valley': 1410,
    'cal-st.-bakersfield': 1167,
    'tarleton-st.': 1470,
    'saint-francis': 1384,
    'texas-a&m-commerce': 1477,
    'mississippi-valley-st.': 1290
}
for team, team_id in manual_team_ids.items():
    men_merged_data.loc[men_merged_data['team'] == team, 'TeamID'] = team_id

# Convert TeamID to an integer (without decimals)
men_merged_data['TeamID'] = men_merged_data['TeamID'].astype(int)
# Print the unique values in the 'TeamID' column
men_merged_data = men_merged_data.sort_values(by='TeamID')

# Verify if the changes were made
#print(men_merged_data[men_merged_data['TeamID'].isna()][['team', 'TeamID']])

### Team ID's for women's

women_data['team'] = women_data['team'].str.lower()
women_team_id['TeamNameSpelling'] = women_team_id["TeamNameSpelling"].str.lower()
# Merge based on matching team names
women_merged_data = women_data.merge(women_team_id, left_on="team", right_on="TeamNameSpelling", how="left")

# Drop the original TeamNameSpelling column (optional)
women_merged_data.drop(columns=["TeamNameSpelling"], inplace=True)

w_teams_with_nan_id = women_merged_data[women_merged_data['TeamID'].isna()]['team']
w_updated_teams = w_teams_with_nan_id.str.replace(' ', '-', regex=False)
women_merged_data.loc[women_merged_data['team'].isin(w_teams_with_nan_id), 'team'] = w_updated_teams
women_merged_data = women_merged_data.merge(women_team_id, left_on="team", right_on="TeamNameSpelling", how="left")
women_merged_data.drop(columns=["TeamNameSpelling", "TeamID_x"], inplace=True)
# Rename TeamID_Y to TeamID because it created a second TeamID
women_merged_data.rename(columns={'TeamID_y': 'TeamID'}, inplace=True)
#print(women_merged_data[women_merged_data['TeamID'].isna()][['team', 'TeamID']])
women_manual_team_ids = {
    'texas-a&m-corpus-chris': 3394,
    'southeast-missouri-st.': 3369,
    'queens': 3474,
    'ut-rio-grande-valley': 3410,
    'cal-st.-bakersfield': 3167,
    'tarleton-st.': 3470,
    'saint-francis': 3384,
    'texas-a&m-commerce': 3477,
    'mississippi-valley-st.': 3290
}
for team, team_id in women_manual_team_ids.items():
    women_merged_data.loc[women_merged_data['team'] == team, 'TeamID'] = team_id

women_merged_data['TeamID'] = women_merged_data['TeamID'].astype(int)
women_merged_data = women_merged_data.sort_values(by='TeamID')

#print(women_merged_data[men_merged_data['TeamID'].isna()][['team', 'TeamID']])

def simulate_game(team1, team2, mean1, mean2, std1, std2):
    # Calculate the mean and std of the difference
    diff_mean = mean1 - mean2
    diff_std = np.sqrt(std1 ** 2 + std2 ** 2)

    # Calculate the probability that team1_strength is greater than team2_strength
    Pred = 1 - norm.cdf(0, loc=diff_mean, scale=diff_std)
    ID = f"2025_{team1}_{team2}"
    return ID, Pred


# Function to simulate matchups in a dataset (men or women)
def simulate_all_matchups(data, adjustedStd = True):
    results = []
    # Iterate over all pairs of teams in the dataset
    for i in range(len(data)):
        for j in range(i + 1, len(data)):  # Only simulate matchups once (team1 vs team2 and not team2 vs team1 again)
            team1 = data.iloc[i]['TeamID']  # Use iloc to select row by index
            team2 = data.iloc[j]['TeamID']  # Use iloc to select row by index

            # Extract the predicted means for each team
            mean1 = data.iloc[i]['Predicted_BARTHAG_Mean']
            mean2 = data.iloc[j]['Predicted_BARTHAG_Mean']

            # Extract the standard deviations
            if adjustedStd:
                std1 = data.iloc[i]['Adjusted_Std_Dev']
                std2 = data.iloc[j]['Adjusted_Std_Dev']
            else:
                std1 = data.iloc[i]['Predicted_BARTHAG_Std']
                std2 = data.iloc[j]['Predicted_BARTHAG_Std']

            # Simulate the game and append the result
            result = simulate_game(team1, team2, mean1, mean2, std1, std2)
            results.append(result)

    # Convert the results to a DataFrame
    result_df = pd.DataFrame(results, columns=['ID','Pred'])
    return result_df


# Simulate all matchups for men and women
men_results = simulate_all_matchups(men_merged_data, False)
women_results = simulate_all_matchups(women_merged_data, False)
#test_results = simulate_all_matchups(test_data)
#test_results.to_csv("test.csv")
# Combine the men and women results into one DataFrame
combined_results = pd.concat([men_results, women_results], ignore_index=True)

# Save the combined results to a CSV file
combined_results.to_csv("C:/Users/ASUS/OneDrive/Desktop/Python/NCAA_Project/NCAA_march_madness/MarchMadnessMania_2025/MarchMadnessMania_2025_predictions.csv", index=False)
