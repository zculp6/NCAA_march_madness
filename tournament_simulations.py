import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import math

# Streamlit app layout
st.title('NCAA Tournament Simulation with Interactive Bracket')
# Initialize session state for the tournament simulation if it doesn't exist
if 'round_matchups' not in st.session_state:
    st.session_state.round_matchups = {}
if 'current_round_teams' not in st.session_state:
    st.session_state.current_round_teams = []
if 'tournament_results' not in st.session_state:
    st.session_state.tournament_results = {}
if 'tournament_simulated' not in st.session_state:
    st.session_state.tournament_simulated = False
# Initialize simulation_num in session state if it doesn't exist
if 'simulation_num' not in st.session_state:
    st.session_state.simulation_num = 0  # Initialize it to 0 or another starting value
# Function to reset session state when button is pressed
def reset_session_state():
    if 'selected_winners' in st.session_state:
        del st.session_state['selected_winners']
    if 'round_matchups' in st.session_state:
        del st.session_state['round_matchups']
    if 'current_round_teams' in st.session_state:
        del st.session_state['current_round_teams']
    if 'tournament_simulated' in st.session_state:
        del st.session_state['tournament_simulated']

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
# Load team strengths
tournament_df = pd.read_csv(
    "C:/Users/ASUS/OneDrive/Desktop/NCAA_Project/NCAA_march_madness/predicted_barthag_2024.csv")

# Define tournament bracket (64 teams, ordered)
# 68 teams in the tournament
teams = [ "Connecticut", "Stetson", "Northwestern", "Florida Atlantic", "San Diego St.", "UAB", "Yale", "Auburn", "Duquesne",
            "BYU", "Illinois", "Morehead St.", "Washington St.", "Drake", "Iowa St.", "South Dakota St.",
            "North Carolina", "Wagner", "Howard", "Michigan St.", "Mississippi St.", "Grand Canyon", "Saint Mary's",
            "Alabama","College of Charleston", "Clemson", "New Mexico", "Baylor", "Colgate", "Dayton", "Nevada",
            "Arizona", "Long Beach St.", "Houston", "Longwood", "Texas A&M", "Nebraska", "James Madison", "Wisconsin",
            "Duke", "Vermont", "North Carolina St.", "Texas Tech", "Oakland", "Kentucky", "Colorado", "Boise St.",
            "Florida", "Marquette", "Western Kentucky", "Purdue", "Grambling St.", "Montana St.", "Utah St.", "TCU",
            "Gonzaga", "McNeese St.", "Kansas", "Samford", "Oregon", "South Carolina", "Creighton", "Akron", "Texas",
            "Colorado St.", "Virginia", "Tennessee", "Saint Peter's"]
# Filter strengths_df to only include teams in the list
tournament_df = tournament_df[tournament_df['team'].isin(teams)]

# Ensure the column name matches your dataset
tournament_df.rename(columns={'team': 'team_names'}, inplace=True)
# sets the seeds and regions of each team
seed_region_mapping = { "Connecticut": (1, "East"),
                        "Stetson": (16, "East"),
                        "Northwestern": (9, "East"),
                        "Florida Atlantic": (8, "East"),
                        "San Diego St.": (5, "East"),
                        "UAB": (12, "East"),
                        "Yale": (13, "East"),
                        "Auburn": (4, "East"),
                        "Duquesne": (11, "East"),
                        "BYU": (6, "East"),
                        "Illinois": (3, "East"),
                        "Morehead St.": (14, "East"),
                        "Washington St.": (7, "East"),
                        "Drake": (10, "East"),
                        "Iowa St.": (2, "East"),
                        "South Dakota St.": (15, "East"),
                        "North Carolina": (1, "West"),
                        "Wagner": (16, "West"),
                        "Howard": (16, "West"),
                        "Michigan St.": (9, "West"),
                        "Mississippi St.": (8, "West"),
                        "Grand Canyon": (12, "West"),
                        "Saint Mary's": (5, "West"),
                        "Alabama": (4, "West"),
                        "College of Charleston": (13, "West"),
                        "Clemson": (6, "West"),
                        "New Mexico": (11, "West"),
                        "Baylor": (3, "West"),
                        "Colgate": (14, "West"),
                        "Dayton": (7, "West"),
                        "Nevada": (10, "West"),
                        "Arizona": (2, "West"),
                        "Long Beach St.": (15, "West"),
                        "Houston": (1, "South"),
                        "Longwood": (16, "South"),
                        "Texas A&M": (9, "South"),
                        "Nebraska": (8, "South"),
                        "James Madison": (12, "South"),
                        "Wisconsin": (5, "South"),
                        "Duke": (4, "South"),
                        "Vermont": (13, "South"),
                        "North Carolina St.": (11, "South"),
                        "Texas Tech": (6, "South"),
                        "Oakland": (14, "South"),
                        "Kentucky": (3, "South"),
                        "Colorado": (10, "South"),
                        "Boise St.": (10, "South"),
                        "Florida": (7, "South"),
                        "Marquette": (2, "South"),
                        "Western Kentucky": (15, "South"),
                        "Purdue": (1, "Midwest"),
                        "Grambling St.": (16, "Midwest"),
                        "Montana St.": (16, "Midwest"),
                        "Utah St.": (8, "Midwest"),
                        "TCU": (9, "Midwest"),
                        "Gonzaga": (5, "Midwest"),
                        "McNeese St.": (12, "Midwest"),
                        "Kansas": (4, "Midwest"),
                        "Samford": (13, "Midwest"),
                        "Oregon": (11, "Midwest"),
                        "South Carolina": (6, "Midwest"),
                        "Creighton": (3, "Midwest"),
                        "Akron": (14, "Midwest"),
                        "Texas": (7, "Midwest"),
                        "Colorado St.": (10, "Midwest"),
                        "Virginia": (10, "Midwest"),
                        "Tennessee": (2, "Midwest"),
                        "Saint Peter's": (15, "Midwest") }
seed_region_df = pd.DataFrame.from_dict(seed_region_mapping, orient='index', columns=['Seed', 'Region']).reset_index()
seed_region_df.rename(columns={'index': 'team_names'}, inplace=True)


# Merge with team data
tournament_df = tournament_df.merge(seed_region_df, on="team_names", how="left")
tournament_df = tournament_df.set_index('team_names').loc[teams].reset_index()

# Load past seed advancement probabilities
past_results = pd.read_csv("C:/Users/ASUS/OneDrive/Desktop/NCAA_Project/NCAA_march_madness/past_tournament_rounds.csv")
round_columns = past_results.columns[1:]  # All columns except 'Seed'
# Create a copy of the original data (to reference unmodified values)
original_values = past_results[round_columns].copy()
# Compute conditional probabilities (P(advancing | reached the round))
for i in range(len(round_columns) - 1):
    past_results[round_columns[i+1]] = original_values[round_columns[i+1]] / original_values[round_columns[i]]
# Fill NaNs with 0 (in case of division by zero)
past_results.fillna(0, inplace=True)

# Drop the "Round of 64" column; was only necessary for probability calculation
past_results.drop(columns=["Round of 64"], inplace=True)
past_results.drop(index=[16,17,18], inplace=True)
# Display the modified dataframe
#print(past_results)

# Merge probabilities into the tournament dataframe
tournament_df = tournament_df.merge(past_results, on="Seed", how="left")

# Simulate a game
def simulate_game(team1, team2, mean1, mean2, std1, std2, seed1_prob, seed2_prob):
    weight = 0.35
    team1_strength = weight * np.random.normal(mean1, std1) + (1 - weight) * seed1_prob
    team2_strength = weight * np.random.normal(mean2, std2) + (1 - weight) * seed2_prob
    return team1 if team1_strength > team2_strength else team2


def simulate_first_four():
    global tournament_df
    first_four_winners = []
    losers = []

    # Group the DataFrame by 'Region' and 'Seed'
    grouped = tournament_df.groupby(['Region', 'Seed'])

    # Iterate through each group (Region, Seed)
    for (region, seed), group in grouped:
        if len(group) == 2:  # Only simulate if there are exactly two teams in the group
            # Get the two teams
            team1, team2 = group.iloc[0], group.iloc[1]

            # Extract mean and std for each team from tournament_df
            mean1, std1 = team1['Predicted_barthag_Mean'], team1['Predicted_barthag_Std']
            mean2, std2 = team2['Predicted_barthag_Mean'], team2['Predicted_barthag_Std']

            # Simulate the game
            simulated_winner = simulate_game(
                team1['team_names'], team2['team_names'],
                mean1, mean2, std1, std2,
                0, 0  # Default probabilities for seed (not used for simulation)
            )

            loser = team1['team_names'] if simulated_winner != team1['team_names'] else team2['team_names']
            losers.append(loser)
            first_four_winners.append(simulated_winner)

            # Print the results
            st.write(f"### {team1['team_names']} ({team1['Seed']}) vs {team2['team_names']} ({team2['Seed']}) → Winner: {simulated_winner}")

    # Remove the losers from the DataFrame
    tournament_df = tournament_df[~tournament_df['team_names'].isin(losers)]

    return first_four_winners


def simulate_tournament():
    reset_session_state() # to avoid duplicate widget keys when simulating again
    round_names = ["Round of 64", "Round of 32", "Sweet 16", "Elite 8", "Final Four", "Championship"]
    round_name_mapping = {
        "Round of 64": 0,
        "Round of 32": 1,
        "Sweet 16": 2,
        "Elite 8": 3,
        "Final Four": 4,
        "Championship": 5
    }

    round_matchups = {round_name: [] for round_name in round_names}

    st.write("### First Four Matchups")
    first_four_winners = simulate_first_four()  # Get winners from the first four games

    current_round_teams = [team for team in tournament_df['team_names']]

    # Initialize or update matchups in session state
    if 'selected_winners' not in st.session_state:
        st.session_state.selected_winners = {}

    for round_name in round_names:
        st.write(f"### {round_name}")
        round_winners = []

        round_idx = round_names.index(round_name) + 1

        for i in range(0, len(current_round_teams), 2):
            if i + 1 >= len(current_round_teams):
                break

            team1_name, team2_name = current_round_teams[i], current_round_teams[i + 1]

            team1 = tournament_df[tournament_df['team_names'] == team1_name].iloc[0]
            team2 = tournament_df[tournament_df['team_names'] == team2_name].iloc[0]

            mean1, std1 = team1['Predicted_barthag_Mean'], team1['Predicted_barthag_Std']
            mean2, std2 = team2['Predicted_barthag_Mean'], team2['Predicted_barthag_Std']

            seed1_prob = past_results[past_results['Seed'] == team1['Seed']].iloc[0][
                past_results.columns[round_idx]]
            seed2_prob = past_results[past_results['Seed'] == team2['Seed']].iloc[0][
                past_results.columns[round_idx]]

            simulated_winner = simulate_game(team1_name, team2_name, mean1, mean2, std1, std2, seed1_prob, seed2_prob)

            seed1 = team1["Seed"]
            seed2 = team2["Seed"]

            round_matchups[round_name].append((team1_name, team2_name, simulated_winner))

            # Use a radio button for the user to select the winner
            winner = st.radio(
                f"{team1_name} ({seed1}) vs {team2_name} ({seed2})",
                options=[f"{team1_name} ({seed1})", f"{team2_name} ({seed2})"],
                #index=[f"{team1_name} ({seed1})", f"{team2_name} ({seed2})"].index(
                #    f"{simulated_winner} ({team1['Seed'] if simulated_winner == team1_name else team2['Seed']})"),
                key=f"{team1_name}_{team2_name}_{st.session_state.simulation_num}",
            )

            # Update the session state with the selected winner
            st.session_state.selected_winners[f"{team1_name}_{team2_name}"] = winner.split(' (')[0]

            round_winners.append(st.session_state.selected_winners[f"{team1_name}_{team2_name}"])

        current_round_teams = round_winners

        st.session_state.round_matchups = round_matchups
        st.session_state.current_round_teams = current_round_teams
        st.session_state.tournament_simulated = True

    return round_matchups, current_round_teams[0], winner

def reset_simulation():
    if 'selected_winners' in st.session_state:
        st.session_state.selected_winners.clear()  # Reset selected winners

    if 'round_matchups' in st.session_state:
        st.session_state.round_matchups.clear()  # Reset round matchups

    if 'current_round_teams' in st.session_state:
        st.session_state.current_round_teams.clear()  # Reset teams for the next simulation

    if 'tournament_simulated' in st.session_state:
        st.session_state.tournament_simulated = False  # Reset simulation status

    # Ensure that radio buttons' state is reset too by clearing related session state
    if 'radio_button_state' in st.session_state:
        del st.session_state['radio_button_state']  # Remove the stored radio button values to reset widgets
        # Reset simulation number
        st.session_state.simulation_num = 1

# Add a button to simulate a new tournament
if st.button('Simulate New Tournament'):
    reset_simulation() # Reset the flag to trigger a new simulation
    st.session_state.simulation_num += 1
    simulate_tournament()

simulated_tournament = simulate_tournament()
# Assuming simulated_tournament contains round_matchups
round_matchups = simulated_tournament[0]  # This is the round matchups dictionary
# Extracting winners (the third element of each tuple) from the dictionary
# Assuming 'round_matchups' is already defined as provided in your example
winners_each_round = {round_name: [matchup[2] for matchup in matchups] for round_name, matchups in
                      round_matchups.items()}

# Iterating through the rounds
for round_name, matchups in round_matchups.items():
    for matchup in matchups:
        team1_name, team2_name, simulated_winner = matchup
        seed1 = 1  # Example seed, adjust as needed
        seed2 = 2  # Example seed, adjust as needed

        # Check if the winner has already been selected from the previous round
        if simulated_winner is None:
            winner = st.radio(
                f"{team1_name} ({seed1}) vs {team2_name} ({seed2})",
                options=[f"{team1_name} ({seed1})", f"{team2_name} ({seed2})"],
                key=f"{team1_name}_{team2_name}_{simulation_num}_{st.session_state.simulation_num}",
            )
            # If no winner is selected, show a message
            if winner is None:
                st.warning("Please select a winner before proceeding.")
        else:
            # If winner is determined (e.g., from a simulated or previous round), use it
            winner = simulated_winner

        # Update the winner for the round
        # Assuming you are updating the round matchups or something similar
        # In this case, you might want to add the winner to the round's list or perform further actions
        winners_each_round[round_name] = [winner for winner in matchups]

        #st.write(f"Winner of {team1_name} vs {team2_name}: {winner}")


