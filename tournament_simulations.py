import pandas as pd
import numpy as np
import streamlit as st

# Streamlit app layout
st.title('NCAA Tournament Simulation with Interactive Bracket')
# Add a small note
st.markdown("<small style='color:gray;'>*Minor Bugs with Changing Results: must press the button twice*</small>", unsafe_allow_html=True)
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
    "predicted_barthag_men.csv")

# Define tournament bracket (64 teams, ordered)
# 68 teams in the tournament
teams = [
    "Alabama", "Alabama St.", "Akron", "American", "Arkansas", "Auburn", "Baylor",
    "Bryant", "BYU", "Clemson", "Colorado St.", "Creighton", "Drake", "Duke", "Florida", "Georgia",
    "Gonzaga", "High Point", "Houston", "Illinois", "Kansas", "Kentucky", "Liberty", "Lipscomb", "Louisville",
    "Marquette", "McNeese St.", "Michigan", "Michigan St.", "Mississippi St.", "Missouri", "Montana", "Mount St. Mary's",
    "New Mexico", "North Carolina", "Oklahoma", "Nebraska Omaha", "Oregon", "Purdue", "Robert Morris",
    "Saint Francis", "Saint Mary's", "SIU Edwardsville", "St. John's", "Tennessee", "Texas", "Texas A&M",
    "Texas Tech", "Troy", "UCLA", "UC San Diego", "Utah St.", "VCU", "Vanderbilt", "Wofford", "Wisconsin",
    "Xavier", "Yale", "UNC Wilmington", "Grand Canyon", "Maryland", "Memphis", "Connecticut", "Norfolk St.", "Arizona",
    "San Diego St.", "Iowa St.", "Mississippi"
]

# Filter strengths_df to only include teams in the list
tournament_df = tournament_df[tournament_df['team'].isin(teams)]

# Ensure the column name matches your dataset
tournament_df.rename(columns={'team': 'team_names'}, inplace=True)
# sets the seeds and regions of each team
seed_region_mapping = {
    # South Region
    "Auburn": (1, "South"),
    "Alabama St.": (16, "South"),
    "Saint Francis": (16, "South"),
    "Louisville": (8, "South"),
    "Creighton": (9, "South"),
    "Michigan": (5, "South"),
    "UC San Diego": (12, "South"),
    "Texas A&M": (4, "South"),
    "Yale": (13, "South"),
    "Mississippi": (6, "South"),
    "San Diego St.": (11, "South"),
    "North Carolina": (11, "South"),
    "Iowa St.": (3, "South"),
    "Lipscomb": (14, "South"),
    "Marquette": (7, "South"),
    "New Mexico": (10, "South"),
    "Michigan St.": (2, "South"),
    "Bryant": (15, "South"),

    # East Region
    "Duke": (1, "East"),
    "American": (16, "East"),
    "Mount St. Mary's": (16, "East"),
    "Mississippi St.": (8, "East"),
    "Baylor": (9, "East"),
    "Oregon": (5, "East"),
    "Liberty": (12, "East"),
    "Arizona": (4, "East"),
    "Akron": (13, "East"),
    "BYU": (6, "East"),
    "VCU": (11, "East"),
    "Wisconsin": (3, "East"),
    "Montana": (14, "East"),
    "Saint Mary's": (7, "East"),
    "Vanderbilt": (10, "East"),
    "Alabama": (2, "East"),
    "Robert Morris": (15, "East"),

    # Midwest Region
    "Houston": (1, "Midwest"),
    "SIU Edwardsville": (16, "Midwest"),
    "Gonzaga": (8, "Midwest"),
    "Georgia": (9, "Midwest"),
    "Clemson": (5, "Midwest"),
    "McNeese St.": (12, "Midwest"),
    "Purdue": (4, "Midwest"),
    "High Point": (13, "Midwest"),
    "Illinois": (6, "Midwest"),
    "Texas": (11, "Midwest"),
    "Xavier": (11, "Midwest"),
    "Kentucky": (3, "Midwest"),
    "Troy": (14, "Midwest"),
    "UCLA": (7, "Midwest"),
    "Utah St.": (10, "Midwest"),
    "Tennessee": (2, "Midwest"),
    "Wofford": (15, "Midwest"),

    # West Region
    "Florida": (1, "West"),
    "Norfolk St.": (16, "West"),
    "Connecticut": (8, "West"),
    "Oklahoma": (9, "West"),
    "Memphis": (5, "West"),
    "Colorado St.": (12, "West"),
    "Maryland": (4, "West"),
    "Grand Canyon": (13, "West"),
    "Missouri": (6, "West"),
    "Drake": (11, "West"),
    "Texas Tech": (3, "West"),
    "UNC Wilmington": (14, "West"),
    "Kansas": (7, "West"),
    "Arkansas": (10, "West"),
    "St. John's": (2, "West"),
    "Nebraska Omaha": (15, "West"),
}

seed_region_df = pd.DataFrame.from_dict(seed_region_mapping, orient='index', columns=['Seed', 'Region']).reset_index()
seed_region_df.rename(columns={'index': 'team_names'}, inplace=True)


# Merge with team data
tournament_df = tournament_df.merge(seed_region_df, on="team_names", how="left")
tournament_df = tournament_df.set_index('team_names').loc[teams].reset_index()

# Extract the teams in the order of seed_region_mapping
ordered_teams = list(seed_region_mapping.keys())

tournament_df['team_names'] = pd.Categorical(tournament_df['team_names'], categories=ordered_teams, ordered=True)
tournament_df = tournament_df.sort_values(by="team_names")

# Load past seed advancement probabilities
past_results = pd.read_csv("past_tournament_rounds.csv")
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
    weight = 0.95
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
            mean1, std1 = team1['Predicted_BARTHAG_Mean'], team1['Predicted_BARTHAG_Std']
            mean2, std2 = team2['Predicted_BARTHAG_Mean'], team2['Predicted_BARTHAG_Std']

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
            #st.write(f"### {team1['team_names']} ({team1['Seed']}) vs {team2['team_names']} ({team2['Seed']}) → Winner: {simulated_winner}")

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

    #st.write("### First Four Matchups")
    simulate_first_four()  # Get winners from the first four games

    current_round_teams = [team for team in tournament_df['team_names']]

    # Initialize or update matchups in session state
    if 'selected_winners' not in st.session_state:
        st.session_state.selected_winners = {}

    for round_name in round_names:
        #st.write(f"### {round_name}")
        round_winners = []

        round_idx = round_names.index(round_name) + 1

        for i in range(0, len(current_round_teams), 2):
            if i + 1 >= len(current_round_teams):
                break

            team1_name, team2_name = current_round_teams[i], current_round_teams[i + 1]

            team1 = tournament_df[tournament_df['team_names'] == team1_name].iloc[0]
            team2 = tournament_df[tournament_df['team_names'] == team2_name].iloc[0]

            mean1, std1 = team1['Predicted_BARTHAG_Mean'], team1['Predicted_BARTHAG_Std']
            mean2, std2 = team2['Predicted_BARTHAG_Mean'], team2['Predicted_BARTHAG_Std']

            seed1_prob = past_results[past_results['Seed'] == team1['Seed']].iloc[0][
                past_results.columns[round_idx]]
            seed2_prob = past_results[past_results['Seed'] == team2['Seed']].iloc[0][
                past_results.columns[round_idx]]

            simulated_winner = simulate_game(team1_name, team2_name, mean1, mean2, std1, std2, seed1_prob, seed2_prob)

            seed1 = team1["Seed"]
            seed2 = team2["Seed"]

            round_matchups[round_name].append((team1_name, team2_name, simulated_winner))

            # Display the matchup and winner
            #st.write(f"**{team1_name} ({seed1}) vs {team2_name} ({seed2})** → Winner: **{simulated_winner}**")

            round_winners.append(simulated_winner)

        current_round_teams = round_winners

        st.session_state.round_matchups = round_matchups
        st.session_state.current_round_teams = current_round_teams
        st.session_state.tournament_simulated = True

    return round_matchups, current_round_teams[0], simulated_winner


import streamlit as st


def reset_simulation():
    if 'selected_winners' in st.session_state:
        st.session_state.selected_winners.clear()

    if 'round_matchups' in st.session_state:
        del st.session_state.round_matchups  # Remove matchups to force a new simulation

    if 'current_round_teams' in st.session_state:
        st.session_state.current_round_teams.clear()

    if 'tournament_simulated' in st.session_state:
        st.session_state.tournament_simulated = False

    if 'radio_button_state' in st.session_state:
        del st.session_state['radio_button_state']

    st.session_state.simulation_num = 1  # Reset simulation counter


# Button to simulate a new tournament
if st.button('Simulate New Tournament'):
    reset_simulation()
    st.session_state.simulation_num += 1
    st.session_state.tournament_simulated = True

# Run simulation if needed
if st.session_state.tournament_simulated or 'round_matchups' not in st.session_state:
    simulated_tournament = simulate_tournament()
    st.session_state.round_matchups = simulated_tournament[0]  # Store matchups persistently
    st.session_state.tournament_simulated = False  # Reset flag

# Ensure selected winners state is initialized
if 'selected_winners' not in st.session_state:
    st.session_state.selected_winners = {}

# Retrieve stored matchups
round_matchups = st.session_state.round_matchups

# Extract winners (track dynamic selections)
for round_name, matchups in round_matchups.items():
    st.write(f"### {round_name}")
    st.session_state.selected_winners[round_name] = []  # Store winners dynamically

    for i, matchup in enumerate(matchups):
        team1_name, team2_name, simulated_winner = matchup
        seed1, _ = seed_region_mapping[team1_name]
        seed2, _ = seed_region_mapping[team2_name]

        # Default selection (if user hasn't picked, use simulation)
        default_winner = simulated_winner if simulated_winner else team1_name
        selected_winner = st.session_state.selected_winners.get(f"{round_name}_{i}", default_winner)

        # Show radio button
        winner = st.radio(
            f"{team1_name} ({seed1}) vs {team2_name} ({seed2})",
            options=[team1_name, team2_name],
            key=f"{round_name}_{i}_{st.session_state.simulation_num}",
            index=0 if selected_winner == team1_name else 1
        )

        # Immediately update the winner in session state upon selection
        st.session_state.selected_winners[f"{round_name}_{i}"] = winner
        st.session_state.selected_winners[round_name].append(winner)


# Function to propagate winners to all future rounds
def update_future_rounds():
    round_keys = list(st.session_state.round_matchups.keys())

    # Initialize a dictionary to store winners for each round
    round_winners = {round_name: st.session_state.selected_winners.get(round_name, []) for round_name in round_keys}

    # Iterate through all rounds to dynamically update matchups
    for i in range(len(round_keys) - 1):  # Loop over all rounds except the last one
        current_round = round_keys[i]
        next_round = round_keys[i + 1]

        # Get winners from the current round
        current_winners = round_winners[current_round]

        # Create matchups for the next round based on winners of the current round
        next_round_matchups = []
        for j in range(0, len(current_winners), 2):
            if j + 1 < len(current_winners):
                next_round_matchups.append((current_winners[j], current_winners[j + 1], None))  # No predefined winner

        # Store the matchups for the next round
        st.session_state.round_matchups[next_round] = next_round_matchups


# After user selections, update all future rounds dynamically
update_future_rounds()
