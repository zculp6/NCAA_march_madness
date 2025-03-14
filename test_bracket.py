import streamlit as st

# Sample data for teams and seeds (you should replace this with your actual data)
teams = {
    "East": [{"team": "Alabama", "seed": 1}, {"team": "Texas Southern", "seed": 16}, {"team": "Baylor", "seed": 8},
             {"team": "Kentucky", "seed": 9},
             {"team": "Maryland", "seed": 5}, {"team": "West Virginia", "seed": 12}, {"team": "UCLA", "seed": 4},
             {"team": "Colorado", "seed": 13}],
    "West": [{"team": "Gonzaga", "seed": 1}, {"team": "Georgia State", "seed": 16}, {"team": "Kansas", "seed": 8},
             {"team": "Arkansas", "seed": 9},
             {"team": "Indiana", "seed": 5}, {"team": "Toledo", "seed": 12}, {"team": "Tennessee", "seed": 4},
             {"team": "Iowa", "seed": 13}],
    "South": [{"team": "Duke", "seed": 1}, {"team": "CSU Fullerton", "seed": 16}, {"team": "UCLA", "seed": 8},
              {"team": "North Carolina", "seed": 9},
              {"team": "Miami", "seed": 5}, {"team": "Texas", "seed": 12}, {"team": "Vanderbilt", "seed": 4},
              {"team": "Purdue", "seed": 13}],
    "Midwest": [{"team": "Purdue", "seed": 1}, {"team": "Texas", "seed": 16}, {"team": "Miami", "seed": 8},
                {"team": "Arizona", "seed": 9},
                {"team": "Iowa State", "seed": 5}, {"team": "Kansas", "seed": 12}, {"team": "LSU", "seed": 4},
                {"team": "Creighton", "seed": 13}],
}

# Initialize the bracket structure if not already initialized
if "bracket" not in st.session_state:
    st.session_state.bracket = {
        "Round 1": {},
        "Round of 32": {},
        "Sweet 16": {},
        "Elite 8": {},
        "Final 4": {},
        "Championship": {},
        "Winner": None
    }


# Simulate game (you can replace this with your own game simulation logic)
def simulate_game(team1, team2):
    return team1 if team1 < team2 else team2  # Just a placeholder logic


# --- ROUND 1 (Simulate First Round) ---
st.title("March Madness 64-Team Bracket")

if st.button("Simulate First Round"):
    st.session_state.bracket["Round 1"] = {}
    for region, region_teams in teams.items():
        for i in range(0, len(region_teams), 2):
            matchup = f"{region_teams[i]['team']} (Seed {region_teams[i]['seed']}) vs {region_teams[i + 1]['team']} (Seed {region_teams[i + 1]['seed']})"
            winner = simulate_game(region_teams[i]['team'], region_teams[i + 1]['team'])
            st.session_state.bracket["Round 1"][matchup] = winner

# --- ROUND 1 Results and Manual Override ---
st.subheader("Round of 64")
round_1_winners = []
for region, region_teams in teams.items():
    st.write(f"**{region} Region**")
    for i in range(0, len(region_teams), 2):
        matchup = f"{region_teams[i]['team']} (Seed {region_teams[i]['seed']}) vs {region_teams[i + 1]['team']} (Seed {region_teams[i + 1]['seed']})"

        # Default winner from simulation
        if matchup not in st.session_state.bracket["Round 1"]:
            st.session_state.bracket["Round 1"][matchup] = region_teams[i]['team']

        # Display selectbox for each matchup and allow manual override
        winner = st.selectbox(f"Select winner for {matchup}",
                              matchup.split(" vs "),
                              index=0 if st.session_state.bracket["Round 1"][matchup] == region_teams[i]['team'] else 1,
                              key=f"round_1_{region}_{matchup}")

        st.session_state.bracket["Round 1"][matchup] = winner
        round_1_winners.append(winner)

# --- ROUND OF 32 ---
round_of_32_winners = []  # Initialize the list here before using it
if len(round_1_winners) == 32:
    st.subheader("Round of 32")
    for i in range(0, len(round_1_winners), 2):
        matchup = f"{round_1_winners[i]} vs {round_1_winners[i + 1]}"

        winner = st.selectbox(f"Select winner for {matchup}",
                              matchup.split(" vs "),
                              index=0 if st.session_state.bracket["Round of 32"].get(matchup, round_1_winners[i]) ==
                                         round_1_winners[i] else 1,
                              key=f"round_32_{matchup}")

        st.session_state.bracket["Round of 32"][matchup] = winner
        round_of_32_winners.append(winner)

# --- SWEET 16 ---
sweet_16_winners = []  # Initialize the list here before using it
if len(round_of_32_winners) == 16:
    st.subheader("Sweet 16")
    for i in range(0, len(round_of_32_winners), 2):
        matchup = f"{round_of_32_winners[i]} vs {round_of_32_winners[i + 1]}"

        winner = st.selectbox(f"Select winner for {matchup}",
                              matchup.split(" vs "),
                              index=0 if st.session_state.bracket["Sweet 16"].get(matchup, round_of_32_winners[i]) ==
                                         round_of_32_winners[i] else 1,
                              key=f"sweet_16_{matchup}")

        st.session_state.bracket["Sweet 16"][matchup] = winner
        sweet_16_winners.append(winner)

# --- ELITE 8 ---
elite_8_winners = []  # Initialize the list here before using it
if len(sweet_16_winners) == 8:
    st.subheader("Elite 8")
    for i in range(0, len(sweet_16_winners), 2):
        matchup = f"{sweet_16_winners[i]} vs {sweet_16_winners[i + 1]}"

        winner = st.selectbox(f"Select winner for {matchup}",
                              matchup.split(" vs "),
                              index=0 if st.session_state.bracket["Elite 8"].get(matchup, sweet_16_winners[i]) ==
                                         sweet_16_winners[i] else 1,
                              key=f"elite_8_{matchup}")

        st.session_state.bracket["Elite 8"][matchup] = winner
        elite_8_winners.append(winner)

# --- FINAL 4 ---
final_4_winners = []  # Initialize the list here before using it
if len(elite_8_winners) == 4:
    st.subheader("Final 4")
    for i in range(0, len(elite_8_winners), 2):
        matchup = f"{elite_8_winners[i]} vs {elite_8_winners[i + 1]}"

        winner = st.selectbox(f"Select winner for {matchup}",
                              matchup.split(" vs "),
                              index=0 if st.session_state.bracket["Final 4"].get(matchup, elite_8_winners[i]) ==
                                         elite_8_winners[i] else 1,
                              key=f"final_4_{matchup}")

        st.session_state.bracket["Final 4"][matchup] = winner
        final_4_winners.append(winner)

# --- CHAMPIONSHIP ---
if len(final_4_winners) == 2:
    st.subheader("Championship")
    matchup = f"{final_4_winners[0]} vs {final_4_winners[1]}"

    champion = st.selectbox(f"Select winner for {matchup}",
                            matchup.split(" vs "),
                            index=0 if st.session_state.bracket["Championship"].get(matchup, final_4_winners[0]) ==
                                       final_4_winners[0] else 1,
                            key=f"championship_{matchup}")

    st.session_state.bracket["Championship"][matchup] = champion
    st.session_state.bracket["Winner"] = champion

# Show the winner
if st.session_state.bracket["Winner"]:
    st.subheader("Champion!")
    st.write(f"The winner of the tournament is {st.session_state.bracket['Winner']}")
