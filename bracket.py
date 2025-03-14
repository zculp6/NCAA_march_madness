import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import random

# Initialize the Dash app
app = dash.Dash(__name__)

# Teams and their matchups for the bracket
teams = ['Team 1', 'Team 2', 'Team 3', 'Team 4', 'Team 5', 'Team 6', 'Team 7', 'Team 8']

# Define the first round matchups
round_1 = [(teams[i], teams[i+1]) for i in range(0, len(teams), 2)]

# Function to simulate a game
def simulate_game(team1, team2):
    return random.choice([team1, team2])

# Layout of the app
app.layout = html.Div([
    html.H1("8-Team Tournament Bracket"),

    # Tournament Bracket Visualization
    dcc.Graph(id='bracket', style={'height': '800px'}),

    # Hidden Store components to store the simulation state
    dcc.Store(id='round-1-store', data=round_1),
    dcc.Store(id='round-2-store'),
    dcc.Store(id='round-3-store'),

    # Button to simulate the tournament
    html.Button("Simulate Tournament", id='simulate-button', n_clicks=0)
])

# Callback to simulate the tournament and update the bracket dynamically
@app.callback(
    [Output('bracket', 'figure'),
     Output('round-2-store', 'data'),
     Output('round-3-store', 'data')],
    [Input('simulate-button', 'n_clicks')],
    [State('round-1-store', 'data'),
     State('round-2-store', 'data'),
     State('round-3-store', 'data')]
)
def update_bracket(n_clicks, round_1_data, round_2_data, round_3_data):
    # If the button is clicked, simulate the tournament
    if n_clicks > 0:
        # Simulate first round if not already simulated
        if round_2_data is None:
            round_2_data = [simulate_game(matchup[0], matchup[1]) for matchup in round_1_data]

        # Simulate second round if not already simulated
        if round_3_data is None:
            round_3_data = [simulate_game(round_2_data[i], round_2_data[i+1]) for i in range(0, len(round_2_data), 2)]

        # Simulate final round
        final_winner = simulate_game(round_3_data[0], round_3_data[1])
        round_3_data = [final_winner]

    # Create bracket visualization
    fig = go.Figure()

    # Round 1 matchups
    y_offset = 3  # Initial y positioning for Round 1
    for i, matchup in enumerate(round_1_data):
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[y_offset, y_offset],
            mode='lines+text',
            line=dict(color='black', width=2),
            text=[matchup[0], matchup[1]],
            textposition='top center',
            showlegend=False
        ))
        y_offset -= 2  # Move y position down for next matchup

    # Round 2 matchups
    y_offset = 3  # Reset y positioning for Round 2
    for i, winner in enumerate(round_2_data):
        fig.add_trace(go.Scatter(
            x=[2, 3], y=[y_offset, y_offset],
            mode='lines+text',
            line=dict(color='black', width=2),
            text=[winner],
            textposition='top center',
            showlegend=False
        ))
        y_offset -= 4  # Move y position down for next winner

    # Round 3 (final) matchup
    y_offset = 0  # Reset y positioning for Round 3 (final)
    for winner in round_3_data:
        fig.add_trace(go.Scatter(
            x=[4, 5], y=[y_offset, y_offset],
            mode='lines+text',
            line=dict(color='black', width=2),
            text=[winner],
            textposition='top center',
            showlegend=False
        ))

    # Final touches for the plot
    fig.update_layout(
        title="8-Team Tournament Bracket",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, range=[-1, 6]),
        yaxis=dict(showgrid=False, zeroline=False, tickvals=[]),
        height=800
    )

    return fig, round_2_data, round_3_data

if __name__ == '__main__':
    app.run_server(debug=True)
