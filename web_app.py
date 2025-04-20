from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy
import os

app = Flask(__name__)

n = 5
width, height = 8, 8
model_file = 'best_policy_8_8_5.model'

# Initialize board and game
board = Board(width=width, height=height, n_in_row=n)
board.init_board(start_player=0)  # Initialize board properly
game = Game(board)

# Load model
if os.path.exists(model_file):
    try:
        policy_param = pickle.load(open(model_file, 'rb'))
    except:
        policy_param = pickle.load(open(model_file, 'rb'), encoding='bytes')
    best_policy = PolicyValueNetNumpy(width, height, policy_param)
    ai_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)
    ai_player.set_player_ind(1)
else:
    raise FileNotFoundError(f"Model file {model_file} not found.")

# Store current state
game_state = {
    "board": board,
    "game": game,
    "players": [None, ai_player],
    "current_player": 0
}

@app.route('/')
def index():
    return render_template('index.html', width=width, height=height)

@app.route('/play', methods=['POST'])
def play():
    data = request.get_json()
    row, col = data['row'], data['col']
    move = board.location_to_move((row, col))

    if move not in board.availables:
        return jsonify({"status": "invalid"})

    board.do_move(move)

    if board.has_a_winner()[0]:
        return jsonify({"status": "win", "winner": int(board.has_a_winner()[1])})

    # AI move
    ai_move = ai_player.get_action(board)
    board.do_move(ai_move)

    if board.has_a_winner()[0]:
        ai_row, ai_col = board.move_to_location(ai_move)
        return jsonify({
            "status": "lose",
            "ai_move": [int(ai_row), int(ai_col)],
            "winner": int(board.has_a_winner()[1])
        })

    ai_row, ai_col = board.move_to_location(ai_move)
    return jsonify({
        "status": "continue",
        "ai_move": [int(ai_row), int(ai_col)]
    })

if __name__ == '__main__':
    app.run(debug=True)