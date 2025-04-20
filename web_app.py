from flask import Flask, request, jsonify, render_template, session
from flask_session import Session
import pickle
import numpy as np
from uuid import uuid4
import os

from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_numpy import PolicyValueNetNumpy

app = Flask(__name__)

# Session config
app.config['SECRET_KEY'] = 'super-secret-key'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)

n = 5
width, height = 8, 8
model_file = 'best_policy_8_8_5.model'

# Load model
if os.path.exists(model_file):
    try:
        policy_param = pickle.load(open(model_file, 'rb'))
    except:
        policy_param = pickle.load(open(model_file, 'rb'), encoding='bytes')
    best_policy = PolicyValueNetNumpy(width, height, policy_param)
else:
    raise FileNotFoundError(f"Model file {model_file} not found.")

# Per-user game storage (in-memory, consider Redis for production)
user_games = {}

def get_user_game():
    user_id = session.get('user_id')
    if not user_id:
        user_id = str(uuid4())
        session['user_id'] = user_id

    if user_id not in user_games:
        board = Board(width=width, height=height, n_in_row=n)
        board.init_board(start_player=0)
        game = Game(board)
        ai_player = MCTSPlayer(best_policy.policy_value_fn, c_puct=5, n_playout=400)
        ai_player.set_player_ind(1)
        user_games[user_id] = {
            'board': board,
            'ai': ai_player
        }

    return user_games[user_id]['board'], user_games[user_id]['ai']

@app.route('/')
def index():
    return render_template('index.html', width=width, height=height)

@app.route('/play', methods=['POST'])
def play():
    board, ai_player = get_user_game()
    data = request.get_json()
    row, col = data['row'], data['col']
    move = board.location_to_move((row, col))

    if move not in board.availables:
        return jsonify({"status": "invalid"})

    board.do_move(move)

    if board.has_a_winner()[0]:
        return jsonify({"status": "win", "winner": int(board.has_a_winner()[1])})

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

@app.route('/reset', methods=['POST'])
def reset():
    board, _ = get_user_game()
    board.init_board(start_player=0)
    return jsonify({"status": "ok"})

if __name__ == '__main__':
    app.run(debug=True)
