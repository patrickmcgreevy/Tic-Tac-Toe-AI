from IPython.display import HTML, display
from Board import Board, GameResult, CROSS, NAUGHT
from Player import Player
from TQPlayer import TQPlayer
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from MinMaxAgent import MinMaxAgent
from RndMinMaxAgent import RndMinMaxAgent
from RandomPlayer import  RandomPlayer
from TFSessionManager import TFSessionManager
from SimpleNNQPlayer import NNQPlayer
from eNNQPlayer import EGreedyNNQPlayer
from ExpDoubleDuelQPlayer import ExpDoubleDuelQPlayer
from DeepExpDoubleDuelQPlayer import DeepExpDoubleDuelQPlayer
from DirectPolicyAgent import DirectPolicyAgent

def print_board(board):
    display(HTML("""
    <style>
    .rendered_html table, .rendered_html th, .rendered_html tr, .rendered_html td {
      border: 1px  black solid !important;
      color: black !important;
    }
    </style>
    """+board.html_str()))

def play_game(b):
    b.reset()
    finished = False
    while not finished:
        _, result, finished = b.move(b.random_empty_spot(), CROSS)
        #print_board(b)
        if finished:
            if result == GameResult.DRAW:
                print("Game is a draw")
            else:
                print("Cross won!")
        else:
            _, result, finished = b.move(b.random_empty_spot(), NAUGHT)
            #print_board(b)
            if finished:
                if result == GameResult.DRAW:
                    print("Game is a draw")
                else:
                    print("Naught won!")


def play_game(board: Board, player1: Player, player2: Player):
    player1.new_game(CROSS)
    player2.new_game(NAUGHT)
    board.reset()

    finished = False
    while not finished:
        result, finished = player1.move(board)
        if finished:
            if result == GameResult.DRAW:
                final_result = GameResult.DRAW
            else:
                final_result = GameResult.CROSS_WIN
        else:
            result, finished = player2.move(board)
            if finished:
                if result == GameResult.DRAW:
                    final_result = GameResult.DRAW
                else:
                    final_result = GameResult.NAUGHT_WIN

    player1.final_result(final_result)
    player2.final_result(final_result)
    return final_result

def battle(player1: Player, player2: Player, num_games: int = 100, silent: bool = True):
    board = Board()
    draw_count = 0
    cross_count = 0
    naught_count = 0
    for _ in range(num_games):
        result = play_game(board, player1, player2)
        if result == GameResult.CROSS_WIN:
            cross_count += 1
        elif result == GameResult.NAUGHT_WIN:
            naught_count += 1
        else:
            draw_count += 1
    if not silent:
        print("After {} game we have draws: {}, Player 1 wins: {}, and Player 2 wins: {}.".format(num_games, draw_count,
                                                                                              cross_count,
                                                                                              naught_count))

        print("Which gives percentages of draws: {:.2%}, Player 1 wins: {:.2%}, and Player 2 wins:  {:.2%}".format(
        draw_count / num_games, cross_count / num_games, naught_count / num_games))

    return cross_count, naught_count, draw_count

"""
def eval_players(p1 : Player, p2 : Player, num_battles : int, games_per_battle = 100, loc='best'):
    p1_wins = []
    p2_wins = []
    draws = []
    count = []

    for i in range(num_battles):
        p1win, p2win, draw = battle(p1, p2, games_per_battle, True)
        p1_wins.append(p1win*100.0/games_per_battle)
        p2_wins.append(p2win*100.0/games_per_battle)
        draws.append(draw*100.0/games_per_battle)
        count.append(i*games_per_battle)
        p1_wins.append(p1win*100.0/games_per_battle)
        p2_wins.append(p2win*100.0/games_per_battle)
        draws.append(draw*100.0/games_per_battle)
        count.append((i+1)*games_per_battle)

    plt.ylabel('Game outcomes in %')
    plt.xlabel('Game number')

    plt.plot(count, draws, 'r-', label='Draw')
    plt.plot(count, p1_wins, 'g-', label='Player 1 wins')
    plt.plot(count, p2_wins, 'b-', label='Player 2 wins')
    plt.legend(loc=loc, shadow=True, fancybox=True, framealpha =0.7)
"""

def evaluate_players(p1 : Player, p2 : Player, games_per_battle = 100, num_battles = 100):
    board = Board()

    p1_wins = []
    p2_wins = []
    draws = []
    game_number = []
    game_counter = 0

    TFSessionManager.set_session(tf.Session())
    TFSessionManager.get_session().run(tf.global_variables_initializer())

    for i in range (num_battles):
        p1win, p2win, draw = battle(p1, p2, games_per_battle, silent=True)
        p1_wins.append(p1win)
        p2_wins.append(p2win)
        draws.append(draw)
        game_counter=game_counter+1
        game_number.append(game_counter)

    TFSessionManager.set_session(None)
    return game_number, p1_wins, p2_wins, draws

tf.reset_default_graph()

#nnp = EGreedyNNQPlayer('Qlearner1', learning_rate=0.001, reward_discount=0.99, random_move_decrease=0.99)
nnp = DeepExpDoubleDuelQPlayer('Qlearner')
sndp = MinMaxAgent()

TFSessionManager.set_session(tf.Session())
TFSessionManager.get_session().run(tf.global_variables_initializer())
#game_num, NNwins, rndWins, draws = evaluate_players(p1, p2, 100, 100)

#plot = plt.plot(game_num, draws, 'r-', game_num, NNwins, 'g-', game_num, rndWins, 'b-')

game_num, p1, p2, draws = evaluate_players(sndp, nnp, games_per_battle=100, num_battles=100)

p = plt.plot(game_num, draws, 'r-', game_num, p1, 'g-', game_num, p2, 'b-')

plt.show()
TFSessionManager.set_session(None)