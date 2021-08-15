from math import floor

import chess
from line_profiler import LineProfiler
from MonteCarloSimulation import MonteCarlo
from chess import *
import os
from NeuralNet import NeuralNet
from theAgent import BoardInformation
import multiprocessing as mp
import time
from core import Core
from collections import defaultdict
from tensorflow.keras import models
import numpy as np
# from chessboard import display
import pickle
from multiprocessing import Process, Manager
from multiprocessing.managers import BaseManager
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

def addEpisodes(episode, i, path):
    v = str(i)
    with open("episodes//episode" + v + ".pickle", "wb") as handle:
        pickle.dump(episode, handle, protocol=pickle.HIGHEST_PROTOCOL)

def runGame(i, path):
    current_model = models.load_model("base_model1")
    episode = []
    board = Board()
    # display.start(board.fen())
    iterations = 40
    mc = MonteCarlo()
    core = Core(board, None)
    player = True
    start_time = time.time()
    c = 0
    tau = 1
    while not board.is_game_over():
        # if c == 60:
        #     break
        # if c%10 == 0:
        #     print(c)
        # if c == 80:
        #     tau = 0.01
        # display.update(board.fen())
        # if c == 60:
        #     tau = 0.001
        input_state = core.OHVETransform(board, player)
        # inp = np.array([input_state])
        # print(current_model.predict(inp))
        action, new_distribution = mc.runSimAndReturnBest([BoardInformation(board, 1)], iterations, board.legal_moves, current_model, player, tau)
        episode.append({"color": player, "src": input_state, "target": new_distribution})
        board.turn = player
        board.push(action)

        player = not player
        c += 1

    del current_model
    # display.terminate()
    white = 0
    black = 0
    winner = board.result().split("-")[0]
    if winner == "1":
        white = 1
        black = -1
    if winner == "0":
        black = 1
        white = -1
    for ep in episode:
        if ep["color"]:
            ep["target"].append(white)
        else:
            ep["target"].append(black)

    end_time = time.time()
    addEpisodes(episode, i, path)
    print(board, board.result(), end_time-start_time)
    # return episode

def selfPlay(b, e, path):
    current_model = models.load_model("base_model1")
    for i in range(b, e):
        runGame(i, path)



if __name__ == "__main__":
    cores = 4
    pool = mp.Pool(cores)
    # runGame()
    # i = 300
    # current_model = models.load_model("base_model1")
    st = time.time()
    # for i in range(441,541):
    #     runGame(current_model, i, "")
    for i in range(641, 741):
        pool.apply_async(runGame, ( i, ""))
    pool.close()
    pool.join()
    et = time.time()
    print((et-st)/60)


