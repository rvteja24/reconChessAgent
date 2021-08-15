import chess
from NeuralNet import ChessModel
from collections import defaultdict
from chess import *
import pickle
from core import Core
import numpy as np
import os
#
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
if __name__ == "__main__":
    all_moves_ids = defaultdict(int)
    ids_all_moves = defaultdict(Move)
    c = 0
    for i in range(64):
        for j in range(64):
            if i != j:
                all_moves_ids[Move(i, j)] = c
                ids_all_moves[c] = Move(i, j)
                c += 1
    myModel = ChessModel((8, 8, 13), len(all_moves_ids))
    board = chess.Board()
    # ans = []
    # move_list = [
    #     {'e4': 'e5'},
    #     {'Qh5': 'Nc6'},
    #     {'Bc4': 'Nf6'},
    #     'Qxf7'
    # ]
    # for i in range(10000):
    #     ans.append(move_list)
    # with open("test.txt", "wb") as handle:
    #     pickle.dump(ans, handle, protocol=pickle.HIGHEST_PROTOCOL)
    #
    # with open("test.pickle", "rb") as handle:
    #     print(pickle.load(handle))
    # import chess.svg
    #
    # board = chess.Board()
    # chess.svg.board(board, size=350)
    # c += 1
    # while move_list:
    #     board.push_san(move_list.pop(0))
    c = Core(board, False)
    # print(input)
    import tensorflow as tf
    # with open("episode0.pickle", "rb") as handle:
    #     print(pickle.load(handle))
    # #
    # # print("Num GPUs Available: ", (tf.config.list_physical_devices('GPU')))
    inp = np.array([c.OHVETransform(board, True)])
    inp = inp.reshape(1, 8, 8, 13)
    print(inp.shape)
    a, b = myModel(inp, training = False)
    a.numpy()
    b.numpy()
    print(a, b)
    # import tensorflow as tf
    # print(tf.config.list_physical_devices())
    # print(myModel.model().summary())
    # myModel.save('base_model1')
    # print(myModel.model().summary())
    # for layer in myModel.layers:
    #     print(layer.output_shape)
    # print(all_moves_ids, ids_all_moves)
    # alpha = [0.3, 100]
    # v = dirichlet(alpha)
    # print(sum(v.rvs(1)))