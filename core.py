from reconchess import *
from chess import Board, Piece, Square
import numpy as np
from math import floor
import math
from collections import defaultdict
class BoardInformation:
    def __init__(self, board_state, p):
        self.board_state = board_state
        self.p = p

class Core:
    def __init__(self, initial_board_state, color):
        self.initialBoardState = initial_board_state # Capital cased are whites and lower cased are blacks
        self.color = 1 if color else 0 # False is Black; True is white
        self.layerMap = {"K":6, "Q": 7, "R": 8, "N": 9, "B": 10, "P": 11, "k":0, "q": 1, "r": 2, "n": 3, "b": 4, "p": 5, "color": 12}
        self.inverseMap = {6: (Piece.from_symbol("K")), 7: (Piece.from_symbol("Q")), 8: (Piece.from_symbol("R")), 9:(Piece.from_symbol("N")), 10:(Piece.from_symbol("B")), 11: (Piece.from_symbol("P")), 0: (Piece.from_symbol("k")), 1:(Piece.from_symbol("q")), 2:(Piece.from_symbol("r")), 3: (Piece.from_symbol("n")), 4: (Piece.from_symbol("b")), 5: (Piece.from_symbol("p"))}

    def OHVETransform(self, board, color):
        #initialize the 3-d array for representing initial game state
        board = str(board).replace(" ", "").replace("\n", "")
        # print(board)
        initial_board_plane = np.zeros((13,8,8))
        for i in range(len(board)):
            if board[i] in self.layerMap:
                initial_board_plane[self.layerMap[board[i]]][int(floor(i/8))][int(floor(i%8))] = 1
        # Add player's color as final layer
        initial_board_plane[12] = [[color]*8]*8
        initial_board_plane = initial_board_plane.reshape(8, 8, 13)
        return initial_board_plane

    def inverseTransform(self, board):
        # inverseMap = {0: (Piece.from_symbol("K")), 1: (Piece.from_symbol("Q")), 2: (Piece.from_symbol("R")),
        #                    3: (Piece.from_symbol("N")), 4: (Piece.from_symbol("B")), 5: (Piece.from_symbol("P")),
        #                    6: (Piece.from_symbol("k")), 7: (Piece.from_symbol("q")), 8: (Piece.from_symbol("r")),
        #                    9: (Piece.from_symbol("n")), 10: (Piece.from_symbol("b")), 11: (Piece.from_symbol("p"))}
        board_copy = Board()
        positions = defaultdict(Piece)
        for piece_id, each in enumerate(board[:-1]):
            for i in range(8):
                for j in range(8):
                    if each[i][j] == 1:
                        positions[chess.square(j,i)] = self.inverseMap.get(piece_id)
        board_copy.set_piece_map(positions)
        return board_copy

    def chooseGrid(self, boards):
        planes = []
        for each in boards:
            planes.append(self.OHVETransform(each.board_state, self.color))
        bestVal = -math.inf
        bestCount = 0
        minGrid = [-1, -1]
        plane_entropy = np.array([[0] * 8] * 8, dtype='float64')
        for each in planes:
            plane_entropy += np.sum(each[:-1], axis=0)
        plane_entropy = plane_entropy / len(planes)
        plane_entropy = np.flip(plane_entropy, axis=0)
        # print(plane_entropy, self.color, "Color")
        for i in range(3,8):
            for j in range(3,8):
                temp = 0
                c = 0
                for h in range(i-3,i):
                    for k in range(j-3,j):
                        if 0 < plane_entropy[h][k] < 1:
                            c += 1
                            temp += plane_entropy[h][k]
                if c > bestCount:
                    bestCount = c
                    minGrid = [i-1,j-1]
                elif c == bestCount:
                    if temp > bestVal:
                        minGrid = [i - 1, j - 1]
                        bestVal = temp

        chosen = 31
        if minGrid != [-1,-1]:
            l = minGrid[0]
            r = minGrid[1]
            chosen = ((l-1) * 8) + (r-1)
        # print(chosen)
        return chosen




