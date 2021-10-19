import random

from reconchess import *
from chess import Board, Piece, Square
import numpy as np
from math import floor
import math
import time
from collections import defaultdict
from statistics import variance

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
        initial_board_plane = np.zeros((13, 8, 8))
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
                        positions[chess.square(j, i)] = self.inverseMap.get(piece_id)
        board_copy.set_piece_map(positions)
        return board_copy

    def vectorRepr(self, boards):
        planes = []
        for each in boards:
            planes.append(self.OHVETransform(each.board_state, self.color))
        plane_entropy = np.array([[[0 for _ in range(13)] for _ in range(8)] for _ in range(8)], dtype='float64')
        for plane in planes:
            plane_entropy += plane
        plane_entropy /= len(planes)
        return plane_entropy

    def chooseGrid(self, boards):
        planes = []
        for each in boards:
            planes.append(self.OHVETransform(each.board_state, self.color))
        bestVal = -math.inf
        bestCount = -math.inf
        minGrid = 31
        plane_entropy = np.array([[[0] * 8] * 8] * 6, dtype='float64')
        for each in planes:
            each = each.reshape(13, 8, 8)
            if self.color == 1:
                plane_entropy += each[:6]
            else:
                plane_entropy += each[6:-1]
        # print(plane_entropy)
        plane_entropy = plane_entropy / len(planes)
        plane_entropy = np.flip(plane_entropy, axis = 1)
        # print(plane_entropy)
        # print(plane_entropy, self.color, "Color")
        for i in range(3, 9):
            for j in range(3, 9):
                temp = -math.inf
                ent = 0
                c = 0
                for l in range(6):
                    for h in range(i-3, i):
                        for k in range(j-3, j):
                            if 0 < plane_entropy[l][h][k] < 1:
                                ent += plane_entropy[l][h][k]
                                c += 1

                temp = max(temp, c*ent)
                            # if 0 < plane_entropy[l][h][k] < 1:
                            #     c += 1
                            #     temp += plane_entropy[l][h][k]
                if temp > bestCount:
                    bestCount = temp
                    minGrid = ((i-2)*8) + (j-2)
                # elif c == bestCount:
                #     if temp < bestVal:
                #         minGrid = ((i-2)*8) + (j-2)
                #         bestVal = temp
        # print(chosen)
        return minGrid

    def chooseGridV2(self, boards):
        st = time.time()
        senseGridVals  = defaultdict(lambda: defaultdict(int))
        board_strings = []
        n = len(boards)
        for each in boards:
            board_strings.append(self.stringify(each.board_state))

        for i in range(3, 9):
            for j in range(3, 9):
                sense_square = ((i - 2) * 8) + (j - 2)
                for board_string in board_strings:
                    gridValue = ""
                    for h in range(i - 3, i):
                        for k in range(j - 3, j):
                            square = ((h) * 8) + (k)
                            gridValue += board_string[square]
                    senseGridVals[sense_square][gridValue] += 1

        # print(senseGridVals.keys())
        bestSquare = 31
        bestVal = 0
        bestMean = math.inf
        bv  = []
        #Finding max(max(sets for each 3x3 grid))
        for k, v in senseGridVals.items():
            vals = list(v.values())
            delta = n - max(vals)#-sum([(x*math.log(x/n))/n for x in vals])
            # mean = (sum(vals)/len(vals))
            # delta = sum([(x-mean)**2 for x in vals])
            # if len(vals) - 1 > 0:
            #     delta = delta/(len(vals)-1)
            #     if delta != 0:
            #         delta = (n-max(vals))/delta
            #     else:
            #         delta = math.inf
            # else:
            #     delta = 0
            if delta > bestVal:
                # bestSquare = k
                bestVal = delta
                bv = [k]
            elif delta == bestVal:
                bv.append(k)

        # print(len(bv))
        # for each in bv:
        #     print(senseGridVals[each])
        # bestSquare = random.choice(bestSquare)
        return random.choice(bv)
        et = time.time()
        # print(et-st)
        return bestSquare

    def stringify(self, p):
        p = p.piece_map()
        s = ""
        for i in range(64):
            if i in p:
                s += p[i].symbol()
            else:
                s += "."
        return s




