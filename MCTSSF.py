import math
import random
from collections import Counter

import chess.engine

from core import BoardInformation
from chess import *
from chess import engine
from core import *
from numpy.random import dirichlet

STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'
import os


class Node:
    def __init__(self, chosenPath: str = "", transitions: {str: ()} = {},
                 total: int = 0, currentCandidate: Board = Board()):
        # transitions : {str -> action: ((N, Q), Node-> new state)}
        self.transitions = transitions
        self.total = total
        self.path = chosenPath
        self.currentCandidate = currentCandidate


class SamplerRoot:
    def __init__(self, children, distribution):
        self.children = children
        self.distribution = distribution


class MonteCarlo:
    def __init__(self):
        self.actions = None
        self.core = Core(None, None)
        self.all_moves_ids = defaultdict(int)
        self.ids_all_moves = defaultdict(Move)
        self.epsilon = 0.25
        self.color = None
        self.c = 11
        k = 0
        for i in range(64):
            for j in range(64):
                if i != j:
                    self.all_moves_ids[Move(i, j)] = k
                    self.ids_all_moves[k] = Move(i, j)
                    k += 1
        STOCKFISH_ENV_VAR = 'STOCKFISH_EXECUTABLE'
        if STOCKFISH_ENV_VAR not in os.environ:
            raise KeyError(
                'TroutBot requires an environment variable called "{}" pointing to the Stockfish executable'.format(
                    STOCKFISH_ENV_VAR))

        # make sure there is actually a file
        stockfish_path = os.environ[STOCKFISH_ENV_VAR]
        if not os.path.exists(stockfish_path):
            raise ValueError('No stockfish executable found at "{}"'.format(stockfish_path))

        stockfish_path = "C:\\Users\\rvtej\\Downloads\\stockfish_14_win_x64_avx2\\stockfish_14_x64_avx2.exe"
        # initialize the stockfish engine
        self.engine = chess.engine.SimpleEngine.popen_uci(stockfish_path, setpgrp=True)

    # @profile
    def runSimAndReturnBest(self, states: [BoardInformation], iterations: int, actions, model, player, c):
        self.c = c
        self.color = player
        self.actions = set(actions)
        self.visited = set()
        allBoards = []
        for i, each in enumerate(states):
            allBoards.append(each.board_state)

        st = time.time()
        i = 0
        parentNode = Node(chosenPath="r", transitions={})
        while i < iterations:
            i += 1

            boardChosen = random.choice(allBoards)
            parentNode.currentCandidate = boardChosen.copy(stack=False)
            parentNode, _ = self.runOneAndUpdate(parentNode, player, model)
        et = time.time()
        print(et - st)
        # print(parentNode.transitions.items())
        all_actions_distribution = defaultdict(float)
        for key, value in parentNode.transitions.items():
            if key in self.actions:
                all_actions_distribution[key] += value[0][0]

        actionId = random.choice(self.allmax(list(all_actions_distribution.values())))
        action = list(all_actions_distribution.keys())[actionId]
        # print(max(self.visited, key=len))
        return action, []

    # @profile
    def runOneAndUpdate(self, root, player_color, model):
        root.currentCandidate.turn = player_color
        state_value = self.winner(root.currentCandidate, player_color)
        if state_value:
            # root.value += state_value
            return root, -state_value

        if root.path not in self.visited:
            moves = root.currentCandidate.pseudo_legal_moves
            self.visited.add(root.path)
            for each in moves:
                if each not in root.transitions:
                    root.transitions[each] = [[0, 0], Node(chosenPath=root.path,
                                                           transitions={},
                                                           currentCandidate=None)]
            # value = 0
            # for _ in range(1):
            #     value += self.simulate(root.currentCandidate, player_color)
            # root.value = value
            # if player_color == self.color:
            if root.currentCandidate.is_valid():
                p = self.engine.analyse(root.currentCandidate, limit=chess.engine.Limit(depth=5, time=0.0000000001))
                val = p["score"].relative.score()
                if val == None:
                    val = p["score"].relative.score(mate_score=10000)
                value = math.tanh(val / 200)
                # print("SF")
            else:
                # except:
                #     print("NN")
                if root.currentCandidate.is_game_over():
                    v = root.currentCandidate.result()
                    if (v[0] == "0" and player_color == False) or (v[0] == "1" and player_color == True):
                        value = 1
                    else:
                        value = -1
                else:
                    inp = self.core.OHVETransform(root.currentCandidate, player_color)
                    inp = np.array([inp])
                    out = model(inp, training=False)
                    value = out.numpy()[0][0]
            # # print(value)
            #     value = self.simulate(root.currentCandidate, player_color)
            # root.value = value
            return root, -value

        currentActions = root.currentCandidate.pseudo_legal_moves
        existingActions = root.transitions
        # print(existingActions)
        actionInfo = []
        actionList = []
        for act in currentActions:
            if act in existingActions:
                actionInfo.append(existingActions.get(act)[0])
                actionList.append(act)
                # root.transitions[act][1].currentCandidate = root.currentCandidate.copy(stack=False)
            else:
                actionInfo.append([0, 0])
                actionList.append(act)
                root.transitions[act] = [[0, 0], Node(chosenPath=root.path, transitions={},
                                                      currentCandidate=None)]

        if self.color == player_color:
            possibleKingAttack = self.enemySquareAttack(root.currentCandidate, player_color)
            if possibleKingAttack:
                action = possibleKingAttack
            else:
                actionId = self.pickAction(actionInfo, root.total)
                action = actionList[actionId]
        else:
            possibleKingAttack = self.enemySquareAttack(root.currentCandidate, player_color)
            if possibleKingAttack:
                action = possibleKingAttack
            else:
                action = random.choice(actionList)
        if action not in root.transitions:
            root.transitions[action] = [[0, 0], Node(chosenPath=root.path, transitions={},
                                                     currentCandidate=None)]
        root.total += 1
        futureCandidate = root.currentCandidate.copy(stack=False)
        futureCandidate.turn = player_color
        futureCandidate.push(action)
        # root.currentCandidate.push(action)
        root.transitions[act][1].currentCandidate = futureCandidate
        root.transitions[action][1].path = root.path + "-" + action.uci()
        chosenNode = root.transitions[act][1]
        root.transitions[action][1], val = self.runOneAndUpdate(chosenNode, not player_color, model)
        root.transitions[action][0][1] += val
        # root.value += val

        root.transitions[action][0][0] += 1
        return root, -val

    # @profile
    def pickAction(self, action_values, total):
        K = 11
        ucb_values = [self.uctval(a_t, q, K, total) for a_t, q in action_values]
        # ucb_values = np.array([ for a_t, q, p in action_values])
        best_vals = self.allmax(ucb_values)
        if len(best_vals) > 1:
            return random.choice(best_vals)
        else:
            return best_vals[0]

    # @profile
    def uctval(self, a_t, q, c, t):
        c1 = 1.25
        c2 = 19625
        return (q / (1 + a_t)) + ((math.sqrt(t + 1) / (1 + a_t)) * (c1 + math.log((1 + t + c2) / c2)))

    # @profile
    def allmax(self, inp):
        if len(inp) == 0:
            return []
        all_ = [0]
        max_ = inp[0]
        for i in range(1, len(inp)):
            if inp[i] > max_:
                all_ = [i]
                max_ = inp[i]
            elif inp[i] == max_:
                all_.append(i)
        return all_

    # @profile
    def simulate(self, currentCandidate, player_color):
        board = currentCandidate.copy(stack=False)
        current_player = player_color

        while self.winner(board, player_color) == None:
            board.turn = current_player
            act = random.choice(list(board.pseudo_legal_moves))
            board.push(act)
            current_player = not current_player

        val = self.winner(board, player_color)
        return val

    # @profile
    def winner(self, board, player):
        if board.king(player) is None:
            return -1
        elif board.king(not player) is None:
            return 1
        return None

    # @profile
    def enemySquareAttack(self, board, player_color):
        enemy_king_square = board.king(not player_color)
        if enemy_king_square:
            enemy_king_attackers = board.attackers(player_color, enemy_king_square)
            if enemy_king_attackers:
                attacker_square = enemy_king_attackers.pop()
                return chess.Move(attacker_square, enemy_king_square)
        return None

    # @profile
    def stringify(self, p):
        p = p.piece_map()
        s = ""
        for i in range(64):
            if i in p:
                s += p[i].symbol()
            else:
                s += "."
        return s
