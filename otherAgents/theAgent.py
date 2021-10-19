import pickle
import random

import numpy.random
from chess import Move
from numpy.random import dirichlet

from core import *
from ISMCTSV2 import MonteCarlo
from tensorflow.keras import models
import os
import visualizer as v
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

class TheAgent(Player):
    def __init__(self):
        self.all_moves_ids = defaultdict(int)
        self.ids_all_moves = defaultdict(Move)
        k = 0
        for i in range(64):
            for j in range(64):
                if i != j:
                    self.all_moves_ids[Move(i, j)] = k
                    self.ids_all_moves[k] = Move(i, j)
                    k += 1
        # Vector of input I(S), action = "", value = 0, policy = [] at the end of game update value, policy and save episode
        self.episode = []

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.core = Core(board, color)
        self.color = color
        self.currentInp = None
        self.currentPolicy = None
        self.maxStates = 0
        self.PVmodel = models.load_model("base_model1")
        self.alpha = 0.0001
        self.epsilon = 0.25
        self.tau = 1
        self.c = 11
        self.simulations = 2000
        self.mc = MonteCarlo()
        self.chosenSquare = None
        if color:
            self.board_states = [BoardInformation(board, 1)]
        if not color:
            self.board_states = []
            new_board = board
            new_board.turn = not self.color
            for action in new_board.pseudo_legal_moves:
                nb = new_board.copy(stack=False)
                nb.push(action)
                self.board_states.append(BoardInformation(nb, 1))
            self.board_states.append(BoardInformation(board, 1))

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        if captured_my_piece:
            # print(self.color, "states before pruning capture: ", len(self.board_states))
            invalid_ids = set()
            for i, each in enumerate(self.board_states):
                piece_on_board = each.board_state.piece_at(capture_square)
                if piece_on_board == None or piece_on_board.color == self.color:
                    invalid_ids.add(i)

            new_boards = []
            for i, each in enumerate(self.board_states):
                if i not in invalid_ids:
                    new_boards.append(each)
            self.board_states = new_boards
            # print(self.color, "states after pruning capture: ", len(self.board_states))

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        return self.core.chooseGridV2(self.board_states)

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        invalid_ids = set()
        # print(self.color, "states before pruning: ", len(self.board_states))
        for i, each in enumerate(self.board_states):
            for square, piece in sense_result:
                piece_on_board = each.board_state.piece_at(square)
                if piece_on_board != piece:
                    invalid_ids.add(i)
                    break

        new_boards = []
        for i, each in enumerate(self.board_states):
            if i not in invalid_ids:
                new_boards.append(each)
        self.board_states = new_boards
        # print(self.color, "states after pruning: ", len(self.board_states), sense_result)
        self.maxStates = max(self.maxStates, len(self.board_states))

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        # return random.choice(move_actions)
        # print(seconds_left)
        move_set = set(move_actions)
        vector_repr = self.core.vectorRepr(self.board_states)
        self.currentInp = vector_repr
        inp = np.array([vector_repr])
        out, val = self.PVmodel(inp, training=False)
        policy = out.numpy()[0]
        movesList = []
        new_policy = [0] * len(self.all_moves_ids)
        movesWeights = []
        for move in move_actions:
            if move in self.all_moves_ids:
                movesList.append(move)
                id_ = self.all_moves_ids.get(move)
                movesWeights.append(policy[id_])
                new_policy[id_] = policy[id_]
            else:
                move = Move(move.from_square, move.to_square)
                movesList.append(move)
                movesWeights.append(policy[self.all_moves_ids.get(move)])
                id_ = self.all_moves_ids.get(move)
                new_policy[id_] = policy[id_]
        movesWeights/sum(movesWeights)
        dirichlet_noise = dirichlet([0.3] * len(movesList))
        for i, each in enumerate(movesList):
            p = movesWeights[i]
            movesWeights[i] = (1 - self.epsilon) * p + (self.epsilon * dirichlet_noise[i])
        new_policy/sum(new_policy)
        self.currentPolicy = new_policy
        action = numpy.random.choice(movesList, 1, movesWeights)[0]
        # action, _ = self.mc.runSimAndReturnBest(self.board_states, self.simulations, move_set, self.engine, self.color, self.tau, self.c, opponent=False, training=True)
        if not action or action not in move_set:
            action = None
        return action

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        # print(self.color, taken_move, requested_move)
        if captured_opponent_piece:
            # print(self.color, "states before pruning opp capture: ", len(self.board_states))
            invalid_ids = set()
            for i, each in enumerate(self.board_states):
                piece_on_board = each.board_state.piece_at(capture_square)
                if piece_on_board is None or piece_on_board.color == self.color:
                    invalid_ids.add(i)
            new_boards = []
            for i, each in enumerate(self.board_states):
                if i not in invalid_ids:
                    new_boards.append(each)
            self.board_states = new_boards
            # print(self.color, "states after pruning opp capture: ", len(self.board_states))
        if taken_move is not None:
            self.episode.append({"vec": self.currentInp, "act": taken_move, "policy": self.currentPolicy, "val": 0})
        else:
            id_ = self.all_moves_ids[requested_move]
            self.currentPolicy[id_] = (self.alpha * self.currentPolicy[id_])
            self.currentPolicy/sum(self.currentPolicy)
            self.episode.append({"vec": self.currentInp, "act": taken_move, "policy": self.currentPolicy, "val": 0})
        v.update(taken_move, self.color)
        if taken_move is not None:
            new_boards = []
            for each in self.board_states:
                each.board_state.turn = self.color
                if taken_move in set(each.board_state.pseudo_legal_moves):
                    each.board_state.push(taken_move)
                    new_boards.append(each)
            self.board_states = new_boards

        board_map = {}
        for each in self.board_states:
            each.board_state.turn = not self.color
            actions = set(each.board_state.pseudo_legal_moves)
            board_string = self.mc.stringify(each.board_state)
            val = 1
            if board_string not in board_map:
                board_map[board_string] = (each.board_state, val, val)
            for act in actions:
                new_board = each.board_state.copy(stack=False)
                new_board.turn = not self.color
                new_board.push(act)
                board_string = self.mc.stringify(new_board)
                if board_string not in board_map:
                    board_map[board_string] = (new_board, val, val)

        new_boards = []
        for key, value in board_map.items():
            new_boards.append(BoardInformation(value[0], 1))
        self.board_states = new_boards

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        if winner_color == self.color:
            val = 1
        else:
            val = -1
        for i in range(len(self.episode)):
            self.episode[i]["val"] = val
            policy = self.episode[i]["policy"]
            act = self.episode[i]["act"]
            id_ = self.all_moves_ids[act]
            policy[id_] += (self.alpha * val)
            if policy[id_] < 0:
                policy[id_] = 0
            policy/sum(policy)
            self.episode[i]["policy"] = list(policy)
        v = str(time.time())
        with open("episodes//episode" + v + ".pickle", "wb") as handle:
            pickle.dump(self.episode, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print("GAME OVER, FILE SAVED: ", v)
        pass