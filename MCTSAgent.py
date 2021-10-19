import pickle
import random

import numpy.random
from chess import Move

from core import *
from MCTSSF import MonteCarlo
import os
import visualizer as v
from tensorflow.keras import models

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class MCTSAgent(Player):
    def __init__(self):
        self.current_model = models.load_model("value_net_v5")
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
        # self.episode = []

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.core = Core(board, color)
        self.color = color
        self.mc = MonteCarlo()
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

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        return self.core.chooseGridV2(self.board_states)

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        invalid_ids = set()
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
        print(self.color, "states after pruning: ", len(self.board_states), sense_result)

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        move_set = set(move_actions)
        if seconds_left > 300:
            iters = 6000
        else:
            iters = 3500
        action, _ = self.mc.runSimAndReturnBest(self.board_states, iters, move_set, self.current_model,
                                                self.color,
                                                0)
        if not action or action not in move_set:
            action = None
        return action

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):

        if captured_opponent_piece:
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
        random.shuffle(self.board_states)
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
        pass
