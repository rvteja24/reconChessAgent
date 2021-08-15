from core import *
from MonteCarloSimulation import MonteCarlo

class TheAgent(Player):

    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        self.core = Core(board, color)
        self.color = color
        self.simulations = 300
        self.mc = MonteCarlo(color)
        if color:
            self.board_states = [BoardInformation(board, 1)]
        if not color:
            self.board_states = []
            new_board = board
            _, all_actions = self.mc.runSimAndReturnBest([BoardInformation(board, 1)], 100, board.legal_moves, not self.color)
            for action, p in all_actions.items():
                nb = new_board.copy()
                nb.push(action)
                self.board_states.append(BoardInformation(nb, p))
        pass

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        pass

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        return self.core.chooseGrid(self.board_states)
        # return random.choice(sense_actions)

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        invalid_ids = set()
        print(self.color, "states before pruning: ",len(self.board_states))
        for i, each in enumerate(self.board_states):
            for square, piece in sense_result:
                piece_on_board = each.board_state.piece_at(square)
                if piece_on_board != piece:
                    invalid_ids.add(i)
                    break

        new_boards = []
        new_total = 0
        for i, each in enumerate(self.board_states):
            if i not in invalid_ids:
                new_total += each.p
                new_boards.append(each)
        self.board_states = new_boards
        for i, each in enumerate(self.board_states):
            self.board_states[i].p /= new_total
        print(self.color, "states after pruning: ", len(self.board_states), sense_result)
        pass

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        move_set = move_actions
        action, _ = self.mc.runSimAndReturnBest(self.board_states, self.simulations, move_set, self.color)
        if not action:
            action = None
        return action

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        print(self.color, taken_move, requested_move)
        if taken_move is not None:
            new_boards = []
            for each in self.board_states:
                each.board_state.turn = self.color
                if taken_move in set(each.board_state.legal_moves):
                    each.board_state.push(taken_move)
                else:
                    continue
                new_boards.append(each)
            self.board_states = new_boards

        new_boards = []
        board_map = {}
        for each in self.board_states:
            _, actions = self.mc.runSimAndReturnBest([BoardInformation(each.board_state, 1)], 10, each.board_state.legal_moves, not self.color)
            for act, val in actions.items():
                new_board = each.board_state.copy()
                new_board.turn = not self.color
                new_board.push(act)
                if new_board.fen() in board_map:
                    board_map[new_board.fen()] = (board_map[new_board.fen()][0]+1, board_map[new_board.fen()][1] + each.p*val)
                else:
                    board_map[new_board.fen()] = (1, each.p*val)

        for key, value in board_map.items():
            new_boards.append(BoardInformation(Board(key), value[1]/value[0]))
        self.board_states = new_boards
        pass

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        pass