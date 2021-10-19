import random
from reconchess import *
import visualizer as v

class RandomBot(Player):
    def handle_game_start(self, color: Color, board: chess.Board, opponent_name: str):
        # print("handle_game_start: color: ", color, " board state: ", board, " opponent name: ", opponent_name )
        self.color = color
        pass

    def handle_opponent_move_result(self, captured_my_piece: bool, capture_square: Optional[Square]):
        # print("Handle opponent move result: captured a piece: ", captured_my_piece, "capture square: ", capture_square )
        pass

    def choose_sense(self, sense_actions: List[Square], move_actions: List[chess.Move], seconds_left: float) -> \
            Optional[Square]:
        # print("Choose sense: sense_actions: ", sense_actions, "---", move_actions, "---", seconds_left, "---")
        return random.choice(sense_actions)

    def handle_sense_result(self, sense_result: List[Tuple[Square, Optional[chess.Piece]]]):
        # print("Handle sense results: ", sense_result)
        pass

    def choose_move(self, move_actions: List[chess.Move], seconds_left: float) -> Optional[chess.Move]:
        # print("choose move: ", move_actions)
        act = random.choice(move_actions + [None])
        # if act:
        #     v.update(act, self.color)
        return act

    def handle_move_result(self, requested_move: Optional[chess.Move], taken_move: Optional[chess.Move],
                           captured_opponent_piece: bool, capture_square: Optional[Square]):
        v.update(taken_move, self.color)
        # print("handle move result: ", requested_move, "----", taken_move, "----", captured_opponent_piece, "----", capture_square)
        pass

    def handle_game_end(self, winner_color: Optional[Color], win_reason: Optional[WinReason],
                        game_history: GameHistory):
        # print("Result fn: ", win_reason, "---", winner_color, "---", game_history)
        # v.terminate()
        pass