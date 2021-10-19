from chessboard import display
from chess import Board

board = Board()
display.start(board.fen())

def update(action, color):
    # c = 0
    global display, board
    board.turn = color
    if action in board.pseudo_legal_moves:
        board.push(action)
        display.update(board.fen())

def terminate():
    display.terminate()