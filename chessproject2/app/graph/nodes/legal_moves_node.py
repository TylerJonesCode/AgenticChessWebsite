import chess
from app.graph.chess_state import ChessState

def legal_moves_node(state: ChessState) -> dict:
    board = chess.Board(state["current_fen"])
    legal_moves = [board.san(move) for move in board.legal_moves]
    
    return {
        "legal_moves": legal_moves
    }