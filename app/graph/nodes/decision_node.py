# app/graph/node/decision_node.py
import chess
from langchain_core.runnables import RunnableConfig
from app.graph.chess_state import ChessState

async def decision_node(state: ChessState, config: RunnableConfig) -> dict:
    """
    Selects the best move from the evaluated branches.
    White maximizes the evaluation score; Black minimizes it.
    """
    current_fen = state["current_fen"]
    board = chess.Board(current_fen)
    is_white_turn = board.turn == chess.WHITE

    evaluations = state.get("branch_evaluations", [])
    candidates = state.get("selected_candidates", [])

    if not evaluations:
        fallback_move = candidates[0] if candidates else None
        return {"best_move": fallback_move}

    if is_white_turn:
        best_eval = max(evaluations, key=lambda e: e.get("evaluation_score", -float('inf')))
    else:
        best_eval = min(evaluations, key=lambda e: e.get("evaluation_score", float('inf')))

    best_move = best_eval.get("move")

    if not best_move and candidates:
        best_move = candidates[0]

    return {"best_move": best_move}