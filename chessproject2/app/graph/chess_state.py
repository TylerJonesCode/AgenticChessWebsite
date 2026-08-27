# app/graph/chess_state.py
import operator
from typing import Annotated, NotRequired, TypedDict


class ChessState(TypedDict):
    current_fen: str
    depth: int
    legal_moves: list[str]
    tactical_analysis: NotRequired[str]
    positional_analysis: NotRequired[str]
    selected_candidates: NotRequired[list[str]]
    branch_evaluations: Annotated[list[dict], operator.add]
    branch_move: NotRequired[str]
    best_move: NotRequired[str]