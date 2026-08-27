import chess
from langgraph.types import Send
from langchain_core.runnables import RunnableConfig
from app.graph.chess_state import ChessState

def recursion_router(state: ChessState, config: RunnableConfig):
    current_depth = state.get("depth", 0)
    max_depth = config.get("configurable", {}).get("max_depth", 3)
    candidates = state.get("selected_candidates", [])
    
    # Base Case: Reached max depth, evaluate current position
    if current_depth >= max_depth:
        return "evaluate_position"
        
    # Recursive Case: Evaluate recursively in parallel
    sends = []
    board = chess.Board(state["current_fen"])
    
    for move_san in candidates:
        try:
            branch_board = board.copy()
            branch_board.push_san(move_san)
            
            sends.append(
                Send(
                    "evaluate_branches", 
                    {
                        "current_fen": branch_board.fen(),
                        "depth": current_depth + 1,
                        "branch_move": move_san,
                        "branch_evaluations": [] 
                    }
                )
            )
        except ValueError:
            continue
            
    return sends