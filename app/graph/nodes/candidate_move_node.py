import chess
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate

from app.graph.chess_state import ChessState
from app.core.llm import get_llm

class MovePruningOutput(BaseModel):
    selected_moves: list[str] = Field(
        description="A list of up to 3 candidate moves in exact SAN notation from the provided legal moves."
    )
    reasoning: list[str] = Field(
        description="Concise reasoning for why these moves were selected."
    )

CANDIDATE_MOVE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert chess engine move selector. "
        "Your task is to synthesize tactical and positional evaluations to filter the available legal moves "
        "down to the top 3 most promising candidate moves for deep calculation.\n\n"
        "Rules:\n"
        "1. Select up to 3 candidate moves (or fewer if less than three are promising).\n"
        "2. Every selected candidate move MUST be an exact match from the provided Legal Moves list.\n"
    ),
    (
        "human",
        "Current FEN: {current_fen}\n\n"
        "Board Layout:\n{ascii_board}\n\n"
        "Legal Moves: {legal_moves}\n\n"
        "Tactical Analysis:\n{tactical_analysis}\n\n"
        "Positional Analysis:\n{positional_analysis}\n\n"
        "Select the top 3 candidate moves from the legal moves list and provide concise reasoning for each."
    ),
])

def candidate_move_node(state: ChessState, config: RunnableConfig) -> dict:
    # Required keys can be accessed directly
    fen = state["current_fen"]
    legal_moves = state["legal_moves"]
    
    # NotRequired keys must use .get() to avoid KeyErrors
    tactical_analysis = state.get("tactical_analysis", "")
    positional_analysis = state.get("positional_analysis", "")
    
    board = chess.Board(fen)
    ascii_board = str(board)

    llm = get_llm(config)
    pruner_llm = llm.with_structured_output(MovePruningOutput)
    
    chain = CANDIDATE_MOVE_PROMPT | pruner_llm
    
    result: MovePruningOutput = chain.invoke(
        {
            "current_fen": fen,
            "ascii_board": ascii_board,
            "legal_moves": legal_moves,
            "tactical_analysis": tactical_analysis,
            "positional_analysis": positional_analysis,
        },
        config=config
    )
    
    selected_moves = result.selected_moves
    branch_fens = {}
    
    for move_san in selected_moves:
        test_board = board.copy()
        try:
            test_board.push_san(move_san)
            branch_fens[move_san] = test_board.fen()
        except ValueError:
            continue
            
    return {
        "selected_candidates": selected_moves,
        "branch_fens": branch_fens
    }