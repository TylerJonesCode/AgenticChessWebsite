from typing import Literal
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from app.graph.chess_state import ChessState
from app.core.llm import get_llm
from app.core.chess_utils import get_ascii_board

class PositionEvaluationOutput(BaseModel):
    category: Literal[
        "WHITE_WINNING", 
        "WHITE_LARGE_ADVANTAGE",
        "WHITE_SLIGHT_ADVANTAGE", 
        "EQUAL", 
        "BLACK_SLIGHT_ADVANTAGE",
        "BLACK_LARGE_ADVANTAGE", 
        "BLACK_WINNING"
    ] = Field(description="The categorical assessment of the position.")

SCORE_MAPPING = {
    "WHITE_WINNING": 3,
    "WHITE_LARGE_ADVANTAGE": 2,
    "WHITE_SLIGHT_ADVANTAGE": 1,
    "EQUAL": 0,
    "BLACK_SLIGHT_ADVANTAGE": -1,
    "BLACK_LARGE_ADVANTAGE": -2,
    "BLACK_WINNING": -3
}

STATIC_EVAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert chess engine evaluator. "
        "Analyze the board position and categorize the advantage."
    ),
    (
        "human",
        "Board Layout:\n{ascii_board}\n\n"
        "FEN: {current_fen}\n\n"
    ),
])

async def position_evaluation_node(state: ChessState, config: RunnableConfig) -> dict:
    llm = get_llm(config)
    structured_llm = llm.with_structured_output(PositionEvaluationOutput)
    chain = STATIC_EVAL_PROMPT | structured_llm

    current_fen = state["current_fen"]
    
    import chess
    board = chess.Board(current_fen)
    ascii_board = get_ascii_board(board)

    result: PositionEvaluationOutput = await chain.ainvoke(
        {
            "ascii_board": ascii_board,
            "current_fen": current_fen,
        },
        config=config,
    )

    # Convert the categorical output to number
    numerical_score = SCORE_MAPPING[result.category]

    return {
        "branch_evaluations": [{
            "resulting_fen": current_fen,
            "depth": state["depth"],
            "evaluation_score": numerical_score,
            "assessment": result.strategic_assessment,
            "category": result.category
        }]
    }