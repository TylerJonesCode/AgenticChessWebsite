from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from app.graph.chess_state import ChessState
from app.core.llm import get_llm

from app.core.chess_utils import get_ascii_board

class PositionalAnalysis(BaseModel):
    pawn_structure: str = Field(
        description="Assessment of pawn chains, isolated pawns, doubled pawns, and passed pawns."
    )
    piece_activity: str = Field(
        description="Evaluation of piece mobility, outposts, open files, and bishop diagonals."
    )
    center_control_and_space: str = Field(
        description="Evaluation of central square control (e4, d4, e5, d5) and overall space advantage."
    )
    weak_squares: list[str] = Field(
        description="Key light or dark square weaknesses and vulnerable color complexes."
    )
    summary: str = Field(
        description="Concise strategic overview detailing which side holds a positional edge and why."
    )


POSITIONAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert chess engine positional analyzer. "
        "Evaluate the strategic properties of the board position based strictly on static features, "
        "such as pawn structures, piece placement/activity, weak squares, and central territory control."
    ),
    (
        "human",
        "Current FEN: {current_fen}\n"
        "Board Layout: {ascii_board}\n"
        "Legal Moves: {legal_moves}\n\n"
        "Generate a structured positional evaluation of this position."
    ),
])



async def positional_node(state: ChessState, config: RunnableConfig) -> dict:
    llm = get_llm(config)
    structured_llm = llm.with_structured_output(PositionalAnalysis)

    chain = POSITIONAL_PROMPT | structured_llm

    ascii_board = get_ascii_board(state["current_fen"])
    

    result: PositionalAnalysis = await chain.ainvoke(
        {
            "current_fen": state["current_fen"],
            "ascii_board": ascii_board,
            "legal_moves": state.get("legal_moves", []),
        },
        config=config,
    )

    return {"positional_analysis": result.summary}