from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableConfig
from langchain_core.prompts import ChatPromptTemplate
from app.graph.chess_state import ChessState
from app.core.llm import get_llm

from app.core.chess_utils import get_ascii_board


class TacticalAnalysis(BaseModel):
    checks_and_captures: list[str] = Field(
        description="Immediate forcing moves including checks, captures, and direct threats."
    )
    hanging_pieces: list[str] = Field(
        description="Undefended or inadequately defended pieces for both sides."
    )
    pins_and_forks: list[str] = Field(
        description="Tactical motifs present, such as pins, skewers, forks, and discovered attacks."
    )
    king_safety_threats: str = Field(
        description="Assessment of direct attacks or vulnerabilities surrounding both kings."
    )
    summary: str = Field(
        description="Concise summary of the tactical dynamics and critical forcing lines."
    )


TACTICAL_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert chess engine tactical analyzer. "
        "Analyze the provided chess board state strictly for concrete tactical elements, "
        "including forcing moves, hanging pieces, tactical patterns (pins, forks, skewers), "
        "and direct king safety hazards."
    ),
    (
        "human",
        "Current FEN: {current_fen}\n"
        "Board Layout: {ascii_board}\n"
        "Legal Moves: {legal_moves}\n\n"
        "Generate a structured tactical evaluation of this position."
    ),
])


async def tactical_node(state: ChessState, config: RunnableConfig) -> dict:
    llm = get_llm(config)
    structured_llm = llm.with_structured_output(TacticalAnalysis)

    chain = TACTICAL_PROMPT | structured_llm


    ascii_board = get_ascii_board(state["current_fen"])

    result: TacticalAnalysis = await chain.ainvoke(
        {
            "current_fen": state["current_fen"],
            "ascii_board": ascii_board,
            "legal_moves": state.get("legal_moves", []),
        },
        config=config,
    )

    return {"tactical_analysis": result.summary}