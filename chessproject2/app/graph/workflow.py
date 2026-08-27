from langgraph.graph import StateGraph, START, END
from langchain_core.runnables import RunnableConfig
from app.graph.chess_state import ChessState

from app.graph.nodes.legal_moves_node import legal_moves_node

from app.graph.nodes.tactical_node import tactical_node
from app.graph.nodes.positional_node import positional_node

from app.graph.nodes.candidate_move_node import candidate_move_node

from app.graph.nodes.recursive_evaluation_node import recursive_evaluation_node

from app.graph.nodes.position_evaluation_node import position_evaluation_node

from app.graph.nodes.decision_node import decision_node
from app.graph.routers import recursion_router 


workflow = StateGraph(ChessState)

workflow.add_node("legal_moves", legal_moves_node)
workflow.add_node("tactical", tactical_node)
workflow.add_node("positional", positional_node)
workflow.add_node("candidate_moves", candidate_move_node)
workflow.add_node("evaluate_branches", recursive_evaluation_node)
workflow.add_node("evaluate_position", position_evaluation_node)
workflow.add_node("decision", decision_node)

# Extract legal moves, then trigger tactical & positional analysis in parallel
workflow.add_edge(START, "legal_moves")
workflow.add_edge("legal_moves", "tactical")
workflow.add_edge("legal_moves", "positional")

# Evaluate legal moves and analysis to select 3 candidate moves
workflow.add_edge(["tactical", "positional"], "candidate_moves")


# If at max depth, evaluate current position. Otherwise, recursively evaluate
workflow.add_conditional_edges(
    "candidate_moves",
    recursion_router,
    ["evaluate_branches", "evaluate_position"]
)

workflow.add_edge("evaluate_branches", "decision")
workflow.add_edge("evaluate_position", "decision")



workflow.add_edge("decision", END)

chess_agent = workflow.compile()