from langchain_core.runnables import RunnableConfig
from app.graph.chess_state import ChessState

async def recursive_evaluation_node(state: ChessState, config: RunnableConfig) -> dict:
    # Lazy import
    from app.graph.workflow import chess_agent 

    sub_result = await chess_agent.ainvoke(state, config=config)

    best_reply = sub_result.get("best_move")
    sub_evaluations = sub_result.get("branch_evaluations", [])
    
    eval_score = 0.0
    for evaluation in sub_evaluations:
        if evaluation.get("move") == best_reply:
            eval_score = evaluation.get("evaluation_score", 0.0)
            break

    
    return {
        "branch_evaluations": [{
            "move": state.get("branch_move"), 
            "evaluation_score": eval_score
        }]
    }