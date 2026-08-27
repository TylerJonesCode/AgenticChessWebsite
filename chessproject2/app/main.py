from fastapi import FastAPI
from pydantic import BaseModel
from app.graph.workflow import chess_agent 

app = FastAPI()

class EngineRequest(BaseModel):
    fen: str
    selected_model: str
    depth: int

@app.post("/engine/move")
def get_best_move(request: EngineRequest):
    initial_state = {
        "current_fen": request.fen,
        "branch_evaluations": [],
        "depth": 0
    }
    
    runtime_config = {
        "recursion_limit": 150,
        "configurable": {
            "model_name": request.selected_model,
            "max_search_depth": request.depth
        }
    }
    
    final_state = chess_agent.invoke(initial_state, config=runtime_config)
    
    return {"best_move": final_state.get("best_move")}