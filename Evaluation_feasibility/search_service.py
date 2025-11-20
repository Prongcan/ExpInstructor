from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

app = FastAPI()

class SearchRequest(BaseModel):
    node_query: str
    node_k: int = 5
    edge_query: Optional[str] = ""
    edge_k: int = 5

@app.post("/search_node_edge")
def search_similar_node_and_edge_api(req: SearchRequest):
    try:
        from Evaluation_feasibility.ins_model import get_retrieval_system
        retrieval_system = get_retrieval_system()
        results = retrieval_system.search_similar_node_and_edge(
            req.node_query, req.node_k, req.edge_query, req.edge_k
        )
        return {"result": results}
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9876)
