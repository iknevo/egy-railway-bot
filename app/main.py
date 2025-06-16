from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv
import os
from fastapi.middleware.cors import CORSMiddleware  # Add this import

# Load environment variables
load_dotenv()

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

class QueryRequest(BaseModel):
    question: str

# Initialize RAG pipeline on startup
@app.on_event("startup")
async def startup_event():
    from rag_pipeline import initialize_rag_pipeline
    global graph
    graph = initialize_rag_pipeline()

@app.post("/query")
async def handle_query(request: QueryRequest):
    try:
        result = graph.invoke({"question": request.question})
        return {"answer": result["answer"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
