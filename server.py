from fastapi import FastAPI
from langchain_ollama import ChatOllama
from langserve import add_routes
import uvicorn

# ── Config ────────────────────────────────────────────────────────
HOST = "0.0.0.0"
PORT = 8000
MODEL = "mistral"
BASE_URL = "http://localhost:11434"

# ── LLM ───────────────────────────────────────────────────────────
llm = ChatOllama(model=MODEL, base_url=BASE_URL)

# ── FastAPI + LangServe ───────────────────────────────────────────
app = FastAPI(title="LLM API")
add_routes(app, llm, path="/llm")

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)