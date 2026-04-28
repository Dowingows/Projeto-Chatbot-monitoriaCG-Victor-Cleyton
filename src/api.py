import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents.history import init_db, get_messages, clear_user_history
from chatbot import run, ensure_knowledge_base

DEFAULT_USER = "guest"


@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_knowledge_base()
    yield


app = FastAPI(title="Luan.AI", lifespan=lifespan)


# --- Modelos ---

class ChatRequest(BaseModel):
    message: str


# --- Endpoints ---

@app.post("/api/chat")
async def chat(req: ChatRequest):
    result = await run_in_threadpool(run, DEFAULT_USER, req.message)
    result["sources"] = [os.path.basename(s) for s in result["sources"]]
    return result


@app.get("/api/history")
async def history():
    conn = init_db()
    messages = get_messages(conn, DEFAULT_USER)
    conn.close()
    return {"messages": messages}


@app.delete("/api/history")
async def clear_history():
    conn = init_db()
    clear_user_history(conn, DEFAULT_USER)
    conn.close()
    return {"ok": True}


# Serve o frontend (deve ser montado por último)
app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
