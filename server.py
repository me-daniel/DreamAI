from typing import List
from fastapi import FastAPI
from langserve import add_routes

from irt_app import chain_with_message_history

# App definition
app = FastAPI(
  title="LangChain Server",
  version="1.0",
  description="A simple API server using LangChain's Runnable interfaces",
)

# Adding chain route

add_routes(
    app,
    chain_with_message_history,
    path="/chain",
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)