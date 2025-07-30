from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging
import asyncio

from wulf_inference import generate
from scripts.session_logger import log_session
from scripts.fail_log import log_failure

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


class Query(BaseModel):
    prompt: str
    user: str | None = None


@app.post("/generate")
async def run_generation(query: Query):
    try:
        response = await asyncio.to_thread(generate, query.prompt)
        log_session(query.prompt, response, user=query.user)
        return {"response": response}
    except Exception as exc:
        log_failure(query.prompt, exc)
        logger.exception("generation failed")
        raise HTTPException(status_code=500, detail="generation error")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000)
