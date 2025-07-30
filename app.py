from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import logging

from wulf_inference import generate
from scripts.session_logger import log_session
from scripts.fail_log import log_failure

app = FastAPI()
logger = logging.getLogger(__name__)

class PromptRequest(BaseModel):
    user: str
    prompt: str

@app.post("/generate")
async def generate_endpoint(req: PromptRequest):
    full_prompt = f"{req.user}: {req.prompt}"
    try:
        response = generate(full_prompt)
        log_session(full_prompt, response)
        return {"response": response}
    except Exception as exc:
        log_failure(full_prompt, exc)
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail="Generation error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
