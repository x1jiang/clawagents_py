"""Local launcher reuses the previously verified project credential."""

import os
from pathlib import Path
from dotenv import dotenv_values
import uvicorn

if __name__ == "__main__":
    key = dotenv_values(Path(__file__).resolve().parents[4] / ".env").get(
        "OPENAI_API_KEY"
    )
    if key:
        os.environ["OPENAI_API_KEY"] = key
    uvicorn.run(
        "examples.hip_dislocation.web.app:app",
        host="127.0.0.1",
        port=8096,
        access_log=False,
    )
