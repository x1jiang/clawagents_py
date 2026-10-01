"""Run using the existing canonical ClawAgents parent credential configuration."""

import os
import runpy
import sys
from pathlib import Path

from dotenv import dotenv_values


def main():
    here = Path(__file__).resolve().parent
    credential_file = here.parents[1].parent / ".env"
    configured_key = (
        dotenv_values(credential_file).get("OPENAI_API_KEY")
        if credential_file.exists()
        else None
    )
    if configured_key:
        os.environ["OPENAI_API_KEY"] = configured_key
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "No OpenAI key configured in the existing project or environment."
        )
    sys.argv[0] = str(here / "agent.py")
    runpy.run_path(str(here / "agent.py"), run_name="__main__")


if __name__ == "__main__":
    main()
