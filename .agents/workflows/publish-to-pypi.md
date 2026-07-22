---
description: Publish clawagents_py to PyPI
---

Publish clawagents_py to PyPI — Bump the version if needed, build the package, and upload it so that pip install clawagents installs the latest version.
Use PyPI API token configured in environment variable `UV_PUBLISH_TOKEN` or `TWINE_PASSWORD`.

Push clawagents_py to GitHub — Sync the local clawagents_py project to GitHub - x1jiang/clawagents_py, ensuring the remote reflects all local changes.
Push clawagents (JS/Node) to GitHub — Sync the local clawagents project to GitHub - x1jiang/clawagents, and update README.md to reflect the latest local changes.
Verify installations work:
pip install clawagents (from PyPI)
npm install git+<https://github.com/x1jiang/clawagents.git> (from GitHub)

Verify on PyPI — Confirm the latest version of clawagents is visible on pypi.org.
