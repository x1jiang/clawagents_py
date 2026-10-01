"""Create an allowlisted deployment context; no credentials or report artifacts."""

import shutil
import sys
from pathlib import Path

here = Path(__file__).resolve().parent
destination = Path(sys.argv[1]).resolve()
if destination.exists():
    raise SystemExit("Choose a new, empty build directory.")
destination.mkdir(parents=True)
for name in ("Dockerfile", "requirements.txt"):
    shutil.copyfile(here / name, destination / name)
app_dir = destination / "examples" / "hip_dislocation"
app_dir.mkdir(parents=True)
shutil.copyfile(here.parent / "agent.py", app_dir / "agent.py")
web_dir = app_dir / "web"
web_dir.mkdir()
shutil.copyfile(here / "app.py", web_dir / "app.py")
shutil.copytree(here / "static", web_dir / "static")
(destination / ".gcloudignore").write_text(
    ".gcloudignore\n.git\n.env\n**/__pycache__/\n", encoding="utf-8"
)
print("Allowlisted build context:")
for path in sorted(destination.rglob("*")):
    if path.is_file():
        print(path.relative_to(destination))
