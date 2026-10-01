"""Deploy the public app using the named UTHealth account and a scoped secret."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import dotenv_values

ACCOUNT = "xjiang2@uth.edu"
PROJECT = "sbmi-jiang-ai-testing01"
REGION = "us-central1"
SERVICE = "hip-report-review"
SECRET = "hip-report-openai-key"
IDENTITY = f"hip-report-runtime@{PROJECT}.iam.gserviceaccount.com"
HERE = Path(__file__).resolve().parent


def cloud(args, *, capture=False, check=True, input=None):
    return subprocess.run(
        ["gcloud", *args, f"--account={ACCOUNT}", f"--project={PROJECT}", "--quiet"],
        check=check,
        capture_output=capture,
        text=True,
        input=input,
    )


def main():
    # A rejected login stops before uploading code or creating any resource.
    cloud(["projects", "describe", PROJECT, "--format=json"], capture=True)
    services = json.loads(
        cloud(
            ["run", "services", "list", f"--region={REGION}", "--format=json"],
            capture=True,
        ).stdout
    )
    existing = next((s for s in services if s["metadata"]["name"] == SERVICE), None)
    if existing and existing["metadata"].get("labels", {}).get("app") != SERVICE:
        raise SystemExit(
            "An unrelated service already uses this name; choose a new service name before deploying."
        )
    key_file = (
        Path(sys.argv[1]).expanduser()
        if len(sys.argv) > 1
        else HERE.parents[3] / ".env"
    )
    key = dotenv_values(key_file).get("OPENAI_API_KEY")
    if not key:
        raise SystemExit("No OPENAI_API_KEY in the existing project credential file.")
    cloud(
        [
            "services",
            "enable",
            "run.googleapis.com",
            "cloudbuild.googleapis.com",
            "artifactregistry.googleapis.com",
            "secretmanager.googleapis.com",
            "iam.googleapis.com",
        ]
    )
    account = cloud(
        ["iam", "service-accounts", "describe", IDENTITY, "--format=json"],
        capture=True,
        check=False,
    )
    if account.returncode:
        cloud(
            [
                "iam",
                "service-accounts",
                "create",
                "hip-report-runtime",
                "--display-name=Hip report inference runtime",
            ]
        )
    elif (
        json.loads(account.stdout).get("displayName") != "Hip report inference runtime"
    ):
        raise SystemExit(
            "An unrelated service account uses this name; refusing to reuse it."
        )
    secret = cloud(
        ["secrets", "describe", SECRET, "--format=json"], capture=True, check=False
    )
    if secret.returncode:
        cloud(
            [
                "secrets",
                "create",
                SECRET,
                "--replication-policy=user-managed",
                f"--locations={REGION}",
                "--labels=app=hip-report-review",
            ]
        )
    elif json.loads(secret.stdout).get("labels", {}).get("app") != SERVICE:
        raise SystemExit("An unrelated secret uses this name; refusing to replace it.")
    # stdin keeps the credential out of command arguments and build artifacts.
    version = json.loads(
        cloud(
            ["secrets", "versions", "add", SECRET, "--data-file=-", "--format=json"],
            input=key,
            capture=True,
        ).stdout
    )["name"].rsplit("/", 1)[-1]
    key = None
    cloud(
        [
            "secrets",
            "add-iam-policy-binding",
            SECRET,
            f"--member=serviceAccount:{IDENTITY}",
            "--role=roles/secretmanager.secretAccessor",
        ]
    )
    context = Path(tempfile.mkdtemp(prefix="hip-report-build-")) / "source"
    subprocess.run(
        [sys.executable, str(HERE / "build_context.py"), str(context)], check=True
    )
    cloud(
        [
            "beta",
            "run",
            "deploy",
            SERVICE,
            f"--source={context}",
            f"--region={REGION}",
            f"--service-account={IDENTITY}",
            f"--set-secrets=OPENAI_API_KEY={SECRET}:{version}",
            "--no-invoker-iam-check",
            "--no-iap",
            "--cpu=1",
            "--memory=512Mi",
            "--min-instances=0",
            "--max-instances=2",
            "--max=2",
            "--concurrency=1",
            "--timeout=3600",
            "--labels=app=hip-report-review",
        ]
    )
    cloud(
        [
            "run",
            "services",
            "describe",
            SERVICE,
            f"--region={REGION}",
            "--format=value(status.url)",
        ]
    )


if __name__ == "__main__":
    main()
