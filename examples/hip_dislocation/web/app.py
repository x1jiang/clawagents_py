"""Request-bound hip-report annotation. No upload files, job store, or report logs."""

import asyncio
import csv
import io
import json
import logging
import os
import posixpath
import zipfile
from pathlib import Path
from urllib.parse import urlsplit
from xml.etree import ElementTree as ET

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from examples.hip_dislocation.agent import FEATURES, MODEL, annotate
from clawagents.config.features import set_overrides

# One fixed process configuration; never change global flags inside a request.
set_overrides({**FEATURES, "micro_compact": False, "aggressive_tool_crush": False})
logging.disable(logging.CRITICAL)  # Provider/framework errors can contain input.

HERE = Path(__file__).resolve().parent
MAX_BYTES = 5 * 1024 * 1024
MAX_REPORTS = 100
MAX_CHARS = 8000
MAX_TOTAL = 200_000
slots = asyncio.Semaphore(3)
app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)
app.mount("/static", StaticFiles(directory=HERE / "static"), name="static")


@app.middleware("http")
async def protect(request: Request, call_next):
    if request.method == "POST":
        origin = request.headers.get("origin")
        if request.headers.get("x-hip-request") != "1" or (
            origin and urlsplit(origin).netloc != request.url.netloc
        ):
            return JSONResponse({"detail": "Use this app's upload or run form."}, 403)
    response = await call_next(request)
    response.headers.update(
        {
            "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff",
            "Referrer-Policy": "no-referrer",
            "Content-Security-Policy": "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
            "Permissions-Policy": "camera=(), microphone=(), geolocation=()",
        }
    )
    return response


@app.get("/")
async def index():
    return FileResponse(HERE / "static" / "index.html")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": MODEL,
        "configured": bool(os.getenv("OPENAI_API_KEY")),
    }


async def limited_body(request):
    chunks = bytearray()
    async for chunk in request.stream():
        if len(chunks) + len(chunk) > MAX_BYTES:
            raise HTTPException(413, "File or request exceeds 5 MB.")
        chunks.extend(chunk)
    return bytes(chunks)


class FileInputError(ValueError):
    """Intentionally user-visible validation message."""


def xml(data):
    if len(data) > 2_500_000:
        raise FileInputError(
            "A worksheet or shared-text part exceeds 2.5 MB. Upload a smaller values-only workbook."
        )
    if b"<!DOCTYPE" in data.upper() or b"<!ENTITY" in data.upper():
        raise FileInputError("XML entity declarations are not supported.")
    return ET.fromstring(data)


def tabulate(rows):
    if not rows:
        raise FileInputError("No rows found. Put column names in the first row.")
    width = max(len(r) for r in rows)
    if width > 64 or len(rows) > 1001:
        raise FileInputError("Use at most 64 columns and 1,000 rows per sheet.")
    if sum(len(str(value)) for row in rows for value in row) > 2_000_000:
        raise FileInputError(
            "Expanded cell text exceeds two million characters. Upload a smaller workbook."
        )
    headers = [str(v).strip() or f"Column {i + 1}" for i, v in enumerate(rows[0])]
    headers += [f"Column {i + 1}" for i in range(len(headers), width)]
    return {
        "headers": headers,
        "rows": [r + [""] * (width - len(r)) for r in rows[1:] if any(r)],
    }


def parse_file(data, kind):
    if kind == "csv":
        rows = []
        for row in csv.reader(io.StringIO(data.decode("utf-8-sig"))):
            rows.append(row)
            if len(rows) > 1001:
                raise FileInputError("CSV is limited to 1,000 rows.")
        return {"sheets": [{"name": "CSV", **tabulate(rows)}]}
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        if (
            len(z.infolist()) > 2000
            or sum(i.file_size for i in z.infolist()) > 25 * 1024 * 1024
        ):
            raise FileInputError("Workbook expands beyond the 25 MB safety limit.")
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            shared = [
                "".join(t.text or "" for t in s.findall(".//m:t", ns))
                for s in xml(z.read("xl/sharedStrings.xml"))
            ]
        rels = {
            r.attrib["Id"]: r.attrib["Target"]
            for r in xml(z.read("xl/_rels/workbook.xml.rels"))
        }
        sheets = []
        workbook = xml(z.read("xl/workbook.xml"))
        for sheet in workbook.findall("m:sheets/m:sheet", ns):
            target = rels[
                sheet.attrib[
                    "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
                ]
            ]
            path = (
                target.lstrip("/")
                if target.startswith("/")
                else posixpath.normpath("xl/" + target)
            )
            if not path.startswith("xl/worksheets/") or len(sheets) >= 10:
                raise FileInputError(
                    "Use a workbook with at most ten standard worksheets."
                )
            rows = []
            for row in xml(z.read(path)).findall("m:sheetData/m:row", ns):
                values = [""] * 64
                last = 0
                for cell in row.findall("m:c", ns):
                    col = 0
                    for char in cell.attrib.get("r", ""):
                        if not char.isalpha():
                            break
                        col = col * 26 + ord(char.upper()) - 64
                    if not 1 <= col <= 64:
                        raise FileInputError("Use at most 64 columns.")
                    if cell.find("m:f", ns) is not None:
                        raise FileInputError(
                            "Formula cells are not supported. Upload values only."
                        )
                    raw = cell.findtext("m:v", "", ns)
                    if cell.attrib.get("t") == "s":
                        raw = shared[int(raw)]
                    elif cell.attrib.get("t") == "inlineStr":
                        raw = "".join(t.text or "" for t in cell.findall(".//m:t", ns))
                    values[col - 1] = raw
                    last = max(last, col)
                if last:
                    rows.append(values[:last])
                if len(rows) > 1001:
                    raise FileInputError("Use at most 1,000 rows per sheet.")
            if rows:
                sheets.append({"name": sheet.attrib["name"], **tabulate(rows)})
        if not sheets:
            raise FileInputError("Workbook has no nonempty worksheets.")
        if (
            sum(
                len(value) for sheet in sheets for row in sheet["rows"] for value in row
            )
            > 2_000_000
        ):
            raise FileInputError(
                "Expanded cell text exceeds two million characters. Upload a smaller workbook."
            )
        return {"sheets": sheets}


@app.post("/api/parse")
async def parse(request: Request, kind: str = "xlsx"):
    if kind not in {"xlsx", "csv"}:
        raise HTTPException(400, "Upload .xlsx or UTF-8 .csv.")
    data = await limited_body(request)
    try:
        return parse_file(data, kind)
    except FileInputError as exc:
        # Only our validation messages are shown; XML/CSV errors can quote input.
        raise HTTPException(400, str(exc)) from None
    except Exception:
        raise HTTPException(
            400, "Unable to read this file. Upload a values-only .xlsx or UTF-8 .csv."
        ) from None


class Report(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str = Field(max_length=100)
    text: str = Field(min_length=1, max_length=MAX_CHARS)


class Batch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    reports: list[Report] = Field(min_length=1, max_length=MAX_REPORTS)


@app.post("/api/predict")
async def predict(request: Request):
    try:
        batch = Batch.model_validate_json(await limited_body(request))
    except ValidationError:
        raise HTTPException(
            400, "Submit 1–100 reports, each with an ID and 1–8,000 characters of text."
        ) from None
    if sum(len(r.text) for r in batch.reports) > MAX_TOTAL or any(
        not r.text.strip() for r in batch.reports
    ):
        raise HTTPException(
            400, "Remove blank reports and limit the batch to 200,000 characters."
        )
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise HTTPException(
            503, "Model access is not configured. Contact the app owner."
        )

    async def run(index, report):
        async with slots:
            try:
                result = await annotate(
                    report.text, key, Path("/proc/hip-memory-only"), memory_only=True
                )
                return {"index": index, "id": report.id, **result}
            except Exception:
                return {
                    "index": index,
                    "id": report.id,
                    "error": "Prediction unavailable. Retry this report or review manually.",
                }

    async def stream():
        tasks = [asyncio.create_task(run(i, r)) for i, r in enumerate(batch.reports)]
        pending = set(tasks)
        try:
            yield (
                json.dumps({"event": "start", "total": len(tasks), "model": MODEL})
                + "\n"
            )
            while pending:
                done, pending = await asyncio.wait(
                    pending, timeout=10, return_when=asyncio.FIRST_COMPLETED
                )
                if await request.is_disconnected():
                    break
                if not done:
                    yield '{"event":"heartbeat"}\n'
                for task in done:
                    yield json.dumps({"event": "result", **task.result()}) + "\n"
            if not pending:
                yield '{"event":"complete"}\n'
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    return StreamingResponse(stream(), media_type="application/x-ndjson")
