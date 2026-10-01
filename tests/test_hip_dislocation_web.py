import io
import json
import zipfile
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient
from examples.hip_dislocation.web import app as service
from examples.hip_dislocation.agent import ReadReport, SubmitLabel


@pytest.fixture
def client():
    return TestClient(service.app)


def workbook(rows, formula=False):
    output = io.BytesIO()
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr(
            "xl/workbook.xml",
            f'<workbook xmlns="{ns}" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="Reports" sheetId="1" r:id="rId1"/></sheets></workbook>',
        )
        archive.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships><Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        cells = "".join(
            f'<row r="{i}"><c r="A{i}" t="inlineStr"><is><t>{value}</t></is>{"<f>1+1</f>" if formula else ""}</c></row>'
            for i, value in enumerate(rows, 1)
        )
        archive.writestr(
            "xl/worksheets/sheet1.xml",
            f'<worksheet xmlns="{ns}"><sheetData>{cells}</sheetData></worksheet>',
        )
    return output.getvalue()


def test_upload_parses_bytes_without_files(client, monkeypatch):
    monkeypatch.setattr(
        service.Path, "write_bytes", lambda *a, **k: pytest.fail("Upload wrote a file")
    )
    response = client.post(
        "/api/parse",
        content=workbook(["Report", "Posterior hip dislocation."]),
        headers={"x-hip-request": "1"},
    )
    assert response.status_code == 200
    assert response.json()["sheets"][0]["rows"] == [["Posterior hip dislocation."]]
    assert response.headers["cache-control"] == "no-store"


def test_csv_preserves_columns(client):
    response = client.post(
        "/api/parse?kind=csv",
        content=b'ID,Report,Label\n7,"Posterior, with fracture",posterior\n',
        headers={"x-hip-request": "1"},
    )
    assert response.json()["sheets"][0]["rows"][0] == [
        "7",
        "Posterior, with fracture",
        "posterior",
    ]


@pytest.mark.parametrize(
    "content", [b"not an Excel file", workbook(["Report", "Value"], formula=True)]
)
def test_bad_workbooks_rejected(client, content):
    assert (
        client.post(
            "/api/parse", content=content, headers={"x-hip-request": "1"}
        ).status_code
        == 400
    )


def test_xml_entity_rejected():
    with pytest.raises(ValueError):
        service.xml(b'<!DOCTYPE a [<!ENTITY x "hidden">]><a>&x;</a>')


def test_zip_expansion_rejected():
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("bomb", b"x" * (26 * 1024 * 1024))
    with pytest.raises(ValueError, match="25 MB"):
        service.parse_file(output.getvalue(), "xlsx")


def test_upload_size_limit(client):
    response = client.post(
        "/api/parse?kind=csv",
        content=b"x" * (service.MAX_BYTES + 1),
        headers={"x-hip-request": "1"},
    )
    assert response.status_code == 413


@pytest.mark.parametrize(
    "headers", [{}, {"x-hip-request": "1", "origin": "https://attacker.invalid"}]
)
def test_cross_site_rejected(client, headers):
    assert (
        client.post("/api/parse", content=b"data", headers=headers).status_code == 403
    )


@pytest.mark.parametrize(
    "reports",
    [
        [],
        [{"id": "1", "text": "x", "reference": "posterior"}],
        [{"id": "1", "text": "x" * 8001}],
        [{"id": "1", "text": " "}],
    ],
)
def test_prediction_input_contract(client, reports, monkeypatch):
    mock = AsyncMock()
    monkeypatch.setattr(service, "annotate", mock)
    assert (
        client.post(
            "/api/predict", json={"reports": reports}, headers={"x-hip-request": "1"}
        ).status_code
        == 400
    )
    mock.assert_not_called()


def test_no_key_gives_safe_message(client, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert (
        client.post(
            "/api/predict",
            json={"reports": [{"id": "1", "text": "report"}]},
            headers={"x-hip-request": "1"},
        ).status_code
        == 503
    )


def test_stream_uses_memory_agent_and_preserves_rows(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    prediction = {
        "label": "posterior",
        "needs_review": False,
        "obturator_screen": "unlikely",
        "evidence": "posterior",
        "reason": "Posterior displacement.",
    }
    mock = AsyncMock(return_value={"prediction": prediction, "seconds": 1})
    monkeypatch.setattr(service, "annotate", mock)
    response = client.post(
        "/api/predict",
        json={
            "reports": [
                {"id": "7", "text": "posterior"},
                {"id": "7", "text": "posterior again"},
            ]
        },
        headers={"x-hip-request": "1"},
    )
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[0]["total"] == 2 and events[-1]["event"] == "complete"
    assert sorted(e["index"] for e in events if e["event"] == "result") == [0, 1]
    assert all(c.kwargs == {"memory_only": True} for c in mock.await_args_list)


def test_error_never_echoes_report_or_secret(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(
        service,
        "annotate",
        AsyncMock(side_effect=RuntimeError("PRIVATE-REPORT test-key")),
    )
    response = client.post(
        "/api/predict",
        json={"reports": [{"id": "1", "text": "PRIVATE-REPORT"}]},
        headers={"x-hip-request": "1"},
    )
    assert "PRIVATE-REPORT" not in response.text and "test-key" not in response.text
    assert '"error"' in response.text


@pytest.mark.asyncio
async def test_memory_tools_use_text_blocks():
    reader = ReadReport("posterior " * 1000, memory_only=True)
    assert isinstance((await reader.execute({})).output, list)
    submit = SubmitLabel(reader)
    result = await submit.execute(
        {
            "label": "posterior",
            "needs_review": False,
            "obturator_screen": "unlikely",
            "evidence": "posterior",
            "reason": "Direction is explicit.",
        }
    )
    assert result.success and isinstance(result.output, list)


def test_static_hardening(client):
    response = client.get("/")
    assert response.status_code == 200 and "Upload file" in response.text
    assert "frame-ancestors 'none'" in response.headers["content-security-policy"]
    assert client.get("/openapi.json").status_code == 404


def test_generic_file_errors_are_not_echoed(client, monkeypatch):
    def bad_file(*args):
        raise ValueError("private report text")

    monkeypatch.setattr(service, "parse_file", bad_file)
    response = client.post(
        "/api/parse", content=b"data", headers={"x-hip-request": "1"}
    )
    assert response.status_code == 400 and "private report" not in response.text


def test_repeated_shared_text_cannot_expand_response_unbounded():
    with pytest.raises(service.FileInputError, match="two million"):
        service.tabulate([["Report"]] + [["x" * 30_000]] * 100)


def test_single_xml_part_is_bounded():
    with pytest.raises(service.FileInputError, match="2.5 MB"):
        service.xml(b" " * 2_500_001)
