"""Independently verify completed live receipts and recalculated Excel metrics."""

import argparse
import hashlib
import json
from pathlib import Path

from openpyxl import load_workbook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    root = parser.parse_args().run_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    metrics = json.loads((root / "metrics.json").read_text())
    rows = sorted(
        [json.loads(p.read_text()) for p in (root / "predictions").glob("*.json")],
        key=lambda r: r["source_row"],
    )
    assert len(rows) == manifest["n_reports"] and len(
        {r["source_row"] for r in rows}
    ) == len(rows)
    assert (
        sum(r["prediction"]["label"] == r["reference"] for r in rows)
        == metrics["reports"]["correct"]
    )
    assert all(
        r["status"] == "done" and r["tool_calls"] == 2 and r["usage"]["requests"] == 2
        for r in rows
    )
    assert all(
        q["model"] == "gpt-6-luna" for r in rows for q in r["usage"]["per_request"]
    )
    assert (
        hashlib.sha256((root / "agent_snapshot.py").read_bytes()).hexdigest()
        == manifest["code_sha256"]
    )
    assert (
        hashlib.sha256(manifest["prompt"].encode()).hexdigest()
        == manifest["prompt_sha256"]
    )
    source = Path(manifest["source"])
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_sha256"]
    original = load_workbook(source, data_only=True).active
    w = load_workbook(root / "GPT6_Luna_performance.xlsx", data_only=True)
    s = w["Summary"]
    expected = {
        2: "n",
        3: "correct",
        4: "agreement_all",
        5: "assigned",
        6: "coverage",
        7: "assigned_correct",
        8: "selective_agreement",
        11: "obturator_screen_tp",
        12: "obturator_screen_capture",
        13: "obturator_screen_fp",
    }
    for i, key in expected.items():
        for col, scope in zip("BC", ["reports", "records"]):
            assert abs(s[f"{col}{i}"].value - metrics[scope][key]) < 1e-10, (
                i,
                col,
                key,
            )
    assert abs(s["B22"].value - metrics["estimated_standard_usd"]) < 1e-10
    for sheet in w:
        for row in sheet:
            for cell in row:
                assert cell.data_type != "e", (sheet.title, cell.coordinate)
    for i, r in enumerate(rows, 2):
        assert (
            w["Reports"][f"I{i}"].value
            == r["report"]
            == original[f"D{r['source_row']}"].value
        )
    cm = w["Confusion and F1"]
    labels = ["obturator", "posterior", "anterior_superior", "other_unknown"]
    headers = [
        i
        for i in range(1, cm.max_row + 1)
        if str(cm.cell(i, 1).value).endswith("reference × predicted")
    ]
    assert len(headers) == 2
    for start, scope in zip(headers, ["reports", "records"]):
        for i, key in enumerate(labels[:3], start + 1):
            assert cm.cell(i, 1).value == key
            for j, pred in enumerate(labels, 2):
                assert cm.cell(i, j).value == metrics[scope]["confusion"][key][pred], (
                    scope,
                    i,
                    j,
                )
            for col, field in zip("HIJ", ["precision", "recall", "f1"]):
                value = metrics[scope]["per_class"][key][field]
                if value is None:
                    assert cm[f"{col}{i}"].value is None
                else:
                    assert abs(cm[f"{col}{i}"].value - value) < 1e-10, (scope, col, i)
    receipt = {
        "status": "passed",
        "completed_reports": len(rows),
        "checks": [
            "all supplied reports completed",
            "model usage receipts",
            "immutable prompt and code snapshots",
            "original source hash and report preservation",
            "independently counted exact-label agreement",
            "recalculated summary metrics",
            "report and record confusion matrices and F1",
            "zero Excel formula errors",
        ],
    }
    (root / "verification.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
