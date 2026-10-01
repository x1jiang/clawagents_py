"""Create an audited Excel results workbook from completed agent receipts.

Run with an existing Python environment containing openpyxl. The agent itself
requires neither openpyxl nor plotting packages. Recalculate the workbook before
using data_only cached metrics.
"""

import argparse
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = args.run_dir.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    metrics = json.loads((root / "metrics.json").read_text())
    reports = sorted(
        [json.loads(p.read_text()) for p in (root / "predictions").glob("*.json")],
        key=lambda r: r["source_row"],
    )
    records = json.loads((root / "records.json").read_text())
    assert len(reports) == manifest["n_reports"], "Incomplete run"
    n = len(reports) + 1
    nr = len(records) + 1
    labels = ["obturator", "posterior", "anterior_superior", "other_unknown"]
    wb = Workbook()
    readme = wb.active
    readme.title = "Read me"
    for row in [
        ["LIVE GPT-6 LUNA • SELECTED-EXAMPLE BENCHMARK"],
        ["Model", manifest["model"]],
        ["Endpoint", manifest["provider"]],
        ["Reasoning", manifest["reasoning_effort"]],
        ["Source SHA-256", manifest["source_sha256"]],
        ["Prompt SHA-256", manifest["prompt_sha256"]],
        ["Code SHA-256", manifest["code_sha256"]],
        [
            "What was tested",
            "Live two-tool ClawAgents annotation: read one report, then submit a validated literal quote, subtype, review flag, and screening flag.",
        ],
        [
            "Blinding",
            "Only report text reaches the agent. Jack’s label, rationale, linking ID, previous rule outputs, and other reports are excluded. No filesystem, web, shell, memory, or skill access.",
        ],
        [
            "Evidence boundary",
            "Jack’s selected training examples; prompt designed after the dataset had already been inspected by the builder. Reference labels blinded to inference, but this is not an untouched held-out validation set.",
        ],
        [
            "Ground truth",
            "Jack’s Type column; no independent label adjudication available. Other/unknown is an abstention without a reference class in this sample.",
        ],
        [
            "Aggregation",
            "At least one non-review subtype supports a linked record, provided supported reports do not conflict. Otherwise other_unknown/review. This is not initial-report performance.",
        ],
        [
            "Repeated imaging",
            "Report-level rows are correlated within IDs. Linked-record IDs are treated as supplied case links, without independently verified patient identity.",
        ],
        [
            "Performance limits",
            "Only one anterior-superior case and no other/unknown controls, historical, negative, or post-reduction reference examples. Cannot estimate population incidence from selected examples.",
        ],
        [
            "Cost",
            "Token-based standard-price estimate, not a verified bill. Rates: uncached input $0.10/M, cached input $0.01/M, output $0.50/M; provider tiers/adjustments may differ.",
        ],
        [
            "Source text",
            "Original input remains unchanged. Local receipts retain text and linking IDs for audit.",
        ],
        [
            "Source updates",
            "Metrics recalculate from outputs; re-run the agent with a new results folder for changed report text.",
        ],
    ]:
        readme.append(row)
    readme.column_dimensions["A"].width = 27
    readme.column_dimensions["B"].width = 115

    r = wb.create_sheet("Reports")
    r.append(
        [
            "Source row",
            "Linked ID",
            "Jack reference",
            "Model label",
            "Needs review",
            "Obturator screen",
            "Exact evidence quote",
            "Model reason",
            "Report text",
            "Imaging type",
            "Seconds",
            "LLM requests",
            "Input tokens incl cache",
            "Cached tokens",
            "Output tokens",
            "Matches reference",
            "Assigned without review",
            "Correct assigned",
        ]
    )
    for i, x in enumerate(reports, 2):
        p = x["prediction"]
        u = x["usage"]
        r.append(
            [
                x["source_row"],
                x["record_id"],
                x["reference"],
                p["label"],
                int(p["needs_review"]),
                p["obturator_screen"],
                p["evidence"],
                p["reason"],
                x["report"],
                x["imaging_type"],
                x["seconds"],
                u["requests"],
                u["prompt_tokens"],
                u["cached_input_tokens"],
                u["output_tokens"],
                f"=IF(C{i}=D{i},1,0)",
                f'=IF(AND(E{i}=0,D{i}<>"other_unknown"),1,0)',
                f"=IF(AND(P{i}=1,Q{i}=1),1,0)",
            ]
        )
        r.row_dimensions[i].height = 95
    for col, w in zip(
        "ABCDEFGHIJKLMNOPQR",
        [12, 14, 23, 23, 15, 22, 70, 65, 85, 35, 14, 16, 23, 18, 18, 20, 22, 20],
    ):
        r.column_dimensions[col].width = w
    r.auto_filter.ref = r.dimensions

    c = wb.create_sheet("Linked records")
    c.append(
        [
            "Linked ID",
            "Jack reference",
            "Model label",
            "Needs review",
            "Obturator screen",
            "Source rows",
            "Conflicting supported labels",
            "Report failures",
            "Matches reference",
            "Assigned without review",
            "Correct assigned",
        ]
    )
    for i, x in enumerate(records, 2):
        p = x["prediction"]
        c.append(
            [
                x["record_id"],
                x["reference"],
                p["label"],
                int(p["needs_review"]),
                p["obturator_screen"],
                ", ".join(map(str, x["source_rows"])),
                int(x["conflicting_supported_labels"]),
                x["report_failures"],
                f"=IF(B{i}=C{i},1,0)",
                f'=IF(AND(D{i}=0,C{i}<>"other_unknown"),1,0)',
                f"=IF(AND(I{i}=1,J{i}=1),1,0)",
            ]
        )
    for col, w in zip("ABCDEFGHIJK", [15, 24, 24, 15, 22, 25, 28, 20, 23, 23, 23]):
        c.column_dimensions[col].width = w
    c.auto_filter.ref = c.dimensions

    s = wb.create_sheet("Summary", 1)
    s.append(["MEASURE", "Report rows", "Linked records", "Interpretation"])
    s.append(
        [
            "N",
            f"=COUNTA(Reports!A2:A{n})",
            f"=COUNTA('Linked records'!A2:A{nr})",
            "All supplied units.",
        ]
    )
    s.append(
        [
            "Matches Jack label",
            f"=SUM(Reports!P2:P{n})",
            f"=SUM('Linked records'!I2:I{nr})",
            "Unknown/abstention remains in denominator.",
        ]
    )
    s.append(
        [
            "Agreement over all examples",
            "=IFERROR(B3/B2,0)",
            "=IFERROR(C3/C2,0)",
            "Selected-example exact-label agreement; not independent validation.",
        ]
    )
    s.append(
        [
            "Assigned without review",
            f"=SUM(Reports!Q2:Q{n})",
            f"=SUM('Linked records'!J2:J{nr})",
            "Model supplies specific subtype and no review flag.",
        ]
    )
    s.append(
        [
            "Assignment coverage",
            "=IFERROR(B5/B2,0)",
            "=IFERROR(C5/C2,0)",
            "Fraction assigned without review.",
        ]
    )
    s.append(
        [
            "Correct assigned",
            f"=SUM(Reports!R2:R{n})",
            f"=SUM('Linked records'!K2:K{nr})",
            "Assigned subtype agrees with supplied reference.",
        ]
    )
    s.append(
        [
            "Agreement among assigned",
            "=IFERROR(B7/B5,0)",
            "=IFERROR(C7/C5,0)",
            "Conditional on no review; do not confuse with overall agreement.",
        ]
    )
    s.append(
        ["Needs review", "=B2-B5", "=C2-C5", "Includes unknown/uncertain outputs."]
    )
    s.append(
        [
            "Labeled obturator",
            f'=COUNTIF(Reports!C2:C{n},"obturator")',
            f"=COUNTIF('Linked records'!B2:B{nr},\"obturator\")",
            "Jack’s reference-positive units.",
        ]
    )
    s.append(
        [
            "Obturator captured by screen",
            f'=COUNTIFS(Reports!C2:C{n},"obturator",Reports!F2:F{n},"possible")',
            f"=COUNTIFS('Linked records'!B2:B{nr},\"obturator\",'Linked records'!E2:E{nr},\"possible\")",
            "Screen may include ambiguous subtype requiring review.",
        ]
    )
    s.append(
        [
            "Obturator screening capture",
            "=IFERROR(B11/B10,0)",
            "=IFERROR(C11/C10,0)",
            "Selected positive examples only.",
        ]
    )
    s.append(
        [
            "Non-obturator flagged",
            f'=COUNTIFS(Reports!C2:C{n},"<>obturator",Reports!F2:F{n},"possible")',
            f"=COUNTIFS('Linked records'!B2:B{nr},\"<>obturator\",'Linked records'!E2:E{nr},\"possible\")",
            "No other/unknown control cases supplied.",
        ]
    )
    s.append(
        [
            "Review IDs",
            None,
            ", ".join(map(str, metrics["records_needing_review"])),
            "Supplied linking-log IDs.",
        ]
    )
    s.append(
        [
            "LLM requests",
            f"=SUM(Reports!L2:L{n})",
            None,
            "Excludes read-only model availability preflight.",
        ]
    )
    s.append(
        [
            "Prompt tokens incl cache",
            f"=SUM(Reports!M2:M{n})",
            None,
            "Actual returned API usage.",
        ]
    )
    s.append(
        [
            "Cached input tokens",
            f"=SUM(Reports!N2:N{n})",
            None,
            "Actual returned API usage.",
        ]
    )
    s.append(
        [
            "Output tokens incl reasoning",
            f"=SUM(Reports!O2:O{n})",
            None,
            "Actual returned API usage.",
        ]
    )
    s.append(["Uncached input rate / M", 0.1, None, "USD; official model page."])
    s.append(["Cached input rate / M", 0.01, None, "USD; official model page."])
    s.append(["Output rate / M", 0.5, None, "USD; official model page."])
    s.append(
        [
            "Estimated standard cost USD",
            "=((B16-B17)*B19+B17*B20+B18*B21)/1000000",
            None,
            "Estimate only; not billing confirmation.",
        ]
    )
    for i in [4, 6, 8, 12]:
        for col in ["B", "C"]:
            s[f"{col}{i}"].number_format = "0.0%"
    s["B22"].number_format = "$0.0000"
    for col, w in zip("ABCD", [45, 23, 27, 100]):
        s.column_dimensions[col].width = w

    cm = wb.create_sheet("Confusion and F1")
    for section, sheet, ref_col, pred_col, last in [
        ("REPORTS", "Reports", "C", "D", n),
        ("LINKED RECORDS", "Linked records", "B", "C", nr),
    ]:
        header = cm.max_row + 1 if cm.max_row > 1 else 1
        if header > 1:
            cm.append([])
            header += 1
        cm.append(
            [section + " • reference × predicted"]
            + labels
            + ["Reference n", "Predicted n", "Precision", "Recall", "F1"]
        )
        for label in labels[:3]:
            i = cm.max_row + 1
            counts = [
                f"=COUNTIFS('{sheet}'!{ref_col}$2:{ref_col}${last},$A{i},'{sheet}'!{pred_col}$2:{pred_col}${last},{cm.cell(header, j).coordinate})"
                for j in range(2, 6)
            ]
            j = labels.index(label) + 2
            tp_cell = f"{get_column_letter(j)}{i}"
            cm.append(
                [label]
                + counts
                + [
                    f"=COUNTIF('{sheet}'!{ref_col}$2:{ref_col}${last},A{i})",
                    f"=COUNTIF('{sheet}'!{pred_col}$2:{pred_col}${last},A{i})",
                    f'=IFERROR({tp_cell}/G{i},"")',
                    f'=IFERROR({tp_cell}/F{i},"")',
                    f'=IFERROR(2*{tp_cell}/(F{i}+G{i}),"")',
                ]
            )
            for col in "HIJ":
                cm[f"{col}{i}"].number_format = "0.0%"
        cm.append(["Other/unknown = abstention; no reference examples for that class."])
    cm.column_dimensions["A"].width = 56
    for col in "BCDEFGHIJ":
        cm.column_dimensions[col].width = 21

    for sheet in wb:
        sheet.freeze_panes = "B2"
        sheet.sheet_view.showGridLines = False
        for row in sheet:
            for cell in row:
                cell.font = Font(name="Arial", size=11, color="172B3A")
                cell.alignment = Alignment(vertical="top", wrap_text=True)
                if cell.data_type == "f":
                    cell.font = Font(name="Arial", size=11, color="008000")
        for cell in sheet[1]:
            cell.fill = PatternFill("solid", fgColor="17435E")
            cell.font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
        sheet.row_dimensions[1].height = 35
        for i in range(2, sheet.max_row + 1):
            if sheet != r:
                sheet.row_dimensions[i].height = 46
            if i % 2 == 0:
                for cell in sheet[i]:
                    cell.fill = PatternFill("solid", fgColor="EEF4F7")
    for i, x in enumerate(reports, 2):
        if x["prediction"]["needs_review"]:
            r[f"E{i}"].fill = PatternFill("solid", fgColor="FFF1CF")
    target = root / "GPT6_Luna_performance.xlsx"
    wb.save(target)
    print(target)


if __name__ == "__main__":
    main()
