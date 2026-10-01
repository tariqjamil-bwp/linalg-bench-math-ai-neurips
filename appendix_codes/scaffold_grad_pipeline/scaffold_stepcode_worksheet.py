#!/usr/bin/env python3
# Module:    scaffold_stepcode_worksheet.py
# Version:   1.0
"""
scaffold_stepcode_worksheet.py — build the human coding sheets for the
step-error reliability check.

The step-error code is a human judgement: it marks where a derivation
first went wrong, and reporting Cohen's kappa only means something if two
coders apply the rubric independently. This script prepares the sheets;
the filled-in codes feed back through scaffold_build_deliverables.py.

Produces:
    scaffold_stepcodes_coder1.xlsx   every incorrect L3/L4 response
    scaffold_stepcodes_coder2.xlsx   a random 20% sample of the same rows
                                     (minimum 20), for the reliability check

The sample is drawn with a fixed seed (42) so it can be regenerated if a
sheet is lost. The two files are deliberately separate workbooks — coder 2
must not see coder 1's codes, since a shared sheet is not independence and
kappa computed off it means nothing.

Rubric (dropdown-enforced):
    E-SETUP    A - lambda*I written incorrectly; wrong matrix entries
    E-EXPAND   correct A - lambda*I but wrong characteristic polynomial
    E-ROOT     correct p(lambda) but wrong numerical roots
    E-NONE     model did not attempt the assigned procedure

One code per response, for the first step at which the error occurs. The
dropdown is a real Excel data validation, so a typo can't enter the data
and quietly become its own category in the kappa table.

Usage:
    python3 scaffold_stepcode_worksheet.py \\
        --raw ../deliverables/scaffold_results_raw.csv --output ../coding
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd
from openpyxl import Workbook

# Paths go through P.<fn>() rather than being imported by name: scaffold_paths
# has no default root, so a from-import would bind before --projects-root is read.
import scaffold_paths as P
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter
from openpyxl.worksheet.datavalidation import DataValidation

CODES = ["E-SETUP", "E-EXPAND", "E-ROOT", "E-NONE"]
SEED = 42
SAMPLE_FRACTION = 0.20
SAMPLE_MINIMUM = 20

COLUMNS = ["model", "level", "pool_id", "problem_id", "repeat",
           "boxed_content", "response_text", "step_error_code", "coder_notes"]

HEADER_FILL = PatternFill("solid", fgColor="1F3864")
HEADER_FONT = Font(color="FFFFFF", bold=True, size=11)
CODE_FILL = PatternFill("solid", fgColor="FFF2CC")
THIN = Side(style="thin", color="BFBFBF")
BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)

WIDTHS = {"model": 20, "level": 8, "pool_id": 13, "problem_id": 17, "repeat": 8,
          "boxed_content": 42, "response_text": 90, "step_error_code": 16,
          "coder_notes": 34}


def read_coding_sheet(path: str) -> pd.DataFrame:
    """Read a filled-in coding sheet back.

    Shared by scaffold_kappa.py and scaffold_build_deliverables.py so there is
    exactly one reader. It matters because the sheet carries a rubric block
    above the table: a plain read_excel() takes the first rubric line as the
    header and silently returns zero usable codes. The header is found by
    looking for the row whose first cell is "model", which also survives a
    coder inserting a note of their own above the table.
    """
    df = (pd.read_excel(path, header=None) if path.endswith((".xlsx", ".xlsm"))
          else pd.read_csv(path, header=None, dtype=str, keep_default_na=False))
    hdr = next((i for i in range(len(df))
                if str(df.iloc[i, 0]).strip() == "model"), None)
    if hdr is None:
        raise SystemExit(f"{path}: no header row starting with 'model' found.")
    out = df.iloc[hdr + 1:].copy()
    out.columns = [str(c).strip() for c in df.iloc[hdr]]
    out = out.dropna(how="all")
    for c in ("model", "level", "problem_id", "repeat", "step_error_code"):
        if c not in out.columns:
            raise SystemExit(f"{path}: missing column {c!r}")
    out["repeat"] = out["repeat"].astype(str).str.strip()
    out["step_error_code"] = (out["step_error_code"].astype(str)
                              .str.strip().str.upper().replace({"NAN": ""}))
    return out


def _instructions(ws) -> None:
    """A self-contained rubric block, in rows above the table so it can't be
    sorted away from the data."""
    lines = [
        "STEP-ERROR CODING — assign the FIRST step at which the derivation goes wrong.",
        "Exactly one code per response. Use the dropdown in the step_error_code column.",
        "",
        "  E-SETUP    (A - lambda*I) written incorrectly; wrong matrix entries",
        "  E-EXPAND   correct (A - lambda*I) but wrong characteristic polynomial",
        "  E-ROOT     correct p(lambda) but wrong numerical roots",
        "  E-NONE     the model did not attempt the assigned procedure",
        "",
        "Code independently. Do not discuss rows with the other coder before both "
        "sheets are finished — Cohen's kappa is only meaningful if the two passes "
        "were genuinely independent.",
    ]
    for i, text in enumerate(lines, start=1):
        c = ws.cell(row=i, column=1, value=text)
        c.font = Font(bold=(i == 1 or text.strip().startswith("E-")), size=10)
    return len(lines) + 1          # blank row, then the header


def write_sheet(df: pd.DataFrame, path: str, title: str) -> None:
    wb = Workbook()
    ws = wb.active
    ws.title = title

    top = _instructions(ws)
    header_row = top + 1

    for j, name in enumerate(COLUMNS, start=1):
        c = ws.cell(row=header_row, column=j, value=name)
        c.fill, c.font = HEADER_FILL, HEADER_FONT
        c.alignment = Alignment(horizontal="center", vertical="center")
        c.border = BORDER
        ws.column_dimensions[get_column_letter(j)].width = WIDTHS[name]

    for i, row in enumerate(df.itertuples(index=False), start=header_row + 1):
        for j, name in enumerate(COLUMNS, start=1):
            value = "" if name in ("step_error_code", "coder_notes") \
                else getattr(row, name, "")
            c = ws.cell(row=i, column=j, value=value)
            c.border = BORDER
            c.alignment = Alignment(vertical="top",
                                    wrap_text=name in ("boxed_content", "response_text",
                                                       "coder_notes"))
            if name == "step_error_code":
                c.fill = CODE_FILL

    last = header_row + len(df)
    col = get_column_letter(COLUMNS.index("step_error_code") + 1)
    dv = DataValidation(type="list", formula1='"' + ",".join(CODES) + '"',
                        allow_blank=True, showDropDown=False)
    dv.error = "Pick one of the four rubric codes."
    dv.errorTitle = "Not a rubric code"
    ws.add_data_validation(dv)
    dv.add(f"{col}{header_row + 1}:{col}{last}")

    ws.freeze_panes = ws[f"A{header_row + 1}"]
    ws.auto_filter.ref = f"A{header_row}:{get_column_letter(len(COLUMNS))}{last}"
    wb.save(path)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--projects-root", required=True,
                    help="output tree, e.g. ../projects. Must already exist. "
                         "Required: there is no default, so nothing is read from "
                         "or written to a tree you did not name.")
    ap.add_argument("--subcat", default=P.DEFAULT_SUBCAT, choices=sorted(P.SUBCATS),
                    help=f"path segment <projects-root>/{{subcat}}/... "
                         f"(default: {P.DEFAULT_SUBCAT})")
    ap.add_argument("--raw", default=None,
                    help="scaffold_results_raw.csv "
                         "(default: <projects-root>/{subcat}/deliverables/)")
    ap.add_argument("--output", default=None,
                    help="where to write the two sheets "
                         "(default: <projects-root>/{subcat}/coding)")
    ap.add_argument("--truncate-response", type=int, default=6000,
                    help="cap response_text at this many chars (Excel's cell limit "
                         "is 32767; a full L4 derivation can approach it)")
    args = ap.parse_args()

    P.configure(args.projects_root, args.subcat)
    raw_path = args.raw or os.path.join(P.deliverables(), "scaffold_results_raw.csv")
    outdir   = args.output or P.coding()

    print("\nPATHS")
    print(f"  projects root   {P.projects()}")
    print(f"  subcat          {P.subcat()}")
    print(f"  raw             {os.path.abspath(raw_path)}")
    print(f"  coding sheets   {os.path.abspath(outdir)}\n")

    raw = pd.read_csv(raw_path, dtype=str, keep_default_na=False)
    todo = raw[raw.level.isin(["L3", "L4"]) & (raw.accuracy == "0")].copy()
    if todo.empty:
        raise SystemExit("No incorrect L3/L4 responses to code.")

    todo["response_text"] = todo.response_text.str.slice(0, args.truncate_response)
    todo = todo.sort_values(["model", "level", "pool_id", "problem_id", "repeat"])
    todo = todo[COLUMNS[:-2]].assign(step_error_code="", coder_notes="")

    n_sample = max(SAMPLE_MINIMUM, int(round(SAMPLE_FRACTION * len(todo))))
    n_sample = min(n_sample, len(todo))
    sample = todo.sample(n=n_sample, random_state=SEED).sort_index()

    os.makedirs(outdir, exist_ok=True)
    p1 = os.path.join(outdir, "scaffold_stepcodes_coder1.xlsx")
    p2 = os.path.join(outdir, "scaffold_stepcodes_coder2.xlsx")
    write_sheet(todo, p1, "coder1")
    write_sheet(sample, p2, "coder2_sample")

    print(f"  incorrect L3/L4 responses : {len(todo)}")
    print(f"  coder 2 sample            : {len(sample)}  "
          f"({SAMPLE_FRACTION:.0%} or {SAMPLE_MINIMUM} minimum, seed {SEED})")
    print(f"  wrote {p1}")
    print(f"  wrote {p2}")
    print("\n  When both are filled in:")
    print("    python3 scaffold_kappa.py --coder1 ... --coder2 ...")
    print("    python3 scaffold_build_deliverables.py ... --step-codes <coder1> --kappa <k>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
