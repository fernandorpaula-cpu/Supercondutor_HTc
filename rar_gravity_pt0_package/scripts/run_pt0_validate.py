#!/usr/bin/env python3
"""
PT0 validation driver (Step 5 of the mandated order).

Usage:
    python scripts/run_pt0_validate.py --case all
    python scripts/run_pt0_validate.py --case case_56keV
    python scripts/run_pt0_validate.py --case case_300keV

Reads data/crespi_table4_targets.yaml, solves each RAR configuration
(shooting where requested), computes the S2 observables, compares them to
the Crespi targets, writes:
    output/reports/pt0_error_table.md
    output/logs/pt0_go_no_go.txt
    output/figures/*.png
and prints the error table + GO/NO-GO to stdout.
"""
import argparse
from pathlib import Path

import _bootstrap  # noqa: F401

from rar_gravity_pt0.validate import (load_targets, run_case,
                                      aggregate_decision)
from rar_gravity_pt0.report import (error_table_markdown, write_go_no_go,
                                    plot_profiles)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "crespi_table4_targets.yaml"
OUT = ROOT / "output"


def main() -> int:
    ap = argparse.ArgumentParser(description="RAR-GRAVITY PT0 validation")
    ap.add_argument("--case", default="all",
                    help="case id from the YAML, or 'all'")
    ap.add_argument("--targets", default=str(DATA),
                    help="path to crespi_table4_targets.yaml")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    spec = load_targets(args.targets)
    cases = spec.get("cases", {})
    if args.case != "all":
        if args.case not in cases:
            raise SystemExit(f"unknown case {args.case!r}; "
                             f"available: {list(cases)}")
        cases = {args.case: cases[args.case]}

    results = []
    for cid, case in cases.items():
        print(f"\n>>> solving case {cid} (mc^2={case['mc2_keV']} keV) ...")
        res = run_case(cid, case)
        results.append(res)
        print(f"    verdict: {res.verdict}  converged: {res.converged}")

    # error table
    table = error_table_markdown(results)
    print("\n" + table)
    (OUT / "reports").mkdir(parents=True, exist_ok=True)
    (OUT / "reports" / "pt0_error_table.md").write_text(table)

    # GO/NO-GO log
    log_text = write_go_no_go(results, OUT / "logs" / "pt0_go_no_go.txt")
    print(log_text)

    # figures
    if not args.no_figures:
        figs = plot_profiles(results, OUT / "figures")
        for f in figs:
            print(f"figure: {f}")

    decision = aggregate_decision(results)
    print(f"\nOVERALL: {decision}")
    # exit non-zero only on a hard NO-GO so CI can gate on it
    return 2 if decision == "NO-GO" else 0


if __name__ == "__main__":
    raise SystemExit(main())
