"""End-to-end: load targets YAML, run a case, produce a verdict & report."""
from pathlib import Path

import pytest

from rar_gravity_pt0.validate import load_targets, run_case
from rar_gravity_pt0.report import write_go_no_go, error_table_markdown

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "crespi_table4_targets.yaml"


def test_targets_yaml_loads_and_has_two_cases():
    spec = load_targets(DATA)
    cases = spec["cases"]
    assert "case_56keV" in cases
    assert "case_300keV" in cases


def test_run_56kev_converges_and_reports(tmp_path):
    spec = load_targets(DATA)
    case = spec["cases"]["case_56keV"]
    res = run_case("case_56keV", case)
    # the 56 keV core-mass shoot is calibrated to converge
    assert res.converged, res.notes
    # all five observables present
    names = {c.name for c in res.comparisons}
    assert names == {
        "core_mass_msun", "core_radius_pc", "total_mass_msun",
        "extended_mass_within_s2_msun", "s2_precession_arcmin_per_orbit"}
    # with placeholder (null) targets the verdict must be NO-TARGET, never a
    # fabricated GO
    assert res.verdict == "NO-TARGET"
    log = write_go_no_go([res], tmp_path / "go.txt")
    assert "GO / NO-GO" in log
    md = error_table_markdown([res])
    assert "case_56keV" in md


def test_no_fabricated_go_without_targets():
    spec = load_targets(DATA)
    res = run_case("case_300keV", spec["cases"]["case_300keV"])
    # honest reporting: no targets -> cannot be EXCELLENT/GO
    assert res.verdict == "NO-TARGET"
    # 300 keV shoots beta0 and hits the critical-mass ceiling -> non-converged,
    # and the branch maximum must be reported (no silent failure / fake GO)
    assert res.converged is False
    assert any("branch maximum" in n for n in res.notes), res.notes
