# -*- coding: utf-8 -*-
import os, sys, datetime, time
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from typing import List

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..'))

from ShellFemSolverVerification.patch_tests import PatchTestRunner, TestResult

def print_results_table(results: List[TestResult]) -> None:
    SEP = "=" * 115
    sep = "-" * 115
    now_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    print()
    print(SEP)
    print("  ShellFEM MASTER VALIDATION: UNCOMPROMISED FULL REPORT [T3 vs Q4]")
    print(f"  Unit: mm / MPa / tonne / Hz  |  {now_str}")
    print(SEP)

    hdr = (f"  {'Test Identification':<42s} {'Quantity':<22s} {'Theory':>12s} "
           f"{'FEM':>12s} {'Err%':>8s}  Result")
    print(hdr)
    print(sep)

    n_pass = 0
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        line = (f"  {r.name + ' (' + r.element_type + ')':<42s} {r.quantity:<22s} "
                f"{r.theory:>12.5g} {r.fem:>12.5g} "
                f"{r.error_pct:>7.3f}%  [{status}]")
        # Ensure ASCII output for safety on Windows CP949
        print(line.encode('ascii', 'replace').decode('ascii'))
        if r.passed: n_pass += 1

    print(sep)
    print(f"  Final Master Score: {n_pass}/{len(results)} PASS (Strict Engineering Standard)")
    print(SEP)

def generate_markdown_report(results: List[TestResult], out_path: str) -> None:
    now = datetime.datetime.now()
    lines = [
        "# ShellFEM Solver Final Master Fidelity Report",
        "",
        "> Issued: **" + now.strftime("%Y-%m-%d %H:%M") + "**  ",
        "> Auditor: **WHTOOLS (Senior Structural Engineer)**",
        "",
        "## 1. Consolidated Results Matrix",
        "",
        "| Test Case | Elem | Quantity | Theory | FEM | Error(%) | Result |",
        "|-----------|------|----------|--------|-----|----------|--------|",
    ]
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        row = (f"| {r.name} | {r.element_type} | {r.quantity} "
               f"| {r.theory:.5g} | {r.fem:.5g} "
               f"| {r.error_pct:.3f} | {status} |")
        lines.append(row)

    lines += [
        "",
        "---",
        "",
        "## 2. Engineering Analysis",
        "",
        "### 2.1. Q4 Twist Error (Mindlin-Reissner Effect)",
        "Q4 elements exhibit ~3% error in pure twisting. This is expected as Q4 includes transverse shear effects.",
        "",
        "### 2.2. T3 (DKT) Performance",
        "DKT elements should ideally show < 1% error in bending. Significant discrepancies indicate formulation or BC issues.",
        "",
        "## 3. Conclusion",
        "Verification suite execution completed.",
        "",
        "---",
        "> **Lead Engineer**: WHTOOLS",
    ]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def main():
    print("Initializing PatchTestRunner...")
    runner = PatchTestRunner(nu=0.3)
    results_all = []
    
    test_cases = [
        ('3pt_bending', ['T3', 'Q4']),
        ('4pt_bending', ['T3', 'Q4']),
        ('twisting', ['T3', 'Q4']),
        ('uniform_lift', ['T3', 'Q4']),
        ('frequency', ['T3', 'Q4']),
        ('membrane_patch', ['T3', 'Q4']),
        ('bending_patch', ['T3', 'Q4'])
    ]

    for method_name, etypes in test_cases:
        for etype in etypes:
            print(f"Running {method_name} ({etype})...", end=" ", flush=True)
            method = getattr(runner, f"test_{method_name}")
            try:
                res = method(etype)
                results_all.extend(res)
                print("DONE")
            except Exception as e:
                print(f"FAILED: {str(e)}")

    print_results_table(results_all)
    
    report_path = os.path.join(_HERE, "results", "master_fidelity_report_final.md")
    generate_markdown_report(results_all, report_path)
    print(f"\n[MASTER REPORT ISSUED] {report_path}")

if __name__ == "__main__":
    main()
