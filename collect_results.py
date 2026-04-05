import os
import sys
import shutil
import glob
import pandas as pd
import numpy as np
import jax
import jax.numpy as jnp
import optax
import datetime
import traceback

# Add current directory to path
sys.path.append(os.getcwd())

from main_shell_verification import (
    Lx, Ly, Nx_low, Ny_low, Nx_high, Ny_high, 
    EquivalentSheetModel, TwistCase, PureBendingCase, CornerLiftCase, TwoCornerLiftCase
)

# Configuration sync from main_shell_verification
target_config = {
    'pattern': 'ABC', 'base_t': 1.0, 
    'pattern_pz': 'TNYBV', 'bead_pz': {'T': 12.0, 'N': 10.0, 'Y': 15.0, 'B': 12.0, 'V': 4.0},
    'base_rho': 7.85e-9, 'base_E': 210000.0,
}

weights = {
    'static': 1.0, 'reaction': 1.0, 'freq': 1.0, 'mode': 1.0,
    'strain_energy': 1.0, 'mass': 1.0, 'reg': 0.001
}

def generate_premium_html(md_file, html_file, rank, loss):
    if not os.path.exists(md_file): return
    with open(md_file, "r", encoding="utf-8") as f: content = f.read()
    html_body = ""
    lines = content.split("\n")
    in_table = False
    for line in lines:
        if "|" in line and "---" not in line:
            if not in_table:
                html_body += "<table border='1' style='width:100%; border-collapse:collapse;'>"; in_table = True
            cells = [c.strip() for c in line.split("|") if c.strip()]
            html_body += "<tr>" + "".join([f"<td>{c}</td>" for c in cells]) + "</tr>"
        else:
            if in_table: html_body += "</table>"; in_table = False
            if line.startswith("#"): html_body += f"<h3>{line.replace('#','').strip()}</h3>"
            elif line.strip(): html_body += f"<p>{line}</p>"
    html_template = f"<html><body style='font-family:sans-serif;background:#f5f5f5;padding:40px;'><div style='background:white;padding:30px;border-radius:10px;'><h1>Rank {rank} Results</h1><p><b>Total Loss:</b> {loss:.6e}</p>{html_body}<div style='display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-top:20px;'><img src='verify_3d_mode_shapes.png' style='width:100%'><img src='verify_3d_parameters.png' style='width:100%'></div></div></body></html>"
    with open(html_file, "w", encoding="utf-8") as f: f.write(html_template)

def main():
    base_dir = "opt_result"
    if os.path.exists(base_dir): shutil.rmtree(base_dir)
    os.makedirs(base_dir)

    model = EquivalentSheetModel(Lx, Ly, Nx_low, Ny_low)
    for c in [TwistCase("twist_x",'x',1.5), TwistCase("twist_y",'y',1.5), PureBendingCase("bend_y",'y',3.0),
              CornerLiftCase("lift_br",'br',5.0), TwoCornerLiftCase("lift_tl_br",['br','tl'],5.0)]: model.add_case(c)
    
    os.environ["NON_INTERACTIVE"] = "1"
    model.generate_targets(resolution_high=(Nx_high, Ny_high), num_modes_save=5, target_config=target_config, cache_file="ground_truth_cache.pkl")
    
    # 3 Ranks, 2 steps each. Fast but provides folder structure.
    snapshots = []
    current_config = {
        't':   {'opt': True, 'init': 1.0, 'min': 0.1, 'max': 10.0},
        'rho': {'opt': False, 'init': 7.85e-9, 'min': 7.85e-10, 'max': 7.85e-8},
        'E':   {'opt': False, 'init': 210000.0, 'min': 110000, 'max': 300000.0},
        'pz':   {'opt': True, 'init': 0.0, 'min': -20.0, 'max': 20.0},
    }

    for i in range(1, 4):
        print(f"\n[SNAP {i}/3] Optimizing...")
        best_p = model.optimize(current_config, weights, max_iterations=2, init_pz_from_gt=(i==1))
        snapshots.append({'rank': i, 'loss': float(model.history[-1]['Total']), 'params': {k: v.copy() for k, v in best_p.items()}})
        for k in ['t', 'pz']: current_config[k]['init'] = best_p[k]

    snapshots.sort(key=lambda x: x['loss'])
    for i, snap in enumerate(snapshots, 1):
        folder = os.path.join(base_dir, f"c{i}")
        os.makedirs(folder, exist_ok=True)
        model.optimized_params = snap['params']; model.verify()
        for f in glob.glob("verify_3d_*.png") + ["verification_report.md"]:
            if os.path.exists(f): shutil.copy(f, os.path.join(folder, f))
        generate_premium_html(os.path.join(folder, "verification_report.md"), os.path.join(folder, "verification_report.html"), i, snap['loss'])
    
    pd.DataFrame(snapshots).drop(columns=['params']).to_csv(os.path.join(base_dir, "summary.csv"), index=False)
    print("Done.")

if __name__ == "__main__": main()
