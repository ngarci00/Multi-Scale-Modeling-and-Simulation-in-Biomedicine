#Main run script for the Coronary Artery Bypass Model
#It loads the model, runs the solver, and writes the results to a VTK file for visualization
#It also prints out some summary information about the solution.
from __future__ import annotations
from pathlib import Path
from assets.model_solver import run_case

def main() -> None:
    default_output = Path(__file__).resolve().parent / "results" / "coronary_solution.vtk"
    network_data, solved_data, vtk_path = run_case(output_vtk=default_output)
    total_outlet_flow = sum(solved_data.outlet_boundary_flows)

    print(f"\nSolved {len(solved_data.vtk_cells)} branches on {network_data.n_points} nodes, VTK written to {vtk_path}\n")
    print(f"Total outlet flow: {total_outlet_flow:.3e} (mmHg*cm^3/s)\n")
    print("Per-tree outlet flow:")
    for tree_id in sorted(solved_data.tree_outlet_flows):
        print(f"Tree {tree_id}: {solved_data.tree_outlet_flows[tree_id]:.3e}")
    print()

if __name__ == "__main__":
    main()
