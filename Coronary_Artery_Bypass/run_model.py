#Main run script for the Coronary Artery Bypass Model
#It loads the model, runs the solver, and writes the results to a VTK file for visualization
#It also prints out some summary information about the solution.
from __future__ import annotations
from pathlib import Path
from model_solver import run_case
def main() -> None:
    default_output = Path(__file__).resolve().parent / "results" / "coronary_solution.vtk"
    network_data, solved_data, vtk_path = run_case(output_vtk=default_output)
    
    total_outlet_flow = 0.0
    outlet_set = set(network_data.outlet_points)
    for (start, end), flow in zip(solved_data.vtk_cells, solved_data.branch_flows):
        if end in outlet_set:
            total_outlet_flow += flow
        elif start in outlet_set:
            total_outlet_flow -= flow

    print(f"\nSolved {len(solved_data.vtk_cells)} branches on {network_data.n_points} nodes, VTK written to {vtk_path}\n.")
    print(f"Total outlet flow: {total_outlet_flow:.3e}\n (mmHg*cm^3/s)")

if __name__ == "__main__":
    main()
