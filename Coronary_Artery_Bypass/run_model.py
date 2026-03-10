#Main run script for the Coronary Artery Bypass Model
#It loads the model, runs the solver, and writes the results to a VTK file for visualization
#It also prints out some summary information about the solution.
from pathlib import Path
from assets.compare_grafts import compare_graft_options
from assets.model_solver import run_case

#Function to print comparison of graft options
def print_graft_comparisons():
    baseline_flow, comparisons = compare_graft_options()

    print("\nGraft ranking by restored outlet flow:")
    print(f"Baseline (occluded) total outlet flow: {baseline_flow:.3e}\n")
    for rank, comparison in enumerate(comparisons, start=1):
        start, end, radius = comparison.graft
        display_start = start+1 #Convert to 1-based indexing for display
        display_end = end+1
        print(
            f"{rank}. Graft_index={comparison.graft_index+1} "
            f"nodes=({display_start} to {display_end}) radius={radius:.3f}\n "
            f"Total_outlet_flow={comparison.total_outlet_flow:.3e}\n "
            f"Restored_outlet_flow={comparison.restored_outlet_flow:.3e} "
        )
        for tree_id, restored_flow in comparison.restored_tree_flows.items():
            print(f"  Tree {tree_id}: restored_flow={restored_flow:.3e}")
        print("\n")

def main():
    default_output = Path(__file__).resolve().parent / "results" / "coronary_solution.vtk"
    network_data, solved_data, vtk_path = run_case(output_vtk=default_output)
    total_outlet_flow = sum(solved_data.outlet_boundary_flows)

    print(f"\nSolved {len(solved_data.vtk_cells)} branches on {network_data.n_points} nodes, VTK written to {vtk_path}\n")
    print(f"Total outlet flow: {total_outlet_flow:.3e} (mmHg*cm^3/s)\n")
    print_graft_comparisons()


#Run the main function when this script is executed directly
if __name__ == "__main__":
    main()
