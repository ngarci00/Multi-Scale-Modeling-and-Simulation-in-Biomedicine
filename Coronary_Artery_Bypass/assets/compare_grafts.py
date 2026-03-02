#Script that helps compare the different graft options by running the solver for each option and calculating
#the total outlet flow for each of these cases.
from dataclasses import dataclass
#dataclasses is a convienient library for defining simple classes that are primarily used to store data.
from pathlib import Path

#This keeps the file easy to run directly from assets/ while still working as a project import.
from .model_loader import load_arterial_network
from .model_solver import SolverConfig, save_solution_vtk, solve_network

@dataclass
class GraftComparison:
    graft_index: int
    graft: tuple
    total_outlet_flow: float
    restored_outlet_flow: float
    vtk_path: Path

#Helper function to calculate the total outlet flow from the solution.
def total_outlet_flow(network, solution):
    total_flow = 0.0
    outlet_set = set(network.outlet_points)
    for (start, end), flow in zip(solution.vtk_cells, solution.branch_flows):
        if end in outlet_set:
            total_flow += flow #if total flow is positive, flow is going out of the otulet, if negative flow is going into the outlet.
        elif start in outlet_set:
            total_flow -= flow 
    return total_flow

#Function to compare all grafts options by the total outlet flow they restore compared to the baseline.
def compare_graft_options():
    network = load_arterial_network()
    baseline_solution = solve_network(network=network, config=SolverConfig())
    baseline_flow = total_outlet_flow(network, baseline_solution)
    output_dir = Path(__file__).resolve().parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    comparisons = []
    for graft_index, graft in enumerate(network.graft_options):
        solution = solve_network(
            network=network,
            config=SolverConfig(graft_index=graft_index),
        )
        total_flow = total_outlet_flow(network, solution)
        vtk_path = output_dir / f"graft_option_{graft_index}.vtk"
        save_solution_vtk(network, solution, vtk_path)
        comparisons.append(
            GraftComparison(
                graft_index=graft_index,
                graft=graft,
                total_outlet_flow=total_flow,
                restored_outlet_flow=total_flow - baseline_flow,
                vtk_path=vtk_path,
            )
        )
#Sort the comparisons by the restored outlet flow in descending order (best graft first)
    comparisons.sort(key=lambda item: item.restored_outlet_flow, reverse=True) #reverse=True sorts in descending order
    return baseline_flow, comparisons

#Main entry point for comparison script, it also runs the comparison and prints out the results:
def main():
    baseline_flow, comparisons = compare_graft_options()

    print("\nOccluded baseline (no graft)")
    print(f"Total outlet flow: {baseline_flow:.3e}\n")

    print("Graft ranking by restored outlet flow:")
    for rank, comparison in enumerate(comparisons, start=1):
        start, end, radius = comparison.graft
        display_start = start + 1 #Converting back to 1-based indexing for display purposes
        display_end = end + 1
        print(
            f"{rank}. Graft_index={comparison.graft_index} "
            f"nodes=({display_start} to {display_end}) radius={radius:.3f}\n "
            f"Total_outlet_flow={comparison.total_outlet_flow:.3e}\n "
            f"Restored_outlet_flow={comparison.restored_outlet_flow:.3e}\n "
            f"VTK={comparison.vtk_path}"
        )

if __name__ == "__main__":
    main()
