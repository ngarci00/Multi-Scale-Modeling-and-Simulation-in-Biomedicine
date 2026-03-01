#Script that helps compare the different graft options by running the solver for each option and calculating
#the total outlet flow for each of these cases.
from __future__ import annotations
from dataclasses import dataclass
from model_loader import load_arterial_network
from model_solver import SolverConfig, solve_network


@dataclass
class GraftComparison:
    graft_index: int
    graft: tuple[int, int, float]
    total_outlet_flow: float
    restored_outlet_flow: float

#Helper function to calculate the total outlet flow from the solution.
def total_outlet_flow(network, solution) -> float:
    total_flow = 0.0
    outlet_set = set(network.outlet_points)
    for (start, end), flow in zip(solution.vtk_cells, solution.branch_flows):
        if end in outlet_set:
            total_flow += flow
        elif start in outlet_set:
            total_flow -= flow
    return total_flow

#Function to compare all grafts options by the total outlet flow they restore compared to the baseline.
def compare_graft_options() -> tuple[float, list[GraftComparison]]:
    network = load_arterial_network()
    baseline_solution = solve_network(network=network, config=SolverConfig())
    baseline_flow = total_outlet_flow(network, baseline_solution)

    comparisons: list[GraftComparison] = []
    for graft_index, graft in enumerate(network.graft_options):
        solution = solve_network(
            network=network,
            config=SolverConfig(graft_index=graft_index),
        )
        total_flow = total_outlet_flow(network, solution)
        comparisons.append(
            GraftComparison(
                graft_index=graft_index,
                graft=graft,
                total_outlet_flow=total_flow,
                restored_outlet_flow=total_flow - baseline_flow,
            )
        )

    comparisons.sort(key=lambda item: item.restored_outlet_flow, reverse=True)
    return baseline_flow, comparisons

#Main entry point for comparison script, it also runs the comparison and prints out the results:
def main() -> None:
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
            f"nodes=({display_start} to {display_end}) radius={radius:.3f} "
            f"Total_outlet_flow={comparison.total_outlet_flow:.3e} "
            f"Restored_outlet_flow={comparison.restored_outlet_flow:.3e}"
        )

if __name__ == "__main__":
    main()
