#Script to solve the flow in the coronary artery network using Poiseuille's law
#We also save the results in VTK format for visualization in Paraview!
from __future__ import annotations

import math, numpy as np, os
from dataclasses import dataclass
from pathlib import Path

from .dumpVTK import dumpVTK
from .model_loader import ArterialNetwork, load_arterial_network

#Parameter class to hold the desired config for the solver
@dataclass
class SolverConfig:
    viscosity: float = 0.04 #mmHg*s
    inlet_pressure: float = 100.0 #mmHg Mean Aortic Pressure
    outlet_pressure: float = 10.0 #mmHg Venous Pressure
    outlet_resistance: float | None = None #mmHg*s/cm^3 lumped outflow resistance at each outlet
    occlusion_radius_fraction: float = 0.01 #fraction of original radius for occluded branches (%Stensosis)
    graft_index: int | None = None #index of the graft option to use


@dataclass
class FlowSolution:
    pressures: list[float] #pressure at each node
    branch_lengths: list[float] #length of each branch
    branch_radii: list[float] #effective radius of each branch after accounting for occlusion
    branch_resistances: list[float] #resistance of each branch computed using Poiseuille's law
    branch_flows: list[float] #flow through each branch computed from pressure drop and resistance
    branch_pressure_drops: list[float] #pressure drop across each branch (inlet pressure - outlet pressure)
    branch_occluded: list[int] #binary flag indicating whether each branch is occluded (1) or not (0)
    branch_is_graft: list[int] #binary flag indicating whether each branch is a graft (1) or not (0)
    vtk_cells: list[tuple[int, int]] #list of branch connectivity (start node index, end node index) for all branches including graft if used
    outlet_boundary_flows: list[float] #net outflow at each outlet node
    tree_outlet_flows: dict[int, float] #total outlet flow aggregated by arterial tree id
    graft_used: tuple[int, int, float] | None #the graft option used (start point, end point, radius) or None if no graft was used
    config: SolverConfig #the configuration used to solve the network

#Helper function for distance calculation between two points in 3D space
def e_distance(point_a: tuple[float, float, float], point_b: tuple[float, float, float]) -> float:
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    dz = point_a[2] - point_b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)
#returns the distance between two points in 3D space using the Euclidean distance formula.

#Helper function to convert 1-based indices from csv files to 0-based indices for Python
def branch_length(network: ArterialNetwork, start: int, end: int) -> float:
    return e_distance(network.points[start], network.points[end])

#Helper function to compute the Poiseuille resistance of a branch given its l,r and mu values 
def poiseuille_resistance(length: float, radius: float, viscosity: float) -> float:
    if radius <= 0.0:
        raise ValueError(f"Branch radius must be positive. Received {radius}.")
    if length <= 0.0:
        raise ValueError(f"Branch length must be positive. Received {length}.")
    return (8.0 * viscosity * length) / (math.pi * radius**4)


def validate_config(config: SolverConfig) -> None:
    if config.viscosity <= 0.0:
        raise ValueError("Viscosity must be positive.")
    if config.occlusion_radius_fraction <= 0.0 or config.occlusion_radius_fraction > 1.0:
        raise ValueError("occlusion_radius_fraction must be in the interval (0, 1].")
    if config.outlet_resistance is not None and config.outlet_resistance <= 0.0:
        raise ValueError("outlet_resistance must be positive when provided.")

#Buidling the effective network by applying occlusion and grafting mods (if any) to the original network
#as well as computing the lengths, effective radii, and resistances of all branches in the modified network
def effective_networks(
    network: ArterialNetwork, config: SolverConfig
) -> tuple[
    list[tuple[int, int]],
    list[float],
    list[float],
    list[float],
    list[int],
    list[int],
    tuple[int, int, float] | None,
]:
    #Start with the original branches and their properties
    vtk_cells = list(network.branches)
    lengths = [branch_length(network, start, end) for start, end in vtk_cells]
    effective_radii = list(network.branch_radii)
    occluded_flags = [0] * len(vtk_cells)
    graft_flags = [0] * len(vtk_cells)

    #iterate through the occluded branches and apply the occlusion radius fraction to their effective radii,
    #and set the occluded flag to 1 for those branches
    for branch_index in network.occluded_branches:
        effective_radii[branch_index] *= config.occlusion_radius_fraction
        occluded_flags[branch_index] = 1
    #if a graft is specified in the config, we add it as a new branch to the network with its specified radius,
    #and set the graft flag to 1 for that branch, and we also compute its length and resistance based on the start
    #and end points of the graft
    graft_used: tuple[int, int, float] | None = None
    if config.graft_index is not None:
       
        graft_used = network.graft_options[config.graft_index]
        graft_start, graft_end, graft_radius = graft_used
        vtk_cells.append((graft_start, graft_end))
        lengths.append(branch_length(network, graft_start, graft_end))
        effective_radii.append(graft_radius)
        occluded_flags.append(0)
        graft_flags.append(1)

    resistances = [
        poiseuille_resistance(length, radius, config.viscosity)
        for length, radius in zip(lengths, effective_radii)
    ]
    return vtk_cells, lengths, effective_radii, resistances, occluded_flags, graft_flags, graft_used

#function that takes the org networks and solver config, and returns effective network properties
#basically applies the occlusion and grafting modifications to the original network and computes the resulting branch properties for the modified network (:
def known_pressures(network: ArterialNetwork, config: SolverConfig) -> dict[int, float]:
    known: dict[int, float] = {}
    for node in network.inlet_points: #setting known presssure at the inlet
        known[node] = config.inlet_pressure
    if config.outlet_resistance is None:
        for node in network.outlet_points: #setting known pressure at the outlet
            known[node] = config.outlet_pressure
    return known

def solve_pressures(
    network: ArterialNetwork,
    vtk_cells: list[tuple[int, int]],
    resistances: list[float],
    config: SolverConfig,
) -> list[float]:
    known = known_pressures(network, config)
    unknown_nodes = [node for node in range(network.n_points) if node not in known] #List of nodes where pressure is unknown and needs to be solved for 

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(network.n_points)]
    for branch_index, ((start, end), resistance) in enumerate(zip(vtk_cells, resistances)):
        adjacency[start].append((branch_index, resistance))
        adjacency[end].append((branch_index, resistance))

    if not unknown_nodes:
        pressures = [0.0] * network.n_points
        for node, pressure in known.items():
            pressures[node] = pressure
        return pressures

    node_to_row = {node: row for row, node in enumerate(unknown_nodes)}
    matrix = np.zeros((len(unknown_nodes), len(unknown_nodes)), dtype=float)
    rhs = np.zeros(len(unknown_nodes), dtype=float)

    #iterating through the unknown nodes and assembling the linear system based on Kirchhoff's law & Poiseuille's law
    for node in unknown_nodes:
        row = node_to_row[node]
        for branch_index, resistance in adjacency[node]:
            start, end = vtk_cells[branch_index]
            neighbor = end if start == node else start
            conductance = 1.0 / resistance #inverse of resistance, representing flow per unit pressure drop for that branch 
            matrix[row, row] += conductance
            if neighbor in node_to_row:
                matrix[row, node_to_row[neighbor]] -= conductance 
            else:
                rhs[row] += conductance * known[neighbor]

        if config.outlet_resistance is not None and node in network.outlet_points:
            outlet_conductance = 1.0 / config.outlet_resistance
            matrix[row, row] += outlet_conductance
            rhs[row] += outlet_conductance * config.outlet_pressure

 
    solved = np.linalg.solve(matrix, rhs) #solving the linear system to find the pressures at the unknown nodes
   
    pressures = [0.0] * network.n_points
    for node, pressure in known.items():
        pressures[node] = pressure
    for node, row in node_to_row.items():
        pressures[node] = float(solved[row])
    return pressures


def outlet_boundary_flows(
    network: ArterialNetwork,
    solution: FlowSolution | None,
    *,
    pressures: list[float] | None = None,
    branch_flows: list[float] | None = None,
    vtk_cells: list[tuple[int, int]] | None = None,
    config: SolverConfig | None = None,
) -> list[float]:
    if solution is not None:
        pressures = solution.pressures
        branch_flows = solution.branch_flows
        vtk_cells = solution.vtk_cells
        config = solution.config
    if pressures is None or branch_flows is None or vtk_cells is None or config is None:
        raise ValueError("Provide either a FlowSolution or explicit pressures, branch_flows, vtk_cells, and config.")

    outlet_to_index = {node: index for index, node in enumerate(network.outlet_points)}
    flows = [0.0] * len(network.outlet_points)

    if config.outlet_resistance is None:
        for (start, end), flow in zip(vtk_cells, branch_flows):
            if end in outlet_to_index:
                flows[outlet_to_index[end]] += flow
            elif start in outlet_to_index:
                flows[outlet_to_index[start]] -= flow
        return flows

    for outlet_index, node in enumerate(network.outlet_points):
        flows[outlet_index] = (pressures[node] - config.outlet_pressure) / config.outlet_resistance
    return flows


def tree_outlet_flows(
    network: ArterialNetwork,
    branch_flows: list[float],
    vtk_cells: list[tuple[int, int]],
) -> dict[int, float]:
    outlet_set = set(network.outlet_points)
    totals: dict[int, float] = {}

    for branch_index, ((start, end), flow) in enumerate(zip(vtk_cells, branch_flows)):
        if branch_index >= len(network.branch_tree):
            continue
        if end in outlet_set:
            contribution = flow
        elif start in outlet_set:
            contribution = -flow
        else:
            continue
        tree_id = network.branch_tree[branch_index]
        totals[tree_id] = totals.get(tree_id, 0.0) + contribution

    return totals

#Main function to solve the flow in the coronary artery network given an arterial network and solver config
#returns a FlowSolution containing the pressures, flows, and properties of all branches
def solve_network(
    network: ArterialNetwork | None = None,
    config: SolverConfig | None = None,
) -> FlowSolution:
    if network is None:
        network = load_arterial_network()
    if config is None:
        config = SolverConfig()
    validate_config(config)

    (
        vtk_cells,
        branch_lengths,
        branch_radii,
        branch_resistances,
        branch_occluded,
        branch_is_graft,
        graft_used,
    ) = effective_networks(network, config)

    pressures = solve_pressures(network, vtk_cells, branch_resistances, config)
    branch_flows: list[float] = []
    branch_pressure_drops: list[float] = []

    #Iterating through the branches in the effective network and computing the flow and pressure drop
    for (start, end), resistance in zip(vtk_cells, branch_resistances):
        pressure_drop = pressures[start] - pressures[end]
        branch_pressure_drops.append(pressure_drop)
        branch_flows.append(pressure_drop / resistance)

    outlet_flows = outlet_boundary_flows(
        network,
        None,
        pressures=pressures,
        branch_flows=branch_flows,
        vtk_cells=vtk_cells,
        config=config,
    )
    per_tree_outlet_flows = tree_outlet_flows(network, branch_flows, vtk_cells)

    return FlowSolution(
        pressures=pressures,
        branch_lengths=branch_lengths,
        branch_radii=branch_radii,
        branch_resistances=branch_resistances,
        branch_flows=branch_flows,
        branch_pressure_drops=branch_pressure_drops,
        branch_occluded=branch_occluded,
        branch_is_graft=branch_is_graft,
        vtk_cells=vtk_cells,
        outlet_boundary_flows=outlet_flows,
        tree_outlet_flows=per_tree_outlet_flows,
        graft_used=graft_used,
        config=config,
    )

#Helper function to write the solution
def save_solution_vtk(
    network: ArterialNetwork,
    solution: FlowSolution,
    filename: str | os.PathLike,
) -> Path:
    filepath = Path(os.fspath(filename)).expanduser().resolve()
    dumpVTK(filepath, network.n_points, len(solution.vtk_cells), network.points, solution.vtk_cells, solution.branch_flows, vtk_cell_type=3, zero_based=True, scalar_name="flow",
    )

    with filepath.open("a", newline="") as handle:
        write_cell_scalar(handle, "pressure_drop", solution.branch_pressure_drops)
        write_cell_scalar(handle, "resistance", solution.branch_resistances)
        write_cell_scalar(handle, "radius", solution.branch_radii)
        write_cell_scalar(handle, "length", solution.branch_lengths)
        write_cell_scalar(handle, "is_occluded", solution.branch_occluded, integer=True)
        write_cell_scalar(handle, "is_graft", solution.branch_is_graft, integer=True)
        handle.write(f"POINT_DATA {network.n_points}\n")
        handle.write("SCALARS pressure float 1\n")
        handle.write("LOOKUP_TABLE default\n")
        for pressure in solution.pressures:
            handle.write(f"{pressure:.12e}\n")
        write_point_scalar(handle, "point_id", [node + 1 for node in range(network.n_points)], integer=True)
        write_point_scalar(
            handle,
            "is_inlet",
            [1 if node in set(network.inlet_points) else 0 for node in range(network.n_points)],
            integer=True,
        )
        write_point_scalar(
            handle,
            "is_outlet",
            [1 if node in set(network.outlet_points) else 0 for node in range(network.n_points)],
            integer=True,
        )
        outlet_flow_by_node = {node: 0.0 for node in range(network.n_points)}
        for outlet_node, outlet_flow in zip(network.outlet_points, solution.outlet_boundary_flows):
            outlet_flow_by_node[outlet_node] = outlet_flow
        write_point_scalar(
            handle,
            "outlet_boundary_flow",
            [outlet_flow_by_node[node] for node in range(network.n_points)],
        )

    return filepath
#Helper function to write a scalar array to the VTK file for either cell data or point data!
def write_cell_scalar(handle, name: str, values: list[float] | list[int], integer: bool = False) -> None:
    vtk_type = "int" if integer else "float"
    handle.write(f"SCALARS {name} {vtk_type} 1\n")
    handle.write("LOOKUP_TABLE default\n")
    for value in values:
        if integer:
            handle.write(f"{int(value)}\n")
        else:
            handle.write(f"{float(value):.12e}\n")


def write_point_scalar(handle, name: str, values: list[float] | list[int], integer: bool = False) -> None:
    write_cell_scalar(handle, name, values, integer=integer)

#Main function to run the entire case: load the network, solve for the flow, save the vtk solution
def run_case(
    data_dir: str | os.PathLike | None = None,
    *,
    output_vtk: str | os.PathLike | None = None,
    config: SolverConfig | None = None,
) -> tuple[ArterialNetwork, FlowSolution, Path | None]:
    network = load_arterial_network(data_dir)
    solution = solve_network(network=network, config=config)

    vtk_path: Path | None = None
    if output_vtk is not None:
        vtk_path = save_solution_vtk(network, solution, output_vtk)

    return network, solution, vtk_path
