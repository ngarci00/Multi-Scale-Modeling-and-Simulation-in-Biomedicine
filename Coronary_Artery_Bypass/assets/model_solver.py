#Script to solve the flow in the coronary artery network using Poiseuille's law
#We also save the results in VTK format for visualization in Paraview!
import os, math, pathlib, numpy as np
from dataclasses import dataclass
from pathlib import Path
#This keeps the file usable both when imported from run_model.py and when run directly inside assets/.
from .dumpVTK import dumpVTK
from .model_loader import ArterialNetwork, load_arterial_network

#Parameter class to hold the desired config for the solver
@dataclass
class SolverConfig:
    viscosity: float = 0.04 #mmHg*s
    inlet_pressure: float = 100.0 #mmHg Mean Aortic Pressure
    outlet_pressure: float = 10.0 #mmHg Venous Pressure
    outlet_resistance: float| None = None #optional outflow resistance at each outlet
    occlusion_radius_fraction: float = 0.01 #fraction of original radius for occluded branches (%Stenosis)
    graft_index: int | None = None  #index of the graft option to use

@dataclass
class FlowSolution:
    pressures: list #pressure at each node
    branch_lengths: list #length of each branch
    branch_radii: list #effective radius of each branch after accounting for occlusion
    branch_resistances: list #resistance of each branch computed using Poiseuille's law
    branch_flows: list #flow through each branch computed from pressure drop and resistance
    branch_pressure_drops: list #pressure drop across each branch
    branch_occluded: list #1 if branch is occluded, 0 otherwise
    branch_is_graft: list #1 if branch is a graft, 0 otherwise
    vtk_cells: list #branch connectivity for all branches including graft if used
    outlet_boundary_flows: list #net outflow at each outlet node
    tree_outlet_flows: dict #total outlet flow aggregated by arterial tree id
    graft_used: object #the graft option used, or None
    config: SolverConfig #the configuration used to solve the network

#Helper function for distance calculation between two points in 3D space
def e_distance(point_a, point_b):
    dx = point_a[0] - point_b[0]
    dy = point_a[1] - point_b[1]
    dz = point_a[2] - point_b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)

#Compute the length of a branch from its start and end points
def branch_length(network, start, end):
    return e_distance(network.points[start], network.points[end])

#Compute the Poiseuille resistance of a branch
def poiseuille_resistance(length, radius, viscosity):
    return (8.0 * viscosity * length) / (math.pi * radius**4)

#Build the network that will actually be solved after applying occlusion and graft changes
def effective_networks(network, config):
    vtk_cells = list(network.branches)
    lengths = [branch_length(network, start, end) for start, end in vtk_cells]
    effective_radii = list(network.branch_radii)
    occluded_flags = [0] * len(vtk_cells)
    graft_flags = [0] * len(vtk_cells)

    for branch_index in network.occluded_branches:
        effective_radii[branch_index] *= config.occlusion_radius_fraction
        occluded_flags[branch_index] = 1

    graft_used = None
    if config.graft_index is not None:
        graft_used = network.graft_options[config.graft_index]
        graft_start, graft_end, graft_radius = graft_used
        vtk_cells.append((graft_start, graft_end))
        lengths.append(branch_length(network, graft_start, graft_end))
        effective_radii.append(graft_radius)
        occluded_flags.append(0)
        graft_flags.append(1)

    resistances = []
    for length, radius in zip(lengths, effective_radii):
        resistances.append(poiseuille_resistance(length, radius, config.viscosity))

    return vtk_cells, lengths, effective_radii, resistances, occluded_flags, graft_flags, graft_used


#Set the known pressures from the chosen boundary conditions
def known_pressures(network, config):
    known = {}
    for node in network.inlet_points:
        known[node] = config.inlet_pressure
    if config.outlet_resistance is None:
        for node in network.outlet_points:
            known[node] = config.outlet_pressure
    return known


#Assemble and solve the linear system for nodal pressures
def solve_pressures(network, vtk_cells, resistances, config):
    known = known_pressures(network, config)
    unknown_nodes = [node for node in range(network.n_points) if node not in known]

    adjacency = [[] for _ in range(network.n_points)]
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

    for node in unknown_nodes:
        row = node_to_row[node]
        for branch_index, resistance in adjacency[node]:
            start, end = vtk_cells[branch_index]
            neighbor = end if start == node else start
            conductance = 1.0 / resistance
            matrix[row, row] += conductance
            if neighbor in node_to_row:
                matrix[row, node_to_row[neighbor]] -= conductance
            else:
                rhs[row] += conductance * known[neighbor]

        #If outlet resistance is used, each outlet has one extra connection to the venous pressure.
        if config.outlet_resistance is not None and node in network.outlet_points:
            outlet_conductance = 1.0 / config.outlet_resistance
            matrix[row, row] += outlet_conductance
            rhs[row] += outlet_conductance * config.outlet_pressure

    solved = np.linalg.solve(matrix, rhs)

    pressures = [0.0] * network.n_points
    for node, pressure in known.items():
        pressures[node] = pressure
    for node, row in node_to_row.items():
        pressures[node] = float(solved[row])
    return pressures


#Compute the total flow going through each outlet boundary
def outlet_boundary_flows(network, pressures, branch_flows, vtk_cells, config):
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


#Add up outlet flow separately for each arterial tree
def tree_outlet_flows(network, branch_flows, vtk_cells):
    outlet_set = set(network.outlet_points)
    totals = {}

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


#Main function to solve the network and return the full result
def solve_network(network=None, config=None):
    if network is None:
        network = load_arterial_network()
    if config is None:
        config = SolverConfig()

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
    branch_flows = []
    branch_pressure_drops = []

    for (start, end), resistance in zip(vtk_cells, branch_resistances):
        pressure_drop = pressures[start] - pressures[end]
        branch_pressure_drops.append(pressure_drop)
        branch_flows.append(pressure_drop / resistance)

    outlet_flows = outlet_boundary_flows(network, pressures, branch_flows, vtk_cells, config)
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


#Helper function to write a scalar array to the VTK file
def write_scalar(handle, name, values, integer=False):
    vtk_type = "int" if integer else "float"
    handle.write(f"SCALARS {name} {vtk_type} 1\n")
    handle.write("LOOKUP_TABLE default\n")
    for value in values:
        if integer:
            handle.write(f"{int(value)}\n")
        else:
            handle.write(f"{float(value):.12e}\n")


#Write the current solution to a VTK file for Paraview
def save_solution_vtk(network, solution, filename):
    filepath = Path(os.fspath(filename)).expanduser().resolve()
    dumpVTK(
        filepath,
        network.n_points,
        len(solution.vtk_cells),
        network.points,
        solution.vtk_cells,
        solution.branch_flows,
        vtk_cell_type=3,
        zero_based=True,
        scalar_name="flow",
    )

    inlet_set = set(network.inlet_points)
    outlet_set = set(network.outlet_points)

    with filepath.open("a", newline="") as handle:
        write_scalar(handle, "pressure_drop", solution.branch_pressure_drops)
        write_scalar(handle, "resistance", solution.branch_resistances)
        write_scalar(handle, "radius", solution.branch_radii)
        write_scalar(handle, "length", solution.branch_lengths)
        write_scalar(handle, "is_occluded", solution.branch_occluded, integer=True)
        write_scalar(handle, "is_graft", solution.branch_is_graft, integer=True)

        handle.write(f"POINT_DATA {network.n_points}\n")
        write_scalar(handle, "pressure", solution.pressures)
        write_scalar(handle, "point_id", [node + 1 for node in range(network.n_points)], integer=True)
        write_scalar(handle, "is_inlet", [1 if node in inlet_set else 0 for node in range(network.n_points)], integer=True)
        write_scalar(handle, "is_outlet", [1 if node in outlet_set else 0 for node in range(network.n_points)], integer=True)

        outlet_flow_by_node = {node: 0.0 for node in range(network.n_points)}
        for outlet_node, outlet_flow in zip(network.outlet_points, solution.outlet_boundary_flows):
            outlet_flow_by_node[outlet_node] = outlet_flow
        write_scalar(handle, "outlet_boundary_flow", [outlet_flow_by_node[node] for node in range(network.n_points)])

    return filepath


#Main function to run the entire case: load the network, solve for the flow, save the vtk solution
def run_case(data_dir=None, output_vtk=None, config=None):
    network = load_arterial_network(data_dir)
    solution = solve_network(network=network, config=config)

    vtk_path = None
    if output_vtk is not None:
        vtk_path = save_solution_vtk(network, solution, output_vtk)

    return network, solution, vtk_path
