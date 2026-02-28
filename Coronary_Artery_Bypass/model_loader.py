#Script to load the arterial network data from CSV files and represent it as a structured data class:
from __future__ import annotations #this helps with type annotations for forward references, meaning:
#we can use file names and types that are defined later in the code without freaking out the interpreter.
#kinda like saying "Hey, I know this thing is gonna be defined later, just trust me on this one."
import csv, os
from dataclasses import dataclass
#dataclasses is a lib that allows us to create classes that are used to store data without having to write,
#a bounch of code to init the class, so basically automates __init__ and other mdethods.
from pathlib import Path 
#This in adjection to os.path provides a way to handle file paths so there is no compatibility issues
from typing import Iterable
#typing: provides support for type hints, which are a way to indicate the expected types of variables;
#for example, a function that expects int input can be annotated with def func(x:int) -> None: 
#to indicate that x should be an integer and the functions returns nothing!

@dataclass
class ArterialNetwork:
    points: list[tuple[float, float, float]] #coronary.ptxyz.csv
    branches: list[tuple[int, int]] #coronary.elems.csv
    branch_radii: list[float] #coronary.elrad.csv
    branch_tree: list[int] #coronary.eltre.csv
    inlet_points: list[int] #coronary.bcinl.csv
    outlet_points: list[int] #coronary.bcout.csv
    occluded_branches: list[int] #coronary.occlu.csv
    graft_options: list[tuple[int, int, float]] #coronary.graft.csv (start point, end point, radius)
    n_points: int #number of points in the network
    n_branches: int #number of branches in the network
    data_path: Path #directory where the data files are loaded from

#Loading the data from the csv file, and converting into appropriate data structures. 
def data_dir(data_path: str | os.PathLike | None = None) -> Path:
    if data_path is None:
        return (Path(__file__).resolve().parent / "data").resolve()
    return Path(os.fspath(data_path)).expanduser().resolve()

#Reading the rows from the csv file & stripping whitespaces.
def read_rows(path: Path) -> list[list[str]]:
   
    rows: list[list[str]] = []
    
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        for row_number, row in enumerate(reader, start=1):
            if not row:
                raise ValueError(f"{path.name}: empty row at line {row_number}")
            if any(cell.strip() == "" for cell in row):
                raise ValueError(f"{path.name}: blank value at line {row_number}")
            rows.append([cell.strip() for cell in row])
    return rows

#Loading the data from the csv file, and converting into appropriate data structures!
def load_float_matrix(path: Path, expected_cols: int) -> list[tuple[float, ...]]:
    rows = read_rows(path)
    matrix: list[tuple[float, ...]] = []
    for row_number, row in enumerate(rows, start=1):
        if len(row) != expected_cols:
            raise ValueError(
                f"{path.name}: Expected {expected_cols} columns, found {len(row)} at line {row_number}"
            )
        try:
            matrix.append(tuple(float(value) for value in row))
        except ValueError as exc:
            raise ValueError(f"{path.name}: non-numeric value at line {row_number}") from exc
    return matrix

def load_vector(path: Path) -> list[int]:
    rows = read_rows(path)
    values: list[int] = []
    for row_number, row in enumerate(rows, start=1):
        if len(row) != 1:
            raise ValueError(f"{path.name}: expected 1 column, found {len(row)} at line {row_number}")
        try:
            value = int(row[0])
        except ValueError as exc:
            raise ValueError(f"{path.name}: non-integer value at line {row_number}") from exc
        values.append(value)
    return values

#Loading the data from the csv file, and converting into appropriate data structures (:
#expects a matrix of integers, and checks # of col and data types
def load_matrix_int(path: Path, expected_cols: int) -> list[tuple[int, ...]]:
    rows = read_rows(path)
    matrix: list[tuple[int, ...]] = []
    for row_number, row in enumerate(rows, start=1):
        if len(row) != expected_cols:
            raise ValueError(
                f"{path.name}: expected {expected_cols} columns, found {len(row)} at line {row_number}"
            )
        try:
            matrix.append(tuple(int(value) for value in row))
        except ValueError as exc:
            raise ValueError(f"{path.name}: non-integer value at line {row_number}") from exc
    return matrix

#Converts 1-based indices from the CSV file (bc MATLAB uses 1-based indx) to 0-based indices in Python
def _to_zero_based(
    values: Iterable[int], *, name: str, upper_bound: int | None = None
) -> list[int]:
    normalized: list[int] = []
    for index, value in enumerate(values, start=1):
        if upper_bound is not None and not 1 <= value <= upper_bound:
            raise ValueError(
                f"{name}: value {value} at position {index} is outside the valid range 1..{upper_bound}"
            )
        normalized.append(value - 1)
    return normalized

#Main function to load the arterial network data from the csv files and represent it as a structured data class
def load_arterial_network(data_path: str | os.PathLike | None = None) -> ArterialNetwork:
    resolved_data_dir = data_dir(data_path)
    #file paths and names 
    points_path = resolved_data_dir / "coronary.ptxyz.csv"
    branches_path = resolved_data_dir / "coronary.elems.csv"
    radii_path = resolved_data_dir / "coronary.elrad.csv"
    tree_path = resolved_data_dir / "coronary.eltre.csv"
    inlet_path = resolved_data_dir / "coronary.bcinl.csv"
    outlet_path = resolved_data_dir / "coronary.bcout.csv"
    occlusion_path = resolved_data_dir / "coronary.occlu.csv"
    graft_path = resolved_data_dir / "coronary.graft.csv"

    raw_points = load_float_matrix(points_path, expected_cols=3) #coronary.ptxyz.csv: containing the x,y,z coordinates of each point
    raw_branches = load_matrix_int(branches_path, expected_cols=2) #cornary.elems.csv: define the connectivity of the arterial network
    raw_radii_rows = load_float_matrix(radii_path, expected_cols=1) #coronary.elrad.csv: contains the raidi of each branchin the network
    raw_tree = load_vector(tree_path) #coronary.eltre.csv: defines the hirarchy of the branches in the network
    raw_inlets = load_vector(inlet_path) #coronary.bincl.csv: lists the points that serve as the inlet (blood in)
    raw_outlets = load_vector(outlet_path) #coronary.bcout.csv: lists the points that serve as the outlet (blood out)
    raw_occlusions = load_vector(occlusion_path) #coronary.occlu.csv: identifies which branches are occluded (blocked)
    raw_grafts = load_float_matrix(graft_path, expected_cols=3) #coronary.graft.csv: lists potential grafting options, row: start point, column: end point, radius of the graft

    n_points = len(raw_points)
    n_branches = len(raw_branches)
    branch_radii = [row[0] for row in raw_radii_rows] #radius values: first column of the elrad matrix

    if len(branch_radii) != n_branches:
        raise ValueError(
            f"{radii_path.name}: branch radii count {len(branch_radii)} does not match branch count {n_branches}"
        )
    if len(raw_tree) != n_branches:
        raise ValueError(
            f"{tree_path.name}: branch tree count {len(raw_tree)} does not match branch count {n_branches}"
        )

    branches: list[tuple[int, int]] = [] #branch connectivity: pairs of point indices that define each branch
    for row_number, (start, end) in enumerate(raw_branches, start=1): #iterating through the rows of the branch connectivity 
        zero_based_pair = _to_zero_based(
            [start, end],
            name=f"{branches_path.name}: branch endpoints at row {row_number}",
            upper_bound=n_points,
        )
        branches.append((zero_based_pair[0], zero_based_pair[1]))
    #Converting the 1-based indices from the CSV files to 0-based indices for Python
    inlet_points = _to_zero_based(raw_inlets, name=inlet_path.name, upper_bound=n_points)
    outlet_points = _to_zero_based(raw_outlets, name=outlet_path.name, upper_bound=n_points)
    occluded_branches = _to_zero_based(
        raw_occlusions, name=occlusion_path.name, upper_bound=n_branches
    )
    #Graft Options: list of tuples containing the start point indx, and end point indx, 
    #side note: a tuple is an immutable sequence of values, example: (1,2,3) is a tuple of three integers.
    graft_options: list[tuple[int, int, float]] = []
    for row_number, row in enumerate(raw_grafts, start=1):
        start = int(row[0])
        end = int(row[1])
        radius = row[2]
        zero_based_pair = _to_zero_based(
            [start, end],
            name=f"{graft_path.name}: graft nodes at row {row_number}",
            upper_bound=n_points,
        )
        graft_options.append((zero_based_pair[0], zero_based_pair[1], radius))

    points = [(x, y, z) for x, y, z in raw_points]
    branch_tree = list(raw_tree) #branch hierarchy: list of integers where each value indicates the parent branch index (or -1 for root branches)

    return ArterialNetwork(
        points=points,
        branches=branches,
        branch_radii=branch_radii,
        branch_tree=branch_tree,
        inlet_points=inlet_points,
        outlet_points=outlet_points,
        occluded_branches=occluded_branches,
        graft_options=graft_options,
        n_points=n_points,
        n_branches=n_branches,
        data_path=resolved_data_dir,
    )
#Tests the loading function by running the script directly. 
if __name__ == "__main__":
    network = load_arterial_network()
    print(
        f"\nLoaded arterial network from {network.data_path} "
        f"with {network.n_points} points and {network.n_branches} branches.\n"
    )
