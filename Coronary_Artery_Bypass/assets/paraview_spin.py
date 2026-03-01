"""Paraview helper: load a VTK file, render vessels as tubes, and save a spin animation.

Run with:
    pvpython assets/paraview_spin.py

Adjust `INPUT_VTK` below if you want a different dataset.
"""
from pathlib import Path

from paraview.simple import (
    ColorBy,
    GetActiveViewOrCreate,
    GetColorTransferFunction,
    HideScalarBarIfNotNeeded,
    LegacyVTKReader,
    Render,
    ResetCamera,
    SaveAnimation,
    SetActiveSource,
    Show,
    Tube,
)


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_VTK = PROJECT_DIR / "results" / "coronary_solution.vtk"
OUTPUT_ANIMATION = PROJECT_DIR / "results" / "coronary_solution_spin.ogv"
COLOR_SCALAR = "flow"
TUBE_SIDES = 18


def main() -> None:
    source = LegacyVTKReader(FileNames=[str(INPUT_VTK)])
    view = GetActiveViewOrCreate("RenderView")
    display = Show(source, view)
    display.SetRepresentationType("Wireframe")

    tube = Tube(Input=source)
    tube.Scalars = ["CELLS", "radius"]
    tube.VaryRadius = "By Absolute Scalar"
    tube.NumberofSides = TUBE_SIDES
    tube_display = Show(tube, view)

    SetActiveSource(tube)
    ColorBy(tube_display, ("CELLS", COLOR_SCALAR))
    tube_display.RescaleTransferFunctionToDataRange(True, False)
    lut = GetColorTransferFunction(COLOR_SCALAR)
    tube_display.SetScalarBarVisibility(view, True)
    HideScalarBarIfNotNeeded(lut, view)

    view.ViewSize = [1440, 900]
    view.Background = [1.0, 1.0, 1.0]
    ResetCamera(view)
    view.InteractionMode = "3D"
    view.CameraAzimuth(45)
    Render(view)

    SaveAnimation(
        str(OUTPUT_ANIMATION),
        view,
        FrameRate=24,
        FrameWindow=[0, 179],
    )


if __name__ == "__main__":
    main()
