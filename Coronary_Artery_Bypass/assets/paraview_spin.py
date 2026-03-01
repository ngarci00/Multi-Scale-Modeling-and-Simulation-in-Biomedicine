"""Paraview helper: load a VTK file, render vessels as tubes, and save spin frames.

Run with:
    pvpython assets/paraview_spin.py

Adjust `INPUT_VTK` below if you want a different dataset.
"""
from pathlib import Path

try:
    from paraview.simple import (
        ColorBy,
        ExtractSurface,
        GetActiveViewOrCreate,
        GetActiveCamera,
        GetColorTransferFunction,
        HideScalarBarIfNotNeeded,
        LegacyVTKReader,
        Render,
        ResetCamera,
        SaveScreenshot,
        SetActiveSource,
        Show,
        Tube,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - depends on local ParaView install
    raise SystemExit(
        "paraview.simple is not available in this Python interpreter.\n"
        "Run this script with ParaView's bundled pvpython instead, for example:\n"
      "/Applications/ParaView-5.13.3.app/Contents/bin/pvpython Coronary_Artery_Bypass/assets/paraview_spin.py"

    ) from exc


PROJECT_DIR = Path(__file__).resolve().parent.parent
INPUT_VTK = PROJECT_DIR / "results" / "coronary_solution.vtk"
OUTPUT_FRAMES_DIR = PROJECT_DIR / "results" / "coronary_solution_spin_frames"
COLOR_SCALAR = "flow"
TUBE_SIDES = 18
N_FRAMES = 72


def main() -> None:
    source = LegacyVTKReader(FileNames=[str(INPUT_VTK)])
    view = GetActiveViewOrCreate("RenderView")
    display = Show(source, view)
    display.SetRepresentationType("Wireframe")

    geometry = ExtractSurface(Input=source)
    tube = Tube(Input=geometry)
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
    view.EnableRayTracing = 0
    ResetCamera(view)
    view.InteractionMode = "3D"
    camera = GetActiveCamera()
    camera.Azimuth(45)
    Render(view)

    OUTPUT_FRAMES_DIR.mkdir(parents=True, exist_ok=True)
    angle_step = 360.0 / N_FRAMES

    for frame_index in range(N_FRAMES):
        SaveScreenshot(
            str(OUTPUT_FRAMES_DIR / f"frame_{frame_index:03d}.png"),
            view,
            ImageResolution=view.ViewSize,
        )
        camera.Azimuth(angle_step)
        Render(view)


if __name__ == "__main__":
    main()
