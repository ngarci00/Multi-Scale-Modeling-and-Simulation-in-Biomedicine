#Script reads labels and creates a VTK file for visualization in Paraview.
import os
def dumpVTK(filename, npoin, nelem, xyz, ele, clr, *, vtk_cell_type=None, zero_based=True, scalar_name='cell-type'):
    filepath = os.path.abspath(os.path.expanduser(os.fspath(filename)))#Ensuring an abs path for the file
    directory = os.path.dirname(filepath)
    if directory:
        os.makedirs(directory, exist_ok=True)#Makes the directory if it doesn't exist, but does nothing if it does exist

    cells = [tuple(int(node) for node in cell) for cell in ele]
    if nelem != len(cells):
        raise ValueError(f'Expected {nelem} elements, received {len(cells)}.')
    if not cells:
        raise ValueError('At least one element is required to write a VTK file.')

    cell_size = len(cells[0])
    if any(len(cell) != cell_size for cell in cells):
        raise ValueError('All elements must have the same number of nodes.')

    if vtk_cell_type is None:
        if cell_size == 2:
            vtk_cell_type = 3 #VTK_LINE
        elif cell_size == 3:
            vtk_cell_type = 5 #VTK_TRIANGLE
        else:
            raise ValueError(f'Unsupported cell size: {cell_size}.')

    cell_record_size = nelem * (cell_size + 1)

    with open(filepath, 'w') as fp:
        fp.write('# vtk DataFile Version 2.0\n')
        fp.write('vtk output\n')
        fp.write('ASCII\nDATASET UNSTRUCTURED_GRID\n')
        fp.write('POINTS %d float\n' % npoin)
        for point in xyz:
            fp.write('%f %f %f\n' % tuple(point))
        fp.write('CELLS %d %d\n' % (nelem, cell_record_size))
        for cell in cells:
            nodes = cell if zero_based else tuple(node - 1 for node in cell)
            fp.write('%d %s\n' % (cell_size, ' '.join(str(node) for node in nodes)))
        fp.write('CELL_TYPES %d\n' % nelem)
        for ele in range(nelem):
            fp.write('%d\n' % vtk_cell_type)
        fp.write('CELL_DATA %d\nSCALARS %s float\nLOOKUP_TABLE default\n' % (nelem, scalar_name))
        for c in clr:
            fp.write('%.12e\n' % float(c))
    return 0
