#Script reads labels and creates a VTK file for visualizaiton in Paraview
import numpy as np

def dumpVTK(filename, npoin, nelem, xyz, ele, clr):
    with open(filename,'w') as fp:
        fp.write('# vtk DataFile Version 2.0\n')
        fp.write('vtk output\n')
        fp.write('ASCII\nDATASET UNSTRUCTURED_GRID\n')
        fp.write('POINTS %d float\n' % npoin)
        for point in xyz:
            fp.write('%f %f %f\n' % tuple(point))
        fp.write('CELLS %d %d\n' % (nelem, 4*nelem)) #4 = 1 (numpoin) + 3 (nodes)
        for cell in ele:
            fp.write('3 %d %d %d\n' % tuple(cell - 1)) # -1 for Fortran to C indexing
        fp.write('CELL_TYPES %d\n' % nelem)
        for ele in range(nelem):
            fp.write('5\n') # VTK_TRIANGLE
        fp.write('CELL_DATA %d\nSCALARS cell-type float\nLOOKUP_TABLE default\n' % nelem)
        for c in clr:
            fp.write('%d\n' % clr)
    return 0