function err = dumpVTK(name,npoin,nelem,xyz,ele,clr)
% Script reads labels and creates a VTK file for visualization in Paraview - Nicolas Garcia Callejas
    fp = fopen(name,'w');
    fprintf(fp,'# vtk DataFile Version 2.0\n');
    fprintf(fp,'test\n');
    fprintf(fp,'ASCII\nDATASET UNSTRUCTURED_GRID\n');
    fprintf(fp,'POINTS %d float\n',npoin);
    fprintf(fp,'%e %e %e\n',xyz');
    fprintf(fp,'CELLS %d %d\n',nelem,nelem*4); % 4 = 1 (num nodes) + 3 (nodes)
    fprintf(fp,'3 %d %d %d\n',(ele-1)'); % -1 for Fortran to C indexing
    fprintf(fp,'CELL_TYPES %d\n',nelem);
    fprintf(fp,'%d\n',5*ones(1,nelem));
    fprintf(fp,'CELL_DATA %d\nSCALARS cell-type float\nLOOKUP_TABLE default\n',nelem);
    fprintf(fp,'%d\n', clr');
    fclose(fp);
    err = 0;
end

