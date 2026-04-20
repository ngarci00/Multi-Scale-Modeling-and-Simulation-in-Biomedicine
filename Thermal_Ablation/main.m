%Project 4 - Thermal Ablation model based on bioheat equation / Finite Volume Method
%By Nicolas Garcia Callejas
clear all; close all; clc;

%% Directory setup
projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot, 'assets'));

%% Mesh selection
%Selecting the mesh to load, making it easy for comparison
needleId = 1;        % options: 1, 2, 3, 4
meshKind = 'sour';   % options: 'unif' (uniform) or 'sour' (refined near the needle tip)

%Loading the mesh file and boundary conditions for the selected case
caseName = sprintf('needle%d', needleId);
dataDir = fullfile(projectRoot, 'data', caseName);
elemFile = fullfile(dataDir, sprintf('%s.%s.elems.csv', caseName, meshKind));
pointFile = fullfile(dataDir, sprintf('%s.%s.point.csv', caseName, meshKind));
bcFile = fullfile(dataDir, sprintf('%s.%s.esuelbc.csv', caseName, meshKind));

ele = csvread(elemFile);      %element connectivity: [node1 node2 node3]
xyz = csvread(pointFile);     %point coordinates: [x y z]
esuelbc = csvread(bcFile);    %neighboring elements plus boundary codes

%Number of elements and points
nelem = size(ele, 1);
npoin = size(xyz, 1);

%Printing mesh info to the console
fprintf('Loaded %s %s mesh: %d points, %d triangular elements.\n', caseName, meshKind, npoin, nelem);
%% Parameters:
Tbody = 37; %Body temperature (C)
Ttip = 100; %Needle tip temperature (C)
Tdead = 50; %Cell death threshold temperature (C)

% TODO: choose/confirm material parameters for the bioheat equation:
% rho, cp, k, metabolic heat source, blood-perfusion sink coefficient.

%% Cell geometry for a triangular finite-volume method
area = zeros(nelem, 1); %area of each triangular cell
centroid = zeros(nelem, 3); %centroid calculations for each cell
edgeLength = zeros(nelem, 3); %the length of each edge of the triangle

%For loop: which loops through each element, calculates the area, centroid, and edge lengths
%based on the coordinates of the nodes that make up each element. 
for e = 1:nelem
    nodes = ele(e, :);
    x1 = xyz(nodes(1), :); %first node of the triangle
    x2 = xyz(nodes(2), :); %second node of the triangle
    x3 = xyz(nodes(3), :); %third node of the triangle

    area(e) = 0.5 * norm(cross(x2 - x1, x3 - x1)); %area of triangle using cross product
    centroid(e, :) = (x1 + x2 + x3) / 3; %centroid is the average of the three vertices

    %edge lengths are calculated using the norm of the difference between the vertices
    edgeLength(e, 1) = norm(x2 - x3); %edge opposite to x1
    edgeLength(e, 2) = norm(x3 - x1); %edge opposite to x2
    edgeLength(e, 3) = norm(x1 - x2); %edge opposite to x3
end

fprintf('Area check: min %.3e, max %.3e, total %.3e. \n', min(area), max(area), sum(area));
%% Boundary labels
%esuelbc entries:
%-1 = needle tip, apply constant hot-temperature boundary condition
%-2 = outer box, apply adiabatic/no-flux boundary condition
%-3 = needle body, apply adiabatic/no-flux boundary condition
isTip = any(esuelbc == -1, 2);
isOuterBox = any(esuelbc == -2, 2);
isNeedleBody = any(esuelbc == -3, 2);

%creating a boundary code vector for visualization and later use in the solver, where:
% 0 = interior cell, 1 = outer box, 2 = needle body, 3 = needle tip
boundaryCode = zeros(nelem, 1); %initializing all cells as interior (0)
boundaryCode(isOuterBox) = 1; %outer box is the default boundary label for any cell that has an outer box face
boundaryCode(isNeedleBody) = 2; %needle body overrides outer box if they overlap
boundaryCode(isTip) = 3; %tip overrides other labels if they overlap

fprintf('Boundary elements: tip %d, outer box %d, needle body %d. \n', nnz(isTip), nnz(isOuterBox), nnz(isNeedleBody));
%% Visualize the mesh and boundary labels in MATLAB
figure('Name', sprintf('%s %s boundary check', caseName, meshKind));
triplot(ele, xyz(:, 1), xyz(:, 2), 'Color', [0.72 0.72 0.72]); %plotting the mesh in light gray hence the color [0.72 0.72 0.72]
axis equal tight;
hold on;
scatter(centroid(isOuterBox, 1), centroid(isOuterBox, 2), 8, [0.1 0.35 0.8], 'filled'); %outer box in blue
scatter(centroid(isNeedleBody, 1), centroid(isNeedleBody, 2), 8, [0.3 0.3 0.3], 'filled'); %needle body in gray
scatter(centroid(isTip, 1), centroid(isTip, 2), 12, [0.85 0.1 0.1], 'filled'); %needle tip in red
legend({'mesh', 'outer box', 'needle body', 'needle tip'}, 'Location', 'bestoutside');
title(sprintf('%s %s mesh boundary labels', caseName, meshKind));
xlabel('x');
ylabel('y');
%% Export results as VTK using dumpVTK for visualization in ParaView
outDir = fullfile(projectRoot, 'results');
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

vtkName = fullfile(outDir, sprintf('%s_%s_boundary_check.vtk', caseName, meshKind));
dumpVTK(vtkName, npoin, nelem, xyz, ele, boundaryCode);
fprintf('Wrote boundary-label VTK: %s\n', vtkName);
%% Next steps:
% Implement one explicit Forward Euler step for T_cell:
% 1. Start with T = Tbody * ones(nelem,1).
% 2. For each cell face, use esuelbc(e,i) to decide whether the face is an
%    interior neighbor, the hot tip, or a no-flux boundary.
% 3. Add conduction fluxes through each face using edgeLength(e,i).
% 4. Add metabolic source and blood-perfusion cooling terms.
% 5. Update T with dt and repeat in a time loop.
