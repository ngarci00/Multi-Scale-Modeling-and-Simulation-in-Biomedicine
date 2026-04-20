%Project 4 - Thermal Ablation model based on bioheat equation / Finite Volume Method
%By Nicolas Garcia Callejas
clear all; close all; clc;

%% Directory setup
projectRoot = fileparts(mfilename('fullpath'));
addpath(fullfile(projectRoot, 'assets'));

%% Mesh selection
%Selecting the mesh to load, making it easy for comparison
needleId = 1;        % options: 1, 2 (missing unif), 3, 4 
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
%% Parameters (CGS) units, length = cm, mass = g, time = s, heat = cal, temperature = C.:
Tbody = 37; %Body temperature (C)
Ttip = 100; %Needle tip temperature (C)
Tdead = 50; %Cell death threshold temperature (C)

%Tissue properties for the bioheat equation:
rho = 1.06; %Tissue density: g/cm^3
cp = 0.9; %Tissue specific heat: J/(kg C) -> cal/(g C)
k = 0.0012; %Thermal conductivity: W/(m C) -> cal/(cm s C)

%Metabolic heat is usually small compared with ablation heating.
metabolicHeat = 0; %cal/(cm^3 s)

%Blood properties for the perfusion term in the bioheat equation:
rhoBlood = 1.06; %Blood density: g/cm^3
cpBlood = 0.9; %Blood specific heat: J/(kg C) -> cal/(g C)
omegaBlood = 0.0017; %Blood perfusion rate: 1/s
bloodPerfusion = rhoBlood * cpBlood * omegaBlood; %cal/(cm^3 s C)

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
%Printing area stats:
fprintf('Area check: min %.3e, max %.3e, total %.3e cm^2. \n', min(area), max(area), sum(area));
%% Boundary labels

%esuelbc (aka element-to-element connectivity and boundary condition): 
%-1 = needle tip, apply constant hot-temperature boundary condition
%-2 = outer box, apply adiabatic/no-flux boundary condition
%-3 = needle body, apply adiabatic/no-flux boundary condition
isTip = any(esuelbc == -1, 2);
isOuterBox = any(esuelbc == -2, 2);
isNeedleBody = any(esuelbc == -3, 2);

%Creating a boundary code vector for visualization and later use in the solver, where:
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
xlabel('x (cm)');
ylabel('y (cm)');
%% Export results as VTK using dumpVTK for visualization in ParaView
outDir = fullfile(projectRoot, 'results');
if exist(outDir, 'dir') ~= 7
    mkdir(outDir);
end

vtkName = fullfile(outDir, sprintf('%s_%s_boundary_check.vtk', caseName, meshKind));
dumpVTK(vtkName, npoin, nelem, xyz, ele, boundaryCode, 'boundary_code');
fprintf('Wrote boundary-label VTK: %s\n', vtkName);
%% Thermal Solver Implementation

%Parameters for the thermal solver:
T = Tbody * ones(nelem, 1); %initial temperature vector for all cells
dt = 1e-5; %time step (s), small value for first explicit conduction test
nSteps = 1000; %number of time steps to simulate

%% Animation setup for visualizing temperature evolution in MATLAB
plotEvery = 50; %plot every N time steps
frameDir = fullfile(outDir, sprintf('%s_%s_temperature_frames', caseName, meshKind));

if exist(frameDir, 'dir') ~= 7
    mkdir(frameDir);
end

FrameId = 0; %initialize frame ID for animation
vtkFrameName = fullfile(frameDir, sprintf('%s_%s_temperature_frame_%04d.vtk', caseName, meshKind, FrameId));
dumpVTK(vtkFrameName, npoin, nelem, xyz, ele, T, 'temperature');
fprintf('Wrote initial temperature VTK: %s\n', vtkFrameName);
%% For loop: loops through 1) time steps, 2) elements, and 3) faces of each element to compute Tnew 
%based on conduction fluxes
for step = 1:nSteps

    T_new = T; %initialize new temperature vector for this time step

    for e = 1:nelem
        fluxSum = 0; %net heat flux into element e from all three faces

        for i = 1:3 %loop over the three faces of the triangle
            neighbor = esuelbc(e, i); %neighboring element index or boundary code

            if neighbor > 0 %interior neighbor

                %compute conduction flux between cell e and its neighbor 
                d = norm(centroid(neighbor, :) - centroid(e, :)); %distance between centroids
                flux = k * edgeLength(e, i) * (T(neighbor) - T(e)) / d; %Fourier's law for conduction
                fluxSum = fluxSum + flux; %accumulate flux for cell

            elseif neighbor == -1 %hot tip boundary condition
                %compute conduction flux between cell e and the hot tip (Ttip)
                nodes = ele(e, :);

                if i == 1
                    faceNodes = nodes([2 3]); %face opposite local node 1
                elseif i == 2
                    faceNodes = nodes([3 1]); %face opposite local node 2
                else
                    faceNodes = nodes([1 2]); %face opposite local node 3
                end
                
                %Calculate the midpoint of the face to estimate the distance to the hot tip boundary condition:
                faceMid = 0.5 * (xyz(faceNodes(1), :) + xyz(faceNodes(2), :));
                d = norm(faceMid - centroid(e, :)); %distance from cell center to boundary face
                flux = k * edgeLength(e, i) * (Ttip - T(e)) / d; %Fourier's law for conduction with hot tip temperature
                fluxSum = fluxSum + flux; %accumulate flux for cell

            elseif neighbor == -2 || neighbor == -3 %no-flux boundary condition
            end
        end

        %Bioheat source and sink terms:
        sourceTerm = metabolicHeat * area(e); %metabolic heat generation term, scaled by cell area
        sinkTerm = bloodPerfusion * area(e) * (T(e) - Tbody); %blood perfusion cooling term, scaled by cell area and temperature difference from body temp
        T_new(e) = T(e) + dt * (fluxSum + sourceTerm - sinkTerm) / (rho * cp * area(e)); %updating temp using Euler's method
    end
    T = T_new; %update temperature vector for the next time step

    if mod(step, plotEvery) == 0
        FrameId = FrameId + 1;
        vtkFrameName = fullfile(frameDir, sprintf('%s_%s_temperature_frame_%04d.vtk', caseName, meshKind, FrameId));
        dumpVTK(vtkFrameName, npoin, nelem, xyz, ele, T, 'temperature');
        fprintf('Wrote temperature VTK for step %d: %s\n', step, vtkFrameName);
    end 
end

%Print result updates to the console: 
fprintf('Completed %d time steps of the thermal ablation simulation.\n', nSteps);
fprintf('Final temperature range: min %.2f C, max %.2f C.\n', min(T), max(T));
%Printing the number of cells that are above the cell death threshold:
deadArea = sum(area(T >= Tdead)); %total area of cells above the cell death threshold
fprintf('Total area above cell death threshold (%.2f C): %.3e cm^2\n', Tdead, deadArea);

%% Export final temperature distribution as VTK for visualization in ParaView
vtkName = fullfile(outDir, sprintf('%s_%s_temperature.vtk', caseName, meshKind));
dumpVTK(vtkName, npoin, nelem, xyz, ele, T, 'temperature');
fprintf('Wrote final temperature VTK: %s\n', vtkName);
