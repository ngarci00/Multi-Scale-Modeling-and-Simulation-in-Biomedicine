%Make triangular model an save in VTK format
clear all; close all;
ele = csvread('data/body_elems.csv'); % elem connectivity
xyz = csvread('data/body_point.csv'); % point connectivity
esuel = csvread('data/body_esuel.csv'); % neighboring elem connectivity
nelem = length(ele); % # of elements
npoin = length(xyz);% # of points
state = zeros(nelem,1); % element colors = 0

%Computing geometry
A = zeros(nelem,1);
for e = 1:nelem
    n = ele(e,:); %node incdices of element e
    x1 = xyz(n(1),:); %coordinates of vertex 1
    x2 = xyz(n(2),:); %coordinates of vertex 2
    x3 = xyz(n(3),:); %coordinates of vertex 3
    A(e) = 0.5 * norm(cross(x2-x1, x3-x1)); %Triangle area, using cross product
end

%Picking a single element and setting it to cancer
ic = randi(nelem); % randi can return a random scalar int between 1 & imax (largest int in sample interval
state(ic) = 1; % 1 = cancer cell
%%Parameters/ Probabilities
%To see more/less agressive growth & diff:
p1 = 0.15; %Cancer
p2 = 0.25; %Complex
p3 = 0.01; %Necrotic

pimmune = 0.003; %Probability of immune response
pchemo = 0.005; %Probability of chemo response

%% Chemo & or Immune response:
immune = false;
chemo = false;
%Time step parameter
numsteps = 1000; %Time Step (days/months/years)

% Figure out how to add legend ??

%Adding an animation calling the function animodel here:
fps = 24; %num of frames 
plotEvery = 5; % every 2 timesteps
state = animodel(numsteps,state,esuel,A,p1,p2,p3,ele,xyz,fps,plotEvery,immune,chemo,pimmune,pchemo,'figs/CheckAnimation.mp4'); % function made to declured the code, 
% contains main loops for each case 
%% Saving to VTK file
name = sprintf('figs/CheckInput.vtk');
dumpVTK(name,npoin,nelem,xyz,ele,state); %save into .vtk format

%keeping the figure open
waitfor(gcf);