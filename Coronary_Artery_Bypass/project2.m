elems = csvread('data/coronary.elems.csv'); 
pts = csvread('data/coronary.ptxyz.csv');
elrad = csvread('data/coronary.elrad.csv');
bcinl = csvread('data/coronary.bcinl.csv');
bcout = csvread('data/coronary.bcout.csv');
occlu = csvread('data/coronary.occlu.csv');
pts_rad = csvread('data/radius.csv');

nelem = length(elems);
npoin = length(pts);

% Length and radious
L = zeros(nelem,1);
r = zeros(nelem,1);
for e = 1:nelem
    i = elems(e,1);
    j = elems(e,2);
    L(e) = norm(pts(j,:) - pts(i,:));
    r(e) = elrad(e);
end

mu = 4.0;  % cPoise
Re = 8 * mu .* L ./ (pi * r.^4);

P_in  = 100;   % mmHg inlets
P_out = 10;    % mmHg outlets

A = zeros(npoin,npoin);
b = zeros(npoin,1);

% Caudal in nodes
for e = 1:nelem
    i = elems(e,1);
    j = elems(e,2);
    R = Re(e);

    A(i,i) = A(i,i) +  1/R;
    A(i,j) = A(i,j) + -1/R;
    A(j,j) = A(j,j) +  1/R;
    A(j,i) = A(j,i) + -1/R;
end

% Condiciones de contorno: todos los inlets a 100, todos los outlets a 10
fixed_nodes  = unique([bcinl(:); bcout(:)]);
fixed_values = zeros(size(fixed_nodes));
fixed_values(ismember(fixed_nodes, bcinl(:))) = P_in;  % primeros = inlets <- 
fixed_values(ismember(fixed_nodes, bcout(:))) = P_out;  % resto = outlets <- 

for idx = 1:numel(fixed_nodes)
    k = fixed_nodes(idx);
    A(k,:) = 0;
    A(k,k) = 1;
    b(k)   = fixed_values(idx);
end

% Resolver presiones
P = A\b;

% Caudales y presión media por rama
Q      = zeros(nelem,1);
P_elem = zeros(nelem,1);
for e = 1:nelem
    i = elems(e,1);
    j = elems(e,2);
    R = Re(e);
    Q(e)      = (P(i) - P(j)) / R;
    P_elem(e) = 0.5 * (P(i) + P(j));
end

% Exportar a VTK coloreado por presión de rama
writeNetworkVTK('coronary_pressure.vtk', pts, elems, P_elem);


function writeNetworkVTK(name, pts, elems, elemScalar)
    npoin = size(pts,1);
    nelem = size(elems,1);

    fp = fopen(name,'w');
    fprintf(fp,'# vtk DataFile Version 2.0\n');
    fprintf(fp,'Coronary network\nASCII\n');
    fprintf(fp,'DATASET POLYDATA\n');

    fprintf(fp,'POINTS %d float\n',npoin);
    fprintf(fp,'%e %e %e\n',pts');

    fprintf(fp,'LINES %d %d\n',nelem, nelem*3);
    for e = 1:nelem
        i = elems(e,1)-1;
        j = elems(e,2)-1;
        fprintf(fp,'2 %d %d\n', i, j);
    end

    fprintf(fp,'CELL_DATA %d\n',nelem);
    fprintf(fp,'SCALARS pressure float 1\n');
    fprintf(fp,'LOOKUP_TABLE default\n');
    fprintf(fp,'%e\n', elemScalar);
    fclose(fp);
end