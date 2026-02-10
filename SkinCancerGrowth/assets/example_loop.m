function state = loopingcases(numsteps,state,esuel,A,p1,p2,p3)

nelem = length(state);

for t = 1:numsteps
    old_state = state;
    new_state = old_state;


    for i = 1:nelem
        nbr = esuel(i,:);
        local = [i,nbr];

        switch old_state(i)

            case 0 %Normal Cell to Cancer Cell
                area_total = sum(A(local));
                area_cancer = sum(A(local(old_state(local)==1)));
                P = p1*(area_cancer/area_total);
                if rand < P 
                    new_state(i) = 1;
                end
            case 1 %Cancer Cell to Complex Cell
                if all(old_state(nbr) ~=0)
                    if rand < p2
                        new_state(i) =2;
                    end
                end
            case 2 %Complex Cell to Necrotic Cell
                if rand < p3
                    new_state(i) =3;
                end
            case 3
                % A Necrotic Cell stays Necrotic
        end
    end
    
    state = new_state;

    if mod(t,5)==0
           fprintf('Normal=%d Cancer=%d Complex=%d Necrotic=%d\n',nnz(state==0), nnz(state==1), nnz(state==2), nnz(state==3)); % prints timestep & # of cancer cells 
    end
end
end
