function state = animodel(numsteps,state,esuel, ...
    A,p1,p2,p3, ...
    ele,xyz,fps,plotEvery, ...
    immune,chemo,pimmune, ...
    pchemo, vidName)

nelem = length(state); %# of surface elements

figure(1);
p = trisurf(ele, xyz(:,1), xyz(:,2), xyz(:,3));
daspect([1 1 1]); %fixing an aspect ratio
set(p,'FaceColor', 'flat','EdgeColor','white','FaceVertexCData',state);
colormap([0.85 0.85 0.85;  % 0 normal
    1.00 0.00 0.00;  % 1 cancer
    0.00 0.45 0.75;  % 2 complex
    0.10 0.10 0.10]);% 3 necrotic
clim([ 0 3]); %clim helps to lock the values into a valid state range
colorbar; 
title('Cancer Growth: t = 0');
drawnow;

%Writing the Video
video = VideoWriter(vidName,'MPEG-4');
video.FrameRate = fps;
open(video);

%Getting those initial frames as frame 0
writeVideo(video,getframe(gcf)); 

for t = 1:numsteps %every iteration is equal to one simulation step
    old_state = state; %updates are based on the old_state
    new_state = old_state;


    for i = 1:nelem
        nbr = esuel(i,:);
        local = [i,nbr];

        switch old_state(i) 

            case 0 %Normal Cell to Cancer Cell
                area_total = sum(A(local));
                area_cancer = sum(A(local(old_state(local)==1)));
                P = p1*(area_cancer/area_total); %probability
                if rand < P % rand generates a random # between 0 and 1
                    new_state(i) = 1;
                end
            case 1 %Cancer Cell to Complex Cell

                if chemo && (rand < pchemo) %Cancer to Necrotic 
                    new_state(i) = 3;
                    break
                end
                if immune && (rand < pimmune) %Cancer to Normal 
                    new_state(i) = 0;
                    break
                end

                if all(old_state(nbr) ~=0) %Cancer to Complex
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

    %Updating every iteration
    if mod(t,5)==0
           fprintf('Normal=%d Cancer=%d Complex=%d Necrotic=%d\n',nnz(state==0), nnz(state==1), nnz(state==2), nnz(state==3)); %Prints an update on the amount of normal, cancer, complex 
           % and nectrotic cells in the model
    end

    if mod(t,plotEvery)==0
        set(p,'FaceVertexCData', state); 
        title(sprintf('Model with t = %d of steps', t));
        camorbit(2,0,'data', [0 0 1]);
        drawnow;
        writeVideo(video,getframe(gcf))

    end
end
close(video);
fprintf('Saved the video sucessfullly: %s\n',vidName); % final print 
end
