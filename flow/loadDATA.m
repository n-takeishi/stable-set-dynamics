% code from http://www.dmdbook.com/

clearvars;

ny = 199;  % number of grid points in y-direction
nx = 449;  % number of grid points in x-direction

maxstep = 2000;
s = 600;

% create space for 150 snapshots
VORTALL = zeros(nx*ny,maxstep); % vorticity

% extract data from 150 snapshots files
for count=1:maxstep
    num = (count-1)*10+s; % load every 10th file    
    % load file
    fname = ['./ibpm_result/ibpm',num2str(num,'%05d'),'.plt'];       
    [X,Y,U,V,VORT] = loadIBPM(fname,nx,ny);
    VORTALL(:,count) = reshape(VORT,nx*ny,1); 
end

clear fname
save('ibpm_result.mat');