% code from http://www.dmdbook.com/

function [X,Y,U,V,VORT] = loadIBPM(fname,nx,ny)

fileID = fopen(fname);  % open file
% remove first 6 lines of text in file
TEXT = textscan(fileID,'%s',6,'delimiter',char(460)); 

% pull out all data
FDATA = fscanf(fileID,'%f',[5,nx*ny]);
X = reshape(FDATA(1,:),nx,ny)';    % x positions
Y = reshape(FDATA(2,:),nx,ny)';    % y positions
U = reshape(FDATA(3,:),nx,ny)';    % u velocity
V = reshape(FDATA(4,:),nx,ny)';    % v velocity
VORT = reshape(FDATA(5,:),nx,ny)'; % vorticity
fclose all;