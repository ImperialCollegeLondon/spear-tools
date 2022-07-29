function out = SPEAR_read_target_data(filename)
% Main read function for head-tracking position data from SPEAR


% filename = '/Volumes/SPEAR_SSD3/SPEAR/InitialRelease/Extra/Train/Dataset_2/Reference_PosOri/Session_1/00/refOri_D2_S1_M00_ID4.csv';
A = readmatrix(filename);
A(1,:) = [];

out.t = A(:,2);
out.x = A(:,3);
out.y = A(:,4);
out.z = A(:,5);

end

